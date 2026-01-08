"""VoVNet policy wrapper for cost-aware vision calling."""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.base_vlm import BaseVLM, VLMOutputs
from src.models.uncertainty import entropy_from_logits
from src.models.vision_budget import VisionBudgetController


class Action(IntEnum):
    """Vision action choices."""

    NO_VISION = 0
    COARSE_VISION = 1
    FULL_VISION = 2


@dataclass
class VoVNetOutput:
    """Structured outputs for training and evaluation."""

    logits: Tensor
    action_logits: Tensor
    action_probs: Tensor
    actions: Tensor
    expected_cost: Tensor
    uncertainty: Tensor
    vision_tokens: Tensor


class VoVNet(nn.Module):
    """Value-of-Vision network with cost-aware action policy."""

    def __init__(
        self,
        base_vlm: BaseVLM,
        vision_budget: VisionBudgetController,
        vow_hidden_dim: int = 256,
        gumbel_tau: float = 1.0,
        use_straight_through: bool = True,
        eval_sample: bool = False,
        cost_c1: float = 1.0,
        cost_c2: float = 4.0,
        full_vlm: Optional[BaseVLM] = None,
    ) -> None:
        super().__init__()
        self.base_vlm = base_vlm
        self.full_vlm = full_vlm
        self.vision_budget = vision_budget
        self.gumbel_tau = gumbel_tau
        self.use_straight_through = use_straight_through
        self.eval_sample = eval_sample
        self.register_buffer("costs", torch.tensor([0.0, cost_c1, cost_c2]))

        hidden_size = _infer_hidden_size(base_vlm)
        self.vow_head = nn.Sequential(
            nn.Linear(hidden_size, vow_hidden_dim),
            nn.ReLU(),
            nn.Linear(vow_hidden_dim, vow_hidden_dim),
        )
        self.policy = nn.Linear(vow_hidden_dim, len(Action))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Optional[List[Image.Image]] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Run text-first pass, choose action, and optionally call vision."""
        text_outputs, action_logits = self.text_first(input_ids, attention_mask)
        action_probs, actions = self._select_actions(action_logits)

        uncertainty = self._compute_uncertainty(text_outputs)
        expected_cost = (action_probs * self.costs.to(action_probs.device)).sum(dim=-1)

        if self.training and not self.use_straight_through:
            logits, vision_tokens = self._forward_soft_mixture(
                input_ids,
                attention_mask,
                images,
                text_outputs,
                action_probs,
            )
        else:
            logits, vision_tokens = self._forward_hard_actions(
                input_ids,
                attention_mask,
                images,
                text_outputs,
                actions,
            )

        return {
            "logits": logits,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "actions": actions,
            "expected_cost": expected_cost,
            "uncertainty": uncertainty,
            "vision_tokens": vision_tokens,
            "labels": labels,
        }

    def text_first(self, input_ids: Tensor, attention_mask: Tensor) -> Tuple[VLMOutputs, Tensor]:
        """Run text-only forward pass and return action logits."""
        text_outputs = self.base_vlm.encode_text(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )
        pooled = _pool_hidden(text_outputs.hidden_states[-1], attention_mask)
        vow_features = self.vow_head(pooled)
        action_logits = self.policy(vow_features)
        return text_outputs, action_logits

    def forward_with_actions(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Optional[List[Image.Image]],
        actions: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass with externally provided actions (baselines)."""
        text_outputs, _ = self.text_first(input_ids, attention_mask)
        logits, vision_tokens = self._forward_hard_actions(
            input_ids, attention_mask, images, text_outputs, actions
        )
        action_probs = F.one_hot(actions, num_classes=len(Action)).float()
        expected_cost = (action_probs * self.costs.to(action_probs.device)).sum(dim=-1)
        uncertainty = self._compute_uncertainty(text_outputs)
        return {
            "logits": logits,
            "action_logits": torch.zeros_like(action_probs),
            "action_probs": action_probs,
            "actions": actions,
            "expected_cost": expected_cost,
            "uncertainty": uncertainty,
            "vision_tokens": vision_tokens,
            "labels": labels,
        }

    def _select_actions(self, action_logits: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            if self.use_straight_through:
                probs = F.gumbel_softmax(
                    action_logits, tau=self.gumbel_tau, hard=True
                )
                actions = probs.argmax(dim=-1)
            else:
                probs = F.softmax(action_logits, dim=-1)
                actions = probs.argmax(dim=-1)
        else:
            probs = F.softmax(action_logits, dim=-1)
            if self.eval_sample:
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                actions = probs.argmax(dim=-1)
        return probs, actions

    def _compute_uncertainty(self, outputs: VLMOutputs) -> Tensor:
        logits = outputs.logits[:, -1]
        return entropy_from_logits(logits, dim=-1)

    def _forward_hard_actions(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Optional[List[Image.Image]],
        text_outputs: VLMOutputs,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if images is None:
            return text_outputs.logits, torch.zeros_like(actions, dtype=torch.float)

        unique_actions = actions.unique().tolist()
        if len(unique_actions) == 1 and unique_actions[0] != Action.NO_VISION:
            mode = "coarse" if unique_actions[0] == Action.COARSE_VISION else "full"
            return self._forward_mode(
                input_ids,
                attention_mask,
                images,
                text_outputs,
                mode=mode,
                past_key_values=text_outputs.past_key_values,
            )

        logits_list: List[Tensor] = []
        token_counts: List[int] = []
        for idx, action in enumerate(actions.tolist()):
            if action == Action.NO_VISION:
                logits_list.append(text_outputs.logits[idx : idx + 1])
                token_counts.append(0)
                continue

            mode = "coarse" if action == Action.COARSE_VISION else "full"
            image = self.vision_budget.prepare_image(images[idx], mode)
            pixel_values = self._prepare_pixel_values([image])
            model = self.full_vlm if (action == Action.FULL_VISION and self.full_vlm) else self.base_vlm

            outputs = model.forward_with_vision(
                input_ids=input_ids[idx : idx + 1],
                attention_mask=attention_mask[idx : idx + 1],
                pixel_values=pixel_values,
                past_key_values=None,
            )
            logits_list.append(outputs.logits)
            token_counts.append(self.vision_budget.estimate_visual_tokens(image))

        logits = torch.cat(logits_list, dim=0)
        vision_tokens = torch.tensor(token_counts, device=logits.device, dtype=torch.float)
        return logits, vision_tokens

    def _forward_soft_mixture(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Optional[List[Image.Image]],
        text_outputs: VLMOutputs,
        action_probs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if images is None:
            return text_outputs.logits, torch.zeros(action_probs.shape[0], device=action_probs.device)

        coarse_logits, coarse_tokens = self._forward_mode(
            input_ids, attention_mask, images, text_outputs, mode="coarse"
        )
        full_logits, full_tokens = self._forward_mode(
            input_ids, attention_mask, images, text_outputs, mode="full"
        )

        p0 = action_probs[:, Action.NO_VISION].view(-1, 1, 1)
        p1 = action_probs[:, Action.COARSE_VISION].view(-1, 1, 1)
        p2 = action_probs[:, Action.FULL_VISION].view(-1, 1, 1)
        logits = p0 * text_outputs.logits + p1 * coarse_logits + p2 * full_logits

        expected_tokens = (
            action_probs[:, Action.COARSE_VISION] * coarse_tokens
            + action_probs[:, Action.FULL_VISION] * full_tokens
        )
        return logits, expected_tokens

    def _forward_mode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: List[Image.Image],
        text_outputs: VLMOutputs,
        mode: str,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[Tensor, Tensor]:
        processed = [self.vision_budget.prepare_image(img, mode) for img in images]
        pixel_values = self._prepare_pixel_values(processed)
        model = self.full_vlm if (mode == "full" and self.full_vlm) else self.base_vlm
        # TODO: If full_vlm uses a different processor, update _prepare_pixel_values to select it.
        if past_key_values is None:
            past_key_values = text_outputs.past_key_values
        outputs = model.forward_with_vision(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
        )
        tokens = torch.tensor(
            [self.vision_budget.estimate_visual_tokens(img) for img in processed],
            device=outputs.logits.device,
            dtype=torch.float,
        )
        return outputs.logits, tokens

    def _prepare_pixel_values(self, images: List[Image.Image]) -> Optional[Tensor]:
        processor = self.base_vlm.processor
        if processor is not None:
            outputs = processor(images=images, return_tensors="pt")
            return outputs.get("pixel_values")

        arrays = [np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0 for img in images]
        tensors = [torch.from_numpy(arr).permute(2, 0, 1) for arr in arrays]
        return torch.stack(tensors, dim=0)


def _pool_hidden(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool hidden states using attention mask."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def _infer_hidden_size(base_vlm: BaseVLM) -> int:
    config = getattr(base_vlm.model, "config", None)
    for attr in ("hidden_size", "d_model"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return int(getattr(text_cfg, "hidden_size"))
    raise ValueError("Unable to infer hidden size from model config")
