"""VoVNet policy wrapper for cost-aware vision calling."""
from __future__ import annotations

import math
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
from src.models.uncertainty import entropy_from_logits, margin_from_logits
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
    token_count_coarse: Tensor
    token_count_full: Tensor
    gain_pred: Optional[Tensor] = None
    gain_true: Optional[Tensor] = None
    actions_raw: Optional[Tensor] = None
    fallback_mask: Optional[Tensor] = None
    fallback_entropy_trigger: Optional[Tensor] = None
    fallback_margin_trigger: Optional[Tensor] = None
    margin: Optional[Tensor] = None
    text_logits: Optional[Tensor] = None


@dataclass
class VisionInputs:
    """Prepared vision tensors and token counts."""

    pixel_values: Optional[Tensor]
    token_counts: Tensor


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
        policy_mode: str = "logits",
        fallback_mode: str = "none",
        fallback_entropy_threshold: Optional[float] = None,
        fallback_margin_threshold: Optional[float] = None,
        cost_scale: float = 1.0,
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
        if policy_mode not in {"logits", "gain"}:
            raise ValueError("policy_mode must be logits or gain")
        self.policy_mode = policy_mode
        if fallback_mode not in {"none", "coarse", "full"}:
            raise ValueError("fallback_mode must be none, coarse, or full")
        self.fallback_mode = fallback_mode
        self.fallback_entropy_threshold = fallback_entropy_threshold
        self.fallback_margin_threshold = fallback_margin_threshold
        self.cost_scale = cost_scale
        self.cost_c1 = cost_c1
        self.cost_c2 = cost_c2

        hidden_size = _infer_hidden_size(base_vlm)
        self.vow_head = nn.Sequential(
            nn.Linear(hidden_size, vow_hidden_dim),
            nn.ReLU(),
            nn.Linear(vow_hidden_dim, vow_hidden_dim),
        )
        self.policy = nn.Linear(vow_hidden_dim, len(Action))
        self.gain_head = nn.Linear(vow_hidden_dim, 2)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Optional[List[Image.Image]] = None,
        labels: Optional[Tensor] = None,
        compute_gain: bool = False,
    ) -> Dict[str, Any]:
        """Run text-first pass, choose action, and optionally call vision."""
        text_outputs, action_logits, gain_pred = self.text_first(input_ids, attention_mask)
        action_probs, actions = self._select_actions(action_logits)

        token_count_coarse, token_count_full, coarse_inputs, full_inputs = (
            self._prepare_token_counts(
                images=images,
                device=action_probs.device,
                batch_size=action_probs.shape[0],
            )
        )
        expected_cost = self._compute_expected_cost(
            action_probs, token_count_coarse, token_count_full
        )
        uncertainty = self._compute_uncertainty(text_outputs)
        margin = self._compute_margin(text_outputs)
        actions_raw = actions
        fallback_mask = None
        fallback_entropy_trigger = None
        fallback_margin_trigger = None
        if not self.training and images is not None:
            actions, fallback_mask, fallback_entropy_trigger, fallback_margin_trigger = (
                self._apply_fallback(actions, uncertainty, margin)
            )
        elif not self.training:
            zeros = torch.zeros_like(actions, dtype=torch.bool)
            fallback_mask = zeros
            fallback_entropy_trigger = zeros
            fallback_margin_trigger = zeros
        gain_true = None
        if compute_gain and labels is not None:
            gain_true = self._compute_gain_targets(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                labels=labels,
                text_outputs=text_outputs,
                coarse_inputs=coarse_inputs,
                full_inputs=full_inputs,
            )

        if self.training and not self.use_straight_through:
            logits, vision_tokens = self._forward_soft_mixture(
                input_ids,
                attention_mask,
                images,
                text_outputs,
                action_probs,
                coarse_inputs=coarse_inputs,
                full_inputs=full_inputs,
            )
        else:
            logits, vision_tokens = self._forward_hard_actions(
                input_ids,
                attention_mask,
                images,
                text_outputs,
                actions,
                coarse_inputs=coarse_inputs,
                full_inputs=full_inputs,
            )

        return {
            "logits": logits,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "actions": actions,
            "expected_cost": expected_cost,
            "uncertainty": uncertainty,
            "vision_tokens": vision_tokens,
            "token_count_coarse": token_count_coarse,
            "token_count_full": token_count_full,
            "gain_pred": gain_pred,
            "gain_true": gain_true,
            "actions_raw": actions_raw,
            "fallback_mask": fallback_mask,
            "fallback_entropy_trigger": fallback_entropy_trigger,
            "fallback_margin_trigger": fallback_margin_trigger,
            "margin": margin,
            "text_logits": text_outputs.logits if not self.training else None,
            "labels": labels,
        }

    def text_first(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> Tuple[VLMOutputs, Tensor, Tensor]:
        """Run text-only forward pass and return action logits plus gain prediction."""
        text_outputs = self.base_vlm.encode_text(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )
        pooled = _pool_hidden(text_outputs.hidden_states[-1], attention_mask)
        vow_features = self.vow_head(pooled)
        gain_pred = self.gain_head(vow_features)
        if self.policy_mode == "gain":
            zeros = torch.zeros(
                gain_pred.size(0), 1, device=gain_pred.device, dtype=gain_pred.dtype
            )
            action_logits = torch.cat([zeros, gain_pred], dim=-1)
        else:
            action_logits = self.policy(vow_features)
        return text_outputs, action_logits, gain_pred

    def forward_with_actions(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Optional[List[Image.Image]],
        actions: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass with externally provided actions (baselines)."""
        text_outputs, _, gain_pred = self.text_first(input_ids, attention_mask)
        token_count_coarse, token_count_full, coarse_inputs, full_inputs = (
            self._prepare_token_counts(
                images=images,
                device=input_ids.device,
                batch_size=input_ids.shape[0],
            )
        )
        logits, vision_tokens = self._forward_hard_actions(
            input_ids,
            attention_mask,
            images,
            text_outputs,
            actions,
            coarse_inputs=coarse_inputs,
            full_inputs=full_inputs,
        )
        action_probs = F.one_hot(actions, num_classes=len(Action)).float()
        expected_cost = self._compute_expected_cost(
            action_probs, token_count_coarse, token_count_full
        )
        uncertainty = self._compute_uncertainty(text_outputs)
        margin = self._compute_margin(text_outputs)
        return {
            "logits": logits,
            "action_logits": torch.zeros_like(action_probs),
            "action_probs": action_probs,
            "actions": actions,
            "expected_cost": expected_cost,
            "uncertainty": uncertainty,
            "vision_tokens": vision_tokens,
            "token_count_coarse": token_count_coarse,
            "token_count_full": token_count_full,
            "gain_pred": gain_pred,
            "gain_true": None,
            "actions_raw": actions,
            "fallback_mask": None,
            "fallback_entropy_trigger": None,
            "fallback_margin_trigger": None,
            "margin": margin,
            "text_logits": text_outputs.logits if not self.training else None,
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

    def _compute_margin(self, outputs: VLMOutputs) -> Tensor:
        logits = outputs.logits[:, -1]
        return margin_from_logits(logits, dim=-1)

    def _apply_fallback(
        self, actions: Tensor, uncertainty: Tensor, margin: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.fallback_mode == "none":
            zeros = torch.zeros_like(actions, dtype=torch.bool)
            return actions, zeros, zeros, zeros

        entropy_trigger = torch.zeros_like(actions, dtype=torch.bool)
        margin_trigger = torch.zeros_like(actions, dtype=torch.bool)

        if self.fallback_entropy_threshold is not None:
            entropy_trigger = uncertainty > float(self.fallback_entropy_threshold)
        if self.fallback_margin_threshold is not None:
            margin_trigger = margin < float(self.fallback_margin_threshold)

        trigger = entropy_trigger | margin_trigger
        fallback_mask = (actions == Action.NO_VISION) & trigger
        if not fallback_mask.any():
            return actions, fallback_mask, entropy_trigger, margin_trigger

        updated = actions.clone()
        if self.fallback_mode == "coarse":
            updated[fallback_mask] = Action.COARSE_VISION
        else:
            updated[fallback_mask] = Action.FULL_VISION
        return updated, fallback_mask, entropy_trigger, margin_trigger

    def _compute_gain_targets(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Optional[List[Image.Image]],
        labels: Tensor,
        text_outputs: VLMOutputs,
        coarse_inputs: Optional[VisionInputs],
        full_inputs: Optional[VisionInputs],
    ) -> Tensor:
        if images is None:
            loss_no = self._per_sample_loss(text_outputs.logits.detach(), labels)
            zeros = torch.zeros_like(loss_no)
            return torch.stack([zeros, zeros], dim=-1)

        if coarse_inputs is None or full_inputs is None:
            _, _, coarse_inputs, full_inputs = self._prepare_token_counts(
                images=images,
                device=input_ids.device,
                batch_size=input_ids.shape[0],
            )

        with torch.no_grad():
            loss_no = self._per_sample_loss(text_outputs.logits.detach(), labels)
            coarse_logits, _ = self._forward_mode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                text_outputs=text_outputs,
                mode="coarse",
                past_key_values=text_outputs.past_key_values,
                vision_inputs=coarse_inputs,
            )
            full_logits, _ = self._forward_mode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                text_outputs=text_outputs,
                mode="full",
                past_key_values=text_outputs.past_key_values,
                vision_inputs=full_inputs,
            )
            loss_coarse = self._per_sample_loss(coarse_logits, labels)
            loss_full = self._per_sample_loss(full_logits, labels)

        gain_coarse = loss_no - loss_coarse
        gain_full = loss_no - loss_full
        return torch.stack([gain_coarse, gain_full], dim=-1)

    @staticmethod
    def _per_sample_loss(logits: Tensor, labels: Tensor) -> Tensor:
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        loss = loss.view(labels.shape)
        mask = labels.ne(-100).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (loss * mask).sum(dim=1) / denom

    def _prepare_token_counts(
        self,
        images: Optional[List[Image.Image]],
        device: torch.device,
        batch_size: int,
    ) -> Tuple[Tensor, Tensor, Optional[VisionInputs], Optional[VisionInputs]]:
        if images is None:
            zeros = torch.zeros(batch_size, device=device)
            return zeros, zeros, None, None

        coarse_inputs = self._prepare_vision_inputs(
            images=images, mode="coarse", model=self.base_vlm
        )
        full_model = self.full_vlm if self.full_vlm is not None else self.base_vlm
        full_inputs = self._prepare_vision_inputs(
            images=images, mode="full", model=full_model
        )
        token_count_coarse = coarse_inputs.token_counts.to(device).float()
        token_count_full = full_inputs.token_counts.to(device).float()
        return token_count_coarse, token_count_full, coarse_inputs, full_inputs

    def _compute_expected_cost(
        self,
        action_probs: Tensor,
        token_count_coarse: Tensor,
        token_count_full: Tensor,
    ) -> Tensor:
        expected_tokens = (
            action_probs[:, Action.COARSE_VISION] * token_count_coarse
            + action_probs[:, Action.FULL_VISION] * token_count_full
        )
        scale = action_probs.new_tensor(self.cost_scale)
        return expected_tokens * scale

    def _prepare_vision_inputs(
        self,
        images: List[Image.Image],
        mode: str,
        model: BaseVLM,
    ) -> VisionInputs:
        processed = [self.vision_budget.prepare_image(img, mode) for img in images]
        pixel_values, token_counts = self._prepare_pixel_values(processed, model=model)
        return VisionInputs(pixel_values=pixel_values, token_counts=token_counts)

    def _forward_hard_actions(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Optional[List[Image.Image]],
        text_outputs: VLMOutputs,
        actions: Tensor,
        coarse_inputs: Optional[VisionInputs] = None,
        full_inputs: Optional[VisionInputs] = None,
    ) -> Tuple[Tensor, Tensor]:
        if images is None:
            return text_outputs.logits, torch.zeros_like(actions, dtype=torch.float)

        if coarse_inputs is None or full_inputs is None:
            _, _, coarse_inputs, full_inputs = self._prepare_token_counts(
                images=images,
                device=input_ids.device,
                batch_size=input_ids.shape[0],
            )

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
                vision_inputs=coarse_inputs if mode == "coarse" else full_inputs,
            )

        logits_list: List[Tensor] = []
        token_counts: List[int] = []
        coarse_pixel_values = (
            coarse_inputs.pixel_values if coarse_inputs is not None else None
        )
        full_pixel_values = full_inputs.pixel_values if full_inputs is not None else None
        coarse_tokens = coarse_inputs.token_counts if coarse_inputs is not None else None
        full_tokens = full_inputs.token_counts if full_inputs is not None else None
        for idx, action in enumerate(actions.tolist()):
            if action == Action.NO_VISION:
                logits_list.append(text_outputs.logits[idx : idx + 1])
                token_counts.append(0)
                continue

            if action == Action.COARSE_VISION:
                pixel_values = (
                    coarse_pixel_values[idx : idx + 1]
                    if coarse_pixel_values is not None
                    else None
                )
                count = (
                    int(coarse_tokens[idx]) if coarse_tokens is not None else 0
                )
                model = self.base_vlm
            else:
                pixel_values = (
                    full_pixel_values[idx : idx + 1]
                    if full_pixel_values is not None
                    else None
                )
                count = int(full_tokens[idx]) if full_tokens is not None else 0
                model = (
                    self.full_vlm if self.full_vlm is not None else self.base_vlm
                )

            outputs = model.forward_with_vision(
                input_ids=input_ids[idx : idx + 1],
                attention_mask=attention_mask[idx : idx + 1],
                pixel_values=pixel_values,
                past_key_values=None,
            )
            logits_list.append(outputs.logits)
            token_counts.append(count)

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
        coarse_inputs: Optional[VisionInputs] = None,
        full_inputs: Optional[VisionInputs] = None,
    ) -> Tuple[Tensor, Tensor]:
        if images is None:
            return text_outputs.logits, torch.zeros(action_probs.shape[0], device=action_probs.device)

        coarse_logits, coarse_tokens = self._forward_mode(
            input_ids,
            attention_mask,
            images,
            text_outputs,
            mode="coarse",
            vision_inputs=coarse_inputs,
        )
        full_logits, full_tokens = self._forward_mode(
            input_ids,
            attention_mask,
            images,
            text_outputs,
            mode="full",
            vision_inputs=full_inputs,
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
        vision_inputs: Optional[VisionInputs] = None,
    ) -> Tuple[Tensor, Tensor]:
        model = self.full_vlm if (mode == "full" and self.full_vlm) else self.base_vlm
        if vision_inputs is None:
            vision_inputs = self._prepare_vision_inputs(
                images=images, mode=mode, model=model
            )
        pixel_values = vision_inputs.pixel_values
        if past_key_values is None:
            past_key_values = text_outputs.past_key_values
        outputs = model.forward_with_vision(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
        )
        tokens = vision_inputs.token_counts.to(outputs.logits.device).float()
        return outputs.logits, tokens

    def _prepare_pixel_values(
        self, images: List[Image.Image], model: BaseVLM
    ) -> Tuple[Optional[Tensor], Tensor]:
        processor = model.processor
        if processor is not None:
            outputs = None
            try:
                outputs = processor(images=images, return_tensors="pt")
            except Exception:
                image_processor = getattr(processor, "image_processor", None) or getattr(
                    processor, "vision_processor", None
                )
                if image_processor is not None:
                    outputs = image_processor(images=images, return_tensors="pt")
                else:
                    try:
                        outputs = processor(
                            text=[""] * len(images), images=images, return_tensors="pt"
                        )
                    except Exception:
                        outputs = None

            if outputs is not None:
                pixel_values = (
                    outputs.get("pixel_values")
                    if hasattr(outputs, "get")
                    else getattr(outputs, "pixel_values", None)
                )
                if pixel_values is not None:
                    token_counts = self._infer_token_counts(
                        outputs=outputs,
                        pixel_values=pixel_values,
                        batch_size=len(images),
                        model=model,
                    )
                    return pixel_values, token_counts

        arrays = [np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0 for img in images]
        tensors = [torch.from_numpy(arr).permute(2, 0, 1) for arr in arrays]
        pixel_values = torch.stack(tensors, dim=0)
        token_counts = self._estimate_tokens_from_pixel_values(pixel_values, model=model)
        return pixel_values, token_counts

    def _infer_token_counts(
        self,
        outputs: Any,
        pixel_values: Optional[Tensor],
        batch_size: int,
        model: BaseVLM,
    ) -> Tensor:
        grid = None
        if outputs is not None and hasattr(outputs, "get"):
            grid = outputs.get("image_grid_thw")
        if grid is not None:
            grid_tensor = torch.as_tensor(grid)
            if grid_tensor.ndim == 1:
                grid_tensor = grid_tensor.unsqueeze(0)
            if grid_tensor.shape[-1] == 3:
                return grid_tensor.long().prod(dim=-1)
        if pixel_values is not None:
            return self._estimate_tokens_from_pixel_values(pixel_values, model=model)
        return torch.zeros(batch_size, dtype=torch.long)

    def _estimate_tokens_from_pixel_values(
        self, pixel_values: Tensor, model: BaseVLM
    ) -> Tensor:
        patch_size = self._get_patch_size(model) or self.vision_budget.patch_size
        patch_size = max(1, int(patch_size))
        if pixel_values.dim() == 3:
            batch = pixel_values.shape[0]
            tokens = int(pixel_values.shape[1])
            return torch.full((batch,), tokens, dtype=torch.long)
        if pixel_values.dim() == 4:
            batch, _, height, width = pixel_values.shape
            grid_h = math.ceil(height / patch_size)
            grid_w = math.ceil(width / patch_size)
            tokens = int(grid_h * grid_w)
            return torch.full((batch,), tokens, dtype=torch.long)
        if pixel_values.dim() == 5:
            batch, frames, _, height, width = pixel_values.shape
            grid_h = math.ceil(height / patch_size)
            grid_w = math.ceil(width / patch_size)
            tokens = int(frames * grid_h * grid_w)
            return torch.full((batch,), tokens, dtype=torch.long)
        return torch.zeros(pixel_values.shape[0], dtype=torch.long)

    def _get_patch_size(self, model: BaseVLM) -> Optional[int]:
        config = getattr(model.model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        for cfg in (vision_config, config):
            if cfg is None:
                continue
            patch_size = getattr(cfg, "patch_size", None)
            if patch_size is not None:
                return int(patch_size)
        return None


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
