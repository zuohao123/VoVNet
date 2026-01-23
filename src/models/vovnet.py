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

from src.models.base_vlm import BaseVLM, VLMOutputs, VisionPruningSpec
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
    image_grid_thw: Optional[Tensor] = None
    input_ids: Optional[Tensor] = None
    attention_mask: Optional[Tensor] = None
    labels: Optional[Tensor] = None


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

    def _get_image_token_id(self) -> Optional[int]:
        tokenizer = self.base_vlm.tokenizer
        if tokenizer is None:
            return None
        image_token_id = getattr(tokenizer, "image_token_id", None)
        if image_token_id is not None:
            return int(image_token_id)
        image_token = getattr(tokenizer, "image_token", None)
        if image_token:
            token_id = tokenizer.convert_tokens_to_ids(image_token)
            if token_id is not None:
                return int(token_id)
        for token in getattr(tokenizer, "additional_special_tokens", []) or []:
            if "image" in token.lower():
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None:
                    return int(token_id)
        return None

    def _get_vision_start_end_ids(self) -> Tuple[Optional[int], Optional[int]]:
        tokenizer = self.base_vlm.tokenizer
        if tokenizer is None:
            return None, None
        start_id = None
        end_id = None
        for token in getattr(tokenizer, "additional_special_tokens", []) or []:
            lowered = token.lower()
            if "vision_start" in lowered:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None:
                    start_id = int(token_id)
            elif "vision_end" in lowered:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None:
                    end_id = int(token_id)
        if start_id is None:
            token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            if token_id is not None:
                start_id = int(token_id)
        if end_id is None:
            token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
            if token_id is not None:
                end_id = int(token_id)
        return start_id, end_id

    def _strip_image_tokens(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor],
        image_token_id: Optional[int],
    ) -> Tuple[List[List[int]], Optional[List[List[int]]]]:
        vision_start_id, vision_end_id = self._get_vision_start_end_ids()
        token_lists: List[List[int]] = []
        label_lists: Optional[List[List[int]]] = [] if labels is not None else None
        for i in range(input_ids.size(0)):
            ids = input_ids[i][attention_mask[i].bool()].tolist()
            lbls = (
                labels[i][attention_mask[i].bool()].tolist() if labels is not None else None
            )
            new_ids: List[int] = []
            new_lbls: Optional[List[int]] = [] if labels is not None else None
            for idx, tok in enumerate(ids):
                if image_token_id is not None and tok == image_token_id:
                    continue
                if vision_start_id is not None and tok == vision_start_id:
                    continue
                if vision_end_id is not None and tok == vision_end_id:
                    continue
                new_ids.append(tok)
                if new_lbls is not None and lbls is not None:
                    new_lbls.append(lbls[idx])
            token_lists.append(new_ids)
            if label_lists is not None:
                label_lists.append(new_lbls or [])
        return token_lists, label_lists

    def _insert_image_tokens(
        self,
        tokens: List[int],
        labels: Optional[List[int]],
        image_token_id: int,
        token_count: int,
        bos_token_id: Optional[int],
    ) -> Tuple[List[int], Optional[List[int]]]:
        if token_count <= 0:
            return tokens, labels
        vision_start_id, vision_end_id = self._get_vision_start_end_ids()
        insert_pos = 0
        if bos_token_id is not None and tokens and tokens[0] == bos_token_id:
            insert_pos = 1
        image_block = [image_token_id] * token_count
        if vision_start_id is not None and vision_end_id is not None:
            image_block = [vision_start_id] + image_block + [vision_end_id]
        new_tokens = tokens[:insert_pos] + image_block + tokens[insert_pos:]
        if labels is None:
            return new_tokens, None
        new_labels = labels[:insert_pos] + [-100] * len(image_block) + labels[insert_pos:]
        return new_tokens, new_labels

    def _pad_sequences(
        self,
        sequences: List[List[int]],
        pad_value: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        batch = len(sequences)
        padded = torch.full((batch, max_len), pad_value, device=device, dtype=dtype)
        attention = torch.zeros((batch, max_len), device=device, dtype=torch.long)
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            length = min(len(seq), max_len)
            padded[i, :length] = torch.tensor(seq[:length], device=device, dtype=dtype)
            attention[i, :length] = 1
        return padded, attention

    def _pad_labels(
        self,
        sequences: Optional[List[List[int]]],
        pad_value: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if sequences is None:
            return None
        batch = len(sequences)
        padded = torch.full((batch, max_len), pad_value, device=device, dtype=dtype)
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            length = min(len(seq), max_len)
            padded[i, :length] = torch.tensor(seq[:length], device=device, dtype=dtype)
        return padded

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Optional[List[Image.Image]] = None,
        labels: Optional[Tensor] = None,
        compute_gain: bool = False,
    ) -> Dict[str, Any]:
        """Run text-first pass, choose action, and optionally call vision."""
        if self.training and not self.use_straight_through and images is not None:
            if self._get_image_token_id() is not None:
                raise RuntimeError(
                    "Soft mixture is not supported with image token expansion; "
                    "enable use_straight_through."
                )

        token_count_coarse, token_count_full, coarse_inputs, full_inputs = (
            self._prepare_token_counts(
                images=images,
                device=attention_mask.device,
                batch_size=attention_mask.shape[0],
            )
            if images is not None
            else (torch.zeros_like(attention_mask[:, 0]), torch.zeros_like(attention_mask[:, 0]), None, None)
        )

        text_input_ids, text_attention_mask, text_labels = self._prepare_text_and_vision_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            coarse_inputs=coarse_inputs,
            full_inputs=full_inputs,
        )

        text_outputs, action_logits, gain_pred = self.text_first(
            text_input_ids, text_attention_mask
        )
        action_probs, actions = self._select_actions(action_logits)

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
        if compute_gain and text_labels is not None:
            gain_true = self._compute_gain_targets(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                images=images,
                labels=text_labels,
                text_outputs=text_outputs,
                coarse_inputs=coarse_inputs,
                full_inputs=full_inputs,
            )

        labels_for_loss = text_labels
        if self.training and not self.use_straight_through:
            logits, vision_tokens = self._forward_soft_mixture(
                text_input_ids,
                text_attention_mask,
                images,
                text_outputs,
                action_probs,
                coarse_inputs=coarse_inputs,
                full_inputs=full_inputs,
            )
        else:
            logits, vision_tokens, labels_for_loss = self._forward_hard_actions(
                text_input_ids,
                text_attention_mask,
                images,
                text_outputs,
                actions,
                text_labels=text_labels,
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
            "labels": labels_for_loss if labels is not None else None,
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
        token_count_coarse, token_count_full, coarse_inputs, full_inputs = (
            self._prepare_token_counts(
                images=images,
                device=attention_mask.device,
                batch_size=attention_mask.shape[0],
            )
            if images is not None
            else (torch.zeros_like(attention_mask[:, 0]), torch.zeros_like(attention_mask[:, 0]), None, None)
        )
        text_input_ids, text_attention_mask, text_labels = self._prepare_text_and_vision_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            coarse_inputs=coarse_inputs,
            full_inputs=full_inputs,
        )
        text_outputs, _, gain_pred = self.text_first(text_input_ids, text_attention_mask)
        logits, vision_tokens, labels_for_loss = self._forward_hard_actions(
            text_input_ids,
            text_attention_mask,
            images,
            text_outputs,
            actions,
            text_labels=text_labels,
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
            "labels": labels_for_loss if labels is not None else None,
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
            coarse_labels = (
                coarse_inputs.labels
                if coarse_inputs is not None and coarse_inputs.labels is not None
                else labels
            )
            full_labels = (
                full_inputs.labels
                if full_inputs is not None and full_inputs.labels is not None
                else labels
            )
            loss_coarse = self._per_sample_loss(coarse_logits, coarse_labels)
            loss_full = self._per_sample_loss(full_logits, full_labels)

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
        token_count_coarse = self._vision_token_counts(coarse_inputs, model=self.base_vlm).to(
            device
        ).float()
        token_count_full = self._vision_token_counts(full_inputs, model=full_model).to(
            device
        ).float()
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

    def _prepare_text_and_vision_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor],
        coarse_inputs: Optional[VisionInputs],
        full_inputs: Optional[VisionInputs],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        image_token_id = self._get_image_token_id()
        tokenizer = self.base_vlm.tokenizer
        pad_id = 0
        bos_id = None
        if tokenizer is not None:
            pad_id = tokenizer.pad_token_id or 0
            bos_id = tokenizer.bos_token_id

        text_tokens, text_labels = self._strip_image_tokens(
            input_ids, attention_mask, labels, image_token_id
        )
        text_max = max((len(seq) for seq in text_tokens), default=0)
        device = input_ids.device
        dtype = input_ids.dtype
        label_dtype = labels.dtype if labels is not None else torch.long

        coarse_tokens: Optional[List[List[int]]] = None
        coarse_labels: Optional[List[List[int]]] = None
        full_tokens: Optional[List[List[int]]] = None
        full_labels: Optional[List[List[int]]] = None

        if image_token_id is not None and coarse_inputs is not None:
            counts_tensor = (
                coarse_inputs.token_counts
                if coarse_inputs.token_counts is not None
                else self._vision_token_counts(coarse_inputs, model=self.base_vlm)
            )
            counts = counts_tensor.tolist()
            coarse_tokens = []
            coarse_labels = [] if labels is not None else None
            for idx, (seq, count) in enumerate(zip(text_tokens, counts)):
                lbls = text_labels[idx] if labels is not None and text_labels is not None else None
                new_seq, new_lbls = self._insert_image_tokens(
                    seq, lbls, image_token_id, count, bos_id
                )
                coarse_tokens.append(new_seq)
                if coarse_labels is not None:
                    coarse_labels.append(new_lbls or [])

        if image_token_id is not None and full_inputs is not None:
            model = self.full_vlm if self.full_vlm is not None else self.base_vlm
            counts_tensor = (
                full_inputs.token_counts
                if full_inputs.token_counts is not None
                else self._vision_token_counts(full_inputs, model=model)
            )
            counts = counts_tensor.tolist()
            full_tokens = []
            full_labels = [] if labels is not None else None
            for idx, (seq, count) in enumerate(zip(text_tokens, counts)):
                lbls = text_labels[idx] if labels is not None and text_labels is not None else None
                new_seq, new_lbls = self._insert_image_tokens(
                    seq, lbls, image_token_id, count, bos_id
                )
                full_tokens.append(new_seq)
                if full_labels is not None:
                    full_labels.append(new_lbls or [])

        max_len = text_max
        if coarse_tokens is not None:
            max_len = max(max_len, max((len(seq) for seq in coarse_tokens), default=0))
        if full_tokens is not None:
            max_len = max(max_len, max((len(seq) for seq in full_tokens), default=0))
        max_len = max(1, max_len)

        text_input_ids, text_attention_mask = self._pad_sequences(
            text_tokens, pad_id, max_len, device, dtype
        )
        text_labels_tensor = self._pad_labels(
            text_labels, -100, max_len, device, label_dtype
        )

        if coarse_inputs is not None and coarse_tokens is not None:
            coarse_ids, coarse_mask = self._pad_sequences(
                coarse_tokens, pad_id, max_len, device, dtype
            )
            coarse_lbls = self._pad_labels(
                coarse_labels, -100, max_len, device, label_dtype
            )
            coarse_inputs.input_ids = coarse_ids
            coarse_inputs.attention_mask = coarse_mask
            coarse_inputs.labels = coarse_lbls

        if full_inputs is not None and full_tokens is not None:
            full_ids, full_mask = self._pad_sequences(
                full_tokens, pad_id, max_len, device, dtype
            )
            full_lbls = self._pad_labels(
                full_labels, -100, max_len, device, label_dtype
            )
            full_inputs.input_ids = full_ids
            full_inputs.attention_mask = full_mask
            full_inputs.labels = full_lbls

        return text_input_ids, text_attention_mask, text_labels_tensor

    def _vision_token_counts(
        self, vision_inputs: VisionInputs, model: Optional[BaseVLM] = None
    ) -> Tensor:
        if vision_inputs.image_grid_thw is not None:
            grid = vision_inputs.image_grid_thw.to(dtype=torch.long)
            if grid.ndim == 1:
                grid = grid.unsqueeze(0)
            grid = self._apply_merge_to_grid(grid, model=model)
            counts = grid.prod(dim=-1)
            return counts.clamp(min=1)
        pixel_values = vision_inputs.pixel_values
        if isinstance(pixel_values, torch.Tensor):
            counts = self._estimate_tokens_from_pixel_values(pixel_values, model=model or self.base_vlm)
            return counts.to(dtype=torch.long).clamp(min=1)
        return vision_inputs.token_counts.to(dtype=torch.long).clamp(min=1)

    def _apply_merge_to_grid(
        self, grid: Tensor, model: Optional[BaseVLM] = None
    ) -> Tensor:
        target_model = model or self.base_vlm
        if target_model is None:
            return grid
        config = getattr(target_model.model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        spatial_merge = None
        temporal_merge = None
        for cfg in (vision_config, config):
            if cfg is None:
                continue
            spatial_merge = (
                getattr(cfg, "spatial_merge_size", None)
                or getattr(cfg, "spatial_merge_factor", None)
                or getattr(cfg, "spatial_merge_ratio", None)
                or getattr(cfg, "spatial_merge", None)
                or getattr(cfg, "merge_size", None)
                or getattr(cfg, "image_merge_size", None)
                or getattr(cfg, "image_merge_ratio", None)
            )
            temporal_merge = (
                getattr(cfg, "temporal_merge_size", None)
                or getattr(cfg, "temporal_merge_factor", None)
                or getattr(cfg, "temporal_merge_ratio", None)
                or getattr(cfg, "temporal_merge", None)
            )
            if spatial_merge or temporal_merge:
                break
        if spatial_merge is None and temporal_merge is None:
            visual = getattr(target_model.model, "visual", None)
            if visual is not None:
                spatial_merge = (
                    getattr(visual, "spatial_merge_size", None)
                    or getattr(visual, "spatial_merge_factor", None)
                    or getattr(visual, "spatial_merge_ratio", None)
                    or getattr(visual, "spatial_merge", None)
                    or getattr(visual, "merge_size", None)
                )
                temporal_merge = (
                    getattr(visual, "temporal_merge_size", None)
                    or getattr(visual, "temporal_merge_factor", None)
                    or getattr(visual, "temporal_merge_ratio", None)
                    or getattr(visual, "temporal_merge", None)
                )
        if isinstance(spatial_merge, (list, tuple)):
            spatial_merge = spatial_merge[0] if spatial_merge else None
        if isinstance(temporal_merge, (list, tuple)):
            temporal_merge = temporal_merge[0] if temporal_merge else None
        spatial_merge = int(spatial_merge) if spatial_merge else 1
        temporal_merge = int(temporal_merge) if temporal_merge else 1
        if spatial_merge <= 1 and temporal_merge <= 1:
            model_type = getattr(config, "model_type", "") if config is not None else ""
            name_hint = getattr(target_model, "model_name", "")
            hint = f"{model_type} {name_hint}".lower()
            if "qwen3_vl" in hint or "qwen2_vl" in hint:
                spatial_merge = 2
        if spatial_merge <= 1 and temporal_merge <= 1:
            return grid
        merged = grid.clone()
        if temporal_merge > 1:
            merged[:, 0] = (merged[:, 0] + temporal_merge - 1) // temporal_merge
        if spatial_merge > 1:
            merged[:, 1] = (merged[:, 1] + spatial_merge - 1) // spatial_merge
            merged[:, 2] = (merged[:, 2] + spatial_merge - 1) // spatial_merge
        return merged

    def _should_use_merged_grid(self, model: BaseVLM) -> bool:
        config = getattr(model.model, "config", None)
        model_type = getattr(config, "model_type", "") if config is not None else ""
        name_hint = getattr(model, "model_name", "")
        hint = f"{model_type} {name_hint}".lower()
        normalized = hint.replace("-", "_")
        if "qwen3_vl" in normalized or "qwen2_vl" in normalized:
            return True
        return "qwen3" in normalized and "vl" in normalized

    def _prepare_vision_inputs(
        self,
        images: List[Image.Image],
        mode: str,
        model: BaseVLM,
    ) -> VisionInputs:
        processed = [self.vision_budget.prepare_image(img, mode) for img in images]
        pixel_values, token_counts, image_grid_thw = self._prepare_pixel_values(
            processed, model=model
        )
        if image_grid_thw is not None:
            grid = image_grid_thw
            if grid.ndim == 1:
                grid = grid.unsqueeze(0)
            merged_grid = self._apply_merge_to_grid(grid, model=model)
            token_counts = merged_grid.long().prod(dim=-1)
        return VisionInputs(
            pixel_values=pixel_values,
            token_counts=token_counts,
            image_grid_thw=image_grid_thw,
        )

    def _forward_hard_actions(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Optional[List[Image.Image]],
        text_outputs: VLMOutputs,
        actions: Tensor,
        text_labels: Optional[Tensor] = None,
        coarse_inputs: Optional[VisionInputs] = None,
        full_inputs: Optional[VisionInputs] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if images is None:
            return (
                text_outputs.logits,
                torch.zeros_like(actions, dtype=torch.float),
                text_labels,
            )

        if coarse_inputs is None or full_inputs is None:
            _, _, coarse_inputs, full_inputs = self._prepare_token_counts(
                images=images,
                device=input_ids.device,
                batch_size=input_ids.shape[0],
            )

        unique_actions = actions.unique().tolist()
        if len(unique_actions) == 1 and unique_actions[0] != Action.NO_VISION:
            mode = "coarse" if unique_actions[0] == Action.COARSE_VISION else "full"
            logits, tokens = self._forward_mode(
                input_ids,
                attention_mask,
                images,
                text_outputs,
                mode=mode,
                past_key_values=text_outputs.past_key_values,
                vision_inputs=coarse_inputs if mode == "coarse" else full_inputs,
            )
            labels = None
            if text_labels is not None:
                if mode == "coarse" and coarse_inputs is not None:
                    labels = coarse_inputs.labels
                elif mode == "full" and full_inputs is not None:
                    labels = full_inputs.labels
            return logits, tokens, labels

        logits_list: List[Tensor] = []
        token_counts: List[int] = []
        label_list: Optional[List[Tensor]] = [] if text_labels is not None else None
        coarse_pixel_values = (
            coarse_inputs.pixel_values if coarse_inputs is not None else None
        )
        full_pixel_values = full_inputs.pixel_values if full_inputs is not None else None
        coarse_tokens = coarse_inputs.token_counts if coarse_inputs is not None else None
        full_tokens = full_inputs.token_counts if full_inputs is not None else None
        coarse_ids = coarse_inputs.input_ids if coarse_inputs is not None else None
        coarse_mask = coarse_inputs.attention_mask if coarse_inputs is not None else None
        full_ids = full_inputs.input_ids if full_inputs is not None else None
        full_mask = full_inputs.attention_mask if full_inputs is not None else None
        for idx, action in enumerate(actions.tolist()):
            if action == Action.NO_VISION:
                logits_list.append(text_outputs.logits[idx : idx + 1])
                token_counts.append(0)
                if label_list is not None and text_labels is not None:
                    label_list.append(text_labels[idx : idx + 1])
                continue

            if action == Action.COARSE_VISION:
                pixel_values = (
                    coarse_pixel_values[idx : idx + 1]
                    if coarse_pixel_values is not None
                    else None
                )
                input_ids_row = (
                    coarse_ids[idx : idx + 1] if coarse_ids is not None else input_ids[idx : idx + 1]
                )
                attention_mask_row = (
                    coarse_mask[idx : idx + 1]
                    if coarse_mask is not None
                    else attention_mask[idx : idx + 1]
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
                input_ids_row = (
                    full_ids[idx : idx + 1] if full_ids is not None else input_ids[idx : idx + 1]
                )
                attention_mask_row = (
                    full_mask[idx : idx + 1]
                    if full_mask is not None
                    else attention_mask[idx : idx + 1]
                )
                count = int(full_tokens[idx]) if full_tokens is not None else 0
                model = (
                    self.full_vlm if self.full_vlm is not None else self.base_vlm
                )

            outputs = model.forward_with_vision(
                input_ids=input_ids_row,
                attention_mask=attention_mask_row,
                pixel_values=pixel_values,
                image_grid_thw=(
                    coarse_inputs.image_grid_thw[idx : idx + 1]
                    if action == Action.COARSE_VISION and coarse_inputs is not None
                    else (
                        full_inputs.image_grid_thw[idx : idx + 1]
                        if action == Action.FULL_VISION and full_inputs is not None
                        else None
                    )
                ),
                past_key_values=None,
            )
            logits_list.append(outputs.logits)
            token_counts.append(count)
            if label_list is not None:
                if action == Action.COARSE_VISION and coarse_inputs is not None:
                    label_list.append(coarse_inputs.labels[idx : idx + 1])
                elif action == Action.FULL_VISION and full_inputs is not None:
                    label_list.append(full_inputs.labels[idx : idx + 1])

        logits = torch.cat(logits_list, dim=0)
        vision_tokens = torch.tensor(token_counts, device=logits.device, dtype=torch.float)
        labels = torch.cat(label_list, dim=0) if label_list is not None else None
        return logits, vision_tokens, labels

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
        vision_pruning: Optional[VisionPruningSpec] = None,
    ) -> Tuple[Tensor, Tensor]:
        model = self.full_vlm if (mode == "full" and self.full_vlm) else self.base_vlm
        if vision_inputs is None:
            vision_inputs = self._prepare_vision_inputs(
                images=images, mode=mode, model=model
            )
        if vision_inputs.input_ids is not None:
            input_ids = vision_inputs.input_ids
            past_key_values = None
        if vision_inputs.attention_mask is not None:
            attention_mask = vision_inputs.attention_mask
        pixel_values = vision_inputs.pixel_values
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(input_ids.device, non_blocking=True)
            if pixel_values.device.type == "cuda":
                pixel_values = pixel_values.contiguous()
        image_grid_thw = vision_inputs.image_grid_thw
        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.to(input_ids.device, non_blocking=True)
            if image_grid_thw.device.type == "cuda":
                image_grid_thw = image_grid_thw.contiguous()
        if past_key_values is None:
            past_key_values = text_outputs.past_key_values
        outputs = model.forward_with_vision(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=past_key_values,
            vision_pruning=vision_pruning,
        )
        tokens = vision_inputs.token_counts.to(outputs.logits.device).float()
        return outputs.logits, tokens

    def _prepare_pixel_values(
        self, images: List[Image.Image], model: BaseVLM
    ) -> Tuple[Optional[Tensor], Tensor, Optional[Tensor]]:
        processor = model.processor
        image_grid_thw = None
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
                grid_thw = (
                    outputs.get("image_grid_thw")
                    if hasattr(outputs, "get")
                    else getattr(outputs, "image_grid_thw", None)
                )
                if grid_thw is not None:
                    image_grid_thw = torch.as_tensor(grid_thw)
                    if image_grid_thw.ndim == 1:
                        image_grid_thw = image_grid_thw.unsqueeze(0)
                if pixel_values is not None:
                    if isinstance(pixel_values, torch.Tensor):
                        pixel_values = pixel_values.contiguous()
                    if image_grid_thw is None and isinstance(pixel_values, torch.Tensor):
                        patch_size = self._get_patch_size(model) or self.vision_budget.patch_size
                        patch_size = max(1, int(patch_size))
                        if pixel_values.dim() == 4:
                            batch, _, height, width = pixel_values.shape
                            frames = 1
                        elif pixel_values.dim() == 5:
                            batch, frames, _, height, width = pixel_values.shape
                        else:
                            batch = pixel_values.shape[0]
                            frames = 1
                            height, width = pixel_values.shape[-2:]
                        grid_h = math.ceil(height / patch_size)
                        grid_w = math.ceil(width / patch_size)
                        image_grid_thw = torch.tensor(
                            [[frames, grid_h, grid_w]] * batch
                        )
                    token_counts = self._infer_token_counts(
                        outputs=outputs,
                        pixel_values=pixel_values,
                        batch_size=len(images),
                        model=model,
                    )
                    return pixel_values, token_counts, image_grid_thw

        arrays = [np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0 for img in images]
        tensors = [torch.from_numpy(arr).permute(2, 0, 1) for arr in arrays]
        pixel_values = torch.stack(tensors, dim=0)
        patch_size = self._get_patch_size(model) or self.vision_budget.patch_size
        patch_size = max(1, int(patch_size))
        height, width = pixel_values.shape[-2:]
        grid_h = math.ceil(height / patch_size)
        grid_w = math.ceil(width / patch_size)
        image_grid_thw = torch.tensor([[1, grid_h, grid_w]] * pixel_values.shape[0])
        token_counts = self._estimate_tokens_from_pixel_values(pixel_values, model=model)
        return pixel_values, token_counts, image_grid_thw

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
                grid_tensor = self._apply_merge_to_grid(grid_tensor, model=model)
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
            grid = torch.tensor([[1, grid_h, grid_w]] * batch)
            grid = self._apply_merge_to_grid(grid, model=model)
            tokens = grid.long().prod(dim=-1)
            return tokens
        if pixel_values.dim() == 5:
            batch, frames, _, height, width = pixel_values.shape
            grid_h = math.ceil(height / patch_size)
            grid_w = math.ceil(width / patch_size)
            grid = torch.tensor([[frames, grid_h, grid_w]] * batch)
            grid = self._apply_merge_to_grid(grid, model=model)
            tokens = grid.long().prod(dim=-1)
            return tokens
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
