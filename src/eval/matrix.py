"""Multi-dataset evaluation utilities for VoVNet."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import os

from src.baselines import normalize_baseline_name, resolve_baseline_actions
from src.baselines.random_policy_matched import (
    bucketize_entropy,
    compute_entropy_thresholds,
    sample_actions_by_bucket,
    sample_actions_from_ratios,
)
from src.baselines.uncertainty_threshold import select_actions_binary
from src.config.config import Config
from src.data.collate import VLMDataCollator
from src.eval.matrix_spec import EvalDatasetSpec, build_dataset, get_metric_fn
from src.models.base_vlm import BaseVLM, VisionPruningSpec
from src.models.uncertainty import expected_calibration_error
from src.models.vovnet import Action, VoVNet, VisionInputs
from src.models.vision_budget import VisionBudgetController
from src.utils.profiling import BatchProfiler




def build_model(cfg: Config) -> VoVNet:
    base_vlm = BaseVLM(
        cfg.model.base_model_name,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=cfg.model.torch_dtype,
    )
    full_vlm = None
    if cfg.model.use_thinking_for_full:
        full_vlm = BaseVLM(
            cfg.model.full_model_name,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=cfg.model.torch_dtype,
        )
    budget = VisionBudgetController(**cfg.vision_budget.__dict__)
    return VoVNet(
        base_vlm=base_vlm,
        full_vlm=full_vlm,
        vision_budget=budget,
        vow_hidden_dim=cfg.policy.vow_hidden_dim,
        gumbel_tau=cfg.policy.gumbel_tau,
        use_straight_through=cfg.policy.use_straight_through,
        eval_sample=cfg.policy.eval_sample,
        policy_mode=cfg.policy.policy_mode,
        fallback_mode=cfg.policy.fallback_mode,
        fallback_entropy_threshold=cfg.policy.fallback_entropy_threshold,
        fallback_margin_threshold=cfg.policy.fallback_margin_threshold,
        cost_scale=cfg.policy.cost_scale,
        cost_c1=cfg.policy.cost_c1,
        cost_c2=cfg.policy.cost_c2,
    )


def _decode_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, tokenizer: Any
) -> List[str]:
    preds: List[str] = []
    pred_ids = logits.argmax(dim=-1)
    for i in range(pred_ids.size(0)):
        mask = labels[i] != -100
        answer_ids = pred_ids[i][mask]
        text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        preds.append(text.strip())
    return preds


def _forward_with_cost(
    model: VoVNet,
    batch: Dict[str, Any],
    cost_weight: Optional[float],
) -> Dict[str, Any]:
    raw_model = getattr(model, "module", model)
    text_outputs, action_logits, _ = raw_model.text_first(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    token_count_coarse, token_count_full, coarse_inputs, full_inputs = (
        raw_model._prepare_token_counts(
            images=batch.get("images"),
            device=batch["input_ids"].device,
            batch_size=batch["input_ids"].size(0),
        )
    )
    if cost_weight is not None:
        zeros = torch.zeros_like(token_count_coarse)
        costs = torch.stack([zeros, token_count_coarse, token_count_full], dim=-1)
        costs = costs * float(getattr(raw_model, "cost_scale", 1.0))
        action_logits = action_logits - float(cost_weight) * costs

    _, actions = raw_model._select_actions(action_logits)
    uncertainty = raw_model._compute_uncertainty(text_outputs)
    margin = raw_model._compute_margin(text_outputs)
    if not raw_model.training and batch.get("images") is not None:
        actions, _, _, _ = raw_model._apply_fallback(actions, uncertainty, margin)

    logits, vision_tokens = raw_model._forward_hard_actions(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        images=batch.get("images"),
        text_outputs=text_outputs,
        actions=actions,
        coarse_inputs=coarse_inputs,
        full_inputs=full_inputs,
    )
    action_probs = F.one_hot(actions, num_classes=len(Action)).float()
    expected_cost = raw_model._compute_expected_cost(
        action_probs, token_count_coarse, token_count_full
    )
    return {
        "logits": logits,
        "actions": actions,
        "expected_cost": expected_cost,
        "vision_tokens": vision_tokens,
    }


def _forward_with_baseline(
    model: VoVNet,
    batch: Dict[str, Any],
    baseline_name: str,
) -> Dict[str, Any]:
    raw_model = getattr(model, "module", model)
    batch_size = batch["input_ids"].size(0)
    actions, action_label, drop_images = resolve_baseline_actions(
        baseline_name, batch_size, batch["input_ids"].device
    )
    images = None if drop_images else batch.get("images")
    outputs = raw_model.forward_with_actions(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        images=images,
        actions=actions,
        labels=batch.get("labels"),
    )
    outputs["action_label"] = action_label
    return outputs


def _forward_with_uncertainty_threshold(
    model: VoVNet,
    batch: Dict[str, Any],
    threshold: float,
    uncertainty_metric: str,
    vision_mode: str,
) -> Dict[str, Any]:
    raw_model = getattr(model, "module", model)
    text_outputs, _, _ = raw_model.text_first(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    if uncertainty_metric == "margin":
        margin = raw_model._compute_margin(text_outputs)
        uncertainty = 1.0 - margin
    else:
        uncertainty = raw_model._compute_uncertainty(text_outputs)
    high_action = Action.FULL_VISION if vision_mode == "full" else Action.COARSE_VISION
    actions = select_actions_binary(uncertainty, threshold, high_action=high_action)

    token_count_coarse, token_count_full, coarse_inputs, full_inputs = (
        raw_model._prepare_token_counts(
            images=batch.get("images"),
            device=batch["input_ids"].device,
            batch_size=batch["input_ids"].size(0),
        )
    )
    logits, vision_tokens = raw_model._forward_hard_actions(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        images=batch.get("images"),
        text_outputs=text_outputs,
        actions=actions,
        coarse_inputs=coarse_inputs,
        full_inputs=full_inputs,
    )
    action_probs = F.one_hot(actions, num_classes=len(Action)).float()
    expected_cost = raw_model._compute_expected_cost(
        action_probs, token_count_coarse, token_count_full
    )
    action_label = f"THRESHOLD_{high_action.name}"
    return {
        "logits": logits,
        "actions": actions,
        "expected_cost": expected_cost,
        "vision_tokens": vision_tokens,
        "action_label": action_label,
    }


def _forward_with_vision_token_pruning_proxy(
    model: VoVNet,
    batch: Dict[str, Any],
    pruning_ratio: float,
    pruning_mode: str,
) -> Dict[str, Any]:
    raw_model = getattr(model, "module", model)
    images = batch.get("images")
    if images is None:
        outputs = _forward_with_cost(raw_model, batch, cost_weight=None)
        outputs["action_label"] = "PRUNING_PROXY_TEXT_ONLY"
        return outputs

    text_outputs, _, _ = raw_model.text_first(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    token_count_coarse, token_count_full, _, full_inputs = (
        raw_model._prepare_token_counts(
            images=images,
            device=batch["input_ids"].device,
            batch_size=batch["input_ids"].size(0),
        )
    )
    if full_inputs is None:
        outputs = _forward_with_cost(raw_model, batch, cost_weight=None)
        outputs["action_label"] = "PRUNING_PROXY_TEXT_ONLY"
        return outputs

    full_model = raw_model.full_vlm if raw_model.full_vlm is not None else raw_model.base_vlm
    keep_counts = full_model.compute_pruned_counts(token_count_full, pruning_ratio)
    keep_counts = keep_counts.to(device=batch["input_ids"].device, dtype=torch.long)

    pruned_inputs = VisionInputs(
        pixel_values=None,
        token_counts=keep_counts,
        image_grid_thw=None,
    )
    raw_model._prepare_text_and_vision_inputs(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch.get("labels"),
        coarse_inputs=None,
        full_inputs=pruned_inputs,
    )
    full_inputs.input_ids = pruned_inputs.input_ids
    full_inputs.attention_mask = pruned_inputs.attention_mask
    full_inputs.labels = pruned_inputs.labels
    full_inputs.token_counts = keep_counts

    pruning_spec = VisionPruningSpec(
        ratio=float(pruning_ratio),
        mode=pruning_mode,
        min_tokens=1,
        keep_counts=keep_counts,
    )
    logits, _ = raw_model._forward_mode(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        images=images,
        text_outputs=text_outputs,
        mode="full",
        vision_inputs=full_inputs,
        vision_pruning=pruning_spec,
    )
    actions = torch.full(
        (batch["input_ids"].size(0),),
        Action.FULL_VISION,
        device=batch["input_ids"].device,
        dtype=torch.long,
    )
    action_probs = F.one_hot(actions, num_classes=len(Action)).float()
    expected_cost = raw_model._compute_expected_cost(
        action_probs, token_count_coarse, keep_counts.float()
    )
    action_label = "PRUNING_PROXY_FULL"
    return {
        "logits": logits,
        "actions": actions,
        "expected_cost": expected_cost,
        "vision_tokens": keep_counts.float(),
        "remaining_vision_tokens": keep_counts.float(),
        "action_label": action_label,
        "labels": full_inputs.labels,
    }


def _forward_with_token_merge_prune_proxy(
    model: VoVNet,
    batch: Dict[str, Any],
    merge_ratio: float,
    merge_mode: str,
    merge_weight: str,
    enable_prune: bool,
    prune_ratio: float,
    prune_mode: str,
) -> Dict[str, Any]:
    raw_model = getattr(model, "module", model)
    images = batch.get("images")
    if images is None:
        outputs = _forward_with_cost(raw_model, batch, cost_weight=None)
        outputs["action_label"] = "MERGE_PROXY_TEXT_ONLY"
        return outputs

    text_outputs, _, _ = raw_model.text_first(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    token_count_coarse, token_count_full, _, full_inputs = (
        raw_model._prepare_token_counts(
            images=images,
            device=batch["input_ids"].device,
            batch_size=batch["input_ids"].size(0),
        )
    )
    if full_inputs is None:
        outputs = _forward_with_cost(raw_model, batch, cost_weight=None)
        outputs["action_label"] = "MERGE_PROXY_TEXT_ONLY"
        return outputs

    full_model = raw_model.full_vlm if raw_model.full_vlm is not None else raw_model.base_vlm
    merge_counts = full_model.compute_pruned_counts(token_count_full, merge_ratio)
    final_counts = merge_counts
    if enable_prune:
        final_counts = full_model.compute_pruned_counts(merge_counts, prune_ratio)

    merge_inputs = VisionInputs(
        pixel_values=None,
        token_counts=final_counts,
        image_grid_thw=None,
    )
    raw_model._prepare_text_and_vision_inputs(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch.get("labels"),
        coarse_inputs=None,
        full_inputs=merge_inputs,
    )
    full_inputs.input_ids = merge_inputs.input_ids
    full_inputs.attention_mask = merge_inputs.attention_mask
    full_inputs.labels = merge_inputs.labels
    full_inputs.token_counts = final_counts

    merge_spec = VisionPruningSpec(
        ratio=float(merge_ratio),
        mode=f"merge_{merge_mode}",
        min_tokens=1,
        keep_counts=merge_counts,
        enable_prune=bool(enable_prune),
        prune_ratio=float(prune_ratio),
        prune_mode=prune_mode,
        merge_weight=merge_weight,
    )
    logits, _ = raw_model._forward_mode(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        images=images,
        text_outputs=text_outputs,
        mode="full",
        vision_inputs=full_inputs,
        vision_pruning=merge_spec,
    )
    actions = torch.full(
        (batch["input_ids"].size(0),),
        Action.FULL_VISION,
        device=batch["input_ids"].device,
        dtype=torch.long,
    )
    action_probs = F.one_hot(actions, num_classes=len(Action)).float()
    expected_cost = raw_model._compute_expected_cost(
        action_probs, token_count_coarse, final_counts.float()
    )
    return {
        "logits": logits,
        "actions": actions,
        "expected_cost": expected_cost,
        "vision_tokens": final_counts.float(),
        "remaining_vision_tokens": final_counts.float(),
        "action_label": "MERGE_PROXY_FULL",
        "labels": full_inputs.labels,
        "merge_ratio": float(merge_ratio),
        "prune_ratio": float(prune_ratio) if enable_prune else 1.0,
        "merge_mode": merge_mode,
        "merge_weight": merge_weight,
        "enable_prune": bool(enable_prune),
        "prune_mode": prune_mode,
    }


def _compute_pooled_counts(
    grid_thw: Optional[torch.Tensor],
    token_counts: torch.Tensor,
    pool_factor: int,
) -> torch.Tensor:
    if pool_factor <= 1:
        return token_counts.to(dtype=torch.long).clamp(min=1)
    if grid_thw is None:
        ratio = 1.0 / float(pool_factor * pool_factor)
        return BaseVLM.compute_pruned_counts(token_counts, ratio)
    grid = grid_thw.to(dtype=torch.long)
    if grid.ndim == 1:
        grid = grid.unsqueeze(0)
    frames = grid[:, 0]
    heights = grid[:, 1]
    widths = grid[:, 2]
    pooled_h = (heights + pool_factor - 1) // pool_factor
    pooled_w = (widths + pool_factor - 1) // pool_factor
    counts = (frames * pooled_h * pooled_w).clamp(min=1)
    return counts.to(device=token_counts.device)


def _forward_with_multi_granularity_proxy(
    model: VoVNet,
    batch: Dict[str, Any],
    pool_factor: int,
) -> Dict[str, Any]:
    raw_model = getattr(model, "module", model)
    images = batch.get("images")
    if images is None:
        outputs = _forward_with_cost(raw_model, batch, cost_weight=None)
        outputs["action_label"] = "GRANULARITY_TEXT_ONLY"
        return outputs

    text_outputs, _, _ = raw_model.text_first(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    token_count_coarse, token_count_full, _, full_inputs = (
        raw_model._prepare_token_counts(
            images=images,
            device=batch["input_ids"].device,
            batch_size=batch["input_ids"].size(0),
        )
    )
    if full_inputs is None:
        outputs = _forward_with_cost(raw_model, batch, cost_weight=None)
        outputs["action_label"] = "GRANULARITY_TEXT_ONLY"
        return outputs

    pool_factor = max(1, int(pool_factor))
    merged_grid = full_inputs.image_grid_thw
    if merged_grid is not None:
        full_model = (
            raw_model.full_vlm if raw_model.full_vlm is not None else raw_model.base_vlm
        )
        merged_grid = raw_model._apply_merge_to_grid(merged_grid, model=full_model)
    pooled_counts = _compute_pooled_counts(
        merged_grid, token_count_full, pool_factor
    ).to(dtype=torch.long)

    pool_inputs = VisionInputs(
        pixel_values=None,
        token_counts=pooled_counts,
        image_grid_thw=None,
    )
    raw_model._prepare_text_and_vision_inputs(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch.get("labels"),
        coarse_inputs=None,
        full_inputs=pool_inputs,
    )
    full_inputs.input_ids = pool_inputs.input_ids
    full_inputs.attention_mask = pool_inputs.attention_mask
    full_inputs.labels = pool_inputs.labels
    full_inputs.token_counts = pooled_counts

    ratio = 1.0 / float(pool_factor * pool_factor) if pool_factor > 1 else 1.0
    pool_spec = VisionPruningSpec(
        ratio=ratio,
        mode="pool",
        min_tokens=1,
        keep_counts=pooled_counts,
        pool_factor=pool_factor,
        image_grid_thw=merged_grid,
    )
    logits, _ = raw_model._forward_mode(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        images=images,
        text_outputs=text_outputs,
        mode="full",
        vision_inputs=full_inputs,
        vision_pruning=pool_spec,
    )
    actions = torch.full(
        (batch["input_ids"].size(0),),
        Action.FULL_VISION,
        device=batch["input_ids"].device,
        dtype=torch.long,
    )
    action_probs = F.one_hot(actions, num_classes=len(Action)).float()
    expected_cost = raw_model._compute_expected_cost(
        action_probs, token_count_coarse, pooled_counts.float()
    )
    return {
        "logits": logits,
        "actions": actions,
        "expected_cost": expected_cost,
        "vision_tokens": pooled_counts.float(),
        "remaining_vision_tokens": pooled_counts.float(),
        "action_label": f"GRANULARITY_{pool_factor}",
        "labels": full_inputs.labels,
        "pool_factor": pool_factor,
    }


def _forward_with_random_policy_matched(
    model: VoVNet,
    batch: Dict[str, Any],
    target_ratios: Optional[List[float]],
    bucket_ratios: Optional[List[List[float]]],
    bucket_thresholds: Optional[tuple[float, float]],
    generator: torch.Generator,
) -> Dict[str, Any]:
    raw_model = getattr(model, "module", model)
    text_outputs, _, _ = raw_model.text_first(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    batch_size = batch["input_ids"].size(0)
    device = batch["input_ids"].device
    if bucket_ratios is not None:
        entropy = raw_model._compute_uncertainty(text_outputs)
        if bucket_thresholds is None:
            raise ValueError("bucket_thresholds required for bucketed random policy")
        bucket_ids = bucketize_entropy(entropy, bucket_thresholds)
        actions = sample_actions_by_bucket(
            bucket_ratios, bucket_ids, generator=generator, device=device
        )
        action_label = "RANDOM_MATCHED_BUCKET"
    else:
        if target_ratios is None:
            raise ValueError("target_ratios required for random_policy_matched")
        actions = sample_actions_from_ratios(
            target_ratios, batch_size, generator=generator, device=device
        )
        action_label = "RANDOM_MATCHED_GLOBAL"

    token_count_coarse, token_count_full, coarse_inputs, full_inputs = (
        raw_model._prepare_token_counts(
            images=batch.get("images"),
            device=batch["input_ids"].device,
            batch_size=batch_size,
        )
    )
    logits, vision_tokens = raw_model._forward_hard_actions(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        images=batch.get("images"),
        text_outputs=text_outputs,
        actions=actions,
        coarse_inputs=coarse_inputs,
        full_inputs=full_inputs,
    )
    action_probs = F.one_hot(actions, num_classes=len(Action)).float()
    expected_cost = raw_model._compute_expected_cost(
        action_probs, token_count_coarse, token_count_full
    )
    return {
        "logits": logits,
        "actions": actions,
        "expected_cost": expected_cost,
        "vision_tokens": vision_tokens,
        "action_label": action_label,
    }


def _compute_bucket_thresholds(
    model: VoVNet,
    loader: DataLoader,
) -> tuple[float, float]:
    raw_model = getattr(model, "module", model)
    entropies: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            text_outputs, _, _ = raw_model.text_first(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            entropy = raw_model._compute_uncertainty(text_outputs)
            entropies.append(entropy)
    return compute_entropy_thresholds(entropies)


def _evaluate_reference_policy(
    model: VoVNet,
    loader: DataLoader,
    metric_fn: Callable[[Iterable[str], Iterable[str]], float],
    cost_weight: Optional[float],
) -> Dict[str, float]:
    raw_model = getattr(model, "module", model)
    total_acc = 0.0
    total_cost = 0.0
    total_count = 0
    tokenizer = raw_model.base_vlm.tokenizer
    with torch.no_grad():
        for batch in loader:
            outputs = _forward_with_cost(raw_model, batch, cost_weight)
            labels = outputs.get("labels") if outputs.get("labels") is not None else batch["labels"]
            preds = _decode_from_logits(outputs["logits"], labels, tokenizer)
            acc = metric_fn(preds, batch.get("answers", []))
            total_acc += acc * len(preds)
            total_cost += outputs["expected_cost"].sum().item()
            total_count += len(preds)
    return {
        "accuracy": total_acc / max(1, total_count),
        "avg_cost": total_cost / max(1, total_count),
    }


def evaluate_dataset(
    model: VoVNet,
    loader: DataLoader,
    metric_fn: Callable[[Iterable[str], Iterable[str]], float],
    cost_weight: Optional[float],
    profile: bool,
    baseline_name: Optional[str],
    baseline_threshold: float,
    baseline_uncertainty: str,
    baseline_vision: str,
    baseline_seed: Optional[int],
    baseline_target_ratios: Optional[List[float]],
    baseline_bucket_ratios: Optional[List[List[float]]],
    baseline_bucket_thresholds: Optional[List[float]],
    baseline_pruning_ratio: float,
    baseline_pruning_mode: str,
    baseline_merge_ratio: float,
    baseline_merge_mode: str,
    baseline_merge_weight: str,
    baseline_enable_prune: bool,
    baseline_prune_ratio: float,
    baseline_prune_mode: str,
    baseline_pool_factor: int,
) -> Dict[str, Any]:
    raw_model = getattr(model, "module", model)
    raw_model.eval()
    total_acc = 0.0
    total_cost = 0.0
    total_vision_tokens = 0.0
    total_remaining_tokens = 0.0
    total_count = 0
    total_ece = 0.0
    total_ece_count = 0
    action_counts = torch.zeros(len(Action), dtype=torch.long)
    action_names = {int(action): action.name.lower() for action in Action}
    profiler = BatchProfiler(profile, action_names=action_names)
    baseline_name = normalize_baseline_name(baseline_name)
    baseline_uncertainty = (baseline_uncertainty or "entropy").strip().lower()
    baseline_vision = (baseline_vision or "full").strip().lower()
    baseline_pruning_mode = (baseline_pruning_mode or "stride").strip().lower()
    baseline_merge_mode = (baseline_merge_mode or "cosine").strip().lower()
    baseline_merge_weight = (baseline_merge_weight or "norm").strip().lower()
    baseline_prune_mode = (baseline_prune_mode or "topk_norm").strip().lower()
    if baseline_pruning_mode == "topk":
        baseline_pruning_mode = "topk_norm"
    if baseline_prune_mode == "topk":
        baseline_prune_mode = "topk_norm"
    action_label: Optional[str] = None
    bucket_thresholds: Optional[tuple[float, float]] = None
    generator: Optional[torch.Generator] = None

    if baseline_name == "random_policy_matched":
        seed = baseline_seed if baseline_seed is not None else 0
        rank = int(os.environ.get("RANK", "0"))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + rank)
        if baseline_bucket_ratios is not None:
            if baseline_bucket_thresholds is not None:
                bucket_thresholds = (
                    float(baseline_bucket_thresholds[0]),
                    float(baseline_bucket_thresholds[1]),
                )
            else:
                bucket_thresholds = _compute_bucket_thresholds(raw_model, loader)

    tokenizer = raw_model.base_vlm.tokenizer
    with torch.no_grad():
        for batch in loader:
            if profile:
                profiler.start()
            if baseline_name == "uncertainty_threshold":
                outputs = _forward_with_uncertainty_threshold(
                    raw_model,
                    batch,
                    threshold=baseline_threshold,
                    uncertainty_metric=baseline_uncertainty,
                    vision_mode=baseline_vision,
                )
                action_label = outputs.get("action_label")
            elif baseline_name == "vision_token_pruning_proxy":
                outputs = _forward_with_vision_token_pruning_proxy(
                    raw_model,
                    batch,
                    pruning_ratio=baseline_pruning_ratio,
                    pruning_mode=baseline_pruning_mode,
                )
                action_label = outputs.get("action_label")
            elif baseline_name == "token_merge_prune_proxy":
                outputs = _forward_with_token_merge_prune_proxy(
                    raw_model,
                    batch,
                    merge_ratio=baseline_merge_ratio,
                    merge_mode=baseline_merge_mode,
                    merge_weight=baseline_merge_weight,
                    enable_prune=baseline_enable_prune,
                    prune_ratio=baseline_prune_ratio,
                    prune_mode=baseline_prune_mode,
                )
                action_label = outputs.get("action_label")
            elif baseline_name == "multi_granularity_proxy":
                outputs = _forward_with_multi_granularity_proxy(
                    raw_model,
                    batch,
                    pool_factor=baseline_pool_factor,
                )
                action_label = outputs.get("action_label")
            elif baseline_name == "random_policy_matched":
                outputs = _forward_with_random_policy_matched(
                    raw_model,
                    batch,
                    target_ratios=baseline_target_ratios,
                    bucket_ratios=baseline_bucket_ratios,
                    bucket_thresholds=bucket_thresholds,
                    generator=generator if generator is not None else torch.Generator(),
                )
                action_label = outputs.get("action_label")
            elif baseline_name:
                outputs = _forward_with_baseline(raw_model, batch, baseline_name)
                action_label = outputs.get("action_label")
            else:
                outputs = _forward_with_cost(raw_model, batch, cost_weight)
            if profile:
                profiler.stop(
                    batch_size=batch["input_ids"].size(0),
                    seq_len=batch["input_ids"].size(1),
                    actions=outputs["actions"],
                )
            labels = outputs.get("labels") if outputs.get("labels") is not None else batch["labels"]
            preds = _decode_from_logits(outputs["logits"], labels, tokenizer)
            acc = metric_fn(preds, batch.get("answers", []))
            total_acc += acc * len(preds)
            total_cost += outputs["expected_cost"].sum().item()
            total_count += len(preds)
            action_counts += torch.bincount(
                outputs["actions"].detach().cpu(), minlength=len(Action)
            )
            vision_tokens = outputs.get("vision_tokens")
            if vision_tokens is not None:
                total_vision_tokens += vision_tokens.sum().item()
            remaining_tokens = outputs.get("remaining_vision_tokens")
            if remaining_tokens is not None:
                total_remaining_tokens += remaining_tokens.sum().item()
            labels = batch.get("labels")
            if labels is not None:
                mask = labels.ne(-100)
                if mask.any():
                    flat_logits = outputs["logits"][mask]
                    flat_labels = labels[mask]
                    probs = torch.softmax(flat_logits, dim=-1)
                    ece, _ = expected_calibration_error(probs, flat_labels)
                    total_ece += float(ece.item()) * int(mask.sum().item())
                    total_ece_count += int(mask.sum().item())

    action_rates = (action_counts.float() / max(1, total_count)).tolist()
    summary = profiler.summary() if profile else {}
    latency_stats = summary.get("overall", {}).get("latency_ms", {})
    mem_peak = summary.get("overall", {}).get("mem_mb", {})
    avg_cost = total_cost / max(1, total_count)
    avg_vision_tokens = total_vision_tokens / max(1, total_count)
    avg_remaining_tokens = total_remaining_tokens / max(1, total_count)
    results = {
        "accuracy": total_acc / max(1, total_count),
        "action_rates": action_rates,
        "action_ratio": action_rates,
        "action": action_label or "policy",
        "baseline_name": baseline_name,
        "avg_cost": avg_cost,
        "avg_token_cost": avg_cost,
        "avg_vision_token_count": avg_vision_tokens,
        "remaining_vision_tokens": avg_remaining_tokens,
        "latency_stats": latency_stats,
        "latency": latency_stats,
        "mem_peak": mem_peak,
        "profile": summary if profile else None,
        "ece": (total_ece / max(1, total_ece_count)),
    }
    if baseline_name == "uncertainty_threshold":
        results["threshold"] = baseline_threshold
        results["uncertainty_metric"] = baseline_uncertainty
        results["vision_mode"] = baseline_vision
    if baseline_name == "vision_token_pruning_proxy":
        results["pruning_ratio"] = baseline_pruning_ratio
        results["pruning_mode"] = baseline_pruning_mode
    if baseline_name == "token_merge_prune_proxy":
        results["merge_ratio"] = baseline_merge_ratio
        results["merge_mode"] = baseline_merge_mode
        results["merge_weight"] = baseline_merge_weight
        results["enable_prune"] = baseline_enable_prune
        results["prune_ratio"] = baseline_prune_ratio
        results["prune_mode"] = baseline_prune_mode
    if baseline_name == "multi_granularity_proxy":
        results["pool_factor"] = baseline_pool_factor
    if baseline_name == "random_policy_matched":
        reference = _evaluate_reference_policy(
            raw_model, loader, metric_fn, cost_weight
        )
        results["vovnet_accuracy"] = reference.get("accuracy", 0.0)
        results["vovnet_avg_cost"] = reference.get("avg_cost", 0.0)
        results["accuracy_gap_vs_vovnet"] = (
            results["accuracy"] - results["vovnet_accuracy"]
        )
        results["cost_gap_vs_vovnet"] = results["avg_cost"] - results["vovnet_avg_cost"]
    if baseline_name == "vision_token_pruning_proxy":
        reference = _evaluate_reference_policy(
            raw_model, loader, metric_fn, cost_weight
        )
        results["vovnet_accuracy"] = reference.get("accuracy", 0.0)
        results["vovnet_avg_cost"] = reference.get("avg_cost", 0.0)
        results["accuracy_gap_vs_vovnet"] = (
            results["accuracy"] - results["vovnet_accuracy"]
        )
        results["cost_gap_vs_vovnet"] = results["avg_cost"] - results["vovnet_avg_cost"]
    return results


def rows_from_results(
    dataset: str, metric_name: str, results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        action_rates = item.get("action_rates", [0.0, 0.0, 0.0])
        latency = item.get("latency_stats", item.get("latency", {}))
        mem_peak = item.get("mem_peak", {})
        avg_cost = item.get("avg_cost", item.get("avg_token_cost"))
        avg_vision_tokens = item.get("avg_vision_token_count", 0.0)
        remaining_tokens = item.get("remaining_vision_tokens")
        rows.append(
            {
                "dataset": dataset,
                "metric": metric_name,
                "lambda_cost": item.get("lambda_cost", 0.0),
                "baseline_name": item.get("baseline_name"),
                "action": item.get("action", "policy"),
                "threshold": item.get("threshold"),
                "pruning_ratio": item.get("pruning_ratio"),
                "remaining_vision_tokens": remaining_tokens,
                "merge_ratio": item.get("merge_ratio"),
                "merge_mode": item.get("merge_mode"),
                "merge_weight": item.get("merge_weight"),
                "enable_prune": item.get("enable_prune"),
                "prune_ratio": item.get("prune_ratio"),
                "prune_mode": item.get("prune_mode"),
                "pool_factor": item.get("pool_factor"),
                "resolution_long_side": item.get("resolution_long_side"),
                "resolution_max_pixels": item.get("resolution_max_pixels"),
                "uncertainty_metric": item.get("uncertainty_metric"),
                "vision_mode": item.get("vision_mode"),
                "accuracy": item.get("accuracy"),
                "action_rate_no": action_rates[0],
                "action_rate_coarse": action_rates[1],
                "action_rate_full": action_rates[2],
                "avg_cost": avg_cost,
                "avg_token_cost": item.get("avg_token_cost"),
                "avg_vision_token_count": avg_vision_tokens,
                "vovnet_accuracy": item.get("vovnet_accuracy"),
                "vovnet_avg_cost": item.get("vovnet_avg_cost"),
                "accuracy_gap_vs_vovnet": item.get("accuracy_gap_vs_vovnet"),
                "cost_gap_vs_vovnet": item.get("cost_gap_vs_vovnet"),
                "latency_p50_ms": latency.get("p50", 0.0),
                "latency_p90_ms": latency.get("p90", 0.0),
                "mem_peak_p50_mb": mem_peak.get("p50", 0.0),
                "mem_peak_p90_mb": mem_peak.get("p90", 0.0),
                "ece": item.get("ece"),
            }
        )
    return rows
