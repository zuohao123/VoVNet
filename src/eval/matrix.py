"""Multi-dataset evaluation utilities for VoVNet."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.collate import VLMDataCollator
from src.eval.matrix_spec import EvalDatasetSpec, build_dataset, get_metric_fn
from src.models.base_vlm import BaseVLM
from src.models.uncertainty import expected_calibration_error
from src.models.vovnet import Action, VoVNet
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


def evaluate_dataset(
    model: VoVNet,
    loader: DataLoader,
    metric_fn: Callable[[Iterable[str], Iterable[str]], float],
    cost_weight: Optional[float],
    profile: bool,
) -> Dict[str, Any]:
    raw_model = getattr(model, "module", model)
    raw_model.eval()
    total_acc = 0.0
    total_cost = 0.0
    total_count = 0
    total_ece = 0.0
    total_ece_count = 0
    action_counts = torch.zeros(len(Action), dtype=torch.long)
    action_names = {int(action): action.name.lower() for action in Action}
    profiler = BatchProfiler(profile, action_names=action_names)

    tokenizer = raw_model.base_vlm.tokenizer
    with torch.no_grad():
        for batch in loader:
            if profile:
                profiler.start()
            outputs = _forward_with_cost(raw_model, batch, cost_weight)
            if profile:
                profiler.stop(
                    batch_size=batch["input_ids"].size(0),
                    seq_len=batch["input_ids"].size(1),
                    actions=outputs["actions"],
                )
            preds = _decode_from_logits(outputs["logits"], batch["labels"], tokenizer)
            acc = metric_fn(preds, batch.get("answers", []))
            total_acc += acc * len(preds)
            total_cost += outputs["expected_cost"].sum().item()
            total_count += len(preds)
            action_counts += torch.bincount(
                outputs["actions"].detach().cpu(), minlength=len(Action)
            )
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
    latency = summary.get("overall", {}).get("latency_ms", {})
    return {
        "accuracy": total_acc / max(1, total_count),
        "action_rates": action_rates,
        "avg_token_cost": total_cost / max(1, total_count),
        "latency": latency,
        "profile": summary if profile else None,
        "ece": (total_ece / max(1, total_ece_count)),
    }


def rows_from_results(
    dataset: str, metric_name: str, results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        action_rates = item.get("action_rates", [0.0, 0.0, 0.0])
        latency = item.get("latency", {})
        rows.append(
            {
                "dataset": dataset,
                "metric": metric_name,
                "lambda_cost": item.get("lambda_cost", 0.0),
                "accuracy": item.get("accuracy"),
                "action_rate_no": action_rates[0],
                "action_rate_coarse": action_rates[1],
                "action_rate_full": action_rates[2],
                "avg_token_cost": item.get("avg_token_cost"),
                "latency_p50_ms": latency.get("p50", 0.0),
                "latency_p90_ms": latency.get("p90", 0.0),
                "ece": item.get("ece"),
            }
        )
    return rows
