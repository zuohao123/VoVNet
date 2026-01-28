"""Train VoVNet with native PyTorch DDP (no accelerate)."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import normalize_baseline_name, resolve_baseline_actions
from src.config.config import Config
from src.data.adapters.hf_dataset import HFDatasetAdapter
from src.data.adapters.jsonl import JsonlVQADataset
from src.data.collate import VLMDataCollator
from src.eval.metrics import exact_match_score
from src.models.base_vlm import BaseVLM
from src.models.uncertainty import expected_calibration_error
from src.models.vision_budget import VisionBudgetController
from src.models.vovnet import Action, VoVNet
from src.training.losses import (
    compute_entropy_loss,
    compute_policy_targets,
    compute_task_loss_per_sample,
    compute_total_loss,
)
from src.training.schedulers import build_scheduler
from src.training.trainer import _compute_ece, _format_budget, _format_duration
from src.utils.logging import setup_logging
from src.utils.profiling import BatchProfiler
from src.utils.run_metadata import collect_dataset_metadata, write_run_metadata
from src.utils.seed import set_seed
from src.utils.stats import pearsonr, spearmanr

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VoVNet with DDP")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        help="Only load model weights from checkpoint (ignore optimizer/scheduler).",
    )
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def _normalize_dataset_name(name: Any) -> str:
    if name is None:
        return "unknown"
    return str(name).strip().lower()


def _extract_dataset_name(item: Any) -> str:
    if isinstance(item, dict):
        name = item.get("dataset") or item.get("source")
        if name is None:
            meta = item.get("meta")
            if isinstance(meta, dict):
                name = meta.get("dataset")
        return _normalize_dataset_name(name)
    return "unknown"


def _build_sample_weights(dataset: Any, ratios: Dict[str, float]) -> List[float]:
    ratio_map = {_normalize_dataset_name(k): float(v) for k, v in ratios.items()}
    total = sum(ratio_map.values())
    if total <= 0:
        raise ValueError("data.sample_ratios must sum to a positive value")
    ratio_map = {k: v / total for k, v in ratio_map.items()}

    if hasattr(dataset, "items"):
        names = [_extract_dataset_name(item) for item in dataset.items]
    else:
        names = [_extract_dataset_name(dataset[i]) for i in range(len(dataset))]

    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1

    missing = {k: v for k, v in counts.items() if k not in ratio_map}
    if missing:
        logger.warning("sample_ratios missing datasets=%s; they will be dropped", missing)

    weights: List[float] = []
    for name in names:
        ratio = ratio_map.get(name, 0.0)
        denom = counts.get(name, 1)
        weights.append(ratio / max(1, denom))

    return weights


class DistributedWeightedSampler(Sampler[int]):
    """Weighted sampler compatible with DDP."""

    def __init__(
        self,
        weights: List[float],
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=g
        ).tolist()
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def build_dataset(cfg: Config, split: str) -> torch.utils.data.Dataset:
    if split == "train" and cfg.data.train_jsonl:
        return JsonlVQADataset(
            cfg.data.train_jsonl,
            text_field=cfg.data.text_field,
            answer_field=cfg.data.answer_field,
            image_field=cfg.data.image_field,
            max_samples=cfg.data.max_samples,
        )
    if split == "eval" and cfg.data.eval_jsonl:
        return JsonlVQADataset(
            cfg.data.eval_jsonl,
            text_field=cfg.data.text_field,
            answer_field=cfg.data.answer_field,
            image_field=cfg.data.image_field,
            max_samples=cfg.data.max_samples,
        )
    if cfg.data.hf_dataset_name:
        return HFDatasetAdapter(
            cfg.data.hf_dataset_name,
            split=cfg.data.hf_dataset_split,
            text_field=cfg.data.text_field,
            answer_field=cfg.data.answer_field,
            image_field=cfg.data.image_field,
            max_samples=cfg.data.max_samples,
        )
    raise ValueError("No dataset configured")


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

    if cfg.model.use_lora:
        for param in base_vlm.model.parameters():
            param.requires_grad = False
        base_vlm.apply_lora(
            r=cfg.model.lora_r,
            alpha=cfg.model.lora_alpha,
            dropout=cfg.model.lora_dropout,
            target_modules=cfg.model.lora_target_modules,
        )

    if cfg.model.freeze_vision_encoder:
        base_vlm.freeze_vision_encoder()
        if full_vlm is not None:
            full_vlm.freeze_vision_encoder()

    if cfg.training.gradient_checkpointing:
        base_vlm.enable_gradient_checkpointing()
        if full_vlm is not None:
            full_vlm.enable_gradient_checkpointing()

    budget = VisionBudgetController(**cfg.vision_budget.__dict__)
    model = VoVNet(
        base_vlm=base_vlm,
        full_vlm=full_vlm,
        vision_budget=budget,
        vow_hidden_dim=cfg.policy.vow_hidden_dim,
        gumbel_tau=cfg.policy.gumbel_tau,
        use_straight_through=cfg.policy.use_straight_through,
        eval_sample=cfg.policy.eval_sample,
        explore_prob=cfg.policy.explore_prob,
        policy_mode=cfg.policy.policy_mode,
        fallback_mode=cfg.policy.fallback_mode,
        fallback_entropy_threshold=cfg.policy.fallback_entropy_threshold,
        fallback_margin_threshold=cfg.policy.fallback_margin_threshold,
        cost_scale=cfg.policy.cost_scale,
        cost_c1=cfg.policy.cost_c1,
        cost_c2=cfg.policy.cost_c2,
    )
    return model


def build_stage_schedule(cfg: Config) -> List[Dict[str, Any]]:
    stages: List[Dict[str, Any]] = []
    stage1_epochs = int(cfg.training.stage1_epochs)
    if stage1_epochs > 0:
        stage1_baseline = cfg.training.stage1_baseline_name
        if not stage1_baseline or str(stage1_baseline).strip().lower() in {"none", "null"}:
            stage1_baseline = "always_full"
        stage1_lambda = (
            float(cfg.training.stage1_lambda_cost)
            if cfg.training.stage1_lambda_cost is not None
            else 0.0
        )
        stages.append(
            {
                "name": "stage1_full",
                "epochs": stage1_epochs,
                "baseline_name": stage1_baseline,
                "lambda_cost": stage1_lambda,
                "max_steps": cfg.training.stage1_max_steps,
            }
        )

    stage2_epochs = int(cfg.training.epochs) - stage1_epochs
    if stage2_epochs > 0:
        stages.append(
            {
                "name": "stage2_policy",
                "epochs": stage2_epochs,
                "baseline_name": cfg.policy.baseline_name,
                "lambda_cost": float(cfg.policy.lambda_cost),
                "max_steps": cfg.training.stage2_max_steps,
            }
        )
    return stages


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for key in ("input_ids", "attention_mask", "labels", "has_choices"):
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def _setup_autocast(mixed_precision: str):
    if not torch.cuda.is_available():
        return nullcontext(), None
    if mixed_precision == "fp16":
        return torch.cuda.amp.autocast(dtype=torch.float16), torch.cuda.amp.GradScaler()
    if mixed_precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            logger.warning("bf16 not supported on this GPU; falling back to fp16.")
            return torch.cuda.amp.autocast(dtype=torch.float16), torch.cuda.amp.GradScaler()
        return torch.cuda.amp.autocast(dtype=torch.bfloat16), None
    return nullcontext(), None


def _configure_cuda(rank: int) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if rank == 0:
        logger.info(
            "CUDA config: cudnn_enabled=%s cudnn_available=%s cudnn_version=%s tf32_matmul=%s tf32_cudnn=%s",
            torch.backends.cudnn.enabled,
            torch.backends.cudnn.is_available(),
            torch.backends.cudnn.version(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )


def _schedule_value(start: float, end: float, warmup_steps: int, step: int) -> float:
    if warmup_steps > 0:
        progress = min(1.0, (step + 1) / float(warmup_steps))
    else:
        progress = 1.0
    return float(start) + progress * (float(end) - float(start))


def _guard_needed_now(
    *,
    ratio: float,
    guard_window: int,
    guard_seen: int,
    current_count: int,
    batch_size: int,
) -> int:
    """Compute how many samples must be forced now to hit a window quota."""
    if guard_window <= 0 or ratio <= 0.0:
        return 0
    required = int(math.ceil(float(ratio) * float(guard_window)))
    remaining_after = max(0, guard_window - (guard_seen + batch_size))
    return max(0, required - (current_count + remaining_after))


def _apply_guard_quotas(
    policy_targets: torch.Tensor,
    loss_triplet: torch.Tensor,
    *,
    guard_window: int,
    guard_seen: int,
    guard_counts: list[int],
    min_full_ratio: float,
    min_coarse_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor | None, int, list[int]]:
    """Enforce per-window minimum action ratios, even with batch size 1."""
    if guard_window <= 0 or policy_targets.numel() == 0:
        return policy_targets, None, guard_seen, guard_counts
    if min_full_ratio <= 0.0 and min_coarse_ratio <= 0.0:
        return policy_targets, None, guard_seen, guard_counts

    batch_size = int(policy_targets.shape[0])
    forced_mask = torch.zeros_like(policy_targets, dtype=torch.bool)

    def _force_action(action: Action, need: int) -> None:
        nonlocal forced_mask, policy_targets
        if need <= 0:
            return
        candidates = policy_targets != action
        # Do not override FULL when satisfying COARSE quota.
        if action == Action.COARSE_VISION:
            candidates = candidates & (policy_targets != Action.FULL_VISION)
        idx = candidates.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        action_losses = loss_triplet[idx, int(action)]
        order = torch.argsort(action_losses)
        pick = idx[order[:need]]
        policy_targets[pick] = int(action)
        forced_mask[pick] = True

    need_full = _guard_needed_now(
        ratio=min_full_ratio,
        guard_window=guard_window,
        guard_seen=guard_seen,
        current_count=int(guard_counts[int(Action.FULL_VISION)]),
        batch_size=batch_size,
    )
    _force_action(Action.FULL_VISION, need_full)

    need_coarse = _guard_needed_now(
        ratio=min_coarse_ratio,
        guard_window=guard_window,
        guard_seen=guard_seen,
        current_count=int(guard_counts[int(Action.COARSE_VISION)]),
        batch_size=batch_size,
    )
    _force_action(Action.COARSE_VISION, need_coarse)

    counts_batch = torch.bincount(policy_targets, minlength=len(Action)).tolist()
    guard_counts = [int(guard_counts[i]) + int(counts_batch[i]) for i in range(len(Action))]
    guard_seen += batch_size
    if guard_seen >= guard_window:
        guard_seen = 0
        guard_counts = [0] * len(Action)

    if not forced_mask.any():
        forced_mask = None
    return policy_targets, forced_mask, guard_seen, guard_counts


def _log_step(
    avg_losses: Dict[str, float],
    outputs: Dict[str, Any],
    progress: Dict[str, Any],
    window_stats: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    lambda_cost: float,
    train_profiler: BatchProfiler,
    model: nn.Module,
    *,
    argmax_actions: torch.Tensor | None = None,
    policy_targets: torch.Tensor | None = None,
    loss_triplet: torch.Tensor | None = None,
    lambda_policy: float | None = None,
    lambda_entropy: float | None = None,
    open_quantile: float | None = None,
) -> None:
    actions = outputs["actions"]
    num_actions = len(Action)
    action_rates = torch.bincount(actions, minlength=num_actions).float() / max(1, actions.numel())
    if argmax_actions is None:
        action_logits = outputs.get("action_logits")
        if action_logits is not None:
            argmax_actions = action_logits.argmax(dim=-1)
        else:
            argmax_actions = actions
    argmax_rates = torch.bincount(argmax_actions, minlength=num_actions).float() / max(
        1, argmax_actions.numel()
    )
    target_rates = None
    if policy_targets is not None and policy_targets.numel() > 0:
        target_rates = torch.bincount(policy_targets, minlength=num_actions).float() / float(
            policy_targets.numel()
        )
    action_probs = outputs.get("action_probs")
    if action_probs is not None:
        entropy = -(action_probs * (action_probs + 1e-8).log()).sum(dim=-1)
        action_entropy = entropy.mean().item()
    else:
        action_entropy = 0.0
    mean_loss_triplet = None
    if loss_triplet is not None and loss_triplet.numel() > 0:
        mean_loss_triplet = loss_triplet.float().mean(dim=0).tolist()
    vision_tokens = outputs.get("vision_tokens")
    vision_mean = vision_tokens.float().mean().item() if vision_tokens is not None else 0.0
    token_count_coarse = outputs.get("token_count_coarse")
    token_count_full = outputs.get("token_count_full")
    coarse_mean = (
        token_count_coarse.float().mean().item() if token_count_coarse is not None else 0.0
    )
    full_mean = (
        token_count_full.float().mean().item() if token_count_full is not None else 0.0
    )
    expected_cost_mean = outputs["expected_cost"].float().mean().item()
    seq_len = outputs["logits"].shape[1]
    flops_proxy = vision_mean * seq_len
    labels = outputs.get("labels")
    ece_value = _compute_ece(outputs["logits"], labels)
    latency_ms = 0.0
    mem_mb = 0.0
    tokens_s = 0.0
    if train_profiler.enabled and train_profiler.last_batch is not None:
        profile = train_profiler.last_batch
        latency_ms = profile.latency_ms
        mem_mb = profile.max_mem_mb
        tokens_s = profile.tokens_s

    gain_pred = outputs.get("gain_pred")
    gain_true = outputs.get("gain_true")
    gain_corr = ""
    if gain_pred is not None and gain_true is not None:
        gain_pred = gain_pred.detach()
        gain_true = gain_true.detach()
        coarse_pearson = pearsonr(gain_pred[:, 0], gain_true[:, 0])
        full_pearson = pearsonr(gain_pred[:, 1], gain_true[:, 1])
        coarse_spearman = spearmanr(gain_pred[:, 0], gain_true[:, 0])
        full_spearman = spearmanr(gain_pred[:, 1], gain_true[:, 1])
        gain_corr = (
            f" gain_corr_pearson=[{coarse_pearson:.3f},{full_pearson:.3f}]"
            f" gain_corr_spearman=[{coarse_spearman:.3f},{full_spearman:.3f}]"
        )

    steps_per_epoch = progress.get("steps_per_epoch")
    total_steps = progress.get("total_steps")
    step_in_epoch = progress.get("step_in_epoch")
    epoch = progress.get("epoch")
    epochs = progress.get("epochs")
    step_text = (
        f"{step_in_epoch}/{steps_per_epoch}"
        if steps_per_epoch is not None
        else f"{step_in_epoch}"
    )
    global_text = (
        f"{progress.get('global_step')}/{total_steps}"
        if total_steps
        else f"{progress.get('global_step')}"
    )
    progress_pct = progress.get("progress_pct")
    progress_text = f"{progress_pct:.2f}%" if progress_pct is not None else "n/a"
    eta_text = _format_duration(progress.get("eta_seconds"))
    elapsed_text = _format_duration(progress.get("elapsed_seconds"))
    stage_name = progress.get("stage", "stage")
    lr = float(optimizer.param_groups[0].get("lr", 0.0))
    raw_model = getattr(model, "module", model)
    budget_text = _format_budget(getattr(raw_model, "vision_budget", None))
    lambda_policy_value = float(lambda_policy) if lambda_policy is not None else float("nan")
    lambda_entropy_value = float(lambda_entropy) if lambda_entropy is not None else float("nan")
    open_quantile_value = float(open_quantile) if open_quantile is not None else float("nan")

    logger.info(
        (
            "stage=%s epoch=%s/%s step=%s global_step=%s progress=%s eta=%s elapsed=%s "
            "window_samples=%s window_samples_s=%.2f window_tokens_s=%.2f "
            "avg_total=%.4f avg_task=%.4f avg_cost=%.4f avg_cal=%.4f "
            "avg_gain=%.4f avg_policy=%.4f "
            "avg_entropy_loss=%.4f "
            "lr=%.6g budget=%s lambda_cost=%.4f lambda_policy=%.4f lambda_entropy=%.4f open_q=%.3f "
            "action_entropy=%.4f action_ratio=%s action_ratio_argmax=%s target_ratio=%s "
            "loss_triplet_mean=%s "
            "vision_tokens=%.2f token_count_coarse=%.2f token_count_full=%.2f "
            "expected_cost=%.2f flops_proxy=%.2f ece=%.4f "
            "latency_ms=%.2f mem_peak_mb=%.2f tokens_s=%.2f%s"
        ),
        stage_name,
        epoch,
        epochs,
        step_text,
        global_text,
        progress_text,
        eta_text,
        elapsed_text,
        window_stats["samples"],
        window_stats["samples_s"],
        window_stats["tokens_s"],
        avg_losses["total_loss"],
        avg_losses["task_loss"],
        avg_losses["cost_loss"],
        avg_losses.get("calibration_loss", 0.0),
        avg_losses.get("gain_loss", 0.0),
        avg_losses.get("policy_loss", 0.0),
        avg_losses.get("entropy_loss", 0.0),
        lr,
        budget_text,
        lambda_cost,
        lambda_policy_value,
        lambda_entropy_value,
        open_quantile_value,
        action_entropy,
        action_rates.tolist(),
        argmax_rates.tolist(),
        target_rates.tolist() if target_rates is not None else None,
        mean_loss_triplet,
        vision_mean,
        coarse_mean,
        full_mean,
        expected_cost_mean,
        flops_proxy,
        ece_value,
        latency_ms,
        mem_mb,
        tokens_s,
        gain_corr,
    )


def _soft_mixture_forward(
    model: nn.Module,
    batch: Dict[str, Any],
    *,
    lambda_cost: float,
    lambda_entropy: float,
    action_temperature: float,
    cost_normalize: bool,
    mixture_branches: str,
    mixture_subsample_every: int,
    step_idx: int,
) -> tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    raw_model = getattr(model, "module", model)
    images = batch.get("images")
    token_count_coarse, token_count_full, coarse_inputs, full_inputs = (
        raw_model._prepare_token_counts(
            images=images,
            device=batch["attention_mask"].device,
            batch_size=batch["attention_mask"].shape[0],
        )
        if images is not None
        else (
            torch.zeros_like(batch["attention_mask"][:, 0]),
            torch.zeros_like(batch["attention_mask"][:, 0]),
            None,
            None,
        )
    )
    text_input_ids, text_attention_mask, text_labels = raw_model._prepare_text_and_vision_inputs(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch.get("labels"),
        coarse_inputs=coarse_inputs,
        full_inputs=full_inputs,
    )
    text_outputs, action_logits, _ = raw_model.text_first(
        text_input_ids, text_attention_mask
    )
    temperature = max(float(action_temperature), 1e-6)
    action_probs = torch.softmax(action_logits / temperature, dim=-1)

    compute_coarse = True
    compute_full = True
    if mixture_branches == "NF":
        compute_coarse = False
    elif mixture_branches == "NCF_subsample":
        interval = max(0, int(mixture_subsample_every))
        if interval > 0 and (step_idx % interval != 0):
            # Skip the most expensive branch on non-interval steps.
            compute_full = False

    if not compute_coarse or not compute_full:
        action_probs = action_probs.clone()
        if not compute_coarse:
            action_probs[:, Action.COARSE_VISION] = 0.0
        if not compute_full:
            action_probs[:, Action.FULL_VISION] = 0.0
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    logits_no = text_outputs.logits
    labels_no = text_labels
    loss_no = compute_task_loss_per_sample(logits_no, labels_no)

    logits_coarse = logits_no
    labels_coarse = labels_no
    loss_coarse = loss_no
    if compute_coarse and images is not None:
        logits_coarse, _ = raw_model._forward_mode(
            text_input_ids,
            text_attention_mask,
            images,
            text_outputs,
            mode="coarse",
            vision_inputs=coarse_inputs,
        )
        if coarse_inputs is not None and coarse_inputs.labels is not None:
            labels_coarse = coarse_inputs.labels
        loss_coarse = compute_task_loss_per_sample(logits_coarse, labels_coarse)

    logits_full = logits_no
    labels_full = labels_no
    loss_full = loss_no
    if compute_full and images is not None:
        logits_full, _ = raw_model._forward_mode(
            text_input_ids,
            text_attention_mask,
            images,
            text_outputs,
            mode="full",
            vision_inputs=full_inputs,
        )
        if full_inputs is not None and full_inputs.labels is not None:
            labels_full = full_inputs.labels
        loss_full = compute_task_loss_per_sample(logits_full, labels_full)

    loss_triplet = torch.stack([loss_no, loss_coarse, loss_full], dim=-1)
    task_loss = (action_probs * loss_triplet).sum(dim=-1).mean()

    expected_cost = raw_model._compute_expected_cost(
        action_probs, token_count_coarse, token_count_full
    )
    if cost_normalize:
        expected_cost = expected_cost / token_count_full.clamp(min=1)
    cost_loss = expected_cost.mean() * float(lambda_cost)
    entropy_loss = compute_entropy_loss(action_probs, float(lambda_entropy))
    total_loss = task_loss + cost_loss + entropy_loss

    p0 = action_probs[:, Action.NO_VISION].view(-1, 1, 1)
    p1 = action_probs[:, Action.COARSE_VISION].view(-1, 1, 1)
    p2 = action_probs[:, Action.FULL_VISION].view(-1, 1, 1)
    logits = p0 * logits_no + p1 * logits_coarse + p2 * logits_full
    expected_tokens = (
        action_probs[:, Action.COARSE_VISION] * token_count_coarse
        + action_probs[:, Action.FULL_VISION] * token_count_full
    )

    outputs = {
        "logits": logits,
        "labels": labels_no,
        "action_logits": action_logits,
        "action_probs": action_probs,
        "actions": action_probs.argmax(dim=-1),
        "expected_cost": expected_cost,
        "vision_tokens": expected_tokens,
        "token_count_coarse": token_count_coarse,
        "token_count_full": token_count_full,
        "loss_triplet": loss_triplet,
    }
    losses = {
        "total_loss": total_loss,
        "task_loss": task_loss,
        "cost_loss": cost_loss,
        "calibration_loss": torch.tensor(0.0, device=total_loss.device),
        "gain_loss": torch.tensor(0.0, device=total_loss.device),
        "entropy_loss": entropy_loss,
        "policy_loss": torch.tensor(0.0, device=total_loss.device),
        "prior_loss": torch.tensor(0.0, device=total_loss.device),
    }
    return outputs, losses


def _enforce_min_full_ratio_soft(soft_targets: torch.Tensor, min_ratio: float) -> torch.Tensor:
    if min_ratio <= 0.0:
        return soft_targets
    full_idx = int(Action.FULL_VISION)
    mean_full = soft_targets[:, full_idx].mean()
    if float(mean_full.item()) >= min_ratio:
        return soft_targets
    denom = max(1e-8, float(1.0 - mean_full.item()))
    blend = max(0.0, min(1.0, float((min_ratio - float(mean_full.item())) / denom)))
    if blend <= 0.0:
        return soft_targets
    full_onehot = torch.zeros_like(soft_targets)
    full_onehot[:, full_idx] = 1.0
    mixed = (1.0 - blend) * soft_targets + blend * full_onehot
    return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    lambda_cost: float,
    baseline_name: Optional[str],
    device: torch.device,
    distributed: bool,
) -> Dict[str, Any]:
    raw_model = getattr(model, "module", model)
    raw_model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    total_ece = 0.0
    total_ece_count = 0
    total_cost = 0.0
    total_vision_tokens = 0.0
    action_counts = torch.zeros(len(Action), dtype=torch.long, device=device)
    tokenizer = raw_model.base_vlm.tokenizer

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            batch_size = batch["input_ids"].size(0)
            actions, _, drop_images = resolve_baseline_actions(
                baseline_name, batch_size, batch["input_ids"].device
            )
            if actions is None:
                outputs = raw_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    images=batch.get("images"),
                    labels=batch.get("labels"),
                )
            else:
                images = None if drop_images else batch.get("images")
                outputs = raw_model.forward_with_actions(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    images=images,
                    actions=actions,
                    labels=batch.get("labels"),
                )
            losses = compute_total_loss(
                outputs["logits"],
                outputs.get("labels"),
                outputs["expected_cost"],
                lambda_cost,
            )
            total_loss += losses["task_loss"].item() * batch_size
            total_cost += outputs["expected_cost"].sum().item()
            vision_tokens = outputs.get("vision_tokens")
            if vision_tokens is not None:
                total_vision_tokens += vision_tokens.sum().item()
            action_counts += torch.bincount(
                outputs["actions"].detach(), minlength=len(Action)
            )
            labels = outputs.get("labels") if outputs.get("labels") is not None else batch.get("labels")
            if labels is not None and tokenizer is not None:
                mask = labels.ne(-100)
                if mask.any():
                    flat_logits = outputs["logits"][mask]
                    flat_labels = labels[mask]
                    probs = torch.softmax(flat_logits, dim=-1)
                    ece, _ = expected_calibration_error(probs, flat_labels)
                    total_ece += float(ece.item()) * int(mask.sum().item())
                    total_ece_count += int(mask.sum().item())
                preds = _decode_from_logits(outputs["logits"], labels, tokenizer)
                acc = exact_match_score(preds, batch.get("answers", []))
                total_acc += acc * len(preds)
                total_count += len(preds)

    if distributed:
        totals = torch.tensor(
            [
                total_loss,
                total_acc,
                total_count,
                total_ece,
                total_ece_count,
                total_cost,
                total_vision_tokens,
            ],
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        (
            total_loss,
            total_acc,
            total_count,
            total_ece,
            total_ece_count,
            total_cost,
            total_vision_tokens,
        ) = totals.tolist()
        dist.all_reduce(action_counts, op=dist.ReduceOp.SUM)

    avg_loss = total_loss / max(1, total_count or 1)
    avg_acc = total_acc / max(1, total_count)
    avg_cost = total_cost / max(1, total_count)
    avg_vision_tokens = total_vision_tokens / max(1, total_count)
    action_rates = (action_counts.float() / max(1, total_count)).tolist()
    metrics = {
        "val_loss": avg_loss,
        "val_accuracy": avg_acc,
        "val_ece": total_ece / max(1, total_ece_count),
        "avg_cost": avg_cost,
        "avg_vision_token_count": avg_vision_tokens,
        "action_ratio": action_rates,
    }
    return metrics


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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    _configure_cuda(rank)

    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        if cfg.model.torch_dtype == "bfloat16":
            if rank == 0:
                logger.warning("GPU does not support bf16; switching model dtype to fp16.")
            cfg.model.torch_dtype = "float16"
        if cfg.training.mixed_precision == "bf16":
            if rank == 0:
                logger.warning("GPU does not support bf16; switching mixed_precision to fp16.")
            cfg.training.mixed_precision = "fp16"

    stages = build_stage_schedule(cfg)
    for stage in stages:
        stage_baseline = normalize_baseline_name(stage["baseline_name"])
        if stage_baseline in {
            "uncertainty_threshold",
            "random_policy_matched",
            "token_merge_prune_proxy",
            "resolution_scaling",
            "multi_granularity_proxy",
        }:
            raise RuntimeError(f"{stage_baseline} baseline is eval-only; skip training")
        if stage_baseline == "vision_token_pruning_proxy" and not cfg.policy.finetune_pruning:
            raise RuntimeError(
                "vision_token_pruning_proxy is eval-only unless finetune_pruning=true"
            )

    seed = cfg.training.seed + rank
    set_seed(seed)

    output_dir = Path(cfg.training.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    if model.base_vlm.tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded; check model name")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_base = [
        p for p in model.base_vlm.model.parameters() if p.requires_grad
    ]
    if rank == 0:
        logger.info(
            "Trainable params: total=%s base_vlm=%s",
            len(trainable_params),
            len(trainable_base),
        )
    if not trainable_params:
        raise RuntimeError(
            "No trainable parameters found. Check LoRA target_modules or disable use_lora."
        )
    if not trainable_base:
        for stage in stages:
            if (
                stage["name"].startswith("stage1")
                and float(stage["lambda_cost"]) == 0.0
                and not cfg.policy.gain_supervision
            ):
                raise RuntimeError(
                    "Stage1 has no trainable base_vlm params (LoRA not attached). "
                    "Update LoRA target_modules or disable use_lora before running stage1."
                )

    train_dataset = build_dataset(cfg, "train")
    eval_dataset = build_dataset(cfg, "eval") if cfg.data.eval_jsonl else None

    collator = VLMDataCollator(
        tokenizer=model.base_vlm.tokenizer,
        prompt_template=cfg.data.prompt_template,
    )

    train_sampler: Sampler[int] | None = None
    sample_weights = None
    if cfg.data.sample_ratios:
        sample_weights = _build_sample_weights(train_dataset, cfg.data.sample_ratios)
        if distributed:
            train_sampler = DistributedWeightedSampler(
                sample_weights, num_replicas=world_size, rank=rank, seed=cfg.training.seed
            )
        else:
            train_sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )
    elif distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.per_device_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_sampler = (
            DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if distributed
            else None
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            sampler=eval_sampler,
            collate_fn=collator,
        )

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    num_training_steps = cfg.training.epochs * len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=num_training_steps,
    )

    resume_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(checkpoint.get("model", {}), strict=False)
        if not args.resume_model_only:
            optimizer_state = checkpoint.get("optimizer")
            scheduler_state = checkpoint.get("scheduler")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            if scheduler_state is not None and scheduler is not None:
                scheduler.load_state_dict(scheduler_state)
            resume_step = int(checkpoint.get("global_step", 0))
        if rank == 0:
            logger.info(
                "Loaded checkpoint %s (model_only=%s, resume_step=%s)",
                ckpt_path,
                args.resume_model_only,
                resume_step,
            )

    if rank == 0:
        dataset_meta: dict[str, Any] = {}
        dataset_meta["train"] = collect_dataset_metadata(
            train_dataset,
            {
                "split": "train",
                "source": "jsonl" if cfg.data.train_jsonl else "hf",
                "jsonl": cfg.data.train_jsonl,
                "hf_dataset_name": cfg.data.hf_dataset_name,
                "hf_dataset_split": cfg.data.hf_dataset_split,
                "max_samples": cfg.data.max_samples,
            },
        )
        if eval_dataset is not None:
            dataset_meta["eval"] = collect_dataset_metadata(
                eval_dataset,
                {
                    "split": "eval",
                    "source": "jsonl" if cfg.data.eval_jsonl else "hf",
                    "jsonl": cfg.data.eval_jsonl,
                    "hf_dataset_name": cfg.data.hf_dataset_name,
                    "hf_dataset_split": cfg.data.hf_dataset_split,
                    "max_samples": cfg.data.max_samples,
                },
            )
        write_run_metadata(
            output_dir=output_dir,
            stage="train_ddp",
            cfg=cfg,
            config_paths=args.config,
            datasets=dataset_meta,
            extra={"output_dir": str(output_dir)},
        )

    mixed_precision = cfg.training.mixed_precision
    autocast_ctx, scaler = _setup_autocast(mixed_precision)
    grad_accum = max(1, int(cfg.training.gradient_accumulation))
    steps_per_epoch = len(train_loader)
    total_steps = 0
    for stage in stages:
        stage_max_steps = stage.get("max_steps")
        if stage_max_steps is not None and stage_max_steps > 0:
            total_steps += int(stage_max_steps)
        else:
            total_steps += int(stage.get("epochs", 0)) * steps_per_epoch

    action_names = {int(action): action.name.lower() for action in Action}
    train_profiler = BatchProfiler(cfg.training.profile, action_names=action_names)
    train_mode = str(getattr(cfg.training, "train_mode", "teacher_policy")).strip().lower()
    mixture_branches = str(getattr(cfg.policy, "mixture_branches", "NCF"))
    mixture_subsample_every = int(getattr(cfg.policy, "mixture_subsample_every", 0) or 0)
    action_temperature = float(getattr(cfg.policy, "action_temperature", 1.0))

    if rank == 0:
        logger.info(
            "Train setup: epochs=%s steps_per_epoch=%s total_steps=%s world_size=%s grad_accum=%s",
            cfg.training.epochs,
            steps_per_epoch,
            total_steps,
            world_size,
            grad_accum,
        )
        logger.info("Stage schedule: %s", stages)

    global_step = resume_step if resume_step > 0 else 0
    train_start = time.time()
    window_start = train_start
    window_samples = 0
    window_tokens = 0
    window_batches = 0
    accum_has_grad = False
    window_losses: Dict[str, float] = {
        "total_loss": 0.0,
        "task_loss": 0.0,
        "cost_loss": 0.0,
        "calibration_loss": 0.0,
        "gain_loss": 0.0,
        "entropy_loss": 0.0,
        "policy_loss": 0.0,
        "prior_loss": 0.0,
    }
    summary_every = int(getattr(cfg.training, "summary_every", 0) or 0)
    period_samples = torch.zeros(1, device=device)
    period_action_counts = torch.zeros(len(Action), device=device)
    period_argmax_counts = torch.zeros(len(Action), device=device)
    period_cost_sum = torch.zeros(1, device=device)
    period_task_sum = torch.zeros(1, device=device)
    period_policy_sum = torch.zeros(1, device=device)
    period_entropy_sum = torch.zeros(1, device=device)
    period_action_prob_sum = torch.zeros(len(Action), device=device)
    period_target_counts = torch.zeros(len(Action), device=device)
    period_target_samples = torch.zeros(1, device=device)
    period_loss_triplet_sum = torch.zeros(len(Action), device=device)
    period_triplet_samples = torch.zeros(1, device=device)
    period_token_coarse_sum = torch.zeros(1, device=device)
    period_token_full_sum = torch.zeros(1, device=device)
    period_label_tokens = torch.zeros(1, device=device)
    period_label_total = torch.zeros(1, device=device)

    epoch_idx = 0
    for stage in stages:
        stage_name = stage["name"]
        stage_epochs = int(stage["epochs"])
        stage_baseline = normalize_baseline_name(stage["baseline_name"])
        stage_lambda_cost = float(stage["lambda_cost"])
        stage_max_steps = stage.get("max_steps")
        if stage_max_steps is not None:
            stage_max_steps = int(stage_max_steps)
        if (
            cfg.training.gradient_checkpointing
            and stage_name == "stage2_policy"
            and stage_baseline is None
        ):
            raw_model = model.module if isinstance(model, DDP) else model
            if raw_model.full_vlm is None:
                if rank == 0:
                    logger.warning(
                        "Disabling gradient checkpointing for stage2_policy to avoid "
                        "DDP reentrant backward errors when reusing the base VLM."
                    )
                raw_model.base_vlm.disable_gradient_checkpointing()
        if rank == 0:
            logger.info(
                "Starting %s: epochs=%s baseline=%s lambda_cost=%.4f max_steps=%s",
                stage_name,
                stage_epochs,
                stage_baseline,
                stage_lambda_cost,
                stage_max_steps,
            )
        stage_steps = 0
        collapse_window_steps = int(cfg.policy.collapse_warn_window_steps)
        collapse_warn_thresh = float(cfg.policy.collapse_warn_ratio_threshold)
        collapse_warn_enabled = (
            stage_baseline is None and rank == 0 and collapse_window_steps > 0
        )
        collapse_counters_sampled = [0] * len(Action)
        collapse_counters_argmax = [0] * len(Action)
        collapse_counters_target = [0] * len(Action)
        guard_window = int(getattr(cfg.policy, "policy_guard_window", 0) or 0)
        guard_seen = 0
        guard_counts = [0] * len(Action)
        if rank == 0 and stage_baseline is None and guard_window > 0:
            logger.info(
                "Policy guard enabled: window=%d min_full=%.3f min_coarse=%.3f",
                guard_window,
                float(cfg.policy.policy_min_full_ratio or 0.0),
                float(getattr(cfg.policy, "policy_min_coarse_ratio", 0.0) or 0.0),
            )

        def _maybe_warn_collapse(kind: str, ratios: list[float], counters: list[int]) -> None:
            if not collapse_warn_enabled:
                return
            for i, ratio in enumerate(ratios):
                if ratio < collapse_warn_thresh:
                    counters[i] += 1
                    if counters[i] == collapse_window_steps:
                        logger.warning(
                            "policy-collapse-warning window=%d kind=%s threshold=%.3f action=%s ratio=%.4f",
                            collapse_window_steps,
                            kind,
                            collapse_warn_thresh,
                            Action(i).name,
                            ratio,
                        )
                else:
                    counters[i] = 0

        for local_epoch in range(stage_epochs):
            epoch = epoch_idx + local_epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            for step, batch in enumerate(train_loader):
                if stage_max_steps is not None and stage_steps >= stage_max_steps:
                    break
                if train_profiler.enabled:
                    train_profiler.start()
                batch = _move_batch(batch, device)
                batch_size = batch["input_ids"].size(0)
                actions, _, drop_images = resolve_baseline_actions(
                    stage_baseline, batch_size, batch["input_ids"].device
                )
                lambda_cost = stage_lambda_cost
                if cfg.policy.cost_warmup_steps > 0 and stage_lambda_cost > 0:
                    warmup = float(cfg.policy.cost_warmup_steps)
                    progress = min(1.0, (stage_steps + 1) / warmup)
                    lambda_cost = stage_lambda_cost * progress
                is_last = step + 1 == steps_per_epoch
                sync_grad = ((step + 1) % grad_accum == 0) or is_last
                if distributed and isinstance(model, DDP) and not sync_grad:
                    sync_ctx = model.no_sync()
                else:
                    sync_ctx = nullcontext()

                with sync_ctx:
                    with autocast_ctx:
                        policy_ce_start = (
                            cfg.policy.policy_ce_weight_start
                            if cfg.policy.policy_ce_weight_start is not None
                            else cfg.policy.policy_ce_weight
                        )
                        policy_ce_end = (
                            cfg.policy.policy_ce_weight_end
                            if cfg.policy.policy_ce_weight_end is not None
                            else cfg.policy.policy_ce_weight
                        )
                        lambda_policy_step = _schedule_value(
                            policy_ce_start,
                            policy_ce_end,
                            int(cfg.policy.policy_ce_weight_warmup_steps),
                            int(stage_steps),
                        )
                        entropy_start = (
                            cfg.policy.entropy_weight_start
                            if cfg.policy.entropy_weight_start is not None
                            else cfg.policy.entropy_weight
                        )
                        entropy_end = (
                            cfg.policy.entropy_weight_end
                            if cfg.policy.entropy_weight_end is not None
                            else cfg.policy.entropy_weight
                        )
                        lambda_entropy_step = _schedule_value(
                            entropy_start,
                            entropy_end,
                            int(cfg.policy.entropy_weight_warmup_steps),
                            int(stage_steps),
                        )
                        open_q_start = (
                            cfg.policy.policy_open_quantile_start
                            if cfg.policy.policy_open_quantile_start is not None
                            else cfg.policy.policy_open_quantile
                        )
                        open_q_end = (
                            cfg.policy.policy_open_quantile_end
                            if cfg.policy.policy_open_quantile_end is not None
                            else cfg.policy.policy_open_quantile
                        )
                        open_quantile_step = _schedule_value(
                            float(open_q_start),
                            float(open_q_end),
                            int(cfg.policy.policy_open_quantile_warmup_steps),
                            int(stage_steps),
                        )
                        open_quantile_step = float(min(1.0, max(0.0, open_quantile_step)))
                        policy_targets_hard = None
                        policy_targets_soft = None
                        loss_triplet = None

                        if actions is None and train_mode == "soft_mixture":
                            outputs, losses = _soft_mixture_forward(
                                model,
                                batch,
                                lambda_cost=lambda_cost,
                                lambda_entropy=lambda_entropy_step,
                                action_temperature=action_temperature,
                                cost_normalize=cfg.policy.cost_normalize,
                                mixture_branches=mixture_branches,
                                mixture_subsample_every=mixture_subsample_every,
                                step_idx=int(stage_steps),
                            )
                            loss_triplet = outputs.get("loss_triplet")
                        else:
                            if actions is None:
                                compute_loss_triplet = cfg.policy.policy_target_mode != "none"
                                outputs = model(
                                    input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    images=batch.get("images"),
                                    labels=batch.get("labels"),
                                    compute_gain=cfg.policy.gain_supervision,
                                    compute_loss_triplet=compute_loss_triplet,
                                )
                            else:
                                images = None if drop_images else batch.get("images")
                                raw_model = model.module if isinstance(model, DDP) else model
                                outputs = raw_model.forward_with_actions(
                                    input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    images=images,
                                    actions=actions,
                                    labels=batch.get("labels"),
                                )
                            if actions is None and cfg.policy.policy_target_mode != "none":
                                loss_triplet = outputs.get("loss_triplet")
                                if loss_triplet is not None:
                                    # Cost-aware targets: penalize expensive actions in the triplet
                                    # using the same cost scale as expected_cost.
                                    token_count_coarse = outputs.get("token_count_coarse")
                                    token_count_full = outputs.get("token_count_full")
                                    if (
                                        token_count_coarse is not None
                                        and token_count_full is not None
                                        and float(lambda_cost) > 0.0
                                    ):
                                        raw_model = model.module if hasattr(model, "module") else model
                                        cost_scale = float(getattr(raw_model, "cost_scale", 1.0))
                                        zeros = torch.zeros_like(token_count_coarse)
                                        cost_triplet = torch.stack(
                                            [zeros, token_count_coarse, token_count_full], dim=-1
                                        )
                                        if cfg.policy.cost_normalize:
                                            denom = token_count_full.clamp(min=1).unsqueeze(-1)
                                            cost_triplet = cost_triplet / denom
                                        loss_scale = (
                                            loss_triplet.detach()
                                            .mean(dim=-1, keepdim=True)
                                            .clamp(min=1e-6)
                                        )
                                        loss_triplet = loss_triplet + cost_triplet * cost_scale * float(
                                            lambda_cost
                                        ) * loss_scale
                                    delta_no_start = (
                                        cfg.policy.policy_delta_no_start
                                        if cfg.policy.policy_delta_no_start is not None
                                        else cfg.policy.policy_delta_start
                                    )
                                    delta_no_end = (
                                        cfg.policy.policy_delta_no_end
                                        if cfg.policy.policy_delta_no_end is not None
                                        else cfg.policy.policy_delta_end
                                    )
                                    delta_coarse_start = (
                                        cfg.policy.policy_delta_coarse_start
                                        if cfg.policy.policy_delta_coarse_start is not None
                                        else cfg.policy.policy_delta_start
                                    )
                                    delta_coarse_end = (
                                        cfg.policy.policy_delta_coarse_end
                                        if cfg.policy.policy_delta_coarse_end is not None
                                        else cfg.policy.policy_delta_end
                                    )
                                    delta_no_warmup = int(
                                        cfg.policy.policy_delta_no_warmup_steps
                                        if cfg.policy.policy_delta_no_warmup_steps is not None
                                        else cfg.policy.policy_delta_warmup_steps
                                    )
                                    delta_coarse_warmup = int(
                                        cfg.policy.policy_delta_coarse_warmup_steps
                                        if cfg.policy.policy_delta_coarse_warmup_steps is not None
                                        else cfg.policy.policy_delta_warmup_steps
                                    )
                                    delta_no = _schedule_value(
                                        delta_no_start,
                                        delta_no_end,
                                        delta_no_warmup,
                                        int(stage_steps),
                                    )
                                    delta_coarse = _schedule_value(
                                        delta_coarse_start,
                                        delta_coarse_end,
                                        delta_coarse_warmup,
                                        int(stage_steps),
                                    )
                                    open_mask = None
                                    has_choices = batch.get("has_choices")
                                    if cfg.policy.policy_open_enable and has_choices is not None:
                                        open_mask = ~has_choices.bool()
                                    no_bias = _schedule_value(
                                        float(cfg.policy.policy_no_bias_start),
                                        float(cfg.policy.policy_no_bias_end),
                                        int(cfg.policy.policy_no_bias_warmup_steps),
                                        int(stage_steps),
                                    )
                                    open_visual_bias = _schedule_value(
                                        float(cfg.policy.policy_open_visual_bias_start),
                                        float(cfg.policy.policy_open_visual_bias_end),
                                        int(cfg.policy.policy_open_visual_bias_warmup_steps),
                                        int(stage_steps),
                                    )
                                    if no_bias > 0 or (
                                        open_visual_bias > 0 and open_mask is not None and open_mask.any()
                                    ):
                                        loss_triplet = loss_triplet.clone()
                                    if no_bias > 0:
                                        loss_triplet[:, Action.NO_VISION] = (
                                            loss_triplet[:, Action.NO_VISION] + no_bias
                                        )
                                    if (
                                        open_visual_bias > 0
                                        and open_mask is not None
                                        and open_mask.any()
                                    ):
                                        loss_triplet[open_mask, Action.COARSE_VISION] = (
                                            loss_triplet[open_mask, Action.COARSE_VISION]
                                            - open_visual_bias
                                        )
                                        loss_triplet[open_mask, Action.FULL_VISION] = (
                                            loss_triplet[open_mask, Action.FULL_VISION]
                                            - open_visual_bias
                                        )
                                    if cfg.policy.policy_target_mode == "loss_margin":
                                        policy_targets_hard = compute_policy_targets(
                                            loss_triplet, (delta_coarse, delta_no)
                                        )
                                    else:
                                        raise ValueError(
                                            f"Unknown policy_target_mode: {cfg.policy.policy_target_mode}"
                                        )
                                    open_targets = None
                                    if cfg.policy.policy_open_enable and open_mask is not None and open_mask.any():
                                        loss_no = loss_triplet[:, Action.NO_VISION]
                                        loss_coarse = loss_triplet[:, Action.COARSE_VISION]
                                        loss_full = loss_triplet[:, Action.FULL_VISION]
                                        best_vis = torch.where(
                                            loss_coarse <= loss_full,
                                            torch.full(
                                                loss_coarse.shape,
                                                int(Action.COARSE_VISION),
                                                device=loss_coarse.device,
                                                dtype=torch.long,
                                            ),
                                            torch.full(
                                                loss_full.shape,
                                                int(Action.FULL_VISION),
                                                device=loss_full.device,
                                                dtype=torch.long,
                                            ),
                                        )
                                        best_vis_loss = torch.minimum(loss_coarse, loss_full)
                                        force_steps = int(cfg.policy.policy_open_force_visual_warmup_steps)
                                        if force_steps > 0 and (stage_steps + 1) <= force_steps:
                                            action_name = cfg.policy.policy_open_force_visual_action
                                            if action_name == "best_vis":
                                                open_targets = best_vis
                                            elif action_name == "coarse":
                                                open_targets = torch.full(
                                                    best_vis.shape,
                                                    int(Action.COARSE_VISION),
                                                    device=best_vis.device,
                                                    dtype=torch.long,
                                                )
                                            else:
                                                open_targets = torch.full(
                                                    best_vis.shape,
                                                    int(Action.FULL_VISION),
                                                    device=best_vis.device,
                                                    dtype=torch.long,
                                                )
                                        else:
                                            diff = loss_no - best_vis_loss
                                            diff_open = diff[open_mask]
                                            if diff_open.numel() == 1:
                                                threshold = float(diff_open.detach().item())
                                            else:
                                                threshold = float(
                                                    torch.quantile(
                                                        diff_open.detach(),
                                                        open_quantile_step,
                                                    ).item()
                                                )
                                            margin = float(cfg.policy.policy_open_margin)
                                            if margin > 0:
                                                threshold = min(threshold, margin)
                                            allow_no = diff <= threshold
                                            open_targets = torch.where(
                                                allow_no,
                                                torch.full(
                                                    best_vis.shape,
                                                    int(Action.NO_VISION),
                                                    device=best_vis.device,
                                                    dtype=torch.long,
                                                ),
                                                best_vis if cfg.policy.policy_open_use_best_vis else torch.full(
                                                    best_vis.shape,
                                                    int(Action.FULL_VISION),
                                                    device=best_vis.device,
                                                    dtype=torch.long,
                                                ),
                                            )
                                        policy_targets_hard = policy_targets_hard.clone()
                                        policy_targets_hard[open_mask] = open_targets[open_mask]
                                    min_full_ratio_step = float(cfg.policy.policy_min_full_ratio or 0.0)
                                    if min_full_ratio_step > 0 and cfg.policy.policy_min_full_warmup_steps > 0:
                                        warmup = float(cfg.policy.policy_min_full_warmup_steps)
                                        min_full_ratio_step = min_full_ratio_step * min(
                                            1.0, (stage_steps + 1) / warmup
                                        )
                                    min_coarse_ratio_step = float(
                                        getattr(cfg.policy, "policy_min_coarse_ratio", 0.0) or 0.0
                                    )
                                    min_coarse_warmup = int(
                                        getattr(cfg.policy, "policy_min_coarse_warmup_steps", 0) or 0
                                    )
                                    if min_coarse_ratio_step > 0 and min_coarse_warmup > 0:
                                        min_coarse_ratio_step = min_coarse_ratio_step * min(
                                            1.0, (stage_steps + 1) / float(min_coarse_warmup)
                                        )
                                    guard_forced_mask = None
                                    if policy_targets_hard is not None and guard_window > 0:
                                        (
                                            policy_targets_hard,
                                            guard_forced_mask,
                                            guard_seen,
                                            guard_counts,
                                        ) = _apply_guard_quotas(
                                            policy_targets_hard,
                                            loss_triplet,
                                            guard_window=guard_window,
                                            guard_seen=guard_seen,
                                            guard_counts=guard_counts,
                                            min_full_ratio=float(min_full_ratio_step),
                                            min_coarse_ratio=float(min_coarse_ratio_step),
                                        )
                                    if cfg.policy.enable_soft_targets and policy_targets_hard is not None:
                                        temp = float(cfg.policy.soft_target_temperature)
                                        policy_targets_soft = torch.softmax(
                                            -loss_triplet / temp, dim=-1
                                        )
                                        if (
                                            open_targets is not None
                                            and open_mask is not None
                                            and open_mask.any()
                                        ):
                                            open_onehot = F.one_hot(
                                                open_targets[open_mask], num_classes=len(Action)
                                            ).float()
                                            policy_targets_soft = policy_targets_soft.clone()
                                            policy_targets_soft[open_mask] = open_onehot
                                        if guard_forced_mask is not None and guard_forced_mask.any():
                                            guard_onehot = F.one_hot(
                                                policy_targets_hard[guard_forced_mask],
                                                num_classes=len(Action),
                                            ).float()
                                            policy_targets_soft = policy_targets_soft.clone()
                                            policy_targets_soft[guard_forced_mask] = guard_onehot
                                        if min_full_ratio_step > 0:
                                            policy_targets_soft = _enforce_min_full_ratio_soft(
                                                policy_targets_soft, float(min_full_ratio_step)
                                            )
                            policy_sample_weights = None
                            if cfg.policy.policy_loss_weights:
                                weight_map = {
                                    _normalize_dataset_name(k): float(v)
                                    for k, v in cfg.policy.policy_loss_weights.items()
                                }
                                default_weight = weight_map.get("default", 1.0)
                                batch_datasets = batch.get("dataset") or []
                                if batch_datasets:
                                    weights = [
                                        weight_map.get(_normalize_dataset_name(name), default_weight)
                                        for name in batch_datasets
                                    ]
                                    policy_sample_weights = torch.tensor(
                                        weights, device=outputs["logits"].device, dtype=torch.float32
                                    )
                            expected_cost_for_loss = outputs["expected_cost"]
                            if cfg.policy.cost_normalize:
                                token_count_full = outputs.get("token_count_full")
                                if token_count_full is not None:
                                    expected_cost_for_loss = expected_cost_for_loss / token_count_full.clamp(
                                        min=1
                                    )
                            prior_weight_start = (
                                cfg.policy.policy_prior_weight_start
                                if cfg.policy.policy_prior_weight_start is not None
                                else cfg.policy.policy_prior_weight
                            )
                            prior_weight_end = (
                                cfg.policy.policy_prior_weight_end
                                if cfg.policy.policy_prior_weight_end is not None
                                else cfg.policy.policy_prior_weight
                            )
                            prior_weight_step = _schedule_value(
                                float(prior_weight_start),
                                float(prior_weight_end),
                                int(cfg.policy.policy_prior_weight_warmup_steps),
                                int(stage_steps),
                            )
                            prior_probs = None
                            if cfg.policy.policy_prior_probs:
                                prior_probs = torch.tensor(
                                    cfg.policy.policy_prior_probs,
                                    device=outputs["logits"].device,
                                    dtype=torch.float32,
                                )
                                if prior_probs.numel() != len(Action):
                                    if rank == 0:
                                        logger.warning(
                                            "policy_prior_probs length=%s != num_actions=%s; ignoring",
                                            prior_probs.numel(),
                                            len(Action),
                                        )
                                    prior_probs = None
                            losses = compute_total_loss(
                                outputs["logits"],
                                outputs.get("labels"),
                                expected_cost_for_loss,
                                lambda_cost,
                                action_probs=outputs.get("action_probs"),
                                lambda_entropy=lambda_entropy_step,
                                action_logits=outputs.get("action_logits"),
                                action_targets=policy_targets_hard,
                                action_targets_soft=policy_targets_soft,
                                lambda_policy=lambda_policy_step,
                                policy_sample_weights=policy_sample_weights,
                                policy_prior=prior_probs,
                                policy_prior_weight=prior_weight_step,
                                calibration_value=None,
                                lambda_cal=cfg.policy.calibration_lambda,
                                gain_pred=outputs.get("gain_pred"),
                                gain_true=outputs.get("gain_true"),
                                gain_loss_type=cfg.policy.gain_loss_type,
                                lambda_gain=cfg.policy.gain_loss_weight,
                                gain_margin=cfg.policy.gain_margin,
                            )
                        loss = losses["total_loss"] / grad_accum

                    if loss.requires_grad:
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        did_backward = True
                        accum_has_grad = True
                    elif rank == 0:
                        logger.warning(
                            "Skipping backward because loss has no grad_fn; check labels/LoRA."
                        )

                if sync_grad:
                    if accum_has_grad:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.max_grad_norm
                        )
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    elif rank == 0:
                        logger.warning(
                            "Skipping optimizer step because no gradients were accumulated."
                        )
                    accum_has_grad = False

                if train_profiler.enabled:
                    seq_len = batch["input_ids"].size(1)
                    train_profiler.stop(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        actions=outputs["actions"],
                    )

                action_logits = outputs.get("action_logits")
                if action_logits is not None and stage_baseline is None:
                    argmax_actions = action_logits.argmax(dim=-1)
                else:
                    argmax_actions = outputs["actions"]
                action_probs_for_log = outputs.get("action_probs")
                sampled_actions = outputs["actions"]
                if train_mode == "soft_mixture" and action_probs_for_log is not None:
                    sampled_actions = torch.multinomial(
                        action_probs_for_log.detach(), num_samples=1
                    ).squeeze(-1)
                loss_triplet = outputs.get("loss_triplet")

                if collapse_warn_enabled:
                    sampled_counts = torch.bincount(
                        sampled_actions.detach(), minlength=len(Action)
                    ).float()
                    sampled_ratios = (sampled_counts / max(1, batch_size)).tolist()
                    _maybe_warn_collapse("sampled", sampled_ratios, collapse_counters_sampled)

                    argmax_counts = torch.bincount(
                        argmax_actions.detach(), minlength=len(Action)
                    ).float()
                    argmax_ratios = (argmax_counts / max(1, batch_size)).tolist()
                    _maybe_warn_collapse("argmax", argmax_ratios, collapse_counters_argmax)

                    if policy_targets_hard is not None:
                        target_counts = torch.bincount(
                            policy_targets_hard.detach(), minlength=len(Action)
                        ).float()
                        target_ratios = (target_counts / max(1, batch_size)).tolist()
                        _maybe_warn_collapse(
                            "target", target_ratios, collapse_counters_target
                        )

                window_samples += batch_size
                window_tokens += int(batch["input_ids"].numel())
                window_batches += 1
                window_losses["total_loss"] += float(losses["total_loss"].item()) * batch_size
                window_losses["task_loss"] += float(losses["task_loss"].item()) * batch_size
                window_losses["cost_loss"] += float(losses["cost_loss"].item()) * batch_size
                window_losses["calibration_loss"] += float(
                    losses["calibration_loss"].item()
                ) * batch_size
                window_losses["gain_loss"] += float(losses["gain_loss"].item()) * batch_size
                window_losses["entropy_loss"] += float(
                    losses.get("entropy_loss", 0.0).item()
                ) * batch_size
                window_losses["policy_loss"] += float(
                    losses.get("policy_loss", 0.0).item()
                ) * batch_size
                window_losses["prior_loss"] += float(
                    losses.get("prior_loss", 0.0).item()
                ) * batch_size

                if summary_every > 0:
                    period_samples += batch_size
                    period_action_counts += torch.bincount(
                        sampled_actions.detach(), minlength=len(Action)
                    ).float()
                    period_argmax_counts += torch.bincount(
                        argmax_actions.detach(), minlength=len(Action)
                    ).float()
                    expected_cost = outputs.get("expected_cost")
                    if expected_cost is not None:
                        period_cost_sum += expected_cost.detach().sum()
                    period_task_sum += losses["task_loss"].detach() * batch_size
                    period_policy_sum += losses.get(
                        "policy_loss", torch.tensor(0.0, device=device)
                    ).detach() * batch_size
                    action_probs = outputs.get("action_probs")
                    if action_probs is not None:
                        ent = -(action_probs * (action_probs + 1e-8).log()).sum(dim=-1)
                        period_entropy_sum += ent.detach().sum()
                        period_action_prob_sum += action_probs.detach().sum(dim=0)
                    token_count_coarse = outputs.get("token_count_coarse")
                    token_count_full = outputs.get("token_count_full")
                    if token_count_coarse is not None:
                        period_token_coarse_sum += token_count_coarse.detach().sum()
                    if token_count_full is not None:
                        period_token_full_sum += token_count_full.detach().sum()
                    if policy_targets_hard is not None:
                        period_target_counts += torch.bincount(
                            policy_targets_hard.detach(), minlength=len(Action)
                        ).float()
                        period_target_samples += float(policy_targets_hard.numel())
                    if loss_triplet is not None:
                        period_loss_triplet_sum += loss_triplet.detach().sum(dim=0)
                        period_triplet_samples += float(loss_triplet.shape[0])
                    labels = batch.get("labels")
                    if labels is not None:
                        period_label_tokens += labels.ne(-100).sum()
                        period_label_total += labels.numel()

                if global_step % cfg.training.log_every == 0 and rank == 0:
                    avg_losses = {
                        key: value / max(1, window_samples)
                        for key, value in window_losses.items()
                    }
                    elapsed = time.time() - train_start
                    step_count = global_step + 1
                    progress_pct = 100.0 * step_count / total_steps if total_steps else None
                    eta_seconds = (
                        (total_steps - step_count) * (elapsed / max(1, step_count))
                        if total_steps
                        else None
                    )
                    window_elapsed = max(1e-9, time.time() - window_start)
                    window_samples_global = window_samples * world_size
                    window_tokens_global = window_tokens * world_size
                    window_stats = {
                        "samples": window_samples_global,
                        "tokens": window_tokens_global,
                        "batches": window_batches,
                        "seconds": window_elapsed,
                        "samples_s": window_samples_global / window_elapsed,
                        "tokens_s": window_tokens_global / window_elapsed,
                        "world_size": world_size,
                    }
                    progress = {
                        "stage": stage_name,
                        "epoch": epoch + 1,
                        "epochs": cfg.training.epochs,
                        "step_in_epoch": step + 1,
                        "steps_per_epoch": steps_per_epoch,
                        "global_step": step_count,
                        "total_steps": total_steps,
                        "progress_pct": progress_pct,
                        "eta_seconds": eta_seconds,
                        "elapsed_seconds": elapsed,
                    }
                    _log_step(
                        avg_losses,
                        outputs,
                        progress,
                        window_stats,
                        optimizer,
                        lambda_cost,
                        train_profiler,
                        model,
                        argmax_actions=argmax_actions,
                        policy_targets=policy_targets_hard,
                        loss_triplet=loss_triplet,
                        lambda_policy=lambda_policy_step,
                        lambda_entropy=lambda_entropy_step,
                        open_quantile=open_quantile_step,
                    )
                    window_start = time.time()
                    window_samples = 0
                    window_tokens = 0
                    window_batches = 0
                    for key in window_losses:
                        window_losses[key] = 0.0

                if (
                    cfg.training.save_every > 0
                    and global_step > 0
                    and global_step % cfg.training.save_every == 0
                    and rank == 0
                ):
                    ckpt_path = output_dir / f"checkpoint-{global_step}.pt"
                    raw_model = model.module if isinstance(model, DDP) else model
                    torch.save(
                        {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "global_step": global_step,
                        },
                        ckpt_path,
                    )
                    logger.info("Saved checkpoint to %s", ckpt_path)

                if summary_every > 0 and global_step > 0 and global_step % summary_every == 0:
                    sample_pred = None
                    sample_label = None
                    if rank == 0:
                        labels = batch.get("labels")
                        logits = outputs.get("logits")
                        if labels is not None and logits is not None:
                            try:
                                pred_ids = logits.argmax(dim=-1)
                                mask = labels.ne(-100)
                                if mask.any():
                                    tok = (model.module if isinstance(model, DDP) else model).base_vlm.tokenizer
                                    if tok is not None:
                                        label_ids = labels[0][mask[0]].detach().cpu().tolist()
                                        pred_ids_0 = pred_ids[0][mask[0]].detach().cpu().tolist()
                                        sample_label = tok.decode(label_ids[:32], skip_special_tokens=True)
                                        sample_pred = tok.decode(pred_ids_0[:32], skip_special_tokens=True)
                            except Exception:
                                sample_pred = None
                                sample_label = None
                    if distributed:
                        dist.all_reduce(period_samples, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_action_counts, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_argmax_counts, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_cost_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_task_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_policy_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_entropy_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_action_prob_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_target_counts, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_target_samples, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_loss_triplet_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_triplet_samples, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_token_coarse_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_token_full_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_label_tokens, op=dist.ReduceOp.SUM)
                        dist.all_reduce(period_label_total, op=dist.ReduceOp.SUM)
                    if rank == 0:
                        samples = max(1.0, float(period_samples.item()))
                        action_ratio = (period_action_counts / samples).tolist()
                        argmax_ratio = (period_argmax_counts / samples).tolist()
                        avg_cost = float(period_cost_sum.item()) / samples
                        avg_task = float(period_task_sum.item()) / samples
                        avg_policy = float(period_policy_sum.item()) / samples
                        avg_entropy = float(period_entropy_sum.item()) / samples
                        mean_probs = (period_action_prob_sum / samples).tolist()
                        target_ratio = None
                        target_samples = float(period_target_samples.item())
                        if target_samples > 0:
                            target_ratio = (
                                period_target_counts / max(1.0, target_samples)
                            ).tolist()
                        mean_loss_triplet = None
                        triplet_samples = float(period_triplet_samples.item())
                        if triplet_samples > 0:
                            mean_loss_triplet = (
                                period_loss_triplet_sum / max(1.0, triplet_samples)
                            ).tolist()
                        mean_token_coarse = float(period_token_coarse_sum.item()) / samples
                        mean_token_full = float(period_token_full_sum.item()) / samples
                        token_ratio = (
                            (mean_token_coarse / mean_token_full)
                            if mean_token_full > 0
                            else None
                        )
                        label_valid_ratio = 0.0
                        if period_label_total.item() > 0:
                            label_valid_ratio = float(period_label_tokens.item()) / float(
                                period_label_total.item()
                            )
                        delta_no_start = (
                            cfg.policy.policy_delta_no_start
                            if cfg.policy.policy_delta_no_start is not None
                            else cfg.policy.policy_delta_start
                        )
                        delta_no_end = (
                            cfg.policy.policy_delta_no_end
                            if cfg.policy.policy_delta_no_end is not None
                            else cfg.policy.policy_delta_end
                        )
                        delta_coarse_start = (
                            cfg.policy.policy_delta_coarse_start
                            if cfg.policy.policy_delta_coarse_start is not None
                            else cfg.policy.policy_delta_start
                        )
                        delta_coarse_end = (
                            cfg.policy.policy_delta_coarse_end
                            if cfg.policy.policy_delta_coarse_end is not None
                            else cfg.policy.policy_delta_end
                        )
                        delta_no_warmup = int(
                            cfg.policy.policy_delta_no_warmup_steps
                            if cfg.policy.policy_delta_no_warmup_steps is not None
                            else cfg.policy.policy_delta_warmup_steps
                        )
                        delta_coarse_warmup = int(
                            cfg.policy.policy_delta_coarse_warmup_steps
                            if cfg.policy.policy_delta_coarse_warmup_steps is not None
                            else cfg.policy.policy_delta_warmup_steps
                        )
                        delta_no_step = _schedule_value(
                            delta_no_start,
                            delta_no_end,
                            delta_no_warmup,
                            int(stage_steps),
                        )
                        delta_coarse_step = _schedule_value(
                            delta_coarse_start,
                            delta_coarse_end,
                            delta_coarse_warmup,
                            int(stage_steps),
                        )
                        no_bias_step = _schedule_value(
                            float(cfg.policy.policy_no_bias_start),
                            float(cfg.policy.policy_no_bias_end),
                            int(cfg.policy.policy_no_bias_warmup_steps),
                            int(stage_steps),
                        )
                        open_bias_step = _schedule_value(
                            float(cfg.policy.policy_open_visual_bias_start),
                            float(cfg.policy.policy_open_visual_bias_end),
                            int(cfg.policy.policy_open_visual_bias_warmup_steps),
                            int(stage_steps),
                        )
                        open_q_start = (
                            cfg.policy.policy_open_quantile_start
                            if cfg.policy.policy_open_quantile_start is not None
                            else cfg.policy.policy_open_quantile
                        )
                        open_q_end = (
                            cfg.policy.policy_open_quantile_end
                            if cfg.policy.policy_open_quantile_end is not None
                            else cfg.policy.policy_open_quantile
                        )
                        open_quantile_step = _schedule_value(
                            float(open_q_start),
                            float(open_q_end),
                            int(cfg.policy.policy_open_quantile_warmup_steps),
                            int(stage_steps),
                        )
                        open_quantile_step = float(min(1.0, max(0.0, open_quantile_step)))
                        budget_ratio = None
                        if cfg.vision_budget.coarse_ratio is not None:
                            budget_ratio = float(cfg.vision_budget.coarse_ratio)
                        elif cfg.vision_budget.coarse_budget_mode is not None:
                            mode = str(cfg.vision_budget.coarse_budget_mode).strip().lower()
                            if mode == "half":
                                budget_ratio = 0.5
                            elif mode == "quarter":
                                budget_ratio = 0.25
                        elif cfg.vision_budget.full_max_pixels > 0:
                            budget_ratio = float(cfg.vision_budget.coarse_max_pixels) / float(
                                cfg.vision_budget.full_max_pixels
                            )
                        if (
                            token_ratio is not None
                            and budget_ratio is not None
                            and budget_ratio > 0
                        ):
                            deviation = abs(token_ratio - budget_ratio) / budget_ratio
                            if deviation > 0.2:
                                logger.warning(
                                    "budget-ratio-warning observed=%.3f target=%.3f",
                                    token_ratio,
                                    budget_ratio,
                                )
                        policy_ce_start = (
                            cfg.policy.policy_ce_weight_start
                            if cfg.policy.policy_ce_weight_start is not None
                            else cfg.policy.policy_ce_weight
                        )
                        policy_ce_end = (
                            cfg.policy.policy_ce_weight_end
                            if cfg.policy.policy_ce_weight_end is not None
                            else cfg.policy.policy_ce_weight
                        )
                        lambda_policy_step = _schedule_value(
                            float(policy_ce_start),
                            float(policy_ce_end),
                            int(cfg.policy.policy_ce_weight_warmup_steps),
                            int(stage_steps),
                        )
                        entropy_start = (
                            cfg.policy.entropy_weight_start
                            if cfg.policy.entropy_weight_start is not None
                            else cfg.policy.entropy_weight
                        )
                        entropy_end = (
                            cfg.policy.entropy_weight_end
                            if cfg.policy.entropy_weight_end is not None
                            else cfg.policy.entropy_weight
                        )
                        lambda_entropy_step = _schedule_value(
                            float(entropy_start),
                            float(entropy_end),
                            int(cfg.policy.entropy_weight_warmup_steps),
                            int(stage_steps),
                        )
                        collapse_window = int(cfg.policy.collapse_warn_window_steps or 0)
                        collapse_thresh = float(cfg.policy.collapse_warn_ratio_threshold)
                        collapse_ready = (
                            stage_baseline is None
                            and collapse_window > 0
                            and summary_every >= collapse_window
                        )
                        if collapse_ready:
                            def _warn(kind: str, ratios: list[float] | None) -> None:
                                if ratios is None:
                                    return
                                low = [
                                    Action(i).name
                                    for i, r in enumerate(ratios)
                                    if r < collapse_thresh
                                ]
                                if low:
                                    logger.warning(
                                        "policy-collapse-warning window=%d kind=%s threshold=%.3f low_actions=%s ratios=%s",
                                        summary_every,
                                        kind,
                                        collapse_thresh,
                                        low,
                                        ratios,
                                    )

                            _warn("sampled", action_ratio)
                            _warn("argmax", argmax_ratio)
                            _warn("target", target_ratio)
                        logger.info(
                            "SUMMARY@%d samples=%d action_ratio=%s argmax_ratio=%s target_ratio=%s "
                            "mean_probs=%s mean_loss_triplet=%s "
                            "token_coarse=%.2f token_full=%.2f token_ratio=%s target_ratio_budget=%s "
                            "label_valid_ratio=%.4f delta_no=%.4g delta_coarse=%.4g no_bias=%.4g open_bias=%.4g "
                            "open_q=%.3f lambda_policy=%.4f lambda_entropy=%.4f "
                            "avg_cost=%.4f avg_task=%.4f avg_policy=%.4f avg_entropy=%.4f "
                            "sample_pred=%s sample_label=%s",
                            global_step,
                            int(samples),
                            action_ratio,
                            argmax_ratio,
                            target_ratio,
                            mean_probs,
                            mean_loss_triplet,
                            mean_token_coarse,
                            mean_token_full,
                            f"{token_ratio:.3f}" if token_ratio is not None else "None",
                            f"{budget_ratio:.3f}" if budget_ratio is not None else "None",
                            label_valid_ratio,
                            delta_no_step,
                            delta_coarse_step,
                            no_bias_step,
                            open_bias_step,
                            open_quantile_step,
                            lambda_policy_step,
                            lambda_entropy_step,
                            avg_cost,
                            avg_task,
                            avg_policy,
                            avg_entropy,
                            sample_pred,
                            sample_label,
                        )
                    period_samples.zero_()
                    period_action_counts.zero_()
                    period_argmax_counts.zero_()
                    period_cost_sum.zero_()
                    period_task_sum.zero_()
                    period_policy_sum.zero_()
                    period_entropy_sum.zero_()
                    period_action_prob_sum.zero_()
                    period_target_counts.zero_()
                    period_target_samples.zero_()
                    period_loss_triplet_sum.zero_()
                    period_triplet_samples.zero_()
                    period_token_coarse_sum.zero_()
                    period_token_full_sum.zero_()
                    period_label_tokens.zero_()
                    period_label_total.zero_()

                global_step += 1
                stage_steps += 1

            if stage_max_steps is not None and stage_steps >= stage_max_steps:
                break

            if eval_loader is not None:
                if distributed:
                    dist.barrier()
                metrics = _evaluate(
                    model=model,
                    loader=eval_loader,
                    lambda_cost=stage_lambda_cost,
                    baseline_name=stage_baseline,
                    device=device,
                    distributed=distributed,
                )
                if rank == 0:
                    logger.info("Eval metrics: %s", metrics)

        epoch_idx += stage_epochs

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
