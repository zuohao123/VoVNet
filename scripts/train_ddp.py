"""Train VoVNet with native PyTorch DDP (no accelerate)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

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
from src.training.losses import compute_total_loss
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
            }
        )
    return stages


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for key in ("input_ids", "attention_mask", "labels"):
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


def _log_step(
    avg_losses: Dict[str, float],
    outputs: Dict[str, Any],
    progress: Dict[str, Any],
    window_stats: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    lambda_cost: float,
    train_profiler: BatchProfiler,
    model: nn.Module,
) -> None:
    actions = outputs["actions"]
    action_rates = torch.bincount(actions, minlength=3).float() / actions.numel()
    action_probs = outputs.get("action_probs")
    if action_probs is not None:
        entropy = -(action_probs * (action_probs + 1e-8).log()).sum(dim=-1)
        action_entropy = entropy.mean().item()
    else:
        action_entropy = 0.0
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

    logger.info(
        (
            "stage=%s epoch=%s/%s step=%s global_step=%s progress=%s eta=%s elapsed=%s "
            "window_samples=%s window_samples_s=%.2f window_tokens_s=%.2f "
            "avg_total=%.4f avg_task=%.4f avg_cost=%.4f avg_cal=%.4f avg_gain=%.4f "
            "lr=%.6g budget=%s lambda_cost=%.4f action_entropy=%.4f action_ratio=%s "
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
        lr,
        budget_text,
        lambda_cost,
        action_entropy,
        action_rates.tolist(),
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
        if stage_baseline in {"uncertainty_threshold", "random_policy_matched"}:
            raise RuntimeError(f"{stage_baseline} baseline is eval-only; skip training")

    seed = cfg.training.seed + rank
    set_seed(seed)

    output_dir = Path(cfg.training.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    if model.base_vlm.tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded; check model name")

    train_dataset = build_dataset(cfg, "train")
    eval_dataset = build_dataset(cfg, "eval") if cfg.data.eval_jsonl else None

    collator = VLMDataCollator(
        tokenizer=model.base_vlm.tokenizer,
        prompt_template=cfg.data.prompt_template,
    )

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
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
    total_steps = steps_per_epoch * cfg.training.epochs

    action_names = {int(action): action.name.lower() for action in Action}
    train_profiler = BatchProfiler(cfg.training.profile, action_names=action_names)

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

    global_step = 0
    train_start = time.time()
    window_start = train_start
    window_samples = 0
    window_tokens = 0
    window_batches = 0
    window_losses: Dict[str, float] = {
        "total_loss": 0.0,
        "task_loss": 0.0,
        "cost_loss": 0.0,
        "calibration_loss": 0.0,
        "gain_loss": 0.0,
    }

    epoch_idx = 0
    for stage in stages:
        stage_name = stage["name"]
        stage_epochs = int(stage["epochs"])
        stage_baseline = normalize_baseline_name(stage["baseline_name"])
        stage_lambda_cost = float(stage["lambda_cost"])
        if rank == 0:
            logger.info(
                "Starting %s: epochs=%s baseline=%s lambda_cost=%.4f",
                stage_name,
                stage_epochs,
                stage_baseline,
                stage_lambda_cost,
            )
        for local_epoch in range(stage_epochs):
            epoch = epoch_idx + local_epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            for step, batch in enumerate(train_loader):
                if train_profiler.enabled:
                    train_profiler.start()
                batch = _move_batch(batch, device)
                batch_size = batch["input_ids"].size(0)
                actions, _, drop_images = resolve_baseline_actions(
                    stage_baseline, batch_size, batch["input_ids"].device
                )
                is_last = step + 1 == steps_per_epoch
                sync_grad = ((step + 1) % grad_accum == 0) or is_last
                if distributed and isinstance(model, DDP) and not sync_grad:
                    sync_ctx = model.no_sync()
                else:
                    sync_ctx = nullcontext()

                with sync_ctx:
                    with autocast_ctx:
                        if actions is None:
                            outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                images=batch.get("images"),
                                labels=batch.get("labels"),
                                compute_gain=cfg.policy.gain_supervision,
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
                        losses = compute_total_loss(
                            outputs["logits"],
                            outputs.get("labels"),
                            outputs["expected_cost"],
                            stage_lambda_cost,
                            calibration_value=None,
                            lambda_cal=cfg.policy.calibration_lambda,
                            gain_pred=outputs.get("gain_pred"),
                            gain_true=outputs.get("gain_true"),
                            gain_loss_type=cfg.policy.gain_loss_type,
                            lambda_gain=cfg.policy.gain_loss_weight,
                            gain_margin=cfg.policy.gain_margin,
                        )
                        loss = losses["total_loss"] / grad_accum

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if sync_grad:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if train_profiler.enabled:
                    seq_len = batch["input_ids"].size(1)
                    train_profiler.stop(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        actions=outputs["actions"],
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
                        stage_lambda_cost,
                        train_profiler,
                        model,
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

                global_step += 1

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
