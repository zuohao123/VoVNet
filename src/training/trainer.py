"""Training loop for VoVNet."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.eval.metrics import exact_match_score
from src.training.losses import compute_total_loss
from src.utils.logging import setup_logging
from src.utils.profiling import BatchProfiler
from src.utils.stats import pearsonr, spearmanr
from src.models.uncertainty import expected_calibration_error
from src.models.vovnet import Action

logger = setup_logging(__name__)


class Trainer:
    """Lightweight trainer for VoVNet with accelerate."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        accelerator: Any,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        output_dir: str,
        lambda_cost: float,
        lambda_cal: float = 0.0,
        log_every: int = 10,
        save_every: int = 500,
        max_grad_norm: float = 1.0,
        profile_train: bool = False,
        profile_eval: bool = False,
        gain_supervision: bool = False,
        gain_loss_type: str = "mse",
        gain_loss_weight: float = 0.0,
        gain_margin: float = 0.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.output_dir = Path(output_dir)
        self.lambda_cost = lambda_cost
        self.lambda_cal = lambda_cal
        self.log_every = log_every
        self.save_every = save_every
        self.max_grad_norm = max_grad_norm
        self.output_dir.mkdir(parents=True, exist_ok=True)
        action_names = {int(action): action.name.lower() for action in Action}
        self.train_profiler = BatchProfiler(profile_train, action_names=action_names)
        self.eval_profiler = BatchProfiler(profile_eval, action_names=action_names)
        self.gain_supervision = gain_supervision
        self.gain_loss_type = gain_loss_type
        self.gain_loss_weight = gain_loss_weight
        self.gain_margin = gain_margin

    def train(self, epochs: int) -> None:
        """Run the training loop."""
        global_step = 0
        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(self.train_loader):
                if self.train_profiler.enabled:
                    self.train_profiler.start()
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        images=batch.get("images"),
                        labels=batch.get("labels"),
                        compute_gain=self.gain_supervision,
                    )
                    losses = compute_total_loss(
                        outputs["logits"],
                        outputs.get("labels"),
                        outputs["expected_cost"],
                        self.lambda_cost,
                        calibration_value=None,
                        lambda_cal=self.lambda_cal,
                        gain_pred=outputs.get("gain_pred"),
                        gain_true=outputs.get("gain_true"),
                        gain_loss_type=self.gain_loss_type,
                        lambda_gain=self.gain_loss_weight,
                        gain_margin=self.gain_margin,
                    )
                    loss = losses["total_loss"]
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.train_profiler.enabled:
                    batch_size = batch["input_ids"].size(0)
                    seq_len = batch["input_ids"].size(1)
                    self.train_profiler.stop(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        actions=outputs["actions"],
                    )

                if global_step % self.log_every == 0:
                    self._log_step(global_step, losses, outputs)
                if global_step > 0 and global_step % self.save_every == 0:
                    self._save_checkpoint(global_step)
                global_step += 1

            if self.eval_loader is not None:
                self.evaluate()

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on the validation set."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0
        total_ece = 0.0
        total_ece_count = 0
        base_acc = 0.0
        base_cost = 0.0
        final_cost = 0.0
        fallback_count = 0
        fallback_entropy_count = 0
        fallback_margin_count = 0
        fallback_both_count = 0
        if self.eval_profiler.enabled:
            self.eval_profiler.reset()

        tokenizer = getattr(self.model.base_vlm, "tokenizer", None)
        with torch.no_grad():
            for batch in self.eval_loader or []:
                if self.eval_profiler.enabled:
                    self.eval_profiler.start()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    images=batch.get("images"),
                    labels=batch.get("labels"),
                )
                if self.eval_profiler.enabled:
                    batch_size = batch["input_ids"].size(0)
                    seq_len = batch["input_ids"].size(1)
                    self.eval_profiler.stop(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        actions=outputs["actions"],
                    )
                losses = compute_total_loss(
                    outputs["logits"],
                    outputs.get("labels"),
                    outputs["expected_cost"],
                    self.lambda_cost,
                )
                total_loss += losses["task_loss"].item() * batch["input_ids"].size(0)
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

                if tokenizer is not None:
                    preds = _decode_from_logits(
                        outputs["logits"], outputs.get("labels"), tokenizer
                    )
                    acc = exact_match_score(preds, batch.get("answers", []))
                    total_acc += acc * len(preds)
                    total_count += len(preds)

                    actions_raw = outputs.get("actions_raw")
                    text_logits = outputs.get("text_logits")
                    if actions_raw is not None and text_logits is not None:
                        fallback_mask = outputs.get("fallback_mask")
                        if fallback_mask is not None and fallback_mask.any():
                            logits_before = outputs["logits"].clone()
                            logits_before[fallback_mask] = text_logits[fallback_mask]
                        else:
                            logits_before = outputs["logits"]
                        base_preds = _decode_from_logits(
                            logits_before, outputs.get("labels"), tokenizer
                        )
                        base_acc += exact_match_score(base_preds, batch.get("answers", [])) * len(
                            base_preds
                        )

                        token_count_coarse = outputs.get("token_count_coarse")
                        token_count_full = outputs.get("token_count_full")
                        if token_count_coarse is not None and token_count_full is not None:
                            cost_scale = float(getattr(self.model, "cost_scale", 1.0))
                            base_cost += (
                                _action_cost(actions_raw, token_count_coarse, token_count_full)
                                * cost_scale
                            ).sum().item()
                            final_cost += (
                                _action_cost(
                                    outputs["actions"], token_count_coarse, token_count_full
                                )
                                * cost_scale
                            ).sum().item()

                        if fallback_mask is not None:
                            fallback_count += int(fallback_mask.sum().item())
                            entropy_trigger = outputs.get("fallback_entropy_trigger")
                            margin_trigger = outputs.get("fallback_margin_trigger")
                            if entropy_trigger is not None:
                                entropy_hits = fallback_mask & entropy_trigger
                                fallback_entropy_count += int(entropy_hits.sum().item())
                            if margin_trigger is not None:
                                margin_hits = fallback_mask & margin_trigger
                                fallback_margin_count += int(margin_hits.sum().item())
                            if entropy_trigger is not None and margin_trigger is not None:
                                both_hits = fallback_mask & entropy_trigger & margin_trigger
                                fallback_both_count += int(both_hits.sum().item())

        avg_loss = total_loss / max(1, total_count or 1)
        avg_acc = total_acc / max(1, total_count)
        metrics = {
            "val_loss": avg_loss,
            "val_accuracy": avg_acc,
            "val_ece": total_ece / max(1, total_ece_count),
        }
        if total_count > 0:
            base_accuracy = base_acc / max(1, total_count)
            metrics["fallback"] = {
                "fallback_count": fallback_count,
                "fallback_rate": fallback_count / max(1, total_count),
                "entropy_trigger_count": fallback_entropy_count,
                "margin_trigger_count": fallback_margin_count,
                "both_trigger_count": fallback_both_count,
                "accuracy_before": base_accuracy,
                "accuracy_after": avg_acc,
                "fallback_gain": avg_acc - base_accuracy,
                "cost_before": base_cost / max(1, total_count),
                "cost_after": final_cost / max(1, total_count),
                "cost_increase": (final_cost - base_cost) / max(1, total_count),
            }
        if self.eval_profiler.enabled:
            metrics["profile"] = self.eval_profiler.summary()
        logger.info("Eval metrics: %s", metrics)
        return metrics

    def _log_step(self, step: int, losses: Dict[str, torch.Tensor], outputs: Dict[str, Any]) -> None:
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
        if self.train_profiler.enabled and self.train_profiler.last_batch is not None:
            profile = self.train_profiler.last_batch
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

        logger.info(
            (
                "step=%s total=%.4f task_loss=%.4f cost_loss=%.4f gain_loss=%.4f "
                "lambda_cost=%.4f action_entropy=%.4f action_ratio=%s "
                "vision_tokens=%.2f token_count_coarse=%.2f token_count_full=%.2f "
                "expected_cost=%.2f flops_proxy=%.2f ece=%.4f "
                "latency_ms=%.2f mem_peak_mb=%.2f tokens_s=%.2f%s"
            ),
            step,
            losses["total_loss"].item(),
            losses["task_loss"].item(),
            losses["cost_loss"].item(),
            losses.get("gain_loss", torch.tensor(0.0)).item(),
            self.lambda_cost,
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

    def _save_checkpoint(self, step: int) -> None:
        path = self.output_dir / f"checkpoint-{step}"
        self.accelerator.save_state(path)


def _decode_from_logits(logits: torch.Tensor, labels: torch.Tensor, tokenizer: Any) -> list[str]:
    """Decode answer tokens based on label mask."""
    preds: list[str] = []
    pred_ids = logits.argmax(dim=-1)
    for i in range(pred_ids.size(0)):
        mask = labels[i] != -100
        answer_ids = pred_ids[i][mask]
        text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        preds.append(text.strip())
    return preds


def _action_cost(
    actions: torch.Tensor,
    token_count_coarse: torch.Tensor,
    token_count_full: torch.Tensor,
) -> torch.Tensor:
    costs = torch.zeros_like(actions, dtype=torch.float)
    costs = costs.to(token_count_coarse.device)
    costs[actions == Action.COARSE_VISION] = token_count_coarse[actions == Action.COARSE_VISION]
    costs[actions == Action.FULL_VISION] = token_count_full[actions == Action.FULL_VISION]
    return costs


def _compute_ece(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> float:
    """Compute token-level ECE for the labeled positions."""
    if labels is None:
        return 0.0
    mask = labels.ne(-100)
    if not mask.any():
        return 0.0
    flat_logits = logits[mask]
    flat_labels = labels[mask]
    probs = torch.softmax(flat_logits, dim=-1)
    ece, _ = expected_calibration_error(probs, flat_labels)
    return float(ece.item())
