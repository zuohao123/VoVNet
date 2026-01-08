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

    def train(self, epochs: int) -> None:
        """Run the training loop."""
        global_step = 0
        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(self.train_loader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        images=batch.get("images"),
                        labels=batch.get("labels"),
                    )
                    losses = compute_total_loss(
                        outputs["logits"],
                        outputs.get("labels"),
                        outputs["expected_cost"],
                        self.lambda_cost,
                        calibration_value=None,
                        lambda_cal=self.lambda_cal,
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

                if global_step % self.log_every == 0:
                    self._log_step(global_step, losses, outputs)
                if global_step > 0 and global_step % self.save_every == 0:
                    self._save_checkpoint(global_step)
                global_step += 1

            if self.eval_loader is not None:
                self.evaluate()

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on the validation set."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0

        tokenizer = getattr(self.model.base_vlm, "tokenizer", None)
        with torch.no_grad():
            for batch in self.eval_loader or []:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    images=batch.get("images"),
                    labels=batch.get("labels"),
                )
                losses = compute_total_loss(
                    outputs["logits"],
                    outputs.get("labels"),
                    outputs["expected_cost"],
                    self.lambda_cost,
                )
                total_loss += losses["task_loss"].item() * batch["input_ids"].size(0)

                if tokenizer is not None:
                    preds = _decode_from_logits(
                        outputs["logits"], outputs.get("labels"), tokenizer
                    )
                    acc = exact_match_score(preds, batch.get("answers", []))
                    total_acc += acc * len(preds)
                    total_count += len(preds)

        avg_loss = total_loss / max(1, total_count or 1)
        avg_acc = total_acc / max(1, total_count)
        metrics = {"val_loss": avg_loss, "val_accuracy": avg_acc}
        logger.info("Eval metrics: %s", metrics)
        return metrics

    def _log_step(self, step: int, losses: Dict[str, torch.Tensor], outputs: Dict[str, Any]) -> None:
        actions = outputs["actions"]
        action_rates = torch.bincount(actions, minlength=3).float() / actions.numel()
        vision_tokens = outputs.get("vision_tokens")
        vision_mean = vision_tokens.float().mean().item() if vision_tokens is not None else 0.0
        seq_len = outputs["logits"].shape[1]
        flops_proxy = vision_mean * seq_len
        logger.info(
            "step=%s total=%.4f task=%.4f cost=%.4f actions=%s vision_tokens=%.2f flops_proxy=%.2f",
            step,
            losses["total_loss"].item(),
            losses["task_loss"].item(),
            losses["cost_loss"].item(),
            action_rates.tolist(),
            vision_mean,
            flops_proxy,
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
