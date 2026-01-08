"""Loss utilities for VoVNet."""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from torch.nn import functional as F


def compute_task_loss(logits: Tensor, labels: Tensor | None) -> Tensor:
    """Compute token-level cross entropy loss."""
    if labels is None:
        return torch.tensor(0.0, device=logits.device)
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(
        logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100
    )
    return loss


def compute_cost_loss(expected_cost: Tensor, lambda_cost: float) -> Tensor:
    """Compute expected cost penalty."""
    return expected_cost.mean() * lambda_cost


def compute_calibration_loss(calibration_value: Tensor, lambda_cal: float) -> Tensor:
    """Optional calibration loss placeholder."""
    return calibration_value * lambda_cal


def compute_total_loss(
    logits: Tensor,
    labels: Tensor | None,
    expected_cost: Tensor,
    lambda_cost: float,
    calibration_value: Tensor | None = None,
    lambda_cal: float = 0.0,
) -> Dict[str, Tensor]:
    """Compute combined losses."""
    task_loss = compute_task_loss(logits, labels)
    cost_loss = compute_cost_loss(expected_cost, lambda_cost)
    cal_loss = (
        compute_calibration_loss(calibration_value, lambda_cal)
        if calibration_value is not None
        else torch.tensor(0.0, device=logits.device)
    )
    total_loss = task_loss + cost_loss + cal_loss
    return {
        "total_loss": total_loss,
        "task_loss": task_loss,
        "cost_loss": cost_loss,
        "calibration_loss": cal_loss,
    }
