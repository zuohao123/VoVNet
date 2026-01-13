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


def compute_gain_loss(
    gain_pred: Tensor | None,
    gain_true: Tensor | None,
    loss_type: str = "mse",
    margin: float = 0.0,
) -> Tensor:
    """Compute gain regression or ranking loss."""
    if gain_pred is None or gain_true is None:
        device = gain_pred.device if gain_pred is not None else torch.device("cpu")
        return torch.tensor(0.0, device=device)

    if loss_type == "mse":
        return F.mse_loss(gain_pred, gain_true)
    if loss_type == "huber":
        return F.smooth_l1_loss(gain_pred, gain_true)

    delta_true = gain_true[:, 1] - gain_true[:, 0]
    delta_pred = gain_pred[:, 1] - gain_pred[:, 0]
    sign = torch.sign(delta_true)
    valid = sign != 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=gain_pred.device)
    sign = sign[valid]
    delta_pred = delta_pred[valid]

    if loss_type == "rank_hinge":
        return F.relu(margin - sign * delta_pred).mean()
    if loss_type == "rank_logistic":
        return F.softplus(-sign * delta_pred).mean()

    raise ValueError(f"Unknown gain loss type: {loss_type}")


def compute_total_loss(
    logits: Tensor,
    labels: Tensor | None,
    expected_cost: Tensor,
    lambda_cost: float,
    calibration_value: Tensor | None = None,
    lambda_cal: float = 0.0,
    gain_pred: Tensor | None = None,
    gain_true: Tensor | None = None,
    gain_loss_type: str = "mse",
    lambda_gain: float = 0.0,
    gain_margin: float = 0.0,
) -> Dict[str, Tensor]:
    """Compute combined losses."""
    task_loss = compute_task_loss(logits, labels)
    cost_loss = compute_cost_loss(expected_cost, lambda_cost)
    cal_loss = (
        compute_calibration_loss(calibration_value, lambda_cal)
        if calibration_value is not None
        else torch.tensor(0.0, device=logits.device)
    )
    gain_loss = compute_gain_loss(
        gain_pred=gain_pred,
        gain_true=gain_true,
        loss_type=gain_loss_type,
        margin=gain_margin,
    )
    total_loss = task_loss + cost_loss + cal_loss + lambda_gain * gain_loss
    return {
        "total_loss": total_loss,
        "task_loss": task_loss,
        "cost_loss": cost_loss,
        "calibration_loss": cal_loss,
        "gain_loss": gain_loss,
    }
