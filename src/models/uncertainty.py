"""Uncertainty and calibration utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


def entropy_from_logits(logits: Tensor, dim: int = -1) -> Tensor:
    """Compute predictive entropy from logits."""
    probs = torch.softmax(logits, dim=dim)
    log_probs = torch.log_softmax(logits, dim=dim)
    return -(probs * log_probs).sum(dim=dim)


def margin_from_logits(logits: Tensor, dim: int = -1) -> Tensor:
    """Compute margin between top-1 and top-2 probabilities."""
    probs = torch.softmax(logits, dim=dim)
    top2 = torch.topk(probs, k=2, dim=dim).values
    return top2[..., 0] - top2[..., 1]


def expected_calibration_error(
    probs: Tensor, labels: Tensor, n_bins: int = 10
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute ECE and reliability diagram data.

    Returns:
        ece: scalar tensor
        stats: dict with bin_acc, bin_conf, bin_count
    """
    probs = probs.detach()
    labels = labels.detach()
    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    bin_acc = torch.zeros(n_bins, device=probs.device)
    bin_conf = torch.zeros(n_bins, device=probs.device)
    bin_count = torch.zeros(n_bins, device=probs.device)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = mask.sum()
        bin_count[i] = count
        if count > 0:
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            bin_acc[i] = acc
            bin_conf[i] = conf
            ece += (count.float() / labels.numel()) * torch.abs(acc - conf)

    stats = {"bin_acc": bin_acc, "bin_conf": bin_conf, "bin_count": bin_count}
    return ece, stats
