"""Lightweight statistics helpers for correlation metrics."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def pearsonr(x: Tensor, y: Tensor, eps: float = 1e-8) -> float:
    """Compute Pearson correlation for 1D tensors."""
    x = x.float().view(-1)
    y = y.float().view(-1)
    if x.numel() < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()) + eps
    if denom.item() == 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def spearmanr(x: Tensor, y: Tensor) -> float:
    """Compute Spearman correlation via rank transform."""
    rx = _rankdata(x)
    ry = _rankdata(y)
    return pearsonr(rx, ry)


def _rankdata(x: Tensor) -> Tensor:
    x = x.float().view(-1)
    if x.numel() == 0:
        return x
    # Approximate ranks with argsort; ties will receive arbitrary ordering.
    sorted_idx = torch.argsort(x)
    ranks = torch.empty_like(sorted_idx, dtype=torch.float)
    ranks[sorted_idx] = torch.arange(x.numel(), device=x.device, dtype=torch.float)
    return ranks
