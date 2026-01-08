"""Calibration utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from src.models.uncertainty import expected_calibration_error


def compute_ece(logits: Tensor, labels: Tensor, n_bins: int = 10) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute ECE from logits and labels."""
    probs = torch.softmax(logits, dim=-1)
    return expected_calibration_error(probs, labels, n_bins=n_bins)
