"""Uncertainty-threshold gating baseline."""
from __future__ import annotations

import torch

from src.models.vovnet import Action


def select_actions(
    uncertainty: torch.Tensor, t1: float, t2: float
) -> torch.Tensor:
    """Select actions based on uncertainty thresholds.

    If uncertainty <= t1: NO_VISION
    If t1 < uncertainty <= t2: COARSE_VISION
    If uncertainty > t2: FULL_VISION
    """
    actions = torch.full_like(uncertainty, Action.FULL_VISION, dtype=torch.long)
    actions = torch.where(uncertainty <= t2, Action.COARSE_VISION, actions)
    actions = torch.where(uncertainty <= t1, Action.NO_VISION, actions)
    return actions
