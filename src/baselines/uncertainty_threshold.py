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


def select_actions_binary(
    uncertainty: torch.Tensor,
    threshold: float,
    high_action: Action = Action.FULL_VISION,
) -> torch.Tensor:
    """Select actions with a single threshold.

    If uncertainty < threshold: NO_VISION
    Else: high_action (FULL_VISION by default).
    """
    actions = torch.full_like(uncertainty, int(high_action), dtype=torch.long)
    actions = torch.where(uncertainty < float(threshold), Action.NO_VISION, actions)
    return actions
