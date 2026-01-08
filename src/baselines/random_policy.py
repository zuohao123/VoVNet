"""Random policy baseline."""
from __future__ import annotations

import torch

from src.models.vovnet import Action


def select_actions(batch_size: int, device: torch.device) -> torch.Tensor:
    """Randomly sample an action per example."""
    choices = torch.randint(low=0, high=len(Action), size=(batch_size,), device=device)
    return choices
