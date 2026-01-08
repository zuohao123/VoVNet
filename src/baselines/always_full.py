"""Always use full vision baseline."""
from __future__ import annotations

import torch

from src.models.vovnet import Action


def select_actions(batch_size: int, device: torch.device) -> torch.Tensor:
    """Return FULL_VISION for every sample."""
    return torch.full((batch_size,), Action.FULL_VISION, device=device, dtype=torch.long)
