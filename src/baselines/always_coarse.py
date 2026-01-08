"""Always use coarse vision baseline."""
from __future__ import annotations

import torch

from src.models.vovnet import Action


def select_actions(batch_size: int, device: torch.device) -> torch.Tensor:
    """Return COARSE_VISION for every sample."""
    return torch.full((batch_size,), Action.COARSE_VISION, device=device, dtype=torch.long)
