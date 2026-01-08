"""Scheduler helpers."""
from __future__ import annotations

from typing import Optional

from torch.optim import Optimizer


def build_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    name: str = "linear",
):
    """Create a transformers scheduler."""
    from transformers import get_scheduler

    return get_scheduler(
        name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
