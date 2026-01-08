"""Distributed helpers."""
from __future__ import annotations

import os

import torch


def get_rank() -> int:
    """Return the global rank from environment, defaulting to 0."""
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    """Return the world size from environment, defaulting to 1."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    """Return True if running on rank 0."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes if distributed is initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
