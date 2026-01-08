"""Logging helpers for VoVNet."""
from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with rank-aware verbosity."""
    rank = int(os.environ.get("RANK", "0"))
    log_level = level if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
