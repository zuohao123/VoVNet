"""FSDP utilities."""
from __future__ import annotations

from typing import Any, Optional


def build_fsdp_plugin() -> Optional[Any]:
    """Build an accelerate FSDP plugin if available."""
    try:
        from accelerate import FullyShardedDataParallelPlugin
    except Exception:
        return None

    return FullyShardedDataParallelPlugin(
        state_dict_type="full",
        limit_all_gathers=True,
    )
