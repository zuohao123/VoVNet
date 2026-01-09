"""Stub adapter for MMMU-Pro."""
from __future__ import annotations

from ..common import StubAdapter


class MMMUProAdapter(StubAdapter):
    """MMMU-Pro adapter stub.

    TODO: If MMMU-Pro becomes available on HF, implement load/normalize.
    Official instructions: https://github.com/MMMU-Benchmark/mmmu
    """

    name = "mmmu_pro"
    reason = "MMMU-Pro is not available on HuggingFace. Use the official release and add an adapter."
