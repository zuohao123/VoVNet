"""Stub adapter for SEED-Bench."""
from __future__ import annotations

from ..common import StubAdapter


class SeedBenchAdapter(StubAdapter):
    """SEED-Bench adapter stub.

    TODO: If SEED-Bench becomes available on HF, implement load/normalize.
    Official instructions: https://github.com/AILab-CVC/SEED-Bench
    """

    name = "seed_bench"
    reason = "SEED-Bench is not available on HuggingFace. Use the official release and add an adapter."
