"""Stub adapter for MMBench."""
from __future__ import annotations

from ..common import StubAdapter


class MMBenchAdapter(StubAdapter):
    """MMBench adapter stub.

    TODO: If MMBench becomes available on HF, implement load/normalize.
    Official instructions: https://github.com/open-compass/MMBench
    """

    name = "mmbench"
    reason = "MMBench is not available on HuggingFace. Use the official release and add an adapter."
