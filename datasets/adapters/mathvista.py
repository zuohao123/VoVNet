"""Stub adapter for MathVista."""
from __future__ import annotations

from ..common import StubAdapter


class MathVistaAdapter(StubAdapter):
    """MathVista adapter stub.

    TODO: If MathVista becomes available on HF, implement load/normalize.
    Official instructions: https://github.com/lupantech/MathVista
    """

    name = "mathvista"
    reason = "MathVista is not available on HuggingFace. Use the official release and add an adapter."
