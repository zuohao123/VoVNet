"""Evaluation metrics."""
from __future__ import annotations

from typing import Iterable, List


def normalize_text(text: str) -> str:
    """Basic normalization for exact match."""
    return " ".join(text.lower().strip().split())


def exact_match_score(preds: Iterable[str], refs: Iterable[str]) -> float:
    """Compute exact match accuracy."""
    preds = list(preds)
    refs = list(refs)
    if not preds:
        return 0.0
    correct = 0
    for pred, ref in zip(preds, refs):
        if normalize_text(pred) == normalize_text(ref):
            correct += 1
    return correct / len(preds)
