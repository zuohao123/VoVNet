"""Pareto sweep utilities."""
from __future__ import annotations

from typing import Callable, Iterable, List, Dict, Any


def run_pareto_sweep(
    values: Iterable[float],
    eval_fn: Callable[[float], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sweep over values (e.g., lambda_cost) and collect metrics."""
    results: List[Dict[str, Any]] = []
    for value in values:
        metrics = eval_fn(value)
        metrics["value"] = value
        results.append(metrics)
    return results
