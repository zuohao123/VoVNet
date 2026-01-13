"""Profiling utilities for latency, memory, and throughput."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(math.floor(q * (len(values) - 1)))
    return values[max(0, min(idx, len(values) - 1))]


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "count": 0}
    mean = float(sum(values) / len(values))
    return {
        "mean": mean,
        "p50": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
        "count": len(values),
    }


@dataclass
class BatchMetrics:
    """Profiling metrics for a single batch."""

    latency_ms: float
    max_mem_mb: float
    tokens_s: float


class BatchProfiler:
    """Collect per-batch profiling metrics with optional action splits."""

    def __init__(
        self,
        enabled: bool,
        action_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.enabled = enabled
        self.action_names = action_names or {
            0: "no_vision",
            1: "coarse",
            2: "full",
        }
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
        self._start_time: Optional[float] = None
        self.last_batch: Optional[BatchMetrics] = None
        self.reset()

    def reset(self) -> None:
        self._overall = {"latency_ms": [], "mem_mb": [], "tokens_s": []}
        self._by_action = {
            name: {"latency_ms": [], "mem_mb": [], "tokens_s": []}
            for name in self.action_names.values()
        }
        self._batch_count = 0
        self._sample_count = 0

    def start(self) -> None:
        if not self.enabled:
            return
        self._start_event = None
        self._end_event = None
        self._start_time = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self, batch_size: int, seq_len: int, actions: torch.Tensor) -> Optional[BatchMetrics]:
        if not self.enabled:
            return None
        latency_ms = self._elapsed_ms()
        max_mem_mb = self._max_mem_mb()
        tokens_s = self._tokens_per_second(batch_size, seq_len, latency_ms)
        metrics = BatchMetrics(
            latency_ms=latency_ms, max_mem_mb=max_mem_mb, tokens_s=tokens_s
        )
        self.last_batch = metrics
        self._record(metrics, actions, batch_size)
        return metrics

    def summary(self) -> Dict[str, Any]:
        overall = {k: _summarize(v) for k, v in self._overall.items()}
        by_action = {
            action: {k: _summarize(v) for k, v in metrics.items()}
            for action, metrics in self._by_action.items()
        }
        return {
            "overall": overall,
            "by_action": by_action,
            "counts": {
                "batches": self._batch_count,
                "samples": self._sample_count,
            },
        }

    @staticmethod
    def summary_to_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        overall = summary.get("overall", {})
        for metric, stats in overall.items():
            rows.append({"scope": "overall", "action": "all", "metric": metric, **stats})
        for action, metrics in summary.get("by_action", {}).items():
            for metric, stats in metrics.items():
                rows.append(
                    {"scope": "by_action", "action": action, "metric": metric, **stats}
                )
        return rows

    def _elapsed_ms(self) -> float:
        if self._start_event is not None and self._end_event is not None:
            self._end_event.record()
            torch.cuda.synchronize()
            return float(self._start_event.elapsed_time(self._end_event))
        start_time = self._start_time or time.perf_counter()
        return float((time.perf_counter() - start_time) * 1000.0)

    def _max_mem_mb(self) -> float:
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024.0**2)
        return 0.0

    def _tokens_per_second(self, batch_size: int, seq_len: int, latency_ms: float) -> float:
        tokens = max(1, int(batch_size * seq_len))
        denom = max(latency_ms / 1000.0, 1e-9)
        return float(tokens / denom)

    def _record(self, metrics: BatchMetrics, actions: torch.Tensor, batch_size: int) -> None:
        self._overall["latency_ms"].append(metrics.latency_ms)
        self._overall["mem_mb"].append(metrics.max_mem_mb)
        self._overall["tokens_s"].append(metrics.tokens_s)
        self._batch_count += 1
        self._sample_count += batch_size

        if actions.numel() == 0:
            return
        actions_list = actions.detach().cpu().tolist()
        for action in actions_list:
            name = self.action_names.get(int(action), f"action_{action}")
            self._by_action.setdefault(
                name, {"latency_ms": [], "mem_mb": [], "tokens_s": []}
            )
            self._by_action[name]["latency_ms"].append(metrics.latency_ms)
            self._by_action[name]["mem_mb"].append(metrics.max_mem_mb)
            self._by_action[name]["tokens_s"].append(metrics.tokens_s)
