"""Latency profiling utilities."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import torch


def measure_latency(model: Any, batch: Dict[str, Any], iters: int = 10) -> Dict[str, float]:
    """Measure latency with CUDA events when available."""
    model.eval()
    latencies: List[float] = []
    use_cuda = torch.cuda.is_available()

    for _ in range(iters):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                model(**batch)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end) / 1000.0)
        else:
            t0 = time.time()
            with torch.no_grad():
                model(**batch)
            latencies.append(time.time() - t0)

    latencies.sort()
    p50 = latencies[int(0.5 * len(latencies))]
    p95 = latencies[int(0.95 * len(latencies))]
    return {"p50": p50, "p95": p95}
