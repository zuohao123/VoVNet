"""DeepSpeed configuration helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def build_deepspeed_config(
    stage: int,
    micro_batch_size: int,
    gradient_accumulation: int,
    mixed_precision: str,
) -> Dict[str, Any]:
    """Build a minimal DeepSpeed config for ZeRO-2/3."""
    if stage not in (2, 3):
        raise ValueError("stage must be 2 or 3")

    fp16 = mixed_precision == "fp16"
    bf16 = mixed_precision == "bf16"

    return {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        },
        "fp16": {"enabled": fp16},
        "bf16": {"enabled": bf16},
        "gradient_clipping": 1.0,
    }


def write_deepspeed_config(path: str | Path, config: Dict[str, Any]) -> None:
    """Write the DeepSpeed config to a JSON file."""
    Path(path).write_text(json.dumps(config, indent=2))
