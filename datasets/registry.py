"""Dataset adapter registry."""
from __future__ import annotations

from typing import Dict

from .adapters.mmbench import MMBenchAdapter, MMBenchLiteAdapter
from .adapters.llava_instruct import LLaVAInstructAdapter
from .adapters.mmmu import MMMUAdapter
from .adapters.textvqa import TextVQAAdapter


_REGISTRY = {
    "textvqa": TextVQAAdapter,
    "text_heavy_vqa": TextVQAAdapter,
    "mmbench": MMBenchAdapter,
    "mmbench_lite": MMBenchLiteAdapter,
    "mmmu": MMMUAdapter,
    "llava_instruct": LLaVAInstructAdapter,
}


def get_adapter(name: str):
    """Return adapter instance by name."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown dataset: {name}. Options: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[key]()


def list_adapters() -> Dict[str, str]:
    """List available adapters with their classes."""
    return {name: cls.__name__ for name, cls in _REGISTRY.items()}
