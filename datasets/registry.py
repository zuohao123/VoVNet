"""Dataset adapter registry."""
from __future__ import annotations

from typing import Dict

from .adapters.gqa import GQAAdapter
from .adapters.llava_instruct_150k import LLaVAInstructAdapter
from .adapters.mathvista import MathVistaAdapter
from .adapters.mmbench import MMBenchAdapter
from .adapters.mmmu_pro import MMMUProAdapter
from .adapters.open_orca import OpenOrcaAdapter
from .adapters.seed_bench import SeedBenchAdapter
from .adapters.textvqa import TextVQAAdapter
from .adapters.vizwiz import VizWizAdapter
from .adapters.vqa_v2 import VQAv2Adapter


_REGISTRY = {
    "vqa_v2": VQAv2Adapter,
    "gqa": GQAAdapter,
    "textvqa": TextVQAAdapter,
    "vizwiz": VizWizAdapter,
    "llava_instruct_150k": LLaVAInstructAdapter,
    "open_orca": OpenOrcaAdapter,
    "mmbench": MMBenchAdapter,
    "mmmu_pro": MMMUProAdapter,
    "mathvista": MathVistaAdapter,
    "seed_bench": SeedBenchAdapter,
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
