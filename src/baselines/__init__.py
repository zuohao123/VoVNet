"""Package."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.baselines.always_coarse import select_actions as always_coarse
from src.baselines.always_full import select_actions as always_full
from src.baselines.no_vision import select_actions as no_vision

_BASELINE_ALIASES = {
    "always_full": "always_full",
    "full": "always_full",
    "always_coarse": "always_coarse",
    "coarse": "always_coarse",
    "no_vision": "no_vision",
    "no_vision_only": "no_vision",
    "no": "no_vision",
    "uncertainty_threshold": "uncertainty_threshold",
    "uncertainty": "uncertainty_threshold",
    "threshold": "uncertainty_threshold",
    "random_policy_matched": "random_policy_matched",
    "random_matched": "random_policy_matched",
    "vision_token_pruning_proxy": "vision_token_pruning_proxy",
    "pruning_proxy": "vision_token_pruning_proxy",
    "vision_pruning": "vision_token_pruning_proxy",
    "resolution_scaling": "resolution_scaling",
    "token_merge_prune_proxy": "token_merge_prune_proxy",
    "token_merge": "token_merge_prune_proxy",
    "multi_granularity_proxy": "multi_granularity_proxy",
}

_BASELINE_LABELS = {
    "always_full": "FULL",
    "always_coarse": "COARSE",
    "no_vision": "NO",
    "resolution_scaling": "RESOLUTION",
}


def normalize_baseline_name(name: Optional[str]) -> Optional[str]:
    """Normalize baseline name and handle empty/none values."""
    if name is None:
        return None
    normalized = name.strip().lower()
    if not normalized or normalized in {"none", "null"}:
        return None
    return _BASELINE_ALIASES.get(normalized, normalized)


def resolve_baseline_actions(
    name: Optional[str],
    batch_size: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[str], bool]:
    """Return baseline actions, action label, and whether to drop images."""
    normalized = normalize_baseline_name(name)
    if normalized is None:
        return None, None, False
    if normalized == "always_full":
        return always_full(batch_size, device), _BASELINE_LABELS[normalized], False
    if normalized == "always_coarse":
        return always_coarse(batch_size, device), _BASELINE_LABELS[normalized], False
    if normalized == "no_vision":
        return no_vision(batch_size, device), _BASELINE_LABELS[normalized], True
    if normalized == "uncertainty_threshold":
        raise ValueError("uncertainty_threshold requires model uncertainty; eval-only")
    if normalized == "vision_token_pruning_proxy":
        raise ValueError("vision_token_pruning_proxy requires vision token pruning; eval-only")
    if normalized == "random_policy_matched":
        raise ValueError("random_policy_matched requires configured ratios; eval-only")
    if normalized == "resolution_scaling":
        return always_full(batch_size, device), _BASELINE_LABELS[normalized], False
    if normalized == "token_merge_prune_proxy":
        raise ValueError("token_merge_prune_proxy requires token merging; eval-only")
    if normalized == "multi_granularity_proxy":
        raise ValueError("multi_granularity_proxy requires pooling; eval-only")
    raise ValueError(f"Unknown baseline_name: {name}")
