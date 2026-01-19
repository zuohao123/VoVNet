"""Random policy matched to target action ratios."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch

from src.models.vovnet import Action


def normalize_ratios(ratios: Sequence[float]) -> torch.Tensor:
    """Normalize ratios to a probability vector of length 3."""
    if len(ratios) != len(Action):
        raise ValueError("target_ratios must have length 3 (no/coarse/full)")
    probs = torch.tensor(ratios, dtype=torch.float32)
    probs = torch.clamp(probs, min=0.0)
    total = float(probs.sum().item())
    if total <= 0.0:
        raise ValueError("target_ratios must sum to a positive value")
    return probs / total


def sample_actions_from_ratios(
    ratios: Sequence[float],
    size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample actions according to target ratios."""
    probs = normalize_ratios(ratios)
    indices = torch.multinomial(probs, num_samples=size, replacement=True, generator=generator)
    return indices.to(device=device, dtype=torch.long)


def bucketize_entropy(entropy: torch.Tensor, thresholds: Tuple[float, float]) -> torch.Tensor:
    """Assign bucket ids (0=low, 1=mid, 2=high) from entropy."""
    t1, t2 = thresholds
    buckets = torch.zeros_like(entropy, dtype=torch.long)
    buckets = torch.where(entropy > t1, torch.ones_like(buckets), buckets)
    buckets = torch.where(entropy > t2, torch.full_like(buckets, 2), buckets)
    return buckets


def sample_actions_by_bucket(
    ratios_by_bucket: Sequence[Sequence[float]],
    bucket_ids: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample actions per bucket using bucket-specific ratios."""
    if len(ratios_by_bucket) != 3:
        raise ValueError("baseline_bucket_ratios must have three buckets")
    actions = torch.empty_like(bucket_ids, dtype=torch.long, device=device)
    for bucket_idx, ratios in enumerate(ratios_by_bucket):
        mask = bucket_ids == bucket_idx
        if not mask.any():
            continue
        sampled = sample_actions_from_ratios(
            ratios, int(mask.sum().item()), generator, device
        )
        actions[mask] = sampled
    return actions


def compute_entropy_thresholds(values: Iterable[torch.Tensor]) -> Tuple[float, float]:
    """Compute 1/3 and 2/3 quantile thresholds from entropy tensors."""
    items = [v.detach().float().cpu().view(-1) for v in values if v is not None]
    if not items:
        return 0.0, 0.0
    all_values = torch.cat(items, dim=0)
    if all_values.numel() == 0:
        return 0.0, 0.0
    q1 = float(torch.quantile(all_values, torch.tensor(1.0 / 3.0)).item())
    q2 = float(torch.quantile(all_values, torch.tensor(2.0 / 3.0)).item())
    return q1, q2
