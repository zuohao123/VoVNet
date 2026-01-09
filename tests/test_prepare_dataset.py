"""Smoke tests for dataset adapters."""
from __future__ import annotations

import pytest

from datasets.validators import validate_record
from datasets.registry import get_adapter, list_adapters


@pytest.mark.parametrize("name", list(list_adapters().keys()))
def test_adapter_smoke(name: str) -> None:
    adapter = get_adapter(name)
    ok, reason = adapter.smoke_test()
    if not ok:
        pytest.skip(f"{name}: {reason}")

    dataset = adapter.load(None, "train")
    try:
        count = min(5, len(dataset))
        dataset = dataset.select(range(count))
    except Exception:
        dataset = dataset

    for ex in dataset:
        record = adapter.normalize_example(ex, split="train")
        record.image = None
        valid, msg = validate_record(record.to_dict())
        assert valid, msg
