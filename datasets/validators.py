"""Schema validation utilities."""
from __future__ import annotations

from typing import Any, Dict, Tuple


def validate_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """Lightweight schema validation for normalized records."""
    required = [
        "id",
        "dataset",
        "split",
        "task_type",
        "image",
        "question",
        "context",
        "choices",
        "answer",
        "meta",
    ]
    for key in required:
        if key not in record:
            return False, f"missing key: {key}"
    if not isinstance(record["id"], str):
        return False, "id must be str"
    if not isinstance(record["question"], str):
        return False, "question must be str"
    if record["choices"] is not None and not isinstance(record["choices"], list):
        return False, "choices must be list or null"
    if not isinstance(record["answer"], dict):
        return False, "answer must be dict"
    if not isinstance(record["meta"], dict):
        return False, "meta must be dict"
    return True, "ok"
