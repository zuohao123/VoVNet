"""I/O utilities."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: str | Path) -> Dict[str, Any]:
    """Read a JSON file."""
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Write a JSON file with indentation."""
    Path(path).write_text(json.dumps(data, indent=2))


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    items: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, items: Iterable[Dict[str, Any]]) -> None:
    """Write items to a JSONL file."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows to a CSV file using dict keys as header."""
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
