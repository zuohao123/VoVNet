"""Export utilities for normalized datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from tqdm import tqdm


def export_jsonl(
    records: Iterable[Dict[str, Any]],
    path: Path,
    total: Optional[int] = None,
) -> int:
    """Export records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, total=total, desc=f"write {path.name}"):
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def export_parquet(
    records: Iterable[Dict[str, Any]],
    path: Path,
    total: Optional[int] = None,
    batch_size: int = 1000,
) -> int:
    """Export records to Parquet (requires pyarrow)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required for Parquet export") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    count = 0
    batch: list[Dict[str, Any]] = []

    for record in tqdm(records, total=total, desc=f"write {path.name}"):
        batch.append(record)
        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            count += len(batch)
            batch = []

    if batch:
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
        count += len(batch)

    if writer is not None:
        writer.close()
    return count
