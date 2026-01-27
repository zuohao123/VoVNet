"""Relabel missing `dataset` fields in mixed training JSONL files.

This is a lightweight, heuristic relabeler intended to fix cases where
`dataset` is null/None for large portions of a merged training set.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def _image_path(ex: Dict[str, Any]) -> str:
    img = ex.get("image")
    if isinstance(img, dict):
        return str(img.get("path") or "")
    return str(img or "")


def _infer_dataset(ex: Dict[str, Any]) -> str:
    # Keep existing dataset labels when present.
    ds = ex.get("dataset")
    if isinstance(ds, str) and ds.strip():
        return ds

    p = _image_path(ex).lower()
    if "/textvqa/" in p:
        return "textvqa"
    if "/mmbench/" in p:
        return "mmbench"
    if "/mmmu/" in p:
        return "mmmu"

    # OCR-heavy samples often come from TextVQA-style data.
    meta = ex.get("meta") or {}
    extra = meta.get("extra") if isinstance(meta, dict) else {}
    if isinstance(extra, dict) and extra.get("ocr_tokens"):
        return "textvqa"

    # Fallback: treat as generic open-ended text data.
    return "text"


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel missing dataset fields.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    before = Counter()
    after = Counter()
    changed = 0

    with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            before[str(ex.get("dataset"))] += 1
            new_ds = _infer_dataset(ex)
            if ex.get("dataset") != new_ds:
                changed += 1
            ex["dataset"] = new_ds
            after[new_ds] += 1
            g.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("before:", before.most_common(6))
    print("after :", after.most_common(6))
    print("changed:", changed)
    print("wrote  :", dst)


if __name__ == "__main__":
    main()

