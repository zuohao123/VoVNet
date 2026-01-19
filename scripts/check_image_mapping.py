"""Check JSONL image paths for missing files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.common import resolve_image_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate image paths in a JSONL dataset")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples checked")
    parser.add_argument(
        "--image_root",
        default=None,
        help="Optional image root override (otherwise uses defaults/env)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    image_root = Path(args.image_root).expanduser() if args.image_root else None

    total = 0
    missing = 0
    for line in jsonl_path.open("r", encoding="utf-8"):
        if args.max_samples is not None and total >= args.max_samples:
            break
        obj = json.loads(line)
        image = obj.get("image")
        if resolve_image_field(image, image_root=image_root) is None:
            missing += 1
        total += 1

    missing_rate = (missing / total) * 100.0 if total else 0.0
    print(f"checked={total} missing={missing} missing_rate={missing_rate:.2f}%")


if __name__ == "__main__":
    main()
