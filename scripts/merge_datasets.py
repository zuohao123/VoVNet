"""Merge multiple JSONL datasets into one JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge JSONL datasets")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files (space-separated)",
    )
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--default_dataset",
        default=None,
        help="Default dataset name if missing",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for input_path in args.inputs:
            path = Path(input_path)
            for item in iter_jsonl(path):
                if args.default_dataset and not item.get("dataset"):
                    item["dataset"] = args.default_dataset
                out.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
