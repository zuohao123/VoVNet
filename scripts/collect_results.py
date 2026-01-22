#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_result_files(root: Path):
    seen = set()
    for results_path in root.rglob("results.csv"):
        seen.add(results_path.parent)
        yield results_path
    for matrix_path in root.rglob("eval_matrix.csv"):
        if matrix_path.parent in seen:
            continue
        yield matrix_path


def _load_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row:
                yield row


def collect(root: Path, include_summary: bool):
    rows = []
    for csv_path in _iter_result_files(root):
        run_dir = csv_path.parent
        summary = _read_json(run_dir / "summary.json") if include_summary else None
        summary_checkpoint = summary.get("checkpoint") if summary else None
        summary_pareto = summary.get("pareto") if summary else None
        dataset_metadata = summary.get("dataset_metadata", {}) if summary else {}

        for row in _load_rows(csv_path):
            dataset_name = row.get("dataset") or row.get("dataset_name")
            meta = dataset_metadata.get(dataset_name, {})
            combined = {
                "run_dir": str(run_dir),
                "source_csv": str(csv_path),
                "checkpoint": summary_checkpoint,
                "pareto": json.dumps(summary_pareto, ensure_ascii=True),
                "dataset_file": meta.get("file_path"),
                "dataset_sha1": meta.get("file_sha1"),
                "dataset_num_examples": meta.get("num_examples"),
            }
            combined.update(row)
            rows.append(combined)
    return rows


def write_csv(path: Path, rows):
    fieldnames = []
    fieldset = set()
    for row in rows:
        for key in row.keys():
            if key not in fieldset:
                fieldset.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Collect eval outputs into one CSV.")
    parser.add_argument(
        "--root",
        default="outputs",
        help="Root directory to scan (default: outputs).",
    )
    parser.add_argument(
        "--output",
        default="outputs/all_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Skip reading summary.json metadata.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect(root, include_summary=not args.no_summary)
    write_csv(Path(args.output), rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
