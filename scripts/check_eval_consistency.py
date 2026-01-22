"""Check evaluation runs for prompt/metric/answer-field consistency."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_summary(output_dir: Path) -> Dict[str, Any] | None:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return _load_json(summary_path)
    except Exception:
        return None


def _parse_dataset_filter(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    tokens = [item.strip() for item in raw.replace(",", " ").split()]
    return [token for token in tokens if token]


def _collect_records(root: Path, dataset_filter: Optional[List[str]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for meta_path in sorted(root.rglob("run_metadata_eval.json")):
        output_dir = meta_path.parent
        metadata = _load_json(meta_path)
        summary = _load_summary(output_dir)
        cfg = metadata.get("config", {})
        datasets = metadata.get("datasets", {})
        for name, info in datasets.items():
            if dataset_filter and name not in dataset_filter:
                continue
            source = info.get("source", {})
            metric = None
            if summary:
                metric = summary.get("datasets", {}).get(name, {}).get("metric")
            records.append(
                {
                    "dataset": name,
                    "output_dir": str(output_dir),
                    "prompt_template": cfg.get("data", {}).get("prompt_template"),
                    "text_field": cfg.get("data", {}).get("text_field"),
                    "answer_field": cfg.get("data", {}).get("answer_field"),
                    "image_field": cfg.get("data", {}).get("image_field"),
                    "max_samples": cfg.get("data", {}).get("max_samples"),
                    "metric": metric,
                    "jsonl": source.get("jsonl"),
                    "baseline_name": metadata.get("hparams", {}).get("baseline_name"),
                    "checkpoint": summary.get("checkpoint") if summary else None,
                }
            )
    return records


def _group_by(records: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(key)), []).append(record)
    return grouped


def _signature(record: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    return {field: record.get(field) for field in fields}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    headers = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check eval runs for prompt/metric/answer-field consistency"
    )
    parser.add_argument(
        "--root",
        default="outputs/eval",
        help="Root directory containing eval outputs",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter (comma-separated)",
    )
    parser.add_argument(
        "--output",
        default="outputs/eval/consistency_report.json",
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--csv",
        default="outputs/eval/consistency_report.csv",
        help="Optional CSV report path (empty to disable)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if mismatches are found",
    )
    args = parser.parse_args()

    root = Path(args.root)
    dataset_filter = _parse_dataset_filter(args.dataset)
    records = _collect_records(root, dataset_filter)
    grouped = _group_by(records, "dataset")
    signature_fields = [
        "prompt_template",
        "text_field",
        "answer_field",
        "image_field",
        "metric",
        "jsonl",
    ]

    mismatches: List[Dict[str, Any]] = []
    for dataset, items in grouped.items():
        if not items:
            continue
        reference = _signature(items[0], signature_fields)
        for item in items[1:]:
            candidate = _signature(item, signature_fields)
            diffs = {
                key: {"expected": reference.get(key), "actual": candidate.get(key)}
                for key in signature_fields
                if reference.get(key) != candidate.get(key)
            }
            if diffs:
                mismatches.append(
                    {
                        "dataset": dataset,
                        "output_dir": item.get("output_dir"),
                        "baseline_name": item.get("baseline_name"),
                        "checkpoint": item.get("checkpoint"),
                        "diffs": diffs,
                    }
                )

    report = {
        "root": str(root),
        "datasets_checked": sorted(grouped.keys()),
        "total_runs": len(records),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.csv:
        _write_csv(Path(args.csv), records)

    if mismatches:
        print(f"Found {len(mismatches)} mismatches. See {output_path}.")
    else:
        print(f"No mismatches found. See {output_path}.")
    if args.strict and mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
