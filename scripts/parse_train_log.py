#!/usr/bin/env python3
"""Parse VoVNet training logs into a compact CSV + summary."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional


FLOAT_RE = r"(?:nan|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
LINE_RE = re.compile(
    rf"stage=(?P<stage>\S+).*?"
    rf"global_step=(?P<global_step>\d+)/(?:\d+).*?"
    rf"avg_task=(?P<avg_task>{FLOAT_RE}).*?"
    rf"avg_cost=(?P<avg_cost>{FLOAT_RE}).*?"
    rf"avg_gain=(?P<avg_gain>{FLOAT_RE}).*?"
    rf"lr=(?P<lr>{FLOAT_RE}).*?"
    rf"lambda_cost=(?P<lambda_cost>{FLOAT_RE}).*?"
    rf"action_entropy=(?P<action_entropy>{FLOAT_RE}).*?"
    rf"action_ratio=\[(?P<action_ratio>[^\]]+)\].*?"
    rf"expected_cost=(?P<expected_cost>{FLOAT_RE}).*?"
    rf"vision_tokens=(?P<vision_tokens>{FLOAT_RE}).*?"
    rf"token_count_coarse=(?P<token_count_coarse>{FLOAT_RE}).*?"
    rf"token_count_full=(?P<token_count_full>{FLOAT_RE})"
)


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_action_ratio(value: str) -> List[Optional[float]]:
    parts = [p.strip() for p in value.split(",")]
    ratios: List[Optional[float]] = []
    for part in parts[:3]:
        ratios.append(_to_float(part))
    while len(ratios) < 3:
        ratios.append(None)
    return ratios


def _avg(values: List[Optional[float]]) -> Optional[float]:
    items = [v for v in values if v is not None]
    if not items:
        return None
    return sum(items) / len(items)


def parse_log(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        ratio = _parse_action_ratio(match.group("action_ratio"))
        rows.append(
            {
                "stage": match.group("stage"),
                "global_step": int(match.group("global_step")),
                "avg_task": _to_float(match.group("avg_task")),
                "avg_cost": _to_float(match.group("avg_cost")),
                "avg_gain": _to_float(match.group("avg_gain")),
                "lr": _to_float(match.group("lr")),
                "lambda_cost": _to_float(match.group("lambda_cost")),
                "action_entropy": _to_float(match.group("action_entropy")),
                "action_ratio_no": ratio[0],
                "action_ratio_coarse": ratio[1],
                "action_ratio_full": ratio[2],
                "expected_cost": _to_float(match.group("expected_cost")),
                "vision_tokens": _to_float(match.group("vision_tokens")),
                "token_count_coarse": _to_float(match.group("token_count_coarse")),
                "token_count_full": _to_float(match.group("token_count_full")),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to training log file.")
    parser.add_argument(
        "--out",
        default="outputs/train_log_metrics.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="Window size for summary stats.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    rows = parse_log(log_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        headers = list(rows[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    print(f"rows={len(rows)}")
    if not rows:
        return

    last = rows[-1]
    window = rows[-args.window :]
    print(
        "last:",
        f"stage={last['stage']}",
        f"global_step={last['global_step']}",
        f"avg_task={last['avg_task']}",
        f"avg_cost={last['avg_cost']}",
        f"avg_gain={last['avg_gain']}",
        f"lambda_cost={last['lambda_cost']}",
        f"entropy={last['action_entropy']}",
        f"ratio={last['action_ratio_no']}/{last['action_ratio_coarse']}/{last['action_ratio_full']}",
    )
    print(
        "window_avg:",
        f"avg_task={_avg([r['avg_task'] for r in window])}",
        f"avg_cost={_avg([r['avg_cost'] for r in window])}",
        f"avg_gain={_avg([r['avg_gain'] for r in window])}",
        f"entropy={_avg([r['action_entropy'] for r in window])}",
        f"ratio_no={_avg([r['action_ratio_no'] for r in window])}",
        f"ratio_coarse={_avg([r['action_ratio_coarse'] for r in window])}",
        f"ratio_full={_avg([r['action_ratio_full'] for r in window])}",
    )


if __name__ == "__main__":
    main()
