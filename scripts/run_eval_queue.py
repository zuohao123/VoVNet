"""Run evaluation jobs across GPUs and aggregate results."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_jobs(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if "jobs" not in data:
        raise ValueError("jobs yaml must define a 'jobs' list")
    return data


def _ensure_parent(path: str | None) -> None:
    if not path:
        return
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _read_summary(output_dir: str) -> Dict[str, Any] | None:
    summary_path = Path(output_dir) / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text())
    except Exception:
        return None


def _flatten_results(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    datasets = summary.get("datasets", {})
    for name, payload in datasets.items():
        results = payload.get("results", [])
        for item in results:
            row = {"dataset": name}
            row.update(item)
            rows.append(row)
    return rows


def run_queue(jobs_path: Path, gpus: List[str], aggregate_path: Path, csv_path: Path | None, poll: float) -> None:
    config = _load_jobs(jobs_path)
    job_env = config.get("env", {})
    jobs = deque(config.get("jobs", []))
    available = deque(gpus)
    running: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    while jobs or running:
        while available and jobs:
            gpu = available.popleft()
            job = jobs.popleft()
            cmd = job.get("cmd")
            name = job.get("name", cmd)
            output_dir = job.get("output_dir")
            log_path = job.get("log")
            env = os.environ.copy()
            env.update(job_env)
            env.update(job.get("env", {}))
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            _ensure_parent(log_path)
            log_handle = open(log_path, "w") if log_path else subprocess.DEVNULL
            proc = subprocess.Popen(cmd, shell=True, env=env, stdout=log_handle, stderr=log_handle)
            running.append(
                {
                    "name": name,
                    "cmd": cmd,
                    "output_dir": output_dir,
                    "log": log_path,
                    "gpu": gpu,
                    "process": proc,
                    "log_handle": log_handle,
                    "start_time": datetime.utcnow().isoformat() + "Z",
                }
            )

        time.sleep(poll)
        still_running: List[Dict[str, Any]] = []
        for item in running:
            proc = item["process"]
            ret = proc.poll()
            if ret is None:
                still_running.append(item)
                continue
            if item.get("log_handle") not in (None, subprocess.DEVNULL):
                item["log_handle"].close()
            item["returncode"] = ret
            item["end_time"] = datetime.utcnow().isoformat() + "Z"
            summary = _read_summary(item.get("output_dir", "")) if item.get("output_dir") else None
            item["summary"] = summary
            item["rows"] = _flatten_results(summary) if summary else []
            results.append(item)
            available.append(item["gpu"])
        running = still_running

    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"jobs": results}
    aggregate_path.write_text(json.dumps(payload, indent=2))

    if csv_path is not None:
        rows: List[Dict[str, Any]] = []
        for item in results:
            for row in item.get("rows", []):
                out = dict(row)
                out["job_name"] = item.get("name")
                out["job_cmd"] = item.get("cmd")
                out["job_output_dir"] = item.get("output_dir")
                out["job_gpu"] = item.get("gpu")
                out["job_returncode"] = item.get("returncode")
                rows.append(out)
        if rows:
            headers = sorted({key for row in rows for key in row.keys()})
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for row in rows:
                    f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval jobs across GPUs and aggregate results")
    parser.add_argument("--jobs", required=True, help="Path to jobs YAML")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids")
    parser.add_argument(
        "--aggregate_path",
        default="outputs/eval/quick_200/aggregate_results.json",
        help="Where to write aggregated JSON",
    )
    parser.add_argument(
        "--aggregate_csv",
        default="outputs/eval/quick_200/aggregate_results.csv",
        help="Optional CSV path (set empty to disable)",
    )
    parser.add_argument("--poll", type=float, default=2.0, help="Polling interval (seconds)")
    args = parser.parse_args()

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    csv_path = Path(args.aggregate_csv) if args.aggregate_csv else None
    run_queue(Path(args.jobs), gpus, Path(args.aggregate_path), csv_path, args.poll)


if __name__ == "__main__":
    main()
