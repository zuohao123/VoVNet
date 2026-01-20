"""Run metadata utilities for reproducible experiments."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.config.config import Config
from src.utils.io import write_json


def compute_sha1(path: Path) -> str:
    """Compute SHA1 hash for a file."""
    sha1 = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def get_git_commit(repo_root: Path) -> str:
    """Return the current git commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_package_versions(packages: Iterable[str]) -> Dict[str, str]:
    """Return installed package versions for the provided names."""
    versions: Dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def save_config_files(paths: Iterable[str], output_dir: Path) -> list[Dict[str, str]]:
    """Copy config files into output_dir/configs and return records."""
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    records: list[Dict[str, str]] = []
    for idx, path_str in enumerate(paths):
        path = Path(path_str)
        if not path.exists():
            continue
        target = config_dir / f"{idx:02d}_{path.name}"
        target.write_text(path.read_text())
        records.append({"source": str(path), "saved": str(target)})
    return records


def collect_dataset_metadata(dataset: Any, source: Dict[str, Any]) -> Dict[str, Any]:
    """Collect dataset provenance and versioning metadata."""
    info: Dict[str, Any] = {"source": source, "num_examples": len(dataset)}
    ds = dataset
    if hasattr(dataset, "dataset"):
        ds = dataset.dataset

    fingerprint = getattr(ds, "_fingerprint", None)
    if fingerprint is not None:
        info["fingerprint"] = fingerprint

    ds_info = getattr(ds, "info", None)
    if ds_info is not None:
        info["hf"] = {
            "builder_name": getattr(ds_info, "builder_name", None),
            "config_name": getattr(ds_info, "config_name", None),
            "version": str(getattr(ds_info, "version", "")) or None,
            "description": getattr(ds_info, "description", None),
        }

    path = getattr(dataset, "path", None)
    if path is not None:
        path = Path(path)
        if path.exists():
            info["file_path"] = str(path)
            info["file_sha1"] = compute_sha1(path)
    return info


def collect_model_metadata(cfg: Config) -> Dict[str, Any]:
    """Collect model identifiers and LoRA settings."""
    return {
        "base_model_name": cfg.model.base_model_name,
        "full_model_name": cfg.model.full_model_name,
        "use_thinking_for_full": cfg.model.use_thinking_for_full,
        "torch_dtype": cfg.model.torch_dtype,
        "use_lora": cfg.model.use_lora,
        "lora_r": cfg.model.lora_r,
        "lora_alpha": cfg.model.lora_alpha,
        "lora_dropout": cfg.model.lora_dropout,
        "lora_target_modules": list(cfg.model.lora_target_modules),
        "freeze_vision_encoder": cfg.model.freeze_vision_encoder,
    }


def collect_hparams(cfg: Config) -> Dict[str, Any]:
    """Collect key hyperparameters for reproducibility."""
    return {
        "lr": cfg.training.lr,
        "epochs": cfg.training.epochs,
        "per_device_batch_size": cfg.training.per_device_batch_size,
        "gradient_accumulation": cfg.training.gradient_accumulation,
        "mixed_precision": cfg.training.mixed_precision,
        "max_grad_norm": cfg.training.max_grad_norm,
        "lambda_cost": cfg.policy.lambda_cost,
        "baseline_name": cfg.policy.baseline_name,
        "baseline_threshold": cfg.policy.baseline_threshold,
        "baseline_uncertainty": cfg.policy.baseline_uncertainty,
        "baseline_vision": cfg.policy.baseline_vision,
        "baseline_seed": cfg.policy.baseline_seed,
        "baseline_target_ratios": cfg.policy.baseline_target_ratios,
        "baseline_bucket_ratios": cfg.policy.baseline_bucket_ratios,
        "baseline_bucket_thresholds": cfg.policy.baseline_bucket_thresholds,
        "baseline_pruning_ratio": cfg.policy.baseline_pruning_ratio,
        "baseline_pruning_mode": cfg.policy.baseline_pruning_mode,
        "baseline_merge_ratio": cfg.policy.baseline_merge_ratio,
        "baseline_merge_mode": cfg.policy.baseline_merge_mode,
        "baseline_merge_weight": cfg.policy.baseline_merge_weight,
        "baseline_enable_prune": cfg.policy.baseline_enable_prune,
        "baseline_prune_ratio": cfg.policy.baseline_prune_ratio,
        "baseline_prune_mode": cfg.policy.baseline_prune_mode,
        "baseline_pool_factor": cfg.policy.baseline_pool_factor,
        "policy_mode": cfg.policy.policy_mode,
        "gain_supervision": cfg.policy.gain_supervision,
        "gain_loss_type": cfg.policy.gain_loss_type,
        "gain_loss_weight": cfg.policy.gain_loss_weight,
        "cost_scale": cfg.policy.cost_scale,
        "seed": cfg.training.seed,
    }


def write_run_metadata(
    output_dir: Path,
    stage: str,
    cfg: Config,
    config_paths: Iterable[str],
    datasets: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a reproducibility metadata JSON for a run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_records = save_config_files(config_paths, output_dir)
    repo_root = Path(__file__).resolve().parents[2]

    metadata_payload = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(repo_root),
        "config_files": config_records,
        "config": asdict(cfg),
        "hparams": collect_hparams(cfg),
        "model": collect_model_metadata(cfg),
        "datasets": datasets,
        "environment": {
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "packages": get_package_versions(
                [
                    "torch",
                    "transformers",
                    "datasets",
                    "accelerate",
                    "deepspeed",
                    "peft",
                    "numpy",
                    "pandas",
                    "pyarrow",
                ]
            ),
        },
    }
    if extra:
        metadata_payload["extra"] = extra

    path = output_dir / f"run_metadata_{stage}.json"
    write_json(path, metadata_payload)
    return path
