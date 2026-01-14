"""HuggingFace dataset loading helpers."""
from __future__ import annotations

import logging
import os
import sysconfig
import time
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


def _load_from_paths(search_paths: Sequence[str]) -> Any:
    import importlib.machinery
    import importlib.util

    spec = importlib.machinery.PathFinder.find_spec("datasets", list(search_paths))
    if spec is None or spec.loader is None:
        raise RuntimeError("HuggingFace datasets not installed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discover_vendor_paths() -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = repo_root / "vendor" / "datasets_src"
    search_paths: list[str] = []
    env_path = os.environ.get("VOVNET_HF_DATASETS_PATH")
    if env_path:
        env_candidate = Path(env_path)
        if (env_candidate / "datasets").is_dir():
            search_paths.append(str(env_candidate))
        elif (env_candidate / "__init__.py").exists():
            search_paths.append(str(env_candidate.parent))
    if vendor_root.exists():
        for base in vendor_root.glob("datasets-*"):
            for candidate in (
                base / "datasets",
                base / "src" / "datasets",
            ):
                if candidate.is_dir():
                    search_paths.append(str(candidate.parent))
    return list(dict.fromkeys(search_paths))


def import_hf_datasets() -> Any:
    """Import HuggingFace datasets without shadowing by local package."""
    search_paths = []
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        if paths.get(key):
            search_paths.append(paths[key])

    try:
        return _load_from_paths(search_paths)
    except ModuleNotFoundError as exc:
        if "datasets.arrow_dataset" not in str(exc):
            raise
        logger.warning(
            "HF datasets install missing arrow_dataset; trying vendor fallback."
        )
    except Exception as exc:
        logger.warning("Failed to import HF datasets from site-packages: %s", exc)

    vendor_paths = _discover_vendor_paths()
    if vendor_paths:
        return _load_from_paths(vendor_paths)
    raise RuntimeError(
        "HuggingFace datasets not installed or broken. "
        "Set VOVNET_HF_DATASETS_PATH or populate vendor/datasets_src."
    )


def retry(fn: Any, retries: int = 3, base_delay: float = 1.0) -> Any:
    """Simple exponential backoff retry helper."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("Retry %s after error: %s", attempt + 1, exc)
            time.sleep(delay)


def try_dataset_info(dataset_id: str) -> Optional[str]:
    """Try to fetch dataset info from HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(dataset_id)
        return info.id
    except Exception:
        return None


def safe_load_dataset(dataset_id: str, subset: Optional[str], split: str) -> Any:
    """Load HF dataset with retries and helpful errors."""
    hf_datasets = import_hf_datasets()

    def _load() -> Any:
        if subset:
            return hf_datasets.load_dataset(dataset_id, subset, split=split)
        return hf_datasets.load_dataset(dataset_id, split=split)

    try:
        return retry(_load, retries=3, base_delay=1.0)
    except Exception as exc:
        info = try_dataset_info(dataset_id)
        hint = (
            f"Dataset info: {info}" if info else "Set --subset or check dataset id."
        )
        raise RuntimeError(
            f"Failed to load dataset {dataset_id} (subset={subset}, split={split}). {hint}"
        ) from exc
