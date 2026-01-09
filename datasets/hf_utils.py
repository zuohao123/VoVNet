"""HuggingFace dataset loading helpers."""
from __future__ import annotations

import logging
import sysconfig
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def import_hf_datasets() -> Any:
    """Import HuggingFace datasets without shadowing by local package."""
    import importlib.machinery
    import importlib.util

    search_paths = []
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        if paths.get(key):
            search_paths.append(paths[key])

    spec = importlib.machinery.PathFinder.find_spec("datasets", search_paths)
    if spec is None or spec.loader is None:
        raise RuntimeError("HuggingFace datasets not installed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
