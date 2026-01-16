"""HuggingFace dataset loading helpers."""
from __future__ import annotations

import logging
import os
import sys
import sysconfig
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Sequence

logger = logging.getLogger(__name__)


@contextmanager
def _swap_datasets_module(hf_module: Any) -> Generator[None, None, None]:
    previous = sys.modules.get("datasets")
    sys.modules["datasets"] = hf_module
    try:
        yield
    finally:
        if previous is not None:
            sys.modules["datasets"] = previous
        else:
            sys.modules.pop("datasets", None)


def _load_from_paths(search_paths: Sequence[str]) -> Any:
    import importlib.machinery
    import importlib.util

    spec = importlib.machinery.PathFinder.find_spec("datasets", list(search_paths))
    if spec is None or spec.loader is None:
        raise RuntimeError("HuggingFace datasets not installed")
    module = importlib.util.module_from_spec(spec)
    with _swap_datasets_module(module):
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


def _get_env_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("VOVNET_HF_TOKEN")
    )


def _get_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_load_dataset(
    dataset_id: str,
    subset: Optional[str],
    split: str,
    streaming: bool = False,
    _retry_disable_multiprocessing: bool = True,
) -> Any:
    """Load HF dataset with retries and helpful errors."""
    hf_datasets = import_hf_datasets()
    token = _get_env_token()
    trust_remote_code = _get_env_bool("VOVNET_HF_TRUST_REMOTE_CODE", default=False)

    def _apply_hf_config() -> None:
        if _get_env_bool("VOVNET_HF_DISABLE_MULTIPROCESSING", default=False) or _get_env_bool(
            "HF_DATASETS_DISABLE_MULTIPROCESSING", default=False
        ):
            try:
                hf_datasets.config.HF_DATASETS_DISABLE_MULTIPROCESSING = True
            except Exception:
                pass
        if _get_env_bool("VOVNET_HF_DISABLE_PROGRESS", default=False) or _get_env_bool(
            "HF_DATASETS_DISABLE_PROGRESS_BAR", default=False
        ):
            try:
                hf_datasets.disable_progress_bar()
            except Exception:
                pass

    def _load() -> Any:
        _apply_hf_config()
        kwargs: dict[str, Any] = {
            "split": split,
            "streaming": streaming,
        }
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        if token:
            kwargs["token"] = token
        with _swap_datasets_module(hf_datasets):
            if subset:
                try:
                    return hf_datasets.load_dataset(dataset_id, subset, **kwargs)
                except TypeError:
                    if "token" in kwargs:
                        kwargs["use_auth_token"] = kwargs.pop("token")
                    return hf_datasets.load_dataset(dataset_id, subset, **kwargs)
            try:
                return hf_datasets.load_dataset(dataset_id, **kwargs)
            except TypeError:
                if "token" in kwargs:
                    kwargs["use_auth_token"] = kwargs.pop("token")
                return hf_datasets.load_dataset(dataset_id, **kwargs)

    try:
        return retry(_load, retries=3, base_delay=1.0)
    except Exception as exc:
        if _is_rlock_error(exc) and _retry_disable_multiprocessing:
            logger.warning(
                "HF dataset load hit RLock error; retrying with streaming=True and "
                "multiprocessing disabled for %s.",
                dataset_id,
            )
            os.environ["VOVNET_HF_DISABLE_MULTIPROCESSING"] = "1"
            os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
            return safe_load_dataset(
                dataset_id,
                subset,
                split,
                streaming=True,
                _retry_disable_multiprocessing=False,
            )
        info = try_dataset_info(dataset_id)
        hint = (
            f"Dataset info: {info}" if info else "Set --subset or check dataset id."
        )
        raise RuntimeError(
            f"Failed to load dataset {dataset_id} (subset={subset}, split={split}). {hint}"
        ) from exc


def _is_rlock_error(exc: Exception) -> bool:
    needle = "RLock objects should only be shared between processes through inheritance"
    seen = set()
    current: Optional[BaseException] = exc
    while current and current not in seen:
        seen.add(current)
        if needle in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False
