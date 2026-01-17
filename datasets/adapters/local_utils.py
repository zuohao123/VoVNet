"""Helpers for local dataset adapters."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image

from ..common import resolve_image_field
from ..hf_utils import safe_load_dataset

logger = logging.getLogger(__name__)

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\\s]+$")

def resolve_dataset_root(name: str, env_prefix: str) -> tuple[Path, Optional[Path]]:
    """Resolve dataset root and optional explicit file from env or defaults."""
    env_path = (
        os.environ.get(f"VOVNET_{env_prefix}_PATH")
        or os.environ.get(f"VOVNET_{env_prefix}_ROOT")
    )
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return path.parent, path
        return path, None
    return Path("data/raw") / name, None


def resolve_image_root(root: Path, env_prefix: str) -> Optional[Path]:
    env_path = os.environ.get(f"VOVNET_{env_prefix}_IMAGE_ROOT")
    if env_path:
        return Path(env_path).expanduser()
    candidate = root / "images"
    if candidate.exists():
        return candidate
    if root.exists():
        return root
    return None


def _split_tokens(split: str) -> List[str]:
    key = split.lower()
    if key in {"train", "training"}:
        return ["train", "trn", "dev"]
    if key in {"val", "valid", "validation", "dev"}:
        return ["val", "valid", "validation", "dev"]
    if key in {"test", "testing"}:
        return ["test"]
    if key == "all":
        return []
    return [key]


def _list_data_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    exts = {".json", ".jsonl", ".csv", ".tsv", ".parquet"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def select_data_files(
    root: Path,
    split: str,
    subset: Optional[str],
    explicit_file: Optional[Path],
) -> List[Path]:
    if explicit_file:
        return [explicit_file]

    if subset:
        subset_path = Path(subset).expanduser()
        if subset_path.is_file():
            return [subset_path]
        candidate = root / subset
        if candidate.is_file():
            return [candidate]

    candidates = _list_data_files(root)
    if not candidates:
        raise FileNotFoundError(
            f"No dataset files found under {root}. "
            f"Set VOVNET_{root.name.upper()}_ROOT or VOVNET_{root.name.upper()}_PATH."
        )

    subset_token = subset.lower() if subset else None
    if subset_token:
        filtered = [p for p in candidates if subset_token in p.name.lower()]
        if filtered:
            candidates = filtered

    tokens = _split_tokens(split)
    if tokens:
        split_filtered = [
            p for p in candidates if any(token in p.name.lower() for token in tokens)
        ]
        if split_filtered:
            return split_filtered

    if len(candidates) == 1:
        return candidates

    return candidates


def _extract_records_from_json(data: Any, split: str) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []

    for key in ("data", "samples", "questions", "instances", "annotations"):
        value = data.get(key)
        if isinstance(value, list):
            return value

    split_tokens = _split_tokens(split)
    for token in split_tokens + [split.lower()]:
        for key, value in data.items():
            if key.lower() == token and isinstance(value, list):
                return value

    list_values = [value for value in data.values() if isinstance(value, list)]
    if list_values:
        records: List[Dict[str, Any]] = []
        for value in list_values:
            records.extend(value)
        return records

    return []


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_table(path: Path, sep: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    return df.to_dict(orient="records")


def load_records_from_file(path: Path, split: str) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return _extract_records_from_json(data, split)
    if suffix == ".parquet":
        return _read_parquet(path)
    if suffix == ".tsv":
        return _read_table(path, sep="\t")
    if suffix == ".csv":
        return _read_table(path, sep=",")
    raise ValueError(f"Unsupported dataset file: {path}")


def _read_parquet(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        import pandas as pd

        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    table = pq.read_table(path)
    return table.to_pylist()


def load_records(files: Sequence[Path], split: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in files:
        records.extend(load_records_from_file(path, split))
    return records


def load_dataset_with_fallback(
    name: str,
    env_prefix: str,
    split: str,
    subset: Optional[str],
    streaming: bool = False,
    dataset_id: Optional[str] = None,
) -> Any:
    env_key = f"VOVNET_HF_DATASET_ID_{env_prefix}"
    env_id = dataset_id or os.environ.get(env_key) or os.environ.get("VOVNET_HF_DATASET_ID")
    last_exc: Optional[Exception] = None
    if env_id:
        try:
            return safe_load_dataset(env_id, subset, split, streaming=streaming)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Failed to load HF dataset %s for %s: %s. Falling back to local files.",
                env_id,
                name,
                exc,
            )
            try:
                root, _ = resolve_dataset_root(name, env_prefix)
                download_hf_dataset_files(env_id, subset, split, root)
            except Exception as dl_exc:
                logger.warning("Failed to download HF dataset files: %s", dl_exc)

    root, explicit = resolve_dataset_root(name, env_prefix)
    files = select_data_files(root, split, subset, explicit)
    records = load_records(files, split)
    if not records and last_exc:
        raise RuntimeError(
            f"Failed to load {name} from HF ({env_id}) and no local records found."
        ) from last_exc
    return records


def download_hf_dataset_files(
    dataset_id: str,
    subset: Optional[str],
    split: str,
    target_root: Path,
) -> List[Path]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("huggingface_hub is required to download HF datasets") from exc

    api = HfApi()
    files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
    candidates = _filter_hf_files(files, subset, split)
    if not candidates:
        raise FileNotFoundError(
            f"No matching HF files for {dataset_id} (subset={subset}, split={split})."
        )
    target_root.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for filename in candidates:
        local_path = hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=filename,
            local_dir=target_root,
            local_dir_use_symlinks=False,
        )
        downloaded.append(Path(local_path))
    return downloaded


def _filter_hf_files(
    files: Sequence[str],
    subset: Optional[str],
    split: str,
) -> List[str]:
    split_token = split.lower()
    subset_token = subset.lower() if subset else None
    candidates: List[str] = []
    for filename in files:
        lower = filename.lower()
        if split_token not in lower:
            continue
        if subset_token and subset_token not in lower:
            continue
        if lower.endswith((".parquet", ".jsonl", ".json", ".csv", ".tsv")):
            candidates.append(filename)
    return candidates


def first_non_empty(ex: dict, keys: Sequence[str]) -> Any:
    for key in keys:
        value = ex.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def parse_choice_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        choices = [str(item) for item in value if item not in (None, "")]
        return choices or None
    if isinstance(value, dict):
        items = [value[key] for key in sorted(value.keys())]
        choices = [str(item) for item in items if item not in (None, "")]
        return choices or None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = json.loads(text)
                return parse_choice_list(parsed)
            except Exception:
                return None
        if "||" in text:
            parts = [part.strip() for part in text.split("||") if part.strip()]
            return parts or None
        if "|" in text:
            parts = [part.strip() for part in text.split("|") if part.strip()]
            return parts or None
        if "\n" in text:
            parts = [part.strip() for part in text.splitlines() if part.strip()]
            return parts or None
    return None


def extract_letter_choices(ex: dict, letters: Sequence[str]) -> Optional[List[str]]:
    choices: List[str] = []
    for key in letters:
        value = ex.get(key)
        if value not in (None, ""):
            choices.append(str(value))
    return choices or None


def map_answer_to_choice(
    answer: Any,
    choices: Optional[List[str]],
) -> Tuple[Optional[str], Optional[int]]:
    if answer is None:
        return None, None
    text = str(answer).strip()
    if not text:
        return None, None
    if not choices:
        return text, None

    if len(text) == 1 and text.isalpha():
        idx = ord(text.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx], idx
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= len(choices):
            return choices[idx - 1], idx - 1
        if 0 <= idx < len(choices):
            return choices[idx], idx
    for idx, choice in enumerate(choices):
        if text.lower() == str(choice).strip().lower():
            return str(choice), idx
    return text, None


def maybe_decode_base64_image(value: str) -> Optional[bytes]:
    text = value.strip()
    if text.startswith("data:image"):
        _, _, text = text.partition(",")
        text = text.strip()
    if len(text) < 64 or "." in text:
        return None
    if not _BASE64_RE.match(text):
        return None
    text = "".join(text.split())
    padded = text + "=" * (-len(text) % 4)
    try:
        data = base64.b64decode(padded, validate=False)
    except Exception:
        return None
    if data[:2] == b"\xff\xd8" or data[:8] == b"\x89PNG\r\n\x1a\n":
        return data
    return None


def _resolve_image_value(
    value: Any,
    image_root: Optional[Path],
) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return resolve_image_field({"bytes": bytes(value)}, image_root=image_root)
    if isinstance(value, dict):
        return resolve_image_field(value, image_root=image_root)
    if isinstance(value, str):
        decoded = maybe_decode_base64_image(value)
        if decoded is not None:
            return resolve_image_field({"bytes": decoded}, image_root=image_root)
        return resolve_image_field(value, image_root=image_root)
    return None


def resolve_image_from_record(
    ex: dict,
    fields: Sequence[str],
    image_root: Optional[Path],
) -> Optional[Image.Image]:
    value = first_non_empty(ex, fields)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            image = _resolve_image_value(item, image_root)
            if image is not None:
                return image
        return None
    return _resolve_image_value(value, image_root)
