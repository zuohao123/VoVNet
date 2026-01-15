"""Common schema and helpers for dataset normalization."""
from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image

from .hf_utils import safe_load_dataset
from .image_utils import image_to_jpeg_bytes, load_image_from_path, load_image_from_url, sha1_bytes

logger = logging.getLogger(__name__)

@dataclass
class ImageInfo:
    """Image metadata in the unified schema."""

    source: str
    path: Optional[str]
    url: Optional[str]
    sha1: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "path": self.path,
            "url": self.url,
            "sha1": self.sha1,
        }

@dataclass
class AnswerInfo:
    """Answer metadata in the unified schema."""

    text: Optional[str]
    aliases: Optional[List[str]]
    label: Optional[int]
    raw: Optional[Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "aliases": self.aliases,
            "label": self.label,
            "raw": self.raw,
        }

@dataclass
class MetaInfo:
    """Metadata in the unified schema."""

    language: Optional[str]
    source_fields: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "source_fields": self.source_fields,
            "extra": self.extra,
        }

@dataclass
class UnifiedExample:
    """Unified dataset example."""

    sample_id: str
    dataset: str
    split: str
    task_type: str
    image: Optional[ImageInfo]
    question: str
    context: Optional[str]
    choices: Optional[List[str]]
    answer: AnswerInfo
    meta: MetaInfo

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "task_type": self.task_type,
            "image": self.image.to_dict() if self.image else None,
            "question": self.question,
            "context": self.context,
            "choices": self.choices,
            "answer": self.answer.to_dict(),
            "meta": self.meta.to_dict(),
        }

class DatasetAdapter(Protocol):
    """Protocol for dataset adapters."""

    name: str
    hf_dataset_id: Optional[str]

    def load(self, subset: Optional[str], split: str, streaming: bool = False) -> Any:  # datasets.Dataset
        ...

    def normalize_example(self, ex: Dict[str, Any], split: str) -> UnifiedExample:
        ...

    def get_image(self, ex: Dict[str, Any]) -> Optional[Image.Image]:
        ...

    def smoke_test(self) -> Tuple[bool, str]:
        ...

class HFAdapterBase:
    """Base adapter for HuggingFace datasets."""

    name: str
    hf_dataset_id: Optional[str]
    hf_dataset_id_candidates: Optional[List[str]] = None
    default_subset: Optional[str] = None

    def __init__(self) -> None:
        if not self.hf_dataset_id:
            raise ValueError("hf_dataset_id must be set for HF adapters")

    def _candidate_dataset_ids(self) -> List[str]:
        env_key = f"VOVNET_HF_DATASET_ID_{self.name.upper()}"
        override = os.environ.get(env_key) or os.environ.get("VOVNET_HF_DATASET_ID")
        if override:
            return [override]
        if self.hf_dataset_id_candidates:
            return list(self.hf_dataset_id_candidates)
        return [self.hf_dataset_id]

    def load(self, subset: Optional[str], split: str, streaming: bool = False) -> Any:
        last_exc: Optional[Exception] = None
        for dataset_id in self._candidate_dataset_ids():
            try:
                if subset is None and self.default_subset is not None:
                    try:
                        return safe_load_dataset(
                            dataset_id, self.default_subset, split, streaming=streaming
                        )
                    except Exception:
                        pass
                return safe_load_dataset(dataset_id, subset, split, streaming=streaming)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Failed to load dataset %s with id=%s (subset=%s, split=%s): %s",
                    self.name,
                    dataset_id,
                    subset,
                    split,
                    exc,
                )
                continue
        raise RuntimeError(
            f"Failed to load dataset {self.name} with candidate ids {self._candidate_dataset_ids()}"
        ) from last_exc

    def smoke_test(self) -> Tuple[bool, str]:
        try:
            dataset = self.load(None, "train")
            sample = dataset[0]
            _ = self.normalize_example(sample, split="train")
            return True, "ok"
        except Exception as exc:  # pragma: no cover - external dependency
            return False, str(exc)

class StubAdapter:
    """Stub adapter for datasets not available on HF."""

    name: str = "stub"
    hf_dataset_id: Optional[str] = None
    reason: str = "Dataset not available on HuggingFace."

    def load(self, subset: Optional[str], split: str, streaming: bool = False) -> Any:
        raise NotImplementedError(self.reason)

    def normalize_example(self, ex: Dict[str, Any], split: str) -> UnifiedExample:
        raise NotImplementedError(self.reason)

    def get_image(self, ex: Dict[str, Any]) -> Optional[Image.Image]:
        return None

    def smoke_test(self) -> Tuple[bool, str]:
        return False, self.reason


def majority_vote(answers: List[str]) -> Tuple[Optional[str], List[str]]:
    """Choose a canonical answer and return aliases."""
    clean = [normalize_answer(a) for a in answers if a]
    if not clean:
        return None, []
    counts: Dict[str, int] = {}
    for item in clean:
        counts[item] = counts.get(item, 0) + 1
    best = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    aliases = sorted(set(clean))
    return best, aliases


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def resolve_choices(ex: Dict[str, Any], fields: List[str]) -> Optional[List[str]]:
    for key in fields:
        value = ex.get(key)
        if value:
            return [str(v) for v in value]
    return None


def resolve_answer_list(ex: Dict[str, Any], fields: List[str]) -> List[str]:
    answers: List[str] = []
    for key in fields:
        value = ex.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    ans = item.get("answer")
                    if ans:
                        answers.append(str(ans))
                else:
                    answers.append(str(item))
            return answers
        if isinstance(value, dict):
            ans = value.get("answer")
            if ans:
                return [str(ans)]
        return [str(value)]
    return answers


def build_image_info(
    image: Optional[Image.Image],
    source: str,
    path: Optional[Path],
    url: Optional[str],
) -> ImageInfo:
    if image is None:
        return ImageInfo(source=source, path=str(path) if path else None, url=url, sha1=None)
    data = image_to_jpeg_bytes(image)
    return ImageInfo(
        source=source,
        path=str(path) if path else None,
        url=url,
        sha1=sha1_bytes(data),
    )


def resolve_image_field(value: Any, image_root: Optional[Path] = None) -> Optional[Image.Image]:
    """Resolve image field into PIL.Image when possible."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            path = Path(value["path"])
            if image_root and not path.is_absolute():
                path = image_root / path
            return load_image_from_path(path)
        if value.get("url"):
            return load_image_from_url(value["url"])
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            return load_image_from_url(value)
        path = Path(value)
        if image_root and not path.is_absolute():
            path = image_root / path
        return load_image_from_path(path)
    return None
