"""JSONL dataset adapter."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from datasets.common import resolve_image_field
from torch.utils.data import Dataset

from src.utils.io import read_jsonl

logger = logging.getLogger(__name__)


@dataclass
class JsonlExample:
    """Single JSONL example."""

    question: str
    answer: str
    image: Optional[Image.Image]
    sample_id: str
    meta: Dict[str, Any]


class JsonlVQADataset(Dataset):
    """Dataset for JSONL with {image, question, answer, id, meta}."""

    def __init__(
        self,
        path: str | Path,
        text_field: str = "question",
        answer_field: str = "answer",
        image_field: str = "image",
        max_samples: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.text_field = text_field
        self.answer_field = answer_field
        self.image_field = image_field

        self.items = read_jsonl(self.path)
        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        question = str(item.get(self.text_field, ""))
        answer_value = item.get(self.answer_field)
        answer_info = item.get("answer_info", answer_value)
        if self.answer_field != "answer":
            raw_answer = item.get("answer")
            if isinstance(raw_answer, dict):
                answer_info = raw_answer
        answer = self._resolve_answer(answer_value)
        context = item.get("context") or item.get("hint") or item.get("rationale")
        choices = item.get("choices") or item.get("options") or item.get("candidates")
        image_path = item.get(self.image_field)
        image = self._load_image(image_path) if image_path else None
        sample_id = str(item.get("id", idx))
        meta = dict(item.get("meta", {}))
        dataset_name = item.get("dataset") or item.get("source") or meta.get("dataset")
        if dataset_name is None:
            dataset_name = "unknown"
        meta.setdefault("dataset", dataset_name)
        return {
            "question": question,
            "answer": answer,
            "answer_info": answer_info,
            "image": image,
            "context": context,
            "choices": choices,
            "id": sample_id,
            "meta": meta,
            "dataset": dataset_name,
        }

    def _resolve_answer(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            label = value.get("label")
            if label is not None:
                letter = self._label_to_letter(label)
                if letter is not None:
                    return letter
            text = value.get("text")
            if text not in (None, ""):
                return str(text)
            raw = value.get("raw")
            if isinstance(raw, list) and raw:
                return str(raw[0])
            if raw not in (None, "", []):
                return str(raw)
            return ""
        if isinstance(value, list):
            for item in value:
                if item not in (None, ""):
                    return str(item)
            return ""
        return str(value)

    def _label_to_letter(self, label: Any) -> Optional[str]:
        if label is None:
            return None
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if isinstance(label, int):
            if 0 <= label < len(letters):
                return letters[label]
            if 1 <= label <= len(letters):
                return letters[label - 1]
            return None
        label_str = str(label).strip()
        if label_str.isdigit():
            idx = int(label_str)
            if 0 <= idx < len(letters):
                return letters[idx]
            if 1 <= idx <= len(letters):
                return letters[idx - 1]
        if len(label_str) == 1 and label_str.upper() in letters:
            return label_str.upper()
        return None

    def _load_image(self, image_path: Any) -> Optional[Image.Image]:
        if image_path is None:
            return None
        if isinstance(image_path, dict) or isinstance(image_path, str):
            image = resolve_image_field(image_path, image_root=self.path.parent)
            if image is not None:
                return image
        if isinstance(image_path, str):
            path = Path(image_path)
            if not path.is_absolute():
                path = self.path.parent / path
            try:
                with Image.open(path) as img:
                    return img.convert("RGB")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Failed to load image %s: %s", path, exc)
        return None
