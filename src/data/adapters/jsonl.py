"""JSONL dataset adapter."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
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
        answer = str(item.get(self.answer_field, ""))
        image_path = item.get(self.image_field)
        image = self._load_image(image_path) if image_path else None
        sample_id = str(item.get("id", idx))
        meta = dict(item.get("meta", {}))
        return {
            "question": question,
            "answer": answer,
            "image": image,
            "id": sample_id,
            "meta": meta,
        }

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.path.parent / path
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to load image %s: %s", path, exc)
            return None
