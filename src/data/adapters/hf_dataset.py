"""HuggingFace datasets adapter."""
from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image
from torch.utils.data import Dataset


class HFDatasetAdapter(Dataset):
    """Adapter for HuggingFace datasets with image fields."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        subset: Optional[str] = None,
        text_field: str = "question",
        answer_field: str = "answer",
        image_field: str = "image",
        max_samples: Optional[int] = None,
    ) -> None:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("datasets is required for HF dataset loading") from exc

        if subset:
            self.dataset = load_dataset(dataset_name, subset, split=split)
        else:
            self.dataset = load_dataset(dataset_name, split=split)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(max_samples))

        self.text_field = text_field
        self.answer_field = answer_field
        self.image_field = image_field

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        question = str(item.get(self.text_field, ""))
        answer = str(item.get(self.answer_field, ""))
        image = item.get(self.image_field)
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        return {
            "question": question,
            "answer": answer,
            "image": image,
            "id": str(item.get("id", idx)),
            "meta": dict(item.get("meta", {})),
        }
