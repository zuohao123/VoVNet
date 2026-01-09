"""Adapter for GQA (HuggingFace datasets)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from ..common import AnswerInfo, HFAdapterBase, MetaInfo, UnifiedExample, resolve_image_field


class GQAAdapter(HFAdapterBase):
    """GQA adapter.

    Notes: Available on HF as "gqa". Some subsets may exist.
    """

    name = "gqa"
    hf_dataset_id = "gqa"
    task_type = "vqa"

    def normalize_example(self, ex: Dict[str, Any], split: str) -> UnifiedExample:
        question = str(ex.get("question", ""))
        answer_text = ex.get("answer")
        canonical = str(answer_text) if answer_text is not None else None

        sample_id = str(ex.get("question_id") or ex.get("id") or ex.get("image_id") or "")
        if not sample_id:
            sample_id = str(hash(question))

        meta = MetaInfo(
            language=None,
            source_fields={
                "question": "question",
                "answer": "answer",
                "image": "image",
            },
            extra={},
        )

        return UnifiedExample(
            sample_id=sample_id,
            dataset=self.name,
            split=split,
            task_type=self.task_type,
            image=None,
            question=question,
            context=None,
            choices=None,
            answer=AnswerInfo(text=canonical, aliases=None, label=None, raw=answer_text),
            meta=meta,
        )

    def get_image(self, ex: Dict[str, Any]) -> Optional[Image.Image]:
        return resolve_image_field(ex.get("image"))
