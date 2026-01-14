"""Adapter for TextVQA (HuggingFace datasets)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from ..common import (
    AnswerInfo,
    HFAdapterBase,
    MetaInfo,
    UnifiedExample,
    majority_vote,
    resolve_answer_list,
    resolve_image_field,
)


class TextVQAAdapter(HFAdapterBase):
    """TextVQA adapter.

    Notes: Common HF ids include "facebook/textvqa" or "textvqa".
    Override via VOVNET_HF_DATASET_ID_TEXTVQA if needed.
    """

    name = "textvqa"
    hf_dataset_id = "facebook/textvqa"
    hf_dataset_id_candidates = ["facebook/textvqa", "textvqa"]
    task_type = "ocr_vqa"

    def normalize_example(self, ex: Dict[str, Any], split: str) -> UnifiedExample:
        question = str(ex.get("question", ""))
        answers = resolve_answer_list(ex, ["answers", "answer", "label"])
        canonical, aliases = majority_vote(answers)

        sample_id = str(ex.get("question_id") or ex.get("id") or ex.get("image_id") or "")
        if not sample_id:
            sample_id = str(hash(question))

        extra = {}
        if "ocr_tokens" in ex:
            extra["ocr_tokens"] = ex.get("ocr_tokens")
        if "ocr" in ex:
            extra["ocr"] = ex.get("ocr")

        meta = MetaInfo(
            language=None,
            source_fields={
                "question": "question",
                "answers": "answers",
                "image": "image",
                "ocr_tokens": "ocr_tokens",
            },
            extra=extra,
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
            answer=AnswerInfo(text=canonical, aliases=aliases or None, label=None, raw=ex.get("answers")),
            meta=meta,
        )

    def get_image(self, ex: Dict[str, Any]) -> Optional[Image.Image]:
        return resolve_image_field(ex.get("image"))
