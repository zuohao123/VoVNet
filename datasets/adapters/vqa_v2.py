"""Adapter for VQA v2 (HuggingFace datasets)."""
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


class VQAv2Adapter(HFAdapterBase):
    """VQA v2 adapter.

    Notes: Common HF ids include "HuggingFaceM4/VQAv2" or "vqa" with config "vqa_v2".
    Override via VOVNET_HF_DATASET_ID_VQA_V2 if needed.
    """

    name = "vqa_v2"
    hf_dataset_id = "HuggingFaceM4/VQAv2"
    hf_dataset_id_candidates = ["HuggingFaceM4/VQAv2", "vqa"]
    default_subset = "vqa_v2"
    task_type = "vqa"

    def normalize_example(self, ex: Dict[str, Any], split: str) -> UnifiedExample:
        question = str(ex.get("question", ""))
        answers = resolve_answer_list(ex, ["answers", "answer", "multiple_choice_answer"])
        canonical, aliases = majority_vote(answers)

        sample_id = str(ex.get("question_id") or ex.get("id") or ex.get("image_id") or "")
        if not sample_id:
            sample_id = str(hash(question))

        meta = MetaInfo(
            language=None,
            source_fields={
                "question": "question",
                "answers": "answers",
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
            answer=AnswerInfo(text=canonical, aliases=aliases or None, label=None, raw=ex.get("answers")),
            meta=meta,
        )

    def get_image(self, ex: Dict[str, Any]) -> Optional[Image.Image]:
        return resolve_image_field(ex.get("image"))
