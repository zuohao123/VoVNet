"""Adapter for TextVQA (HuggingFace datasets)."""
from __future__ import annotations

import os
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
from .local_utils import load_dataset_with_fallback, resolve_dataset_root, resolve_image_root


class TextVQAAdapter(HFAdapterBase):
    """TextVQA adapter.

    Notes: Common HF ids include "lmms-lab/textvqa" or "facebook/textvqa".
    Override via VOVNET_HF_DATASET_ID_TEXTVQA if needed.
    """

    name = "textvqa"
    env_prefix = "TEXTVQA"
    hf_dataset_id = "lmms-lab/textvqa"
    hf_dataset_id_candidates = ["lmms-lab/textvqa", "facebook/textvqa", "textvqa"]
    task_type = "ocr_vqa"

    def __init__(self) -> None:
        super().__init__()
        root, _ = resolve_dataset_root(self.name, self.env_prefix)
        self.image_root = resolve_image_root(root, self.env_prefix)

    def load(self, subset: Optional[str], split: str, streaming: bool = False) -> Any:
        env_key = f"VOVNET_HF_DATASET_ID_{self.env_prefix}"
        override = os.environ.get(env_key) or os.environ.get("VOVNET_HF_DATASET_ID")
        candidates = [override] if override else []
        for candidate in self.hf_dataset_id_candidates or []:
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            candidates = [self.hf_dataset_id]

        last_exc: Optional[Exception] = None
        for dataset_id in candidates:
            try:
                return load_dataset_with_fallback(
                    name=self.name,
                    env_prefix=self.env_prefix,
                    split=split,
                    subset=subset,
                    streaming=streaming,
                    dataset_id=dataset_id,
                )
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("Failed to load TextVQA dataset")

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
        return resolve_image_field(ex.get("image"), image_root=self.image_root)
