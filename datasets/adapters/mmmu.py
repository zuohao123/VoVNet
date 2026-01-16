"""Adapter for MMMU (local files or HF override via env)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from ..common import AnswerInfo, MetaInfo, UnifiedExample
from .local_utils import (
    coerce_str,
    extract_letter_choices,
    first_non_empty,
    load_dataset_with_fallback,
    map_answer_to_choice,
    parse_choice_list,
    resolve_dataset_root,
    resolve_image_from_record,
    resolve_image_root,
)


class MMMUAdapter:
    """MMMU adapter."""

    name = "mmmu"
    env_prefix = "MMMU"
    task_type = "multi_choice_vqa"

    def __init__(self) -> None:
        root, _ = resolve_dataset_root(self.name, self.env_prefix)
        self.image_root = resolve_image_root(root, self.env_prefix)

    def load(self, subset: Optional[str], split: str, streaming: bool = False) -> Any:
        return load_dataset_with_fallback(
            name=self.name,
            env_prefix=self.env_prefix,
            split=split,
            subset=subset,
            streaming=streaming,
        )

    def normalize_example(self, ex: Dict[str, Any], split: str) -> UnifiedExample:
        question = coerce_str(
            first_non_empty(ex, ["question", "query", "prompt", "instruction", "text"])
        )
        context = first_non_empty(ex, ["context", "hint", "explanation"])
        context = coerce_str(context) if context is not None else None

        choices = parse_choice_list(first_non_empty(ex, ["choices", "options", "candidates"]))
        if not choices:
            choices = extract_letter_choices(ex, list("ABCDEFGH"))
        if not choices:
            for prefix in ("option", "choice"):
                candidates = [
                    ex.get(f"{prefix}{idx}") for idx in range(1, 9) if ex.get(f"{prefix}{idx}")
                ]
                if candidates:
                    choices = [str(item) for item in candidates]
                    break

        answer_raw = first_non_empty(ex, ["answer", "label", "gt_answer", "correct"])
        answer_text, answer_label = map_answer_to_choice(answer_raw, choices)

        sample_id = coerce_str(first_non_empty(ex, ["id", "question_id", "uid", "index"]))
        if not sample_id:
            sample_id = str(hash(question))

        meta_extra = {}
        for key in ("subject", "subfield", "topic", "difficulty", "source", "image"):
            if key in ex:
                meta_extra[key] = ex.get(key)

        meta = MetaInfo(
            language=coerce_str(ex.get("language")) or None,
            source_fields={
                "question": "question|query|prompt|instruction|text",
                "choices": "choices|options|A..H",
                "answer": "answer|label",
                "image": "image",
            },
            extra=meta_extra,
        )

        return UnifiedExample(
            sample_id=sample_id,
            dataset=self.name,
            split=split,
            task_type=self.task_type,
            image=None,
            question=question,
            context=context,
            choices=choices,
            answer=AnswerInfo(
                text=answer_text,
                aliases=None,
                label=answer_label,
                raw=answer_raw,
            ),
            meta=meta,
        )

    def get_image(self, ex: Dict[str, Any]) -> Optional[Image.Image]:
        return resolve_image_from_record(
            ex,
            fields=[
                "image",
                "image_path",
                "image_file",
                "image_id",
                "img",
                "img_path",
                "image_url",
            ],
            image_root=self.image_root,
        )
