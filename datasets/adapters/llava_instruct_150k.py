"""Adapter for LLaVA-Instruct-150K (HuggingFace datasets)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from ..common import AnswerInfo, HFAdapterBase, MetaInfo, UnifiedExample, resolve_image_field


class LLaVAInstructAdapter(HFAdapterBase):
    """LLaVA-Instruct-150K adapter.

    Notes: Common HF dataset id is "liuhaotian/llava-instruct-150k".
    If the dataset is gated or unavailable, use the official release.
    """

    name = "llava_instruct_150k"
    hf_dataset_id = "liuhaotian/llava-instruct-150k"
    task_type = "instruct_vqa"

    def normalize_example(self, ex: Dict[str, Any], split: str) -> UnifiedExample:
        conversations = ex.get("conversations") or ex.get("messages") or []
        question = _extract_first(conversations, roles=("human", "user"))
        answer = _extract_first(conversations, roles=("gpt", "assistant"))

        sample_id = str(ex.get("id") or ex.get("image") or ex.get("image_id") or "")
        if not sample_id:
            sample_id = str(hash(question))

        meta = MetaInfo(
            language=None,
            source_fields={
                "conversations": "conversations",
                "image": "image",
            },
            extra={"conversations": conversations},
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
            answer=AnswerInfo(text=answer or None, aliases=None, label=None, raw=conversations),
            meta=meta,
        )

    def get_image(self, ex: Dict[str, Any]) -> Optional[Image.Image]:
        return resolve_image_field(ex.get("image"))


def _extract_first(conversations: Any, roles: tuple[str, ...]) -> str:
    if isinstance(conversations, list):
        for item in conversations:
            if not isinstance(item, dict):
                continue
            if item.get("from") in roles or item.get("role") in roles:
                return str(item.get("value") or item.get("content") or "")
    return ""
