"""Adapter for OpenOrca (text-only instruction dataset)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from ..common import AnswerInfo, HFAdapterBase, MetaInfo, UnifiedExample


class OpenOrcaAdapter(HFAdapterBase):
    """OpenOrca adapter (text-only).

    Notes: Common HF dataset id is "Open-Orca/OpenOrca".
    """

    name = "open_orca"
    hf_dataset_id = "Open-Orca/OpenOrca"
    task_type = "text_only"

    def normalize_example(self, ex: Dict[str, Any], split: str) -> UnifiedExample:
        question = _first_non_empty(ex, ["question", "prompt", "instruction", "input"])
        answer = _first_non_empty(ex, ["response", "output", "completion", "answer"])
        context = _first_non_empty(ex, ["system_prompt", "context"]) or None

        sample_id = str(ex.get("id") or ex.get("uid") or "")
        if not sample_id:
            sample_id = str(hash(question))

        meta = MetaInfo(
            language=None,
            source_fields={
                "question": "question|prompt|instruction|input",
                "answer": "response|output|completion|answer",
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
            context=context,
            choices=None,
            answer=AnswerInfo(text=answer or None, aliases=None, label=None, raw=answer),
            meta=meta,
        )

    def get_image(self, ex: Dict[str, Any]) -> Optional[Image.Image]:
        return None


def _first_non_empty(ex: Dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = ex.get(key)
        if value:
            return str(value)
    return ""
