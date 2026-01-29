"""Adapter for LLaVA-Instruct-150K (HuggingFace datasets)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from ..common import AnswerInfo, HFAdapterBase, MetaInfo, UnifiedExample, resolve_image_field


class LLaVAInstructAdapter(HFAdapterBase):
    """LLaVA-Instruct-150K adapter."""

    name = "llava_instruct"
    hf_dataset_id = "liuhaotian/LLaVA-Instruct-150K"
    task_type = "instruction_vqa"

    def load(self, subset: Optional[str], split: str, streaming: bool = False) -> Any:
        # Reuse HFAdapterBase default behavior.
        return super().load(subset=subset, split=split, streaming=streaming)

    def _extract_pair(self, ex: Dict[str, Any]) -> tuple[str, str]:
        conversations = ex.get("conversations") or ex.get("messages") or []
        last_user = None
        for msg in conversations:
            role = (msg.get("from") or msg.get("role") or msg.get("speaker") or "").lower()
            text = str(msg.get("value") or msg.get("content") or msg.get("text") or "").strip()
            if not text:
                continue
            if role in {"human", "user", "instruction"}:
                last_user = text
            elif role in {"assistant", "gpt", "bot"} and last_user:
                return last_user, text
        question = str(ex.get("question") or ex.get("prompt") or "").strip()
        answer = str(ex.get("answer") or ex.get("output") or "").strip()
        return question, answer

    def normalize_example(self, ex: Dict[str, Any], split: str) -> UnifiedExample:
        question, answer = self._extract_pair(ex)
        sample_id = str(ex.get("id") or ex.get("image_id") or ex.get("sample_id") or "")
        if not sample_id:
            sample_id = str(hash(question))
        meta = MetaInfo(
            language=None,
            source_fields={"question": "conversations", "answers": "conversations", "image": "image"},
            extra={"source_id": ex.get("id")},
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
            answer=AnswerInfo(text=answer, aliases=None, label=None, raw=[answer]),
            meta=meta,
        )

    def get_image(self, ex: Dict[str, Any]) -> Optional[Image.Image]:
        return resolve_image_field(ex.get("image") or ex.get("image_path"))
