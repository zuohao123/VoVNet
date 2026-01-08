"""Data collators for VLM training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
from transformers import PreTrainedTokenizerBase


def _blank_image() -> Image.Image:
    return Image.new("RGB", (1, 1), color=0)


@dataclass
class VLMDataCollator:
    """Collator that builds LM-style inputs with optional images."""

    tokenizer: PreTrainedTokenizerBase
    prompt_template: str
    max_length: Optional[int] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        questions = [item.get("question", "") for item in batch]
        answers = [item.get("answer", "") for item in batch]
        prompts = [self.prompt_template.format(question=q) for q in questions]
        full_texts = [f"{p} {a}".strip() for p, a in zip(prompts, answers)]

        encoded = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        labels = torch.full_like(input_ids, fill_value=-100)
        for i, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )["input_ids"]
            cut = min(len(prompt_ids), input_ids.shape[1])
            labels[i, cut:] = input_ids[i, cut:]
        labels = labels.masked_fill(attention_mask == 0, -100)

        images = [item.get("image") for item in batch]
        images = [img if img is not None else _blank_image() for img in images]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images,
            "questions": questions,
            "answers": answers,
            "meta": [item.get("meta", {}) for item in batch],
        }
