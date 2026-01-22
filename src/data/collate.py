"""Data collators for VLM training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
from transformers import PreTrainedTokenizerBase

_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _blank_image() -> Image.Image:
    return Image.new("RGB", (1, 1), color=0)


def _get_image_token(tokenizer: PreTrainedTokenizerBase) -> Optional[str]:
    if hasattr(tokenizer, "image_token"):
        return getattr(tokenizer, "image_token")
    if hasattr(tokenizer, "image_token_id"):
        token = tokenizer.convert_ids_to_tokens(getattr(tokenizer, "image_token_id"))
        if token is not None:
            return token
    tokens = getattr(tokenizer, "additional_special_tokens", None) or []
    for token in tokens:
        if "image" in token.lower():
            return token
    return None


def _format_choices(choices: Any) -> str:
    if not choices or not isinstance(choices, list):
        return ""
    lines: List[str] = []
    for idx, item in enumerate(choices):
        if idx >= len(_CHOICE_LETTERS):
            break
        text = str(item).strip()
        if not text:
            continue
        lines.append(f"{_CHOICE_LETTERS[idx]}. {text}")
    return "\n".join(lines)


def _append_before_answer(prompt: str, addition: str) -> str:
    if not addition:
        return prompt
    marker = "Answer:"
    if marker in prompt:
        return prompt.replace(marker, f"{addition}\n{marker}", 1)
    return f"{prompt}\n{addition}"


@dataclass
class VLMDataCollator:
    """Collator that builds LM-style inputs with optional images."""

    tokenizer: PreTrainedTokenizerBase
    prompt_template: str
    max_length: Optional[int] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        questions = [item.get("question", "") for item in batch]
        answers = [item.get("answer", "") for item in batch]
        image_token = _get_image_token(self.tokenizer)
        prompts = []
        for item, q in zip(batch, questions):
            context = item.get("context") or ""
            choices_text = _format_choices(item.get("choices"))
            prompt = self.prompt_template.format(
                question=q,
                context=context,
                choices=choices_text,
            )
            if context and "{context}" not in self.prompt_template:
                prompt = _append_before_answer(prompt, f"Context: {context}")
            if choices_text and "{choices}" not in self.prompt_template:
                prompt = _append_before_answer(prompt, f"Options:\n{choices_text}")
            if image_token and image_token not in prompt:
                prompt = f"{image_token}\n{prompt}".strip()
            prompts.append(prompt)
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
