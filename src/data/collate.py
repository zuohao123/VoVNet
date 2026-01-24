"""Data collators for VLM training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
from transformers import PreTrainedTokenizerBase

_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _clean_choices(choices: Any) -> List[str]:
    if not choices or not isinstance(choices, list):
        return []
    out: List[str] = []
    for item in choices:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text.lower() == "nan":
            continue
        out.append(text)
    return out


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
    choices = _clean_choices(choices)
    if not choices:
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


def _label_to_letter(label: Any) -> Optional[str]:
    if label is None:
        return None
    if isinstance(label, int):
        if 0 <= label < len(_CHOICE_LETTERS):
            return _CHOICE_LETTERS[label]
        if 1 <= label <= len(_CHOICE_LETTERS):
            return _CHOICE_LETTERS[label - 1]
        return None
    label_str = str(label).strip()
    if label_str.isdigit():
        idx = int(label_str)
        if 0 <= idx < len(_CHOICE_LETTERS):
            return _CHOICE_LETTERS[idx]
        if 1 <= idx <= len(_CHOICE_LETTERS):
            return _CHOICE_LETTERS[idx - 1]
    if len(label_str) == 1 and label_str.upper() in _CHOICE_LETTERS:
        return label_str.upper()
    return None


def _answer_to_letter(answer: Any, choices: List[str]) -> Optional[str]:
    if isinstance(answer, dict):
        letter = _label_to_letter(answer.get("label"))
        if letter:
            return letter
        raw = answer.get("raw")
        if isinstance(raw, str):
            raw_letter = raw.strip().upper()
            if len(raw_letter) == 1 and raw_letter in _CHOICE_LETTERS:
                return raw_letter
        text = answer.get("text") or answer.get("answer") or raw
        answer = text
    if isinstance(answer, list):
        answer = next((item for item in answer if isinstance(item, str)), "")
    if isinstance(answer, str):
        ans = answer.strip()
        if len(ans) == 1 and ans.upper() in _CHOICE_LETTERS:
            return ans.upper()
        for idx, choice in enumerate(choices):
            if ans.lower() == choice.lower():
                return _CHOICE_LETTERS[idx]
    return None


def _coerce_answer_text(answer: Any, choices: List[str]) -> str:
    if choices:
        letter = _answer_to_letter(answer, choices)
        if letter:
            return letter
    if isinstance(answer, dict):
        for key in ("text", "answer", "raw", "label"):
            value = answer.get(key)
            if value not in (None, ""):
                return str(value)
        return ""
    if isinstance(answer, list):
        for item in answer:
            if item not in (None, ""):
                return str(item)
        return ""
    return "" if answer is None else str(answer)


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
    use_sample_prompt: bool = True

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        questions = [item.get("question", "") for item in batch]
        cleaned_choices = [_clean_choices(item.get("choices")) for item in batch]
        answer_texts = [
            _coerce_answer_text(item.get("answer", ""), choices)
            for item, choices in zip(batch, cleaned_choices)
        ]
        answer_refs = []
        for item, answer_text, choices in zip(batch, answer_texts, cleaned_choices):
            ref = item.get("answer_info")
            if ref is None and isinstance(item.get("answer"), dict):
                ref = item.get("answer")
            if isinstance(ref, dict):
                merged = dict(ref)
                if choices and "choices" not in merged:
                    merged["choices"] = choices
                if "text" not in merged and answer_text not in (None, ""):
                    merged["text"] = answer_text
                ref = merged
            elif ref is None:
                ref = answer_text
            answer_refs.append(ref)
        image_token = _get_image_token(self.tokenizer)
        prompts = []
        for item, q, choices in zip(batch, questions, cleaned_choices):
            context = item.get("context") or ""
            choices_text = _format_choices(choices)
            prompt_tpl = (
                item.get("prompt_template")
                if self.use_sample_prompt and item.get("prompt_template")
                else self.prompt_template
            )
            prompt = prompt_tpl.format(
                question=q,
                context=context,
                choices=choices_text,
            )
            if choices_text and "answer:" in prompt.lower() and "letter" not in prompt.lower():
                prompt = prompt.replace("Answer:", "Answer (letter only):", 1)
            if context and "{context}" not in self.prompt_template:
                prompt = _append_before_answer(prompt, f"Context: {context}")
            if choices_text and "{choices}" not in self.prompt_template:
                prompt = _append_before_answer(prompt, f"Options:\n{choices_text}")
            if image_token and image_token not in prompt:
                prompt = f"{image_token}\n{prompt}".strip()
            prompts.append(prompt)
        full_texts = [f"{p} {a}".strip() for p, a in zip(prompts, answer_texts)]

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
            "answers": answer_refs,
            "meta": [item.get("meta", {}) for item in batch],
        }
