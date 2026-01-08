"""Dataset and collator smoke tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.data.adapters.jsonl import JsonlVQADataset
from src.data.collate import VLMDataCollator


class DummyTokenizer:
    def __call__(self, texts: str | List[str], **kwargs: Any) -> Dict[str, Any]:
        if isinstance(texts, str):
            texts = [texts]
        max_len = max(len(t.split()) for t in texts)
        input_ids = []
        attention_mask = []
        for text in texts:
            tokens = list(range(1, len(text.split()) + 1))
            pad_len = max_len - len(tokens)
            input_ids.append(tokens + [0] * pad_len)
            attention_mask.append([1] * len(tokens) + [0] * pad_len)
        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join([str(i) for i in ids])


def test_jsonl_dataset_and_collator(tmp_path: Path) -> None:
    data = [
        {"question": "What?", "answer": "Yes", "image": "missing.jpg", "id": "1"},
        {"question": "Where?", "answer": "Here", "id": "2"},
    ]
    jsonl_path = tmp_path / "data.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in data:
            handle.write(json.dumps(item) + "\n")

    dataset = JsonlVQADataset(jsonl_path)
    assert len(dataset) == 2

    collator = VLMDataCollator(
        tokenizer=DummyTokenizer(),
        prompt_template="Question: {question}\nAnswer:",
    )
    batch = collator([dataset[0], dataset[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
