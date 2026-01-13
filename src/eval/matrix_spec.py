"""Dataset specification utilities for evaluation matrix."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import torch
import yaml

from src.config.config import Config
from src.data.adapters.hf_dataset import HFDatasetAdapter
from src.data.adapters.jsonl import JsonlVQADataset
from src.eval.metrics import exact_match_score


@dataclass
class EvalDatasetSpec:
    """Configuration for a single evaluation dataset."""

    name: str
    source: str = "jsonl"
    jsonl: Optional[str] = None
    hf_name: Optional[str] = None
    subset: Optional[str] = None
    split: str = "validation"
    prompt_template: Optional[str] = None
    metric: str = "exact_match"
    text_field: str = "question"
    answer_field: str = "answer"
    image_field: str = "image"
    max_samples: Optional[int] = None


def get_metric_fn(name: str) -> Callable[[Iterable[str], Iterable[str]], float]:
    name = name.lower()
    if name in {"exact_match", "em", "accuracy"}:
        return exact_match_score
    raise ValueError(f"Unknown metric: {name}")


def load_dataset_specs(path: str | None, cfg: Config) -> List[EvalDatasetSpec]:
    if path is None:
        if cfg.data.eval_jsonl:
            return [
                EvalDatasetSpec(
                    name="eval_jsonl",
                    source="jsonl",
                    jsonl=cfg.data.eval_jsonl,
                    prompt_template=cfg.data.prompt_template,
                    text_field=cfg.data.text_field,
                    answer_field=cfg.data.answer_field,
                    image_field=cfg.data.image_field,
                    max_samples=cfg.data.max_samples,
                )
            ]
        if cfg.data.hf_dataset_name:
            return [
                EvalDatasetSpec(
                    name=cfg.data.hf_dataset_name,
                    source="hf",
                    hf_name=cfg.data.hf_dataset_name,
                    split=cfg.data.hf_dataset_split,
                    prompt_template=cfg.data.prompt_template,
                    text_field=cfg.data.text_field,
                    answer_field=cfg.data.answer_field,
                    image_field=cfg.data.image_field,
                    max_samples=cfg.data.max_samples,
                )
            ]
        raise ValueError("No eval dataset configured")

    data = yaml.safe_load(Path(path).read_text()) or {}
    defaults = data.get("defaults", {})
    datasets = data.get("datasets", [])
    specs: List[EvalDatasetSpec] = []
    for item in datasets:
        merged = {**defaults, **(item or {})}
        specs.append(EvalDatasetSpec(**merged))
    if not specs:
        raise ValueError("dataset_config contains no datasets")
    return specs


def build_dataset(spec: EvalDatasetSpec) -> torch.utils.data.Dataset:
    if spec.source == "jsonl":
        if not spec.jsonl:
            raise ValueError(f"{spec.name} requires jsonl path")
        return JsonlVQADataset(
            spec.jsonl,
            text_field=spec.text_field,
            answer_field=spec.answer_field,
            image_field=spec.image_field,
            max_samples=spec.max_samples,
        )
    if spec.source == "hf":
        if not spec.hf_name:
            raise ValueError(f"{spec.name} requires hf_name")
        return HFDatasetAdapter(
            spec.hf_name,
            split=spec.split,
            subset=spec.subset,
            text_field=spec.text_field,
            answer_field=spec.answer_field,
            image_field=spec.image_field,
            max_samples=spec.max_samples,
        )
    raise ValueError(f"Unknown dataset source: {spec.source}")
