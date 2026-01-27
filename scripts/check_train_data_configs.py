"""Quick sanity checks for training/eval JSONL dataset configs.

This script is intentionally lightweight: it does not load a model/tokenizer.
It checks file existence and streams a subset of samples to verify that
prompts, answers, and (optionally) image paths look consistent.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from src.config.config import Config
from src.data.collate import _clean_choices, _coerce_answer_text


@dataclass
class DatasetStats:
    n: int = 0
    prompt: int = 0
    choices: int = 0
    answer_text: int = 0
    answers_list: int = 0
    answer_types: Counter[str] | None = None
    image_types: Counter[str] | None = None
    image_missing: int = 0
    image_exists: int = 0

    def __post_init__(self) -> None:
        if self.answer_types is None:
            self.answer_types = Counter()
        if self.image_types is None:
            self.image_types = Counter()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _is_nonempty(x: Any) -> bool:
    return x not in (None, "", [], {})


def _answer_text_from_item(item: Dict[str, Any]) -> str:
    choices = _clean_choices(item.get("choices"))
    answer = item.get("answer")
    if answer in (None, "", []):
        answer = item.get("answer_info")
    if answer in (None, "", []):
        answer = item.get("label")
    return _coerce_answer_text(answer, choices)


def analyze_jsonl(
    path: Path,
    *,
    max_samples: int,
    check_images: bool,
    image_check_samples: int,
) -> Dict[str, Any]:
    stats_by_dataset: dict[str, DatasetStats] = defaultdict(DatasetStats)

    total = 0
    image_checked = 0

    for item in _iter_jsonl(path):
        total += 1
        if total > max_samples:
            break

        dataset = str(item.get("dataset") or "unknown")
        st = stats_by_dataset[dataset]
        st.n += 1

        if _is_nonempty(item.get("prompt_template")):
            st.prompt += 1

        choices = _clean_choices(item.get("choices"))
        if choices:
            st.choices += 1

        ans = item.get("answer")
        if isinstance(ans, dict) and isinstance(ans.get("answers"), list) and ans.get("answers"):
            st.answers_list += 1

        ans_text = _answer_text_from_item(item)
        if ans_text.strip():
            st.answer_text += 1

        ans_type = type(ans).__name__
        st.answer_types[ans_type] += 1

        image_value = item.get("image")
        st.image_types[type(image_value).__name__] += 1

        if check_images and image_checked < image_check_samples:
            image_checked += 1
            if isinstance(image_value, str) and image_value:
                if Path(image_value).exists():
                    st.image_exists += 1
                else:
                    st.image_missing += 1

    # Aggregate summary metrics.
    summary: Dict[str, Any] = {
        "path": str(path),
        "processed": total if total < max_samples else max_samples,
        "datasets": {},
        "mc_label_valid_ratio": 0.0,
        "open_answer_valid_ratio": 0.0,
    }

    mc_total = 0
    mc_valid = 0
    open_total = 0
    open_valid = 0

    for name, st in sorted(stats_by_dataset.items(), key=lambda kv: kv[1].n, reverse=True):
        prompt_pct = st.prompt / max(1, st.n)
        choices_pct = st.choices / max(1, st.n)
        answer_pct = st.answer_text / max(1, st.n)
        top_ans_types = st.answer_types.most_common(3)
        image_types = st.image_types.most_common(3)

        summary["datasets"][name] = {
            "n": st.n,
            "prompt_pct": prompt_pct,
            "choices_pct": choices_pct,
            "answer_text_pct": answer_pct,
            "answers_list_pct": st.answers_list / max(1, st.n),
            "answer_types_top3": top_ans_types,
            "image_types_top3": image_types,
            "image_exists": st.image_exists,
            "image_missing": st.image_missing,
        }

        mc_total += st.choices
        mc_valid += min(st.choices, st.answer_text)
        open_total += st.n - st.choices
        open_valid += max(0, st.answer_text - min(st.choices, st.answer_text))

    summary["mc_label_valid_ratio"] = mc_valid / max(1, mc_total)
    summary["open_answer_valid_ratio"] = open_valid / max(1, open_total)

    return summary


def _print_summary(tag: str, summary: Dict[str, Any]) -> None:
    print(f"\n== {tag} ==")
    print("jsonl:", summary["path"])
    print("processed:", summary["processed"])
    print(
        "mc_label_valid_ratio:",
        f"{summary['mc_label_valid_ratio']:.4f}",
        "open_answer_valid_ratio:",
        f"{summary['open_answer_valid_ratio']:.4f}",
    )
    print("-- per dataset --")
    for name, info in summary["datasets"].items():
        print(
            f"{name:12s} n={info['n']:6d} "
            f"prompt%={info['prompt_pct']:.3f} "
            f"choices%={info['choices_pct']:.3f} "
            f"answer%={info['answer_text_pct']:.3f} "
            f"answers_list%={info['answers_list_pct']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check train/eval JSONL configs.")
    parser.add_argument(
        "--config",
        action="append",
        default=[
            "configs/base.yaml",
            "configs/train_mmbench_llava_textvqa_vovnet_v3_2k5k.yaml",
        ],
        help="YAML config(s) to load in order.",
    )
    parser.add_argument("--train_jsonl", type=str, default=None, help="Override train_jsonl.")
    parser.add_argument("--eval_jsonl", type=str, default=None, help="Override eval_jsonl.")
    parser.add_argument("--max_samples", type=int, default=5000, help="Samples to scan per file.")
    parser.add_argument(
        "--check_images",
        action="store_true",
        help="Check a small number of image paths for existence.",
    )
    parser.add_argument(
        "--image_check_samples",
        type=int,
        default=200,
        help="Max image paths to check when --check_images is set.",
    )
    args = parser.parse_args()

    cfg = Config()
    for path in args.config:
        cfg.update_from_yaml(path)

    train_path = Path(args.train_jsonl or (cfg.data.train_jsonl or ""))
    eval_path = Path(args.eval_jsonl or (cfg.data.eval_jsonl or ""))

    if not train_path.exists():
        raise SystemExit(f"train_jsonl not found: {train_path}")
    if eval_path and str(eval_path) and not eval_path.exists():
        print(f"warning: eval_jsonl not found: {eval_path}")

    train_summary = analyze_jsonl(
        train_path,
        max_samples=args.max_samples,
        check_images=args.check_images,
        image_check_samples=args.image_check_samples,
    )
    _print_summary("TRAIN", train_summary)

    if eval_path.exists():
        eval_summary = analyze_jsonl(
            eval_path,
            max_samples=min(2000, args.max_samples),
            check_images=False,
            image_check_samples=0,
        )
        _print_summary("EVAL", eval_summary)

    # Light heuristics for common pitfalls.
    mmbench = train_summary["datasets"].get("mmbench")
    if mmbench and mmbench["choices_pct"] < 0.9:
        print("warning: mmbench choices% is unexpectedly low.")
    textvqa = train_summary["datasets"].get("textvqa")
    if textvqa and textvqa["answers_list_pct"] < 0.5:
        print("warning: textvqa answers_list% is unexpectedly low.")


if __name__ == "__main__":
    main()

