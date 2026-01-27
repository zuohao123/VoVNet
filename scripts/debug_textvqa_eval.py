"""Debug TextVQA-style evaluation mismatches."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.collate import VLMDataCollator
from src.eval import matrix as eval_matrix
from src.eval.matrix_spec import EvalDatasetSpec, build_dataset, load_dataset_specs
from src.eval.metrics import _extract_answer_list, _normalize_vqa, vqa_accuracy_score


def _load_config(paths: Iterable[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def _unwrap_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise TypeError("Unsupported checkpoint format")


def _remap_base_model_keys(state: Dict[str, torch.Tensor], add_base_model: bool) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    needle = ".model.model."
    repl_add = ".model.base_model.model.model."
    repl_remove = ".model.model."
    for key, value in state.items():
        new_key = key
        if add_base_model:
            if repl_add not in new_key and needle in new_key:
                new_key = new_key.replace(needle, repl_add)
        else:
            if repl_add in new_key:
                new_key = new_key.replace(repl_add, repl_remove)
        out[new_key] = value
    return out


def _robust_load(raw_model: torch.nn.Module, checkpoint_path: str) -> Tuple[int, int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = _unwrap_state_dict(ckpt)

    candidates = [
        ("original", state),
        ("add_base_model", _remap_base_model_keys(state, add_base_model=True)),
        ("remove_base_model", _remap_base_model_keys(state, add_base_model=False)),
    ]

    best = None
    best_score = None
    for name, cand in candidates:
        missing, unexpected = raw_model.load_state_dict(cand, strict=False)
        score = len(missing) + len(unexpected)
        if best_score is None or score < best_score:
            best = (name, cand, missing, unexpected)
            best_score = score

    assert best is not None
    name, cand, missing, unexpected = best
    raw_model.load_state_dict(cand, strict=False)
    print(f"load_strategy={name} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("missing sample:", missing[:3])
    if unexpected:
        print("unexpected sample:", unexpected[:3])
    return len(missing), len(unexpected)


def _with_ids(collator: VLMDataCollator):
    def _fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [str(item.get("id", "")) for item in batch]
        out = collator(batch)
        out["ids"] = ids
        return out

    return _fn


def _max_similarity(pred_norm: str, answers_norm: List[str]) -> float:
    if not pred_norm or not answers_norm:
        return 0.0
    return max(SequenceMatcher(None, pred_norm, ans).ratio() for ans in answers_norm)


def _per_sample_stats(pred: str, ref: object) -> Dict[str, Any]:
    answers = _extract_answer_list(ref)
    pred_norm = _normalize_vqa(pred)
    answers_norm = [_normalize_vqa(a) for a in answers if a not in (None, "")]
    match = sum(1 for a in answers_norm if a == pred_norm)
    vqa_score = min(match / 3.0, 1.0)
    fuzzy = _max_similarity(pred_norm, answers_norm)
    return {
        "pred": pred,
        "pred_norm": pred_norm,
        "answers": answers[:10],
        "answers_norm": answers_norm[:10],
        "match_count": match,
        "vqa_score": vqa_score,
        "fuzzy_max": fuzzy,
    }


def _prepare_specs(specs: List[EvalDatasetSpec], max_samples: int | None) -> List[EvalDatasetSpec]:
    if max_samples is None:
        return specs
    prepared: List[EvalDatasetSpec] = []
    for spec in specs:
        spec_dict = asdict(spec)
        cur = spec_dict.get("max_samples")
        spec_dict["max_samples"] = min(cur, max_samples) if cur else max_samples
        prepared.append(EvalDatasetSpec(**spec_dict))
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug TextVQA evaluation mismatches")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Config YAML path (repeatable, applied in order)",
    )
    parser.add_argument(
        "--dataset_config",
        default="configs/eval_textvqa.yaml",
        help="Dataset config YAML (default: eval_textvqa.yaml)",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt path")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples to run")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--baseline",
        default="policy",
        choices=["policy", "always_full", "always_coarse", "no_vision"],
        help="Which action policy to evaluate",
    )
    parser.add_argument(
        "--cost_weight",
        type=float,
        default=None,
        help="Override lambda_cost during eval (default: cfg.training.lambda_cost)",
    )
    parser.add_argument("--top_k", type=int, default=15, help="Number of near-miss examples")
    parser.add_argument(
        "--near_threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for near misses",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON report",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg.data.max_samples = args.max_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = eval_matrix.build_model(cfg).to(device)
    raw_model = getattr(model, "module", model)
    raw_model.eval()
    _robust_load(raw_model, args.checkpoint)

    specs = load_dataset_specs(args.dataset_config, cfg)
    specs = _prepare_specs(specs, args.max_samples)

    tokenizer = raw_model.base_vlm.tokenizer
    max_length = getattr(cfg.data, "max_length", None)
    if max_length is None:
        max_length = getattr(cfg.training, "max_length", None)
    collator = VLMDataCollator(tokenizer, cfg.data.prompt_template, max_length)

    cost_weight = args.cost_weight if args.cost_weight is not None else cfg.policy.lambda_cost
    print(f"cost_weight={cost_weight}")

    report: Dict[str, Any] = {"checkpoint": args.checkpoint, "datasets": {}}

    for spec in specs:
        dataset = build_dataset(spec)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_with_ids(collator),
        )

        action_counts: Counter[str] = Counter()
        vqa_scores: List[float] = []
        fuzzy_scores: List[float] = []
        near_misses: List[Dict[str, Any]] = []
        all_preds: List[str] = []
        all_refs: List[object] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                batch = eval_matrix._move_batch_to_device(batch, device)
                if args.baseline == "policy":
                    outputs = eval_matrix._forward_with_cost(raw_model, batch, cost_weight)
                else:
                    outputs = eval_matrix._forward_with_baseline(raw_model, batch, args.baseline)

                labels = outputs.get("labels") if outputs.get("labels") is not None else batch["labels"]
                preds = eval_matrix._decode_from_logits(outputs["logits"], labels, tokenizer)
                answers = batch.get("answers", [])
                ids = batch.get("ids", [])
                actions = outputs["actions"].detach().cpu().tolist()
                all_preds.extend(preds)
                all_refs.extend(answers)

                for i, (pred, ref) in enumerate(zip(preds, answers)):
                    sid = ids[i] if i < len(ids) else f"{batch_idx}:{i}"
                    action_name = eval_matrix.Action(actions[i]).name if i < len(actions) else "UNK"
                    action_counts[action_name] += 1
                    stats = _per_sample_stats(pred, ref)
                    stats["id"] = sid
                    stats["action"] = action_name
                    vqa_scores.append(float(stats["vqa_score"]))
                    fuzzy_scores.append(float(stats["fuzzy_max"]))
                    if stats["vqa_score"] == 0.0 and stats["fuzzy_max"] >= args.near_threshold:
                        near_misses.append(stats)

        near_misses.sort(key=lambda x: x.get("fuzzy_max", 0.0), reverse=True)
        official_direct = vqa_accuracy_score(all_preds, all_refs)
        near_miss_fuzzy_mean = (
            sum(m.get("fuzzy_max", 0.0) for m in near_misses) / max(1, len(near_misses))
        )

        summary = {
            "dataset": spec.name,
            "metric": spec.metric,
            "baseline": args.baseline,
            "num_samples": len(vqa_scores),
            "vqa_accuracy_mean": sum(vqa_scores) / max(1, len(vqa_scores)),
            "vqa_accuracy_official": official_direct,
            "fuzzy_mean": sum(fuzzy_scores) / max(1, len(fuzzy_scores)),
            "fuzzy_ge_0p5": sum(1 for s in fuzzy_scores if s >= 0.5) / max(1, len(fuzzy_scores)),
            "fuzzy_ge_0p7": sum(1 for s in fuzzy_scores if s >= 0.7) / max(1, len(fuzzy_scores)),
            "fuzzy_ge_0p85": sum(1 for s in fuzzy_scores if s >= 0.85) / max(1, len(fuzzy_scores)),
            "action_counts": dict(action_counts),
            "near_miss_count": len(near_misses),
            "near_miss_fuzzy_mean": near_miss_fuzzy_mean,
            "near_miss_examples": near_misses[: args.top_k],
        }

        report["datasets"][spec.name] = summary

        print("\n=== TextVQA Debug Summary ===")
        for k, v in summary.items():
            if k == "near_miss_examples":
                continue
            print(f"{k}: {v}")
        if summary["near_miss_examples"]:
            print("\n--- Near misses (top) ---")
            for item in summary["near_miss_examples"]:
                ans_preview = item.get("answers", [])[:5]
                print(
                    f"id={item['id']} action={item['action']} "
                    f"vqa={item['vqa_score']:.3f} fuzzy={item['fuzzy_max']:.3f} "
                    f"pred={item['pred']!r} answers={ans_preview!r}"
                )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"\nwrote report to {out_path}")


if __name__ == "__main__":
    main()
