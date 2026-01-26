"""Evaluate policy vs baselines on a dataset config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.collate import VLMDataCollator
from src.eval.matrix import build_model, evaluate_dataset, load_eval_checkpoint
from src.eval.matrix_spec import build_dataset, get_metric_fn, load_dataset_specs
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare policy/baselines accuracy.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def _baseline_target_ratios(cfg: Config) -> List[float]:
    ratios = cfg.policy.baseline_target_ratios
    if ratios:
        return [float(x) for x in ratios]
    return [0.33, 0.33, 0.34]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    if args.batch_size is not None:
        cfg.eval.batch_size = args.batch_size

    specs = load_dataset_specs(args.dataset_config, cfg)
    if args.max_samples is not None:
        for spec in specs:
            spec.max_samples = args.max_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    load_eval_checkpoint(model, args.checkpoint)
    model.to(device)

    strategies = [
        ("policy", None),
        ("no_vision", "no_vision"),
        ("always_coarse", "always_coarse"),
        ("always_full", "always_full"),
        ("random_policy_matched", "random_policy_matched"),
    ]

    results: List[Dict[str, Any]] = []
    for spec in specs:
        dataset = build_dataset(spec)
        collator = VLMDataCollator(
            tokenizer=model.base_vlm.tokenizer,
            prompt_template=spec.prompt_template or cfg.data.prompt_template,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        metric_fn = get_metric_fn(spec.metric)
        for name, baseline_name in strategies:
            ratios = _baseline_target_ratios(cfg) if baseline_name == "random_policy_matched" else None
            metrics = evaluate_dataset(
                model=model,
                loader=loader,
                metric_fn=metric_fn,
                cost_weight=None,
                profile=False,
                dataset_name=spec.name,
                log_pred_interval=0,
                log_pred_examples=1,
                log_pred_max_chars=200,
                baseline_name=baseline_name,
                baseline_threshold=cfg.policy.baseline_threshold,
                baseline_uncertainty=cfg.policy.baseline_uncertainty,
                baseline_vision=cfg.policy.baseline_vision,
                baseline_seed=cfg.policy.baseline_seed or cfg.training.seed,
                baseline_target_ratios=ratios,
                baseline_bucket_ratios=cfg.policy.baseline_bucket_ratios,
                baseline_bucket_thresholds=cfg.policy.baseline_bucket_thresholds,
                baseline_pruning_ratio=cfg.policy.baseline_pruning_ratio,
                baseline_pruning_mode=cfg.policy.baseline_pruning_mode,
                baseline_merge_ratio=cfg.policy.baseline_merge_ratio,
                baseline_merge_mode=cfg.policy.baseline_merge_mode,
                baseline_merge_weight=cfg.policy.baseline_merge_weight,
                baseline_enable_prune=cfg.policy.baseline_enable_prune,
                baseline_prune_ratio=cfg.policy.baseline_prune_ratio,
                baseline_prune_mode=cfg.policy.baseline_prune_mode,
                baseline_pool_factor=cfg.policy.baseline_pool_factor,
            )
            results.append(
                {
                    "dataset": spec.name,
                    "metric": spec.metric,
                    "strategy": name,
                    "baseline_name": baseline_name,
                    "accuracy": metrics.get("accuracy"),
                    "avg_cost": metrics.get("avg_cost"),
                    "action_ratio": metrics.get("action_ratio"),
                }
            )

    print(json.dumps({"results": results}, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
