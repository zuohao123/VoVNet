"""Sweep uncertainty thresholds and write pareto_threshold.csv."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from src.config.config import Config
from src.data.collate import VLMDataCollator
from src.eval.matrix import build_model, evaluate_dataset, rows_from_results
from src.eval.matrix_spec import load_dataset_specs, build_dataset, get_metric_fn
from src.utils.io import write_csv, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pareto sweep for uncertainty-threshold baseline"
    )
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--uncertainty", type=str, default=None)
    parser.add_argument("--vision", type=str, default=None)
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    cfg.policy.baseline_name = "uncertainty_threshold"
    if args.uncertainty:
        cfg.policy.baseline_uncertainty = args.uncertainty.strip().lower()
    if args.vision:
        cfg.policy.baseline_vision = args.vision.strip().lower()

    output_dir = Path(args.output_dir)
    thresholds = args.thresholds

    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)
    model = build_model(cfg)
    if model.base_vlm.tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded; check model name")
    model = accelerator.prepare(model)
    if args.checkpoint:
        accelerator.load_state(args.checkpoint)

    specs = load_dataset_specs(args.dataset_config, cfg)
    all_rows = []
    results_by_dataset = {}

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
        loader = accelerator.prepare(loader)
        metric_fn = get_metric_fn(spec.metric)
        results = []
        for threshold in thresholds:
            metrics = evaluate_dataset(
                model=model,
                loader=loader,
                metric_fn=metric_fn,
                cost_weight=None,
                profile=cfg.eval.profile,
                baseline_name=cfg.policy.baseline_name,
                baseline_threshold=threshold,
                baseline_uncertainty=cfg.policy.baseline_uncertainty,
                baseline_vision=cfg.policy.baseline_vision,
                baseline_seed=cfg.policy.baseline_seed or cfg.training.seed,
                baseline_target_ratios=cfg.policy.baseline_target_ratios,
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
            metrics["threshold"] = threshold
            results.append(metrics)
        results_by_dataset[spec.name] = {"metric": spec.metric, "results": results}
        all_rows.extend(rows_from_results(spec.name, spec.metric, results))

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "pareto_threshold.csv", all_rows)
        write_json(output_dir / "pareto_threshold.json", results_by_dataset)
        logger.info("Saved pareto sweep to %s", output_dir / "pareto_threshold.csv")


if __name__ == "__main__":
    main()
