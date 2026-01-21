"""Sweep resolution budgets for resolution-scaling baseline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

from src.config.config import Config
from src.data.collate import VLMDataCollator
from src.eval.matrix import build_model, evaluate_dataset, load_eval_checkpoint, rows_from_results
from src.eval.matrix_spec import load_dataset_specs, build_dataset, get_metric_fn
from src.utils.io import write_csv, write_json
from src.utils.logging import setup_logging
from src.utils.run_metadata import collect_dataset_metadata, write_run_metadata
from src.utils.seed import set_seed

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pareto sweep for resolution scaling baseline"
    )
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default="outputs/baselines/resolution_scaling"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--baseline_name", type=str, default="resolution_scaling")
    parser.add_argument(
        "--vision_budget_list",
        type=str,
        required=True,
        help='Comma list or JSON list, e.g. "224,336,448" or "[224,336,448]"',
    )
    parser.add_argument(
        "--budget_mode",
        type=str,
        default="long_side",
        choices=["long_side", "max_pixels"],
    )
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def parse_budget_list(raw: str) -> List[int]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        values = json.loads(raw)
    else:
        values = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(float(item)) for item in values]


def apply_budget(
    model: object, budget_value: int, mode: str
) -> Tuple[int, int]:
    raw_model = getattr(model, "module", model)
    budget = raw_model.vision_budget
    if mode == "max_pixels":
        max_pixels = int(budget_value)
        long_side = max(
            int(math.ceil(max_pixels**0.5)), int(budget.full_long_side)
        )
    else:
        long_side = int(budget_value)
        max_pixels = int(long_side * long_side)
    budget.full_long_side = long_side
    budget.full_max_pixels = max_pixels
    budget.coarse_long_side = long_side
    budget.coarse_max_pixels = max_pixels
    return long_side, max_pixels


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.policy.baseline_name = args.baseline_name
    set_seed(cfg.training.seed)

    budgets = parse_budget_list(args.vision_budget_list)
    if not budgets:
        raise ValueError("vision_budget_list is empty")

    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)
    model = build_model(cfg)
    if model.base_vlm.tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded; check model name")
    model = accelerator.prepare(model)
    load_eval_checkpoint(model, args.checkpoint, accelerator)

    output_dir = Path(args.output_dir)
    specs = load_dataset_specs(args.dataset_config, cfg)
    all_rows = []
    results_by_dataset = {}

    for spec in specs:
        dataset = build_dataset(spec)
        dataset_meta = collect_dataset_metadata(
            dataset,
            {
                "name": spec.name,
                "source": spec.source,
                "jsonl": spec.jsonl,
                "hf_name": spec.hf_name,
                "subset": spec.subset,
                "split": spec.split,
                "max_samples": spec.max_samples,
            },
        )
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

        for budget_value in budgets:
            long_side, max_pixels = apply_budget(
                model, budget_value, args.budget_mode
            )
            cfg.vision_budget.full_long_side = long_side
            cfg.vision_budget.full_max_pixels = max_pixels
            cfg.vision_budget.coarse_long_side = long_side
            cfg.vision_budget.coarse_max_pixels = max_pixels
            metrics = evaluate_dataset(
                model=model,
                loader=loader,
                metric_fn=metric_fn,
                cost_weight=None,
                profile=cfg.eval.profile,
                baseline_name=cfg.policy.baseline_name,
                baseline_threshold=cfg.policy.baseline_threshold,
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
            metrics["resolution_long_side"] = long_side
            metrics["resolution_max_pixels"] = max_pixels
            results.append(metrics)

            if accelerator.is_main_process:
                budget_dir = output_dir / spec.name / str(budget_value)
                budget_dir.mkdir(parents=True, exist_ok=True)
                rows = rows_from_results(spec.name, spec.metric, [metrics])
                write_csv(budget_dir / "results.csv", rows)
                write_csv(budget_dir / "eval_matrix.csv", rows)
                write_json(
                    budget_dir / "eval_matrix.json",
                    {"datasets": {spec.name: {"metric": spec.metric, "results": [metrics]}}},
                )
                metadata_path = write_run_metadata(
                    output_dir=budget_dir,
                    stage="eval_resolution_scaling",
                    cfg=cfg,
                    config_paths=args.config,
                    datasets={spec.name: dataset_meta},
                    extra={
                        "checkpoint": args.checkpoint,
                        "resolution_long_side": long_side,
                        "resolution_max_pixels": max_pixels,
                        "budget_mode": args.budget_mode,
                    },
                )
                write_json(
                    budget_dir / "summary.json",
                    {
                        "run_metadata_path": str(metadata_path),
                        "datasets": {spec.name: {"metric": spec.metric, "results": [metrics]}},
                        "dataset_metadata": {spec.name: dataset_meta},
                        "checkpoint": args.checkpoint,
                        "resolution_long_side": long_side,
                        "resolution_max_pixels": max_pixels,
                        "budget_mode": args.budget_mode,
                    },
                )

        results_by_dataset[spec.name] = {"metric": spec.metric, "results": results}
        all_rows.extend(rows_from_results(spec.name, spec.metric, results))

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "pareto_resolution.csv", all_rows)
        write_json(output_dir / "pareto_resolution.json", results_by_dataset)
        logger.info(
            "Saved resolution sweep to %s", output_dir / "pareto_resolution.csv"
        )


if __name__ == "__main__":
    main()
