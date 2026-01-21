"""Sweep pooling factors for multi-granularity proxy baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

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
        description="Pareto sweep for multi-granularity token pooling baseline"
    )
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default="outputs/baselines/multi_granularity"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--baseline_name", type=str, default="multi_granularity_proxy")
    parser.add_argument(
        "--pooling_list",
        type=str,
        required=True,
        help='Comma list or JSON list, e.g. "1,2,4" or "[1,2,4]"',
    )
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def parse_pooling_list(raw: str) -> List[int]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        values = json.loads(raw)
    else:
        values = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    return [int(float(item)) for item in values]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.policy.baseline_name = args.baseline_name
    set_seed(cfg.training.seed)

    pool_factors = parse_pooling_list(args.pooling_list)
    if not pool_factors:
        raise ValueError("pooling_list is empty")

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

        for pool_factor in pool_factors:
            cfg.policy.baseline_pool_factor = int(pool_factor)
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
            metrics["pool_factor"] = int(pool_factor)
            results.append(metrics)

            if accelerator.is_main_process:
                run_dir = output_dir / spec.name / f"pool_{pool_factor}"
                run_dir.mkdir(parents=True, exist_ok=True)
                rows = rows_from_results(spec.name, spec.metric, [metrics])
                write_csv(run_dir / "results.csv", rows)
                write_csv(run_dir / "eval_matrix.csv", rows)
                write_json(
                    run_dir / "eval_matrix.json",
                    {"datasets": {spec.name: {"metric": spec.metric, "results": [metrics]}}},
                )
                write_run_metadata(
                    output_dir=run_dir,
                    stage="eval_multi_granularity",
                    cfg=cfg,
                    config_paths=args.config,
                    datasets={spec.name: dataset_meta},
                    extra={
                        "checkpoint": args.checkpoint,
                        "pool_factor": pool_factor,
                    },
                )
                write_json(
                    run_dir / "summary.json",
                    {
                        "baseline_name": cfg.policy.baseline_name,
                        "pool_factor": pool_factor,
                        "metric": spec.metric,
                        "dataset": spec.name,
                        "results": metrics,
                    },
                )

        results_by_dataset[spec.name] = {"metric": spec.metric, "results": results}
        all_rows.extend(rows_from_results(spec.name, spec.metric, results))

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "pareto_granularity.csv", all_rows)
        write_json(output_dir / "pareto_granularity.json", results_by_dataset)
        logger.info("Saved pareto sweep to %s", output_dir / "pareto_granularity.csv")


if __name__ == "__main__":
    main()
