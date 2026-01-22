"""Sweep merge/prune ratios for token-merge proxy baseline."""
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
        description="Pareto sweep for token-merge prune proxy baseline"
    )
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default="outputs/baselines/token_merge"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--baseline_name", type=str, default="token_merge_prune_proxy")
    parser.add_argument(
        "--merge_ratio_list",
        type=str,
        required=True,
        help='Comma list or JSON list, e.g. "0.25,0.5,0.75,1.0" or "[0.25,0.5]"',
    )
    parser.add_argument(
        "--merge_mode", type=str, default="cosine", choices=["cosine", "l2"]
    )
    parser.add_argument(
        "--merge_weight", type=str, default="norm", choices=["norm", "mean"]
    )
    parser.add_argument(
        "--enable_prune",
        type=str,
        default="false",
        help="Enable post-merge pruning: true/false",
    )
    parser.add_argument(
        "--prune_ratio_list",
        type=str,
        default="1.0",
        help='Comma list or JSON list, e.g. "0.5,0.75,1.0"',
    )
    parser.add_argument(
        "--prune_mode",
        type=str,
        default="topk_norm",
        choices=["stride", "topk_norm", "topk"],
    )
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def _parse_ratio_list(raw: str) -> List[float]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        values = json.loads(raw)
    else:
        values = [item.strip() for item in raw.split(",") if item.strip()]
    return [float(item) for item in values]


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _tag(merge_ratio: float, enable_prune: bool, prune_ratio: float) -> str:
    merge_tag = f"merge_{merge_ratio:.2f}".replace(".", "p")
    if enable_prune:
        prune_tag = f"prune_{prune_ratio:.2f}".replace(".", "p")
        return f"{merge_tag}_{prune_tag}"
    return merge_tag


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.policy.baseline_name = args.baseline_name
    set_seed(cfg.training.seed)

    merge_ratios = _parse_ratio_list(args.merge_ratio_list)
    if not merge_ratios:
        raise ValueError("merge_ratio_list is empty")
    enable_prune = _parse_bool(args.enable_prune)
    prune_ratios = _parse_ratio_list(args.prune_ratio_list) if enable_prune else [1.0]
    if enable_prune and not prune_ratios:
        raise ValueError("prune_ratio_list is empty")

    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)
    use_distributed = accelerator.num_processes > 1
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
        if use_distributed:
            loader = accelerator.prepare(loader)
        metric_fn = get_metric_fn(spec.metric)
        results = []

        for merge_ratio in merge_ratios:
            for prune_ratio in prune_ratios:
                cfg.policy.baseline_merge_ratio = merge_ratio
                cfg.policy.baseline_merge_mode = args.merge_mode
                cfg.policy.baseline_merge_weight = args.merge_weight
                cfg.policy.baseline_enable_prune = enable_prune
                cfg.policy.baseline_prune_ratio = prune_ratio
                cfg.policy.baseline_prune_mode = args.prune_mode
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
                    baseline_merge_ratio=merge_ratio,
                    baseline_merge_mode=args.merge_mode,
                    baseline_merge_weight=args.merge_weight,
                    baseline_enable_prune=enable_prune,
                    baseline_prune_ratio=prune_ratio,
                    baseline_prune_mode=args.prune_mode,
                    baseline_pool_factor=cfg.policy.baseline_pool_factor,
                )
                metrics["merge_ratio"] = merge_ratio
                metrics["prune_ratio"] = prune_ratio if enable_prune else 1.0
                metrics["merge_mode"] = args.merge_mode
                metrics["merge_weight"] = args.merge_weight
                metrics["enable_prune"] = enable_prune
                metrics["prune_mode"] = args.prune_mode
                results.append(metrics)

                if accelerator.is_main_process:
                    tag = _tag(merge_ratio, enable_prune, prune_ratio)
                    run_dir = output_dir / spec.name / tag
                    run_dir.mkdir(parents=True, exist_ok=True)
                    rows = rows_from_results(spec.name, spec.metric, [metrics])
                    write_csv(run_dir / "results.csv", rows)
                    write_csv(run_dir / "eval_matrix.csv", rows)
                    write_json(
                        run_dir / "eval_matrix.json",
                        {"datasets": {spec.name: {"metric": spec.metric, "results": [metrics]}}},
                    )
                    metadata_path = write_run_metadata(
                        output_dir=run_dir,
                        stage="eval_token_merge",
                        cfg=cfg,
                        config_paths=args.config,
                        datasets={spec.name: dataset_meta},
                        extra={
                            "checkpoint": args.checkpoint,
                            "merge_ratio": merge_ratio,
                            "merge_mode": args.merge_mode,
                            "merge_weight": args.merge_weight,
                            "enable_prune": enable_prune,
                            "prune_ratio": prune_ratio,
                            "prune_mode": args.prune_mode,
                        },
                    )
                    write_json(
                        run_dir / "summary.json",
                        {
                            "run_metadata_path": str(metadata_path),
                            "datasets": {spec.name: {"metric": spec.metric, "results": [metrics]}},
                            "dataset_metadata": {spec.name: dataset_meta},
                            "checkpoint": args.checkpoint,
                            "merge_ratio": merge_ratio,
                            "merge_mode": args.merge_mode,
                            "merge_weight": args.merge_weight,
                            "enable_prune": enable_prune,
                            "prune_ratio": prune_ratio,
                            "prune_mode": args.prune_mode,
                        },
                    )

        results_by_dataset[spec.name] = {"metric": spec.metric, "results": results}
        all_rows.extend(rows_from_results(spec.name, spec.metric, results))

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "pareto_merge.csv", all_rows)
        write_json(output_dir / "pareto_merge.json", results_by_dataset)
        logger.info("Saved merge sweep to %s", output_dir / "pareto_merge.csv")


if __name__ == "__main__":
    main()
