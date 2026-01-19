"""Runner for multi-dataset evaluation matrix."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.config import Config
from src.eval.matrix import build_model, evaluate_dataset, rows_from_results
from src.eval.matrix_spec import EvalDatasetSpec, build_dataset, get_metric_fn
from src.data.collate import VLMDataCollator
from src.utils.io import write_csv, write_json
from src.utils.run_metadata import collect_dataset_metadata, write_run_metadata


def _evaluate_single_process(
    cfg: Config,
    specs: List[EvalDatasetSpec],
    pareto: List[float] | None,
    output_dir: Path,
    checkpoint: Optional[str],
    write_outputs: bool = True,
) -> Dict[str, Any]:
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)
    model = build_model(cfg)
    if model.base_vlm.tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded; check model name")
    model = accelerator.prepare(model)
    if checkpoint:
        accelerator.load_state(checkpoint)

    all_results: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    dataset_meta: Dict[str, Any] = {}
    for spec in specs:
        dataset = build_dataset(spec)
        dataset_meta[spec.name] = collect_dataset_metadata(
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
        if pareto:
            results = []
            for value in pareto:
                metrics = evaluate_dataset(
                    model=model,
                    loader=loader,
                    metric_fn=metric_fn,
                    cost_weight=value,
                    profile=cfg.eval.profile,
                    baseline_name=cfg.policy.baseline_name,
                    baseline_threshold=cfg.policy.baseline_threshold,
                    baseline_uncertainty=cfg.policy.baseline_uncertainty,
                    baseline_vision=cfg.policy.baseline_vision,
                    baseline_seed=cfg.policy.baseline_seed or cfg.training.seed,
                    baseline_target_ratios=cfg.policy.baseline_target_ratios,
                    baseline_bucket_ratios=cfg.policy.baseline_bucket_ratios,
                    baseline_bucket_thresholds=cfg.policy.baseline_bucket_thresholds,
                )
                metrics["lambda_cost"] = value
                results.append(metrics)
        else:
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
            )
            metrics["lambda_cost"] = 0.0
            results = [metrics]

        all_results[spec.name] = {"metric": spec.metric, "results": results}
        all_rows.extend(rows_from_results(spec.name, spec.metric, results))

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "eval_matrix.json", {"datasets": all_results})
        write_csv(output_dir / "eval_matrix.csv", all_rows)
    return {"datasets": all_results, "rows": all_rows, "dataset_meta": dataset_meta}


def _worker_entry(
    cfg_paths: List[str],
    spec_dict: Dict[str, Any],
    pareto: List[float] | None,
    checkpoint: Optional[str],
    gpu_id: Optional[str],
) -> Dict[str, Any]:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cfg = Config()
    for path in cfg_paths:
        cfg.update_from_yaml(path)
    spec = EvalDatasetSpec(**spec_dict)
    return _evaluate_single_process(
        cfg=cfg,
        specs=[spec],
        pareto=pareto,
        output_dir=Path(cfg.training.output_dir),
        checkpoint=checkpoint,
        write_outputs=False,
    )


def run_eval_matrix(
    cfg: Config,
    cfg_paths: List[str],
    specs: List[EvalDatasetSpec],
    pareto: List[float] | None,
    output_dir: Path,
    checkpoint: Optional[str],
    parallel: bool,
    num_workers: Optional[int],
    gpus: Optional[List[str]],
) -> Dict[str, Any]:
    if parallel and len(specs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context

        cpu_cap = os.cpu_count() or 1
        gpu_cap = len(gpus) if gpus else cpu_cap
        max_workers = num_workers or min(len(specs), gpu_cap, cpu_cap)
        results: Dict[str, Any] = {"datasets": {}, "rows": [], "dataset_meta": {}}
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=get_context("spawn")
        ) as executor:
            futures = []
            for idx, spec in enumerate(specs):
                gpu_id = gpus[idx % len(gpus)] if gpus else None
                futures.append(
                    executor.submit(
                        _worker_entry,
                        cfg_paths,
                        spec.__dict__,
                        pareto,
                        checkpoint,
                        gpu_id,
                    )
                )
            for future in as_completed(futures):
                item = future.result()
                results["datasets"].update(item["datasets"])
                results["rows"].extend(item["rows"])
                results["dataset_meta"].update(item.get("dataset_meta", {}))

        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "eval_matrix.json", {"datasets": results["datasets"]})
        write_csv(output_dir / "eval_matrix.csv", results["rows"])
        _write_eval_summary(
            output_dir=output_dir,
            cfg=cfg,
            cfg_paths=cfg_paths,
            results=results,
            pareto=pareto,
            checkpoint=checkpoint,
        )
        return results

    results = _evaluate_single_process(
        cfg=cfg,
        specs=specs,
        pareto=pareto,
        output_dir=output_dir,
        checkpoint=checkpoint,
        write_outputs=True,
    )
    _write_eval_summary(
        output_dir=output_dir,
        cfg=cfg,
        cfg_paths=cfg_paths,
        results=results,
        pareto=pareto,
        checkpoint=checkpoint,
    )
    return results


def _write_eval_summary(
    output_dir: Path,
    cfg: Config,
    cfg_paths: List[str],
    results: Dict[str, Any],
    pareto: Optional[List[float]],
    checkpoint: Optional[str],
) -> None:
    metadata_path = write_run_metadata(
        output_dir=output_dir,
        stage="eval",
        cfg=cfg,
        config_paths=cfg_paths,
        datasets=results.get("dataset_meta", {}),
        extra={"checkpoint": checkpoint},
    )
    summary = {
        "run_metadata_path": str(metadata_path),
        "datasets": results.get("datasets", {}),
        "dataset_metadata": results.get("dataset_meta", {}),
        "pareto": pareto or [],
        "checkpoint": checkpoint,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "results.csv", results.get("rows", []))
