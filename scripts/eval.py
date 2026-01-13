"""Evaluate VoVNet with multi-dataset matrix support."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from src.config.config import Config
from src.eval.matrix_spec import load_dataset_specs
from src.eval.matrix_runner import run_eval_matrix
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VoVNet (matrix)")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pareto", type=float, nargs="*", default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None)
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

    specs = load_dataset_specs(args.dataset_config, cfg)
    gpus: Optional[List[str]] = args.gpus.split(",") if args.gpus else None
    pareto = args.pareto if args.pareto else None

    run_eval_matrix(
        cfg=cfg,
        cfg_paths=args.config,
        specs=specs,
        pareto=pareto,
        output_dir=Path(args.output_dir),
        checkpoint=args.checkpoint,
        parallel=args.parallel,
        num_workers=args.num_workers,
        gpus=gpus,
    )


if __name__ == "__main__":
    main()
