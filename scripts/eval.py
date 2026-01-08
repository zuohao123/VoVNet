"""Evaluate VoVNet on a dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.adapters.hf_dataset import HFDatasetAdapter
from src.data.adapters.jsonl import JsonlVQADataset
from src.data.collate import VLMDataCollator
from src.models.base_vlm import BaseVLM
from src.models.vovnet import VoVNet
from src.models.vision_budget import VisionBudgetController
from src.training.trainer import Trainer
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VoVNet")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def build_dataset(cfg: Config) -> torch.utils.data.Dataset:
    if cfg.data.eval_jsonl:
        return JsonlVQADataset(
            cfg.data.eval_jsonl,
            text_field=cfg.data.text_field,
            answer_field=cfg.data.answer_field,
            image_field=cfg.data.image_field,
            max_samples=cfg.data.max_samples,
        )
    if cfg.data.hf_dataset_name:
        return HFDatasetAdapter(
            cfg.data.hf_dataset_name,
            split=cfg.data.hf_dataset_split,
            text_field=cfg.data.text_field,
            answer_field=cfg.data.answer_field,
            image_field=cfg.data.image_field,
            max_samples=cfg.data.max_samples,
        )
    raise ValueError("No eval dataset configured")


def build_model(cfg: Config) -> VoVNet:
    base_vlm = BaseVLM(
        cfg.model.base_model_name,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=cfg.model.torch_dtype,
    )
    full_vlm = None
    if cfg.model.use_thinking_for_full:
        full_vlm = BaseVLM(
            cfg.model.full_model_name,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=cfg.model.torch_dtype,
        )

    budget = VisionBudgetController(**cfg.vision_budget.__dict__)
    return VoVNet(
        base_vlm=base_vlm,
        full_vlm=full_vlm,
        vision_budget=budget,
        vow_hidden_dim=cfg.policy.vow_hidden_dim,
        gumbel_tau=cfg.policy.gumbel_tau,
        use_straight_through=cfg.policy.use_straight_through,
        eval_sample=cfg.policy.eval_sample,
        cost_c1=cfg.policy.cost_c1,
        cost_c2=cfg.policy.cost_c2,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    model = build_model(cfg)
    if model.base_vlm.tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded; check model name")

    dataset = build_dataset(cfg)
    collator = VLMDataCollator(
        tokenizer=model.base_vlm.tokenizer,
        prompt_template=cfg.data.prompt_template,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)
    model, loader = accelerator.prepare(model, loader)

    if args.checkpoint:
        accelerator.load_state(args.checkpoint)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.training.lr
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        accelerator=accelerator,
        train_loader=loader,
        eval_loader=loader,
        output_dir=cfg.training.output_dir,
        lambda_cost=cfg.policy.lambda_cost,
        lambda_cal=cfg.policy.calibration_lambda,
    )
    trainer.evaluate()


if __name__ == "__main__":
    main()
