"""Profile per-action latency."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.adapters.jsonl import JsonlVQADataset
from src.data.collate import VLMDataCollator
from src.eval.latency import measure_latency
from src.models.base_vlm import BaseVLM
from src.models.vovnet import Action, VoVNet
from src.models.vision_budget import VisionBudgetController
from src.utils.io import write_json
from src.utils.logging import setup_logging

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile VoVNet latency")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", type=str, default="outputs/latency.json")
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def build_model(cfg: Config) -> VoVNet:
    base_vlm = BaseVLM(
        cfg.model.base_model_name,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=cfg.model.torch_dtype,
    )
    budget = VisionBudgetController(**cfg.vision_budget.__dict__)
    return VoVNet(
        base_vlm=base_vlm,
        full_vlm=None,
        vision_budget=budget,
        vow_hidden_dim=cfg.policy.vow_hidden_dim,
        gumbel_tau=cfg.policy.gumbel_tau,
        use_straight_through=cfg.policy.use_straight_through,
        eval_sample=cfg.policy.eval_sample,
        policy_mode=cfg.policy.policy_mode,
        fallback_mode=cfg.policy.fallback_mode,
        fallback_entropy_threshold=cfg.policy.fallback_entropy_threshold,
        fallback_margin_threshold=cfg.policy.fallback_margin_threshold,
        cost_scale=cfg.policy.cost_scale,
        cost_c1=cfg.policy.cost_c1,
        cost_c2=cfg.policy.cost_c2,
    )


class ActionWrapper(nn.Module):
    """Wrap VoVNet to force an action during forward."""

    def __init__(self, model: VoVNet, action: Action) -> None:
        super().__init__()
        self.model = model
        self.action = action

    def forward(self, **batch: object) -> dict:
        batch_size = batch["input_ids"].size(0)
        actions = torch.full((batch_size,), self.action, device=batch["input_ids"].device)
        return self.model.forward_with_actions(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=batch.get("images"),
            actions=actions,
            labels=batch.get("labels"),
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model = build_model(cfg)
    if model.base_vlm.tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded; check model name")

    dataset = JsonlVQADataset(
        cfg.data.eval_jsonl or cfg.data.train_jsonl,
        text_field=cfg.data.text_field,
        answer_field=cfg.data.answer_field,
        image_field=cfg.data.image_field,
        max_samples=1,
    )
    collator = VLMDataCollator(
        tokenizer=model.base_vlm.tokenizer,
        prompt_template=cfg.data.prompt_template,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))

    results = {}
    for action in [Action.NO_VISION, Action.COARSE_VISION, Action.FULL_VISION]:
        wrapper = ActionWrapper(model, action)
        results[action.name.lower()] = measure_latency(wrapper, batch, iters=args.iters)
        logger.info("Latency %s: %s", action.name, results[action.name.lower()])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, results)


if __name__ == "__main__":
    main()
