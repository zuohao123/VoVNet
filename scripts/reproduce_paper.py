"""Reproduce core tables/figures data for the VoVNet paper."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from src.baselines.always_coarse import select_actions as always_coarse
from src.baselines.always_full import select_actions as always_full
from src.baselines.no_vision import select_actions as no_vision
from src.baselines.random_policy import select_actions as random_policy
from src.baselines.uncertainty_threshold import select_actions as threshold_policy
from src.config.config import Config
from src.data.adapters.hf_dataset import HFDatasetAdapter
from src.data.adapters.jsonl import JsonlVQADataset
from src.data.collate import VLMDataCollator
from src.eval.metrics import exact_match_score
from src.eval.pareto import run_pareto_sweep
from src.models.base_vlm import BaseVLM
from src.models.vovnet import Action, VoVNet
from src.models.vision_budget import VisionBudgetController
from src.utils.io import write_csv, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce VoVNet results")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", type=str, default="outputs/repro")
    parser.add_argument("--thresholds", type=float, nargs=2, default=[0.5, 1.0])
    parser.add_argument("--pareto", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
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
        policy_mode=cfg.policy.policy_mode,
        fallback_mode=cfg.policy.fallback_mode,
        fallback_entropy_threshold=cfg.policy.fallback_entropy_threshold,
        fallback_margin_threshold=cfg.policy.fallback_margin_threshold,
        cost_scale=cfg.policy.cost_scale,
        cost_c1=cfg.policy.cost_c1,
        cost_c2=cfg.policy.cost_c2,
    )


def decode_answers(logits: torch.Tensor, labels: torch.Tensor, tokenizer: Any) -> List[str]:
    preds: List[str] = []
    pred_ids = logits.argmax(dim=-1)
    for i in range(pred_ids.size(0)):
        mask = labels[i] != -100
        answer_ids = pred_ids[i][mask]
        text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        preds.append(text.strip())
    return preds


def evaluate_policy(model: VoVNet, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    total_acc = 0.0
    total_cost = 0.0
    total_count = 0

    tokenizer = model.base_vlm.tokenizer
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch.get("images"),
                labels=batch.get("labels"),
            )
            preds = decode_answers(outputs["logits"], batch["labels"], tokenizer)
            acc = exact_match_score(preds, batch.get("answers", []))
            total_acc += acc * batch["input_ids"].size(0)
            total_cost += outputs["expected_cost"].sum().item()
            total_count += batch["input_ids"].size(0)

    return {
        "accuracy": total_acc / max(1, total_count),
        "avg_cost": total_cost / max(1, total_count),
    }


def evaluate_policy_with_cost_weight(
    model: VoVNet, loader: DataLoader, cost_weight: float
) -> Dict[str, float]:
    """Recompute actions using a cost-adjusted logit for Pareto sweeps."""
    model.eval()
    total_acc = 0.0
    total_cost = 0.0
    total_count = 0

    tokenizer = model.base_vlm.tokenizer
    with torch.no_grad():
        for batch in loader:
            _, action_logits, _ = model.text_first(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            token_count_coarse, token_count_full, _, _ = model._prepare_token_counts(
                images=batch.get("images"),
                device=action_logits.device,
                batch_size=batch["input_ids"].size(0),
            )
            zeros = torch.zeros_like(token_count_coarse)
            costs = torch.stack([zeros, token_count_coarse, token_count_full], dim=-1)
            costs = costs * float(model.cost_scale)
            adjusted_logits = action_logits - cost_weight * costs
            actions = adjusted_logits.argmax(dim=-1)

            outputs = model.forward_with_actions(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch.get("images"),
                actions=actions,
                labels=batch.get("labels"),
            )
            preds = decode_answers(outputs["logits"], batch["labels"], tokenizer)
            acc = exact_match_score(preds, batch.get("answers", []))
            total_acc += acc * batch["input_ids"].size(0)
            total_cost += outputs["expected_cost"].sum().item()
            total_count += batch["input_ids"].size(0)

    return {
        "accuracy": total_acc / max(1, total_count),
        "avg_cost": total_cost / max(1, total_count),
    }


def evaluate_baseline(
    model: VoVNet,
    loader: DataLoader,
    policy_name: str,
    thresholds: List[float],
) -> Dict[str, float]:
    model.eval()
    total_acc = 0.0
    total_cost = 0.0
    total_count = 0

    tokenizer = model.base_vlm.tokenizer
    with torch.no_grad():
        for batch in loader:
            batch_size = batch["input_ids"].size(0)
            device = batch["input_ids"].device

            if policy_name == "uncertainty_threshold":
                text_outputs = model.base_vlm.encode_text(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,
                )
                uncertainty = model._compute_uncertainty(text_outputs)
                actions = threshold_policy(uncertainty, thresholds[0], thresholds[1])
            elif policy_name == "always_full":
                actions = always_full(batch_size, device)
            elif policy_name == "always_coarse":
                actions = always_coarse(batch_size, device)
            elif policy_name == "no_vision":
                actions = no_vision(batch_size, device)
            elif policy_name == "random_policy":
                actions = random_policy(batch_size, device)
            else:
                raise ValueError(f"Unknown policy {policy_name}")

            outputs = model.forward_with_actions(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch.get("images"),
                actions=actions,
                labels=batch.get("labels"),
            )
            preds = decode_answers(outputs["logits"], batch["labels"], tokenizer)
            acc = exact_match_score(preds, batch.get("answers", []))
            total_acc += acc * batch_size
            total_cost += outputs["expected_cost"].sum().item()
            total_count += batch_size

    return {
        "accuracy": total_acc / max(1, total_count),
        "avg_cost": total_cost / max(1, total_count),
    }


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
    loader = DataLoader(dataset, batch_size=cfg.eval.batch_size, shuffle=False, collate_fn=collator)

    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)
    model, loader = accelerator.prepare(model, loader)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    vov_metrics = evaluate_policy(model, loader)
    baselines = {
        name: evaluate_baseline(model, loader, name, args.thresholds)
        for name in [
            "always_full",
            "always_coarse",
            "no_vision",
            "random_policy",
            "uncertainty_threshold",
        ]
    }

    pareto_results = run_pareto_sweep(
        args.pareto,
        eval_fn=lambda value: {
            "lambda_cost": value,
            **evaluate_policy_with_cost_weight(model, loader, value),
        },
    )

    if accelerator.is_main_process:
        write_json(output_dir / "vovnet.json", vov_metrics)
        write_json(output_dir / "baselines.json", baselines)
        write_json(output_dir / "pareto.json", pareto_results)

        table_rows = [
            {"name": "vovnet", **vov_metrics},
            *[{"name": name, **metrics} for name, metrics in baselines.items()],
        ]
        write_csv(output_dir / "tables.csv", table_rows)

    logger.info("Saved outputs to %s", output_dir)


if __name__ == "__main__":
    main()
