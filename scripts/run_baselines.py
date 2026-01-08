"""Run baseline policies using the shared evaluation pipeline."""
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
from src.models.base_vlm import BaseVLM
from src.models.vovnet import Action, VoVNet
from src.models.vision_budget import VisionBudgetController
from src.utils.io import write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VoVNet baselines")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", type=str, default="outputs/baselines.json")
    parser.add_argument("--thresholds", type=float, nargs=2, default=[0.5, 1.0])
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
    budget = VisionBudgetController(**cfg.vision_budget.__dict__)
    return VoVNet(
        base_vlm=base_vlm,
        full_vlm=None,
        vision_budget=budget,
        vow_hidden_dim=cfg.policy.vow_hidden_dim,
        gumbel_tau=cfg.policy.gumbel_tau,
        use_straight_through=cfg.policy.use_straight_through,
        eval_sample=cfg.policy.eval_sample,
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


def evaluate_baseline(
    model: VoVNet,
    loader: DataLoader,
    policy_name: str,
    thresholds: List[float],
) -> Dict[str, float]:
    model.eval()
    tokenizer = model.base_vlm.tokenizer
    total_acc = 0.0
    total_count = 0
    total_cost = 0.0
    action_counts = torch.zeros(len(Action))

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
            action_counts += torch.bincount(actions.cpu(), minlength=len(Action))

    action_rates = (action_counts / max(1, total_count)).tolist()
    return {
        "accuracy": total_acc / max(1, total_count),
        "avg_cost": total_cost / max(1, total_count),
        "action_rates": action_rates,
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
    loader = DataLoader(
        dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)
    model, loader = accelerator.prepare(model, loader)

    results = {}
    for policy in [
        "always_full",
        "always_coarse",
        "no_vision",
        "random_policy",
        "uncertainty_threshold",
    ]:
        metrics = evaluate_baseline(model, loader, policy, args.thresholds)
        results[policy] = metrics
        logger.info("Baseline %s: %s", policy, metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        write_json(output_path, results)


if __name__ == "__main__":
    main()
