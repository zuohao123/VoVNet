"""Oracle action analysis for VoVNet (eval-only)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Optional

import torch

from src.config.config import Config
from src.data.collate import VLMDataCollator
from src.eval.matrix import build_model
from src.eval.matrix_spec import EvalDatasetSpec, build_dataset, load_dataset_specs
from src.eval.metrics import normalize_text
from src.models.vovnet import Action
from src.utils.io import write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle action analysis")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def _decode_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, tokenizer: Any
) -> List[str]:
    preds: List[str] = []
    pred_ids = logits.argmax(dim=-1)
    for i in range(pred_ids.size(0)):
        mask = labels[i] != -100
        answer_ids = pred_ids[i][mask]
        text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        preds.append(text.strip())
    return preds


def _match_preds(preds: Iterable[str], refs: Iterable[str], device: torch.device) -> torch.Tensor:
    values = [
        normalize_text(pred) == normalize_text(ref)
        for pred, ref in zip(preds, refs)
    ]
    return torch.tensor(values, dtype=torch.bool, device=device)


def _oracle_select(
    correct_mask: torch.Tensor, losses: torch.Tensor, costs: torch.Tensor
) -> torch.Tensor:
    large = torch.full_like(costs, 1e9)
    costs_for_correct = torch.where(correct_mask, costs, large)
    any_correct = correct_mask.any(dim=1)
    best_cost_action = costs_for_correct.argmin(dim=1)
    losses_for_select = losses + costs * 1e-6
    best_loss_action = losses_for_select.argmin(dim=1)
    return torch.where(any_correct, best_cost_action, best_loss_action)


def _evaluate_oracle(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
) -> dict:
    raw_model = getattr(model, "module", model)
    raw_model.eval()
    tokenizer = raw_model.base_vlm.tokenizer

    total_count = 0
    oracle_correct = 0.0
    oracle_cost = 0.0
    vovnet_correct = 0.0
    vovnet_cost = 0.0
    action_counts = torch.zeros(len(Action), dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch.get("labels")
            images = batch.get("images")
            answers = batch.get("answers") or ["" for _ in range(input_ids.size(0))]
            device = input_ids.device
            batch_size = input_ids.size(0)

            token_count_coarse, token_count_full, coarse_inputs, full_inputs = (
                raw_model._prepare_token_counts(
                    images=images,
                    device=device,
                    batch_size=batch_size,
                )
                if images is not None
                else (
                    torch.zeros(batch_size, device=device),
                    torch.zeros(batch_size, device=device),
                    None,
                    None,
                )
            )

            text_ids, text_mask, text_labels = raw_model._prepare_text_and_vision_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                coarse_inputs=coarse_inputs,
                full_inputs=full_inputs,
            )
            text_outputs, action_logits, _ = raw_model.text_first(text_ids, text_mask)

            actions_no = torch.full((batch_size,), Action.NO_VISION, device=device)
            actions_coarse = torch.full((batch_size,), Action.COARSE_VISION, device=device)
            actions_full = torch.full((batch_size,), Action.FULL_VISION, device=device)

            logits_no, _, labels_no = raw_model._forward_hard_actions(
                input_ids=text_ids,
                attention_mask=text_mask,
                images=None,
                text_outputs=text_outputs,
                actions=actions_no,
                text_labels=text_labels,
                coarse_inputs=coarse_inputs,
                full_inputs=full_inputs,
            )
            logits_coarse, _, labels_coarse = raw_model._forward_hard_actions(
                input_ids=text_ids,
                attention_mask=text_mask,
                images=images,
                text_outputs=text_outputs,
                actions=actions_coarse,
                text_labels=text_labels,
                coarse_inputs=coarse_inputs,
                full_inputs=full_inputs,
            )
            logits_full, _, labels_full = raw_model._forward_hard_actions(
                input_ids=text_ids,
                attention_mask=text_mask,
                images=images,
                text_outputs=text_outputs,
                actions=actions_full,
                text_labels=text_labels,
                coarse_inputs=coarse_inputs,
                full_inputs=full_inputs,
            )

            labels_no = labels_no if labels_no is not None else text_labels
            labels_coarse = labels_coarse if labels_coarse is not None else labels_no
            labels_full = labels_full if labels_full is not None else labels_no

            loss_no = raw_model._per_sample_loss(logits_no, labels_no)
            loss_coarse = raw_model._per_sample_loss(logits_coarse, labels_coarse)
            loss_full = raw_model._per_sample_loss(logits_full, labels_full)

            preds_no = _decode_from_logits(logits_no, labels_no, tokenizer)
            preds_coarse = _decode_from_logits(logits_coarse, labels_coarse, tokenizer)
            preds_full = _decode_from_logits(logits_full, labels_full, tokenizer)

            correct_no = _match_preds(preds_no, answers, device)
            correct_coarse = _match_preds(preds_coarse, answers, device)
            correct_full = _match_preds(preds_full, answers, device)

            correct_mask = torch.stack(
                [correct_no, correct_coarse, correct_full], dim=1
            )
            losses = torch.stack([loss_no, loss_coarse, loss_full], dim=1)

            cost_scale = float(getattr(raw_model, "cost_scale", 1.0))
            cost_no = torch.zeros_like(token_count_full) * cost_scale
            cost_coarse = token_count_coarse * cost_scale
            cost_full = token_count_full * cost_scale
            costs = torch.stack([cost_no, cost_coarse, cost_full], dim=1)

            oracle_actions = _oracle_select(correct_mask, losses, costs)
            oracle_correct_batch = correct_mask.gather(
                1, oracle_actions.unsqueeze(1)
            ).squeeze(1)
            oracle_cost_batch = costs.gather(1, oracle_actions.unsqueeze(1)).squeeze(1)

            action_counts += torch.bincount(
                oracle_actions.detach().cpu(), minlength=len(Action)
            )
            oracle_correct += float(oracle_correct_batch.float().sum().item())
            oracle_cost += float(oracle_cost_batch.sum().item())

            action_probs, policy_actions = raw_model._select_actions(action_logits)
            if images is not None:
                uncertainty = raw_model._compute_uncertainty(text_outputs)
                margin = raw_model._compute_margin(text_outputs)
                policy_actions, _, _, _ = raw_model._apply_fallback(
                    policy_actions, uncertainty, margin
                )
            policy_correct_batch = correct_mask.gather(
                1, policy_actions.unsqueeze(1)
            ).squeeze(1)
            policy_cost_batch = costs.gather(1, policy_actions.unsqueeze(1)).squeeze(1)
            vovnet_correct += float(policy_correct_batch.float().sum().item())
            vovnet_cost += float(policy_cost_batch.sum().item())

            total_count += batch_size

    oracle_action_ratio = (action_counts.float() / max(1, total_count)).tolist()
    oracle_accuracy = oracle_correct / max(1, total_count)
    oracle_avg_cost = oracle_cost / max(1, total_count)
    vovnet_accuracy = vovnet_correct / max(1, total_count)
    vovnet_avg_cost = vovnet_cost / max(1, total_count)
    return {
        "oracle_action_ratio": oracle_action_ratio,
        "oracle_accuracy": oracle_accuracy,
        "oracle_avg_cost": oracle_avg_cost,
        "vovnet_accuracy": vovnet_accuracy,
        "vovnet_avg_cost": vovnet_avg_cost,
        "accuracy_gap_vs_vovnet": oracle_accuracy - vovnet_accuracy,
        "cost_gap_vs_vovnet": oracle_avg_cost - vovnet_avg_cost,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    specs = load_dataset_specs(args.dataset_config, cfg)
    if args.max_samples is not None:
        for spec in specs:
            spec.max_samples = args.max_samples

    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)
    model = build_model(cfg)
    if model.base_vlm.tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded; check model name")
    model = accelerator.prepare(model)
    if args.checkpoint:
        accelerator.load_state(args.checkpoint)

    results = {"datasets": {}}
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
        metrics = _evaluate_oracle(model, loader)
        metrics["metric"] = spec.metric
        metrics["max_samples"] = spec.max_samples
        results["datasets"][spec.name] = metrics

    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "analysis_oracle.json", results)
        logger.info("Saved oracle analysis to %s", output_dir / "analysis_oracle.json")


if __name__ == "__main__":
    main()
