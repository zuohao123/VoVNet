"""Train VoVNet with DeepSpeed or FSDP via accelerate."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional

import torch
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.adapters.hf_dataset import HFDatasetAdapter
from src.data.adapters.jsonl import JsonlVQADataset
from src.data.collate import VLMDataCollator
from src.models.base_vlm import BaseVLM
from src.models.vovnet import VoVNet
from src.models.vision_budget import VisionBudgetController
from src.training.deepspeed_utils import build_deepspeed_config, write_deepspeed_config
from src.training.fsdp_utils import build_fsdp_plugin
from src.training.schedulers import build_scheduler
from src.training.trainer import Trainer
from src.utils.logging import setup_logging
from src.utils.run_metadata import collect_dataset_metadata, write_run_metadata
from src.utils.seed import set_seed

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VoVNet")
    parser.add_argument("--config", action="append", required=True)
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def build_dataset(cfg: Config, split: str) -> torch.utils.data.Dataset:
    if split == "train" and cfg.data.train_jsonl:
        return JsonlVQADataset(
            cfg.data.train_jsonl,
            text_field=cfg.data.text_field,
            answer_field=cfg.data.answer_field,
            image_field=cfg.data.image_field,
            max_samples=cfg.data.max_samples,
        )
    if split == "eval" and cfg.data.eval_jsonl:
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
    raise ValueError("No dataset configured")


def build_accelerator(cfg: Config, output_dir: Path) -> Any:
    from accelerate import Accelerator

    ds_plugin = None
    if cfg.training.deepspeed_stage in (2, 3):
        ds_config = build_deepspeed_config(
            cfg.training.deepspeed_stage,
            cfg.training.per_device_batch_size,
            cfg.training.gradient_accumulation,
            cfg.training.mixed_precision,
        )
        ds_path = output_dir / "deepspeed_config.json"
        write_deepspeed_config(ds_path, ds_config)
        try:
            from accelerate import DeepSpeedPlugin

            try:
                ds_plugin = DeepSpeedPlugin(
                    zero_stage=cfg.training.deepspeed_stage,
                    config_file=str(ds_path),
                )
            except TypeError:
                ds_plugin = DeepSpeedPlugin(zero_stage=cfg.training.deepspeed_stage)
        except Exception as exc:
            logger.warning("DeepSpeed plugin unavailable: %s", exc)

    fsdp_plugin = build_fsdp_plugin() if cfg.training.use_fsdp else None

    return Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation,
        mixed_precision=cfg.training.mixed_precision,
        deepspeed_plugin=ds_plugin,
        fsdp_plugin=fsdp_plugin,
    )


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

    if cfg.model.use_lora:
        for param in base_vlm.model.parameters():
            param.requires_grad = False
        base_vlm.apply_lora(
            r=cfg.model.lora_r,
            alpha=cfg.model.lora_alpha,
            dropout=cfg.model.lora_dropout,
            target_modules=cfg.model.lora_target_modules,
        )

    if cfg.model.freeze_vision_encoder:
        base_vlm.freeze_vision_encoder()
        if full_vlm is not None:
            full_vlm.freeze_vision_encoder()

    if cfg.training.gradient_checkpointing:
        base_vlm.enable_gradient_checkpointing()
        if full_vlm is not None:
            full_vlm.enable_gradient_checkpointing()

    budget = VisionBudgetController(**cfg.vision_budget.__dict__)
    model = VoVNet(
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
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    if model.base_vlm.tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded; check model name")

    train_dataset = build_dataset(cfg, "train")
    eval_dataset = build_dataset(cfg, "eval") if cfg.data.eval_jsonl else None

    collator = VLMDataCollator(
        tokenizer=model.base_vlm.tokenizer,
        prompt_template=cfg.data.prompt_template,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        if eval_dataset is not None
        else None
    )

    accelerator = build_accelerator(cfg, output_dir)
    if accelerator.is_main_process:
        dataset_meta: dict[str, Any] = {}
        dataset_meta["train"] = collect_dataset_metadata(
            train_dataset,
            {
                "split": "train",
                "source": "jsonl" if cfg.data.train_jsonl else "hf",
                "jsonl": cfg.data.train_jsonl,
                "hf_dataset_name": cfg.data.hf_dataset_name,
                "hf_dataset_split": cfg.data.hf_dataset_split,
                "max_samples": cfg.data.max_samples,
            },
        )
        if eval_dataset is not None:
            dataset_meta["eval"] = collect_dataset_metadata(
                eval_dataset,
                {
                    "split": "eval",
                    "source": "jsonl" if cfg.data.eval_jsonl else "hf",
                    "jsonl": cfg.data.eval_jsonl,
                    "hf_dataset_name": cfg.data.hf_dataset_name,
                    "hf_dataset_split": cfg.data.hf_dataset_split,
                    "max_samples": cfg.data.max_samples,
                },
            )
        write_run_metadata(
            output_dir=output_dir,
            stage="train",
            cfg=cfg,
            config_paths=args.config,
            datasets=dataset_meta,
            extra={"output_dir": str(output_dir)},
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    num_training_steps = cfg.training.epochs * len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        train_loader=train_loader,
        eval_loader=eval_loader,
        output_dir=str(output_dir),
        lambda_cost=cfg.policy.lambda_cost,
        lambda_cal=cfg.policy.calibration_lambda,
        log_every=cfg.training.log_every,
        save_every=cfg.training.save_every,
        max_grad_norm=cfg.training.max_grad_norm,
        profile_train=cfg.training.profile,
        profile_eval=cfg.eval.profile,
        gain_supervision=cfg.policy.gain_supervision,
        gain_loss_type=cfg.policy.gain_loss_type,
        gain_loss_weight=cfg.policy.gain_loss_weight,
        gain_margin=cfg.policy.gain_margin,
    )
    trainer.train(cfg.training.epochs)


if __name__ == "__main__":
    main()
