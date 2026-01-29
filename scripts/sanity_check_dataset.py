"""Sanity check a JSONL dataset: load images, tokenize, and run a forward pass."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.config import Config
from src.data.adapters.jsonl import JsonlVQADataset
from src.data.collate import VLMDataCollator
from src.models.base_vlm import BaseVLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("sanity_check_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check a JSONL dataset")
    parser.add_argument("--data", required=True, help="Path to converted JSONL")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--config", action="append", default=["configs/base.yaml"])
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(paths: List[str]) -> Config:
    cfg = Config()
    for path in paths:
        cfg.update_from_yaml(path)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset = JsonlVQADataset(
        args.data,
        text_field=cfg.data.text_field,
        answer_field=cfg.data.answer_field,
        image_field=cfg.data.image_field,
        max_samples=args.num_samples,
    )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_vlm = BaseVLM(
        cfg.model.base_model_name,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=cfg.model.torch_dtype,
    )
    base_vlm.eval()
    base_vlm.to(device)

    collator = VLMDataCollator(
        base_vlm.tokenizer,
        cfg.data.prompt_template,
        max_length=getattr(cfg.data, "max_length", None),
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    ok_images = 0
    total_images = 0
    ran_forward = False

    for batch in loader:
        images = batch["images"]
        total_images += len(images)
        ok_images += sum(img is not None for img in images)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pixel_values = None
        image_grid_thw = None
        if base_vlm.processor is not None:
            try:
                proc = base_vlm.processor(images=images, return_tensors="pt")
                if "pixel_values" in proc:
                    pixel_values = proc["pixel_values"].to(device)
                if "image_grid_thw" in proc:
                    image_grid_thw = proc["image_grid_thw"].to(device)
            except Exception as exc:  # pragma: no cover - model-specific
                logger.warning("Processor failed: %s", exc)

        if pixel_values is not None:
            with torch.no_grad():
                _ = base_vlm.forward_with_vision(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=False,
                )
            ran_forward = True
            break

    logger.info("Loaded images: %d/%d", ok_images, total_images)
    if ran_forward:
        logger.info("Forward pass: OK")
    else:
        logger.warning("Forward pass skipped (no processor or failed preprocessing).")


if __name__ == "__main__":
    main()
