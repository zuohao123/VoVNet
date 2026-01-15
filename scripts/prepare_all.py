"""Prepare datasets using a YAML recipe."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.exporters import export_jsonl, export_parquet
from datasets.registry import get_adapter
from scripts.prepare_dataset import build_image_entry, iter_records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("prepare_all")


@dataclass
class ManifestEntry:
    dataset: str
    split: str
    num_examples: int
    schema_version: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "num_examples_written": self.num_examples,
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets from recipe")
    parser.add_argument("--mode", choices=["fast_dev", "paper"], default=None)
    parser.add_argument("--recipe", default="configs/data_recipe.yaml")
    parser.add_argument("--override_max_samples", type=int, default=None)
    parser.add_argument("--download_images", action="store_true")
    parser.add_argument("--export_parquet", action="store_true")
    return parser.parse_args()


def load_recipe(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Recipe must be a dict")
    return data


def resolve_mode(recipe: Dict[str, Any], cli_mode: Optional[str]) -> str:
    mode = cli_mode or recipe.get("mode")
    if mode not in ("fast_dev", "paper"):
        raise ValueError("mode must be fast_dev or paper")
    return mode


def resolve_max_samples(
    entry: Dict[str, Any],
    split: str,
    mode: str,
    override: Optional[int],
) -> Optional[int]:
    if override is not None:
        return override
    max_by_split = entry.get("max_samples_by_split", {})
    split_cfg = max_by_split.get(split)
    if isinstance(split_cfg, dict):
        return split_cfg.get(mode)
    if isinstance(split_cfg, int):
        return split_cfg
    return None


def resolve_download_images(
    entry: Dict[str, Any],
    recipe: Dict[str, Any],
    override: bool,
) -> bool:
    if override:
        return True
    if "download_images" in entry:
        return bool(entry["download_images"])
    return bool(recipe.get("download_images_default", False))


def maybe_shuffle(dataset: Any, seed: int, max_samples: Optional[int]) -> Any:
    if max_samples is None:
        return dataset
    try:
        return dataset.shuffle(seed=seed)
    except Exception:
        return dataset


def shuffle_dataset(dataset: Any, seed: int) -> Any:
    try:
        return dataset.shuffle(seed=seed)
    except Exception:
        return dataset


def export_split(
    adapter: Any,
    dataset: Any,
    split: str,
    output_dir: Path,
    download_images: bool,
    max_samples: Optional[int],
    export_parquet_flag: bool,
) -> int:
    try:
        total = len(dataset)
        if max_samples is not None:
            total = min(total, max_samples)
    except Exception:
        total = None

    image_dir = Path("data/images") / adapter.name
    jsonl_path = output_dir / f"{adapter.name}_{split}.jsonl"
    count = export_jsonl(
        iter_records(adapter, dataset, split, image_dir, download_images, max_samples),
        jsonl_path,
        total=total,
    )

    if export_parquet_flag:
        parquet_path = output_dir / f"{adapter.name}_{split}.parquet"
        export_parquet(
            iter_records(adapter, dataset, split, image_dir, download_images, max_samples),
            parquet_path,
            total=total,
        )
    return count


def export_streaming_train_val(
    adapter: Any,
    dataset: Any,
    val_split_name: str,
    output_dir: Path,
    download_images: bool,
    train_max: Optional[int],
    val_max: Optional[int],
    val_ratio: float,
    seed: int,
    export_parquet_flag: bool,
) -> tuple[int, int]:
    if export_parquet_flag:
        logger.warning("Streaming split does not support parquet export; skipping parquet.")
    image_dir = Path("data/images") / adapter.name
    train_path = output_dir / f"{adapter.name}_train.jsonl"
    val_path = output_dir / f"{adapter.name}_{val_split_name}.jsonl"
    rng = random.Random(seed)
    train_count = 0
    val_count = 0

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open(
        "w", encoding="utf-8"
    ) as val_handle:
        for ex in dataset:
            if train_max is not None and val_max is not None:
                if train_count >= train_max and val_count >= val_max:
                    break

            pick_val = rng.random() < val_ratio
            target = "val" if pick_val else "train"

            if target == "val" and val_max is not None and val_count >= val_max:
                target = "train"
            if target == "train" and train_max is not None and train_count >= train_max:
                target = "val"
            if target == "val" and val_max is not None and val_count >= val_max:
                continue
            if target == "train" and train_max is not None and train_count >= train_max:
                continue

            split_name = val_split_name if target == "val" else "train"
            try:
                unified = adapter.normalize_example(ex, split=split_name)
                image_info = build_image_entry(
                    adapter, ex, image_dir, download_images, unified.sample_id
                )
                unified.image = image_info
                if target == "val":
                    val_handle.write(json.dumps(unified.to_dict()) + "\n")
                    val_count += 1
                else:
                    train_handle.write(json.dumps(unified.to_dict()) + "\n")
                    train_count += 1
            except Exception as exc:  # pragma: no cover - runtime data issues
                logger.warning("Failed to normalize sample: %s", exc)
                continue

    return train_count, val_count


def split_train_val(dataset: Any, val_ratio: float, seed: int) -> tuple[Any, Any]:
    if hasattr(dataset, "train_test_split"):
        split = dataset.train_test_split(test_size=val_ratio, seed=seed, shuffle=True)
        return split["train"], split["test"]
    try:
        dataset = dataset.shuffle(seed=seed)
        total = len(dataset)
        val_size = max(1, int(total * val_ratio))
        val = dataset.select(range(val_size))
        train = dataset.select(range(val_size, total))
        return train, val
    except Exception as exc:
        raise RuntimeError(f"Failed to split train/val: {exc}") from exc


def prepare_entry(
    entry: Dict[str, Any],
    recipe: Dict[str, Any],
    mode: str,
    override_max: Optional[int],
    download_images_override: bool,
    export_parquet_flag: bool,
) -> List[ManifestEntry]:
    name = entry["name"]
    adapter = get_adapter(name)
    subset = entry.get("subset")
    splits = entry.get("splits", [])
    val_from_train = bool(entry.get("val_from_train", False))
    val_ratio = float(entry.get("val_ratio", recipe.get("val_ratio_default", 0.03)))
    seed = int(entry.get("seed", recipe.get("seed", 42)))
    download_images = resolve_download_images(entry, recipe, download_images_override)
    streaming = bool(entry.get("streaming", False))

    output_root = Path(recipe.get("output_root", "data/processed"))
    output_dir = output_root / adapter.name
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[ManifestEntry] = []
    timestamp = datetime.utcnow().isoformat()
    schema_version = str(recipe.get("schema_version", "v1"))

    if val_from_train:
        if "train" not in splits:
            raise ValueError(f"{name} val_from_train requires 'train' in splits")
        val_split_name = next((s for s in splits if s != "train"), None)
        if val_split_name is None:
            raise ValueError(f"{name} val_from_train requires a val split name")

        logger.info("Loading %s split=train (val_from_train)", name)
        dataset = adapter.load(subset=subset, split="train", streaming=streaming)
        if streaming or not hasattr(dataset, "train_test_split"):
            train_max = resolve_max_samples(entry, "train", mode, override_max)
            val_max = resolve_max_samples(entry, val_split_name, mode, override_max)
            train_count, val_count = export_streaming_train_val(
                adapter=adapter,
                dataset=dataset,
                val_split_name=val_split_name,
                output_dir=output_dir,
                download_images=download_images,
                train_max=train_max,
                val_max=val_max,
                val_ratio=val_ratio,
                seed=seed,
                export_parquet_flag=export_parquet_flag,
            )
            manifest.append(
                ManifestEntry(
                    dataset=adapter.name,
                    split="train",
                    num_examples=train_count,
                    schema_version=schema_version,
                    timestamp=timestamp,
                )
            )
            manifest.append(
                ManifestEntry(
                    dataset=adapter.name,
                    split=val_split_name,
                    num_examples=val_count,
                    schema_version=schema_version,
                    timestamp=timestamp,
                )
            )
            return manifest

        dataset = shuffle_dataset(dataset, seed=seed)
        train_ds, val_ds = split_train_val(dataset, val_ratio=val_ratio, seed=seed)

        train_max = resolve_max_samples(entry, "train", mode, override_max)
        train_count = export_split(
            adapter,
            train_ds,
            "train",
            output_dir,
            download_images,
            train_max,
            export_parquet_flag,
        )
        manifest.append(
            ManifestEntry(
                dataset=adapter.name,
                split="train",
                num_examples=train_count,
                schema_version=schema_version,
                timestamp=timestamp,
            )
        )

        val_max = resolve_max_samples(entry, val_split_name, mode, override_max)
        val_count = export_split(
            adapter,
            val_ds,
            val_split_name,
            output_dir,
            download_images,
            val_max,
            export_parquet_flag,
        )
        manifest.append(
            ManifestEntry(
                dataset=adapter.name,
                split=val_split_name,
                num_examples=val_count,
                schema_version=schema_version,
                timestamp=timestamp,
            )
        )

        for split in splits:
            if split in ("train", val_split_name):
                continue
            logger.info("Loading %s split=%s", name, split)
            dataset = adapter.load(subset=subset, split=split, streaming=streaming)
            split_max = resolve_max_samples(entry, split, mode, override_max)
            dataset = maybe_shuffle(dataset, seed=seed, max_samples=split_max)
            count = export_split(
                adapter,
                dataset,
                split,
                output_dir,
                download_images,
                split_max,
                export_parquet_flag,
            )
            manifest.append(
                ManifestEntry(
                    dataset=adapter.name,
                    split=split,
                    num_examples=count,
                    schema_version=schema_version,
                    timestamp=timestamp,
                )
            )
        return manifest

    for split in splits:
        logger.info("Loading %s split=%s", name, split)
        dataset = adapter.load(subset=subset, split=split, streaming=streaming)
        split_max = resolve_max_samples(entry, split, mode, override_max)
        dataset = maybe_shuffle(dataset, seed=seed, max_samples=split_max)
        count = export_split(
            adapter,
            dataset,
            split,
            output_dir,
            download_images,
            split_max,
            export_parquet_flag,
        )
        manifest.append(
            ManifestEntry(
                dataset=adapter.name,
                split=split,
                num_examples=count,
                schema_version=schema_version,
                timestamp=timestamp,
            )
        )
    return manifest


def main() -> None:
    args = parse_args()
    recipe_path = Path(args.recipe)
    recipe = load_recipe(recipe_path)
    mode = resolve_mode(recipe, args.mode)

    output_root = Path(recipe.get("output_root", "data/processed"))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"

    all_entries: List[Dict[str, Any]] = []
    for entry in recipe.get("datasets", []):
        modes = entry.get("modes")
        if modes and mode not in modes:
            continue
        try:
            entries = prepare_entry(
                entry=entry,
                recipe=recipe,
                mode=mode,
                override_max=args.override_max_samples,
                download_images_override=args.download_images,
                export_parquet_flag=args.export_parquet,
            )
            all_entries.extend([item.to_dict() for item in entries])
        except Exception as exc:
            logger.warning("Failed to prepare %s: %s", entry.get("name"), exc)
            continue

    manifest = {
        "schema_version": recipe.get("schema_version", "v1"),
        "mode": mode,
        "generated_at": datetime.utcnow().isoformat(),
        "entries": all_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
