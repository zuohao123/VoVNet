"""Download and normalize multimodal datasets into a unified schema."""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.common import ImageInfo
from datasets.exporters import export_jsonl, export_parquet
from datasets.image_utils import save_image, sha1_path
from datasets.registry import get_adapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("prepare_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multimodal datasets")
    parser.add_argument("--dataset", required=True, help="dataset name (e.g., mmbench)")
    parser.add_argument("--subset", default=None, help="HF subset/config name")
    parser.add_argument("--splits", default="train", help="comma-separated splits")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="output directory (default: data/processed/<dataset>)",
    )
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Drop samples when images cannot be resolved (only when --download-images is set)",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=1,
        help="Number of threads used to download/convert images",
    )
    parser.add_argument("--export-parquet", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def parse_splits(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def maybe_shuffle(dataset: Any, seed: int, max_samples: Optional[int]) -> Any:
    if max_samples is None:
        return dataset
    try:
        return dataset.shuffle(seed=seed)
    except Exception:
        return dataset


def build_image_entry(
    adapter: Any,
    ex: dict,
    image_dir: Path,
    download_images: bool,
    sample_id: str,
) -> Optional[ImageInfo]:
    if not download_images:
        url = None
        if isinstance(ex.get("image"), dict):
            url = ex.get("image", {}).get("url")
        url = url or ex.get("image_url") or ex.get("url")
        if ex.get("image") is None and url is None:
            return None
        return ImageInfo(source="hf", path=None, url=url, sha1=None)

    image = adapter.get_image(ex)
    if image is None:
        return None

    image_path = image_dir / f"{sample_id}.jpg"
    if image_path.exists():
        sha1 = sha1_path(image_path)
    else:
        sha1 = save_image(image, image_path)
    return ImageInfo(source="local", path=str(image_path), url=None, sha1=sha1)


def iter_records(
    adapter: Any,
    dataset: Any,
    split: str,
    image_dir: Path,
    download_images: bool,
    max_samples: Optional[int] = None,
    skip_missing_images: bool = False,
    stats: Optional[dict] = None,
    download_workers: int = 1,
) -> Iterable[dict]:
    count = 0

    def _process(ex: dict) -> tuple[Optional[dict], Optional[Exception], bool]:
        try:
            unified = adapter.normalize_example(ex, split=split)
            image_info = build_image_entry(
                adapter, ex, image_dir, download_images, unified.sample_id
            )
            if skip_missing_images and download_images and image_info is None:
                return None, None, True
            unified.image = image_info
            return unified.to_dict(), None, False
        except Exception as exc:  # pragma: no cover - runtime data issues
            return None, exc, False

    def _record_stats(missing: bool, kept: bool) -> None:
        if stats is None:
            return
        stats["seen"] = stats.get("seen", 0) + 1
        if missing:
            stats["skipped_missing_images"] = stats.get("skipped_missing_images", 0) + 1
        if kept:
            stats["kept"] = stats.get("kept", 0) + 1

    if download_workers <= 1:
        for ex in dataset:
            if max_samples is not None and count >= max_samples:
                break
            record, err, missing = _process(ex)
            if err is not None:
                logger.warning("Failed to normalize sample: %s", err)
                _record_stats(missing=False, kept=False)
                continue
            if missing:
                _record_stats(missing=True, kept=False)
                continue
            _record_stats(missing=False, kept=True)
            yield record
            count += 1
        return

    iterator = dataset
    if max_samples is not None:
        iterator = islice(iterator, max_samples)
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        for record, err, missing in executor.map(_process, iterator):
            if err is not None:
                logger.warning("Failed to normalize sample: %s", err)
                _record_stats(missing=False, kept=False)
                continue
            if missing:
                _record_stats(missing=True, kept=False)
                continue
            _record_stats(missing=False, kept=True)
            yield record


def prepare_split(
    adapter: Any,
    split: str,
    subset: Optional[str],
    output_dir: Path,
    download_images: bool,
    skip_missing_images: bool,
    download_workers: int,
    max_samples: Optional[int],
    seed: int,
    export_parquet_flag: bool,
    streaming: bool,
) -> None:
    logger.info(
        "Loading %s split=%s subset=%s streaming=%s", adapter.name, split, subset, streaming
    )
    dataset = adapter.load(subset=subset, split=split, streaming=streaming)
    dataset = maybe_shuffle(dataset, seed=seed, max_samples=max_samples)
    try:
        total = len(dataset)
        if max_samples is not None:
            total = min(total, max_samples)
    except Exception:
        total = None

    image_dir = Path("data/images") / adapter.name
    jsonl_path = output_dir / f"{adapter.name}_{split}.jsonl"
    stats: dict = {}
    record_iter = iter_records(
        adapter,
        dataset,
        split,
        image_dir,
        download_images,
        max_samples,
        skip_missing_images=skip_missing_images,
        stats=stats,
        download_workers=download_workers,
    )
    written = export_jsonl(
        record_iter,
        jsonl_path,
        total=total,
    )

    if export_parquet_flag:
        parquet_path = output_dir / f"{adapter.name}_{split}.parquet"
        export_parquet(
            iter_records(
                adapter,
                dataset,
                split,
                image_dir,
                download_images,
                max_samples,
                skip_missing_images=skip_missing_images,
                download_workers=download_workers,
            ),
            parquet_path,
            total=total,
        )

    skipped = stats.get("skipped_missing_images", 0)
    logger.info(
        "Saved split %s to %s (written=%s skipped_missing_images=%s)",
        split,
        jsonl_path,
        written,
        skipped,
    )


def main() -> None:
    args = parse_args()
    adapter = get_adapter(args.dataset)
    splits = parse_splits(args.splits)
    output_dir = Path(args.output_dir or f"data/processed/{adapter.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_proc != 1:
        logger.info("num_proc is not used in the current pipeline; processing is sequential")

    for split in splits:
        prepare_split(
            adapter=adapter,
            split=split,
            subset=args.subset,
            output_dir=output_dir,
            download_images=args.download_images,
            skip_missing_images=args.skip_missing_images,
            download_workers=args.download_workers,
            max_samples=args.max_samples,
            seed=args.seed,
            export_parquet_flag=args.export_parquet,
            streaming=args.streaming,
        )


if __name__ == "__main__":
    main()
