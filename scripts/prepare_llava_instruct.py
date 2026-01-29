"""Download LLaVA-Instruct-150K and cache images + raw annotations."""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.image_utils import load_image_from_url, save_image, sha1_path
from datasets.hf_utils import safe_load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("prepare_llava_instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LLaVA-Instruct-150K")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8, help="Download workers (threads).")
    parser.add_argument(
        "--max_inflight",
        type=int,
        default=None,
        help="Max in-flight downloads. Default = num_workers * 4.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=200,
        help="Flush output files every N samples.",
    )
    return parser.parse_args()


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _infer_coco_urls(filename: str) -> list[str]:
    name = filename.strip()
    if not name:
        return []
    if _is_url(name):
        return [name]
    if name.startswith("COCO_train2014_"):
        return [f"http://images.cocodataset.org/train2014/{name}"]
    if name.startswith("COCO_val2014_"):
        return [f"http://images.cocodataset.org/val2014/{name}"]
    if name.startswith("COCO_train2017_"):
        return [f"http://images.cocodataset.org/train2017/{name}"]
    if name.startswith("COCO_val2017_"):
        return [f"http://images.cocodataset.org/val2017/{name}"]
    if name.endswith(".jpg") or name.endswith(".png"):
        # Best-effort fallback: try COCO 2014/2017 splits.
        return [
            f"http://images.cocodataset.org/{split}/{name}"
            for split in ("train2014", "val2014", "train2017", "val2017")
        ]
    return []


def _image_from_any(value: Any) -> Tuple[Optional[Image.Image], list[str]]:
    """Return (image, url_candidates) from a value if possible."""
    if value is None:
        return None, []
    if isinstance(value, Image.Image):
        return value.convert("RGB"), []
    if isinstance(value, dict):
        url = value.get("url") or value.get("image_url")
        if url and _is_url(url):
            return None, [url]
        if value.get("bytes"):
            try:
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB"), []
            except Exception:
                return None, []
        path = value.get("path")
        if path:
            try:
                return Image.open(path).convert("RGB"), []
            except Exception:
                return None, []
    if isinstance(value, str):
        if _is_url(value):
            return None, [value]
        urls = _infer_coco_urls(value)
        if urls:
            return None, urls
    return None, []


def _save_image(
    image: Optional[Image.Image],
    url_candidates: list[str],
    image_dir: Path,
    sample_id: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Save image and return (path, url, sha1)."""
    url_used = None
    if image is None and url_candidates:
        for url in url_candidates:
            image = load_image_from_url(url)
            if image is not None:
                url_used = url
                break
    if image is None:
        return None, url_used or (url_candidates[0] if url_candidates else None), None
    image_path = image_dir / f"{sample_id}.jpg"
    if image_path.exists():
        sha1 = sha1_path(image_path)
    else:
        sha1 = save_image(image, image_path)
    return str(image_path), url_used, sha1


def _process_example(
    idx: int, ex: Dict[str, Any], image_dir: Path
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    sample_id = str(ex.get("id") or ex.get("sample_id") or ex.get("image_id") or idx)
    image_obj = ex.get("image") or ex.get("image_data")
    image_url = ex.get("image_url") or ex.get("url")
    image, url_candidates = _image_from_any(image_obj)
    if image_url and _is_url(image_url):
        url_candidates = [image_url] + url_candidates
    if isinstance(image_obj, str):
        url_candidates += _infer_coco_urls(image_obj)
    if ex.get("image_path"):
        url_candidates += _infer_coco_urls(str(ex.get("image_path")))
    # De-duplicate while preserving order.
    seen_urls = set()
    url_candidates = [u for u in url_candidates if not (u in seen_urls or seen_urls.add(u))]

    image_path, image_url, sha1 = _save_image(image, url_candidates, image_dir, sample_id)

    bad_record = None
    if image_path is None:
        bad_record = {
            "id": sample_id,
            "reason": "missing_image",
            "image_field": ex.get("image"),
            "image_url": image_url,
        }

    record: Dict[str, Any] = {}
    for key, value in ex.items():
        if key == "image":
            continue
        record[key] = value
    record["id"] = sample_id
    record["image_path"] = image_path
    record["image_url"] = image_url
    record["image_sha1"] = sha1
    return record, bad_record


def _load_split(split: str, streaming: bool) -> Any:
    try:
        return safe_load_dataset(
            "liuhaotian/LLaVA-Instruct-150K", None, split=split, streaming=streaming
        )
    except Exception:
        dataset = safe_load_dataset(
            "liuhaotian/LLaVA-Instruct-150K", None, split="train", streaming=streaming
        )
        if isinstance(dataset, dict):
            if split in dataset:
                return dataset[split]
            return next(iter(dataset.values()))
        return dataset


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"
    bad_path = output_dir / "bad_samples.jsonl"

    try:
        dataset = _load_split(args.split, streaming=args.streaming)
    except Exception as exc:
        if not args.streaming:
            logger.warning("Non-streaming load failed: %s; retrying with streaming.", exc)
            dataset = _load_split(args.split, streaming=True)
        else:
            raise
    total = len(dataset) if hasattr(dataset, "__len__") else None
    logger.info("Loaded split=%s size=%s", args.split, total)

    seen = 0
    bad = 0
    max_inflight = args.max_inflight or max(1, args.num_workers * 4)
    with raw_path.open("w", encoding="utf-8") as raw_f, bad_path.open(
        "w", encoding="utf-8"
    ) as bad_f:
        if args.num_workers <= 1:
            for idx, ex in enumerate(dataset):
                if args.max_samples is not None and seen >= args.max_samples:
                    break
                record, bad_record = _process_example(idx, ex, image_dir)
                if bad_record is not None:
                    bad += 1
                    bad_f.write(json.dumps(bad_record) + "\n")
                raw_f.write(json.dumps(record) + "\n")
                seen += 1
                if args.flush_every and seen % args.flush_every == 0:
                    raw_f.flush()
                    bad_f.flush()
                    logger.info("Processed %d samples (bad=%d)", seen, bad)
        else:
            futures = set()
            submitted = 0
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                for idx, ex in enumerate(dataset):
                    if args.max_samples is not None and submitted >= args.max_samples:
                        break
                    futures.add(executor.submit(_process_example, idx, ex, image_dir))
                    submitted += 1
                    if len(futures) >= max_inflight:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for fut in done:
                            record, bad_record = fut.result()
                            if bad_record is not None:
                                bad += 1
                                bad_f.write(json.dumps(bad_record) + "\n")
                            raw_f.write(json.dumps(record) + "\n")
                            seen += 1
                            if args.flush_every and seen % args.flush_every == 0:
                                raw_f.flush()
                                bad_f.flush()
                                logger.info("Processed %d samples (bad=%d)", seen, bad)
                for fut in as_completed(futures):
                    record, bad_record = fut.result()
                    if bad_record is not None:
                        bad += 1
                        bad_f.write(json.dumps(bad_record) + "\n")
                    raw_f.write(json.dumps(record) + "\n")
                    seen += 1
                    if args.flush_every and seen % args.flush_every == 0:
                        raw_f.flush()
                        bad_f.flush()
                        logger.info("Processed %d samples (bad=%d)", seen, bad)

    logger.info("Wrote raw jsonl: %s (seen=%d, bad=%d)", raw_path, seen, bad)


if __name__ == "__main__":
    main()
