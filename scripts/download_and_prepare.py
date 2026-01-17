"""Download and prepare core datasets for VoVNet."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.registry import get_adapter
from scripts.prepare_dataset import prepare_split

logger = logging.getLogger("download_and_prepare")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

MMBENCH_URLS = {
    "en": {
        "train": "http://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv",
        "test": "http://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv",
    },
    "cn": {
        "train": "http://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN.tsv",
        "test": "http://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN.tsv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--raw_root", default="data/raw", help="Raw dataset root")
    parser.add_argument("--output_root", default="data/processed", help="Processed dataset root")
    parser.add_argument("--lang", choices=["en", "cn"], default="en")
    parser.add_argument("--mmbench_splits", default="train,test")
    parser.add_argument("--mmbench_lite_splits", default="train")
    parser.add_argument("--mmmu_splits", default="validation")
    parser.add_argument("--textvqa_splits", default="validation")
    parser.add_argument(
        "--mmbench_hf_id",
        default=os.environ.get("VOVNET_HF_DATASET_ID_MMBENCH"),
        help="Use HF dataset id for MMBench instead of direct TSV download",
    )
    parser.add_argument(
        "--mmbench_lite_hf_id",
        default=os.environ.get("VOVNET_HF_DATASET_ID_MMBENCH_LITE"),
        help="Use HF dataset id for MMBench-Lite instead of direct TSV download",
    )
    parser.add_argument(
        "--mmbench_dev_url",
        default=None,
        help="Override MMBench dev TSV URL or local path",
    )
    parser.add_argument(
        "--mmbench_test_url",
        default=None,
        help="Override MMBench test TSV URL or local path",
    )
    parser.add_argument(
        "--mmbench_lite_url",
        default=os.environ.get("VOVNET_MMBENCH_LITE_URL"),
        help="MMBench-Lite TSV URL or local path",
    )
    parser.add_argument("--skip_mmbench_lite", action="store_true")
    parser.add_argument(
        "--mmmu_hf_id",
        default=os.environ.get("VOVNET_HF_DATASET_ID_MMMU", "MMMU/MMMU"),
    )
    parser.add_argument(
        "--textvqa_hf_id",
        default=os.environ.get("VOVNET_HF_DATASET_ID_TEXTVQA", "lmms-lab/textvqa"),
    )
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--download_images", dest="download_images", action="store_true")
    parser.add_argument("--no_download_images", dest="download_images", action="store_false")
    parser.set_defaults(download_images=True)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--export_parquet", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge_train", dest="merge_train", action="store_true")
    parser.add_argument("--no_merge_train", dest="merge_train", action="store_false")
    parser.set_defaults(merge_train=True)
    parser.add_argument(
        "--mmbench_train_split",
        default="train",
        help="Which MMBench split to treat as training when merging",
    )
    parser.add_argument(
        "--mmbench_lite_train_split",
        default="train",
        help="Which MMBench-Lite split to treat as training when merging",
    )
    parser.add_argument(
        "--merged_train_path",
        default=None,
    )
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def download_file(source: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Skip download (exists): %s", dest)
        return dest
    if _is_url(source):
        logger.info("Downloading %s -> %s", source, dest)
        request = urllib.request.Request(source, headers={"User-Agent": "VoVNet"})
        with urllib.request.urlopen(request) as response, dest.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return dest
    src_path = Path(source).expanduser()
    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    logger.info("Copying %s -> %s", src_path, dest)
    shutil.copyfile(src_path, dest)
    return dest


def download_mmbench(raw_root: Path, lang: str, dev_url: Optional[str], test_url: Optional[str]) -> None:
    urls = MMBENCH_URLS.get(lang, {})
    dev = dev_url or urls.get("train")
    test = test_url or urls.get("test")
    if not dev or not test:
        raise ValueError(f"Missing MMBench URLs for lang={lang}")
    mmbench_root = raw_root / "mmbench"
    download_file(dev, mmbench_root / Path(dev).name)
    download_file(test, mmbench_root / Path(test).name)


def download_mmbench_lite(raw_root: Path, url: str) -> None:
    mmbench_root = raw_root / "mmbench_lite"
    download_file(url, mmbench_root / Path(url).name)


def prepare_dataset(
    name: str,
    splits: Iterable[str],
    subset: Optional[str],
    output_root: Path,
    download_images: bool,
    seed: int,
    export_parquet: bool,
    streaming: bool,
) -> None:
    adapter = get_adapter(name)
    output_dir = output_root / adapter.name
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        prepare_split(
            adapter=adapter,
            split=split,
            subset=subset,
            output_dir=output_dir,
            download_images=download_images,
            max_samples=None,
            seed=seed,
            export_parquet_flag=export_parquet,
            streaming=streaming,
        )


def merge_jsonl(paths: Iterable[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_handle:
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing JSONL: {path}")
            with path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    line = line.strip()
                    if line:
                        out_handle.write(line + "\n")
    logger.info("Merged train JSONL: %s", output_path)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    mmbench_splits = parse_csv(args.mmbench_splits)
    mmbench_lite_splits = parse_csv(args.mmbench_lite_splits)
    mmmu_splits = parse_csv(args.mmmu_splits)
    textvqa_splits = parse_csv(args.textvqa_splits)

    if args.mmbench_hf_id:
        os.environ["VOVNET_HF_DATASET_ID_MMBENCH"] = args.mmbench_hf_id
    if args.mmbench_lite_hf_id:
        os.environ["VOVNET_HF_DATASET_ID_MMBENCH_LITE"] = args.mmbench_lite_hf_id

    if not args.prepare_only:
        if not args.mmbench_hf_id:
            download_mmbench(raw_root, args.lang, args.mmbench_dev_url, args.mmbench_test_url)
        if not args.skip_mmbench_lite and not args.mmbench_lite_hf_id:
            if not args.mmbench_lite_url:
                raise ValueError(
                    "MMBench-Lite URL missing. Set --mmbench_lite_url or VOVNET_MMBENCH_LITE_URL."
                )
            download_mmbench_lite(raw_root, args.mmbench_lite_url)

    if args.download_only:
        return

    if args.mmmu_hf_id:
        os.environ["VOVNET_HF_DATASET_ID_MMMU"] = args.mmmu_hf_id
    if args.textvqa_hf_id:
        os.environ["VOVNET_HF_DATASET_ID_TEXTVQA"] = args.textvqa_hf_id

    prepare_dataset(
        name="mmbench",
        splits=mmbench_splits,
        subset=args.lang,
        output_root=output_root,
        download_images=args.download_images,
        seed=args.seed,
        export_parquet=args.export_parquet,
        streaming=args.streaming,
    )
    if not args.skip_mmbench_lite:
        prepare_dataset(
            name="mmbench_lite",
            splits=mmbench_lite_splits,
            subset=args.lang,
            output_root=output_root,
            download_images=args.download_images,
            seed=args.seed,
            export_parquet=args.export_parquet,
            streaming=args.streaming,
        )
    prepare_dataset(
        name="mmmu",
        splits=mmmu_splits,
        subset=None,
        output_root=output_root,
        download_images=args.download_images,
        seed=args.seed,
        export_parquet=args.export_parquet,
        streaming=args.streaming,
    )
    prepare_dataset(
        name="textvqa",
        splits=textvqa_splits,
        subset=None,
        output_root=output_root,
        download_images=args.download_images,
        seed=args.seed,
        export_parquet=args.export_parquet,
        streaming=args.streaming,
    )

    if args.merge_train:
        merged_path = (
            Path(args.merged_train_path)
            if args.merged_train_path
            else output_root / "mmbench_core" / "mmbench_core_train.jsonl"
        )
        train_paths = [
            output_root / "mmbench" / f"mmbench_{args.mmbench_train_split}.jsonl"
        ]
        if not args.skip_mmbench_lite:
            train_paths.append(
                output_root
                / "mmbench_lite"
                / f"mmbench_lite_{args.mmbench_lite_train_split}.jsonl"
            )
        merge_jsonl(train_paths, merged_path)


if __name__ == "__main__":
    main()
