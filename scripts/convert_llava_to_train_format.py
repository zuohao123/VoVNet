"""Convert LLaVA-Instruct raw.jsonl to VoVNet JSONL schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.common import AnswerInfo, ImageInfo, MetaInfo, UnifiedExample
from datasets.image_utils import sha1_path
from src.utils.io import write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("convert_llava")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LLaVA-Instruct to JSONL")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _strip_image_tokens(text: str) -> str:
    tokens = ["<image>", "<image_start>", "<image_end>", "<image>\n", "<image>\r\n"]
    out = text
    for tok in tokens:
        out = out.replace(tok, " ")
    return " ".join(out.split())


def _extract_pairs(conversations: Iterable[Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    last_user: Optional[str] = None
    for msg in conversations:
        role = (msg.get("from") or msg.get("role") or msg.get("speaker") or "").lower()
        text = _clean_text(msg.get("value") or msg.get("content") or msg.get("text"))
        if not text:
            continue
        if role in {"human", "user", "instruction"}:
            last_user = text
        elif role in {"assistant", "gpt", "bot"}:
            if last_user is not None:
                pairs.append((last_user, text))
                last_user = None
    return pairs


def _stable_hash(*parts: str) -> str:
    payload = "||".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_image_info(path: Optional[str], url: Optional[str]) -> Optional[ImageInfo]:
    if path is None and url is None:
        return None
    sha1 = None
    if path:
        p = Path(path)
        if p.exists():
            sha1 = sha1_path(p)
    return ImageInfo(source="local" if path else "url", path=path, url=url, sha1=sha1)


def _iter_raw(path: Path, max_samples: Optional[int]) -> Iterable[Dict[str, Any]]:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if max_samples is not None and count >= max_samples:
                break


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = input_dir / "raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw.jsonl not found: {raw_path}")

    prompt_template = "Question: {question}\nAnswer:"
    output_path = output_dir / "converted_train.jsonl"
    manifest_path = output_dir / "manifest.json"

    total_raw = 0
    total_pairs = 0
    total_written = 0
    missing_images = 0
    total_q_len = 0
    total_a_len = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for ex in _iter_raw(raw_path, args.max_samples):
            total_raw += 1
            conversations = ex.get("conversations") or ex.get("messages") or []
            pairs = _extract_pairs(conversations)
            if not pairs:
                question = _clean_text(ex.get("question") or ex.get("prompt") or "")
                answer = _clean_text(ex.get("answer") or ex.get("output") or "")
                if question and answer:
                    pairs = [(question, answer)]
            for idx, (question, answer) in enumerate(pairs):
                total_pairs += 1
                q = _strip_image_tokens(question)
                a = _clean_text(answer)

                image_path = ex.get("image_path")
                image_url = ex.get("image_url")
                image_info = _build_image_info(image_path, image_url)
                if image_info is None:
                    missing_images += 1

                base_id = _stable_hash(str(image_path or ""), q, a)[:16]
                extra = {
                    "source_id": ex.get("id"),
                    "conversation_index": idx,
                    "modality_flags": {"has_image": image_info is not None},
                }
                meta = MetaInfo(
                    language=None,
                    source_fields={
                        "question": "conversations",
                        "answers": "conversations",
                        "image": "image",
                    },
                    extra=extra,
                )
                answer_info = AnswerInfo(text=a, aliases=None, label=None, raw=[a])

                vision_ex = UnifiedExample(
                    sample_id=f"{base_id}_vision",
                    dataset="llava_instruct",
                    split=ex.get("split") or "train",
                    task_type="instruction_vqa",
                    image=image_info,
                    question=q,
                    context=None,
                    choices=None,
                    answer=answer_info,
                    meta=meta,
                ).to_dict()
                vision_ex["prompt_template"] = prompt_template
                out_f.write(json.dumps(vision_ex) + "\n")

                text_only_q = f"[NO_IMAGE] {q}".strip()
                text_meta = MetaInfo(
                    language=None,
                    source_fields=meta.source_fields,
                    extra={
                        **extra,
                        "modality_flags": {"has_image": False},
                    },
                )
                text_ex = UnifiedExample(
                    sample_id=f"{base_id}_text",
                    dataset="llava_instruct",
                    split=ex.get("split") or "train",
                    task_type="instruction_vqa",
                    image=None,
                    question=text_only_q,
                    context=None,
                    choices=None,
                    answer=answer_info,
                    meta=text_meta,
                ).to_dict()
                text_ex["prompt_template"] = prompt_template
                out_f.write(json.dumps(text_ex) + "\n")

                total_q_len += len(q)
                total_a_len += len(a)
                total_written += 2
    avg_q = total_q_len / max(total_pairs, 1)
    avg_a = total_a_len / max(total_pairs, 1)
    manifest = {
        "raw_samples": total_raw,
        "conversation_pairs": total_pairs,
        "converted_samples": total_written,
        "vision_samples": total_pairs,
        "text_samples": total_pairs,
        "missing_images": missing_images,
        "avg_question_length": round(avg_q, 2),
        "avg_answer_length": round(avg_a, 2),
        "output_path": str(output_path),
    }
    write_json(manifest_path, manifest)
    logger.info("Wrote converted jsonl: %s", output_path)
    logger.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
