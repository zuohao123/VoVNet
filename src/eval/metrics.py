"""Evaluation metrics."""
from __future__ import annotations

import re
from typing import Iterable, List, Optional


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~]")
_CHOICE_LETTERS = "abcdefgh"


def normalize_text(text: str) -> str:
    """Basic normalization for exact match."""
    return " ".join(str(text).lower().strip().split())


def _normalize_vqa(text: str) -> str:
    text = normalize_text(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def _coerce_ref_text(ref: object) -> str:
    if isinstance(ref, dict):
        for key in ("text", "answer", "label"):
            value = ref.get(key)
            if value not in (None, ""):
                return str(value)
        aliases = ref.get("aliases")
        if aliases:
            return str(aliases[0])
        raw = ref.get("raw")
        if isinstance(raw, list) and raw:
            item = raw[0]
            if isinstance(item, dict):
                return str(item.get("answer") or item.get("text") or item)
            return str(item)
        return ""
    if isinstance(ref, list):
        return str(ref[0]) if ref else ""
    if ref is None:
        return ""
    return str(ref)


def exact_match_score(preds: Iterable[str], refs: Iterable[object]) -> float:
    """Compute exact match accuracy."""
    preds = list(preds)
    refs = list(refs)
    if not preds:
        return 0.0
    correct = 0
    for pred, ref in zip(preds, refs):
        if normalize_text(pred) == normalize_text(_coerce_ref_text(ref)):
            correct += 1
    return correct / len(preds)


def _label_to_letter(label: object) -> Optional[str]:
    if label is None:
        return None
    if isinstance(label, int):
        if 0 <= label < len(_CHOICE_LETTERS):
            return _CHOICE_LETTERS[label]
        return None
    label_str = normalize_text(str(label))
    if label_str.isdigit():
        idx = int(label_str) - 1
        if 0 <= idx < len(_CHOICE_LETTERS):
            return _CHOICE_LETTERS[idx]
    if len(label_str) == 1 and label_str in _CHOICE_LETTERS:
        return label_str
    return None


def _extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    normalized = normalize_text(text)
    token = normalized.split()[0] if normalized else ""
    token = token.strip("()[]{}.,:;")
    if len(token) == 1 and token in _CHOICE_LETTERS:
        return token
    if token.isdigit():
        return _label_to_letter(token)
    match = re.search(r"\b([a-h])\b", normalized)
    if match:
        return match.group(1)
    return None


def _extract_answer_list(ref: object) -> List[str]:
    answers: List[str] = []
    if isinstance(ref, dict):
        raw = ref.get("raw")
        if isinstance(raw, list):
            for item in raw:
                if item is None:
                    continue
                if isinstance(item, dict):
                    ans = item.get("answer") or item.get("text") or item.get("value")
                else:
                    ans = item
                if ans not in (None, ""):
                    answers.append(str(ans))
            if answers:
                return answers
        aliases = ref.get("aliases")
        if isinstance(aliases, list) and aliases:
            return [str(a) for a in aliases if a not in (None, "")]
        text = ref.get("text")
        if text not in (None, ""):
            return [str(text)]
        return []
    if isinstance(ref, list):
        return [str(item) for item in ref if item not in (None, "")]
    if ref is None:
        return []
    return [str(ref)]


def multi_choice_accuracy(preds: Iterable[str], refs: Iterable[object]) -> float:
    preds = list(preds)
    refs = list(refs)
    if not preds:
        return 0.0
    correct = 0
    for pred, ref in zip(preds, refs):
        pred_norm = normalize_text(pred)
        pred_letter = _extract_choice_letter(pred)
        candidates = _extract_answer_list(ref)
        normalized = {normalize_text(item) for item in candidates if item not in (None, "")}
        label_letter = None
        if isinstance(ref, dict):
            label_letter = _label_to_letter(ref.get("label"))
        candidate_letters = {label_letter} if label_letter else set()
        for cand in normalized:
            letter = _extract_choice_letter(cand)
            if letter:
                candidate_letters.add(letter)
        if pred_letter and pred_letter in candidate_letters:
            correct += 1
            continue
        if pred_norm in normalized:
            correct += 1
            continue
    return correct / len(preds)


def vqa_accuracy_score(preds: Iterable[str], refs: Iterable[object]) -> float:
    preds = list(preds)
    refs = list(refs)
    if not preds:
        return 0.0
    total = 0.0
    for pred, ref in zip(preds, refs):
        answers = _extract_answer_list(ref)
        if not answers:
            continue
        pred_norm = _normalize_vqa(pred)
        match = sum(1 for ans in answers if _normalize_vqa(ans) == pred_norm)
        total += min(match / 3.0, 1.0)
    return total / len(preds)
