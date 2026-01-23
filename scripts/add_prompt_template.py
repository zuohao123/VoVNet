import argparse
import json
from pathlib import Path

MC_PROMPT = (
    "Question: {question}\nContext: {context}\nOptions:\n{choices}\nAnswer (letter only):"
)
TEXT_PROMPT = "Question: {question}\nAnswer:"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _clean_choices(choices):
    out = []
    for c in choices or []:
        if not isinstance(c, str):
            continue
        c_strip = c.strip()
        if not c_strip or c_strip.lower() == "nan":
            continue
        out.append(c_strip)
    return out


def _is_mc(example: dict) -> bool:
    choices = _clean_choices(example.get("choices") or [])
    return len(choices) >= 2


def _label_to_letter(label):
    if label is None:
        return None
    if isinstance(label, int):
        if 0 <= label < len(_LETTERS):
            return _LETTERS[label]
        if 1 <= label <= len(_LETTERS):
            return _LETTERS[label - 1]
        return None
    label_str = str(label).strip()
    if label_str.isdigit():
        idx = int(label_str)
        if 0 <= idx < len(_LETTERS):
            return _LETTERS[idx]
        if 1 <= idx <= len(_LETTERS):
            return _LETTERS[idx - 1]
    if len(label_str) == 1 and label_str.upper() in _LETTERS:
        return label_str.upper()
    return None


def _answer_to_letter(answer, choices):
    if isinstance(answer, dict):
        letter = _label_to_letter(answer.get("label"))
        if letter:
            return letter
        raw = answer.get("raw")
        if isinstance(raw, str):
            raw_letter = raw.strip().upper()
            if len(raw_letter) == 1 and raw_letter in _LETTERS:
                return raw_letter
        text = answer.get("text")
        if isinstance(text, str):
            answer = text
        elif isinstance(raw, str):
            answer = raw
    if isinstance(answer, list):
        answer = next((item for item in answer if isinstance(item, str)), "")
    if isinstance(answer, str):
        ans = answer.strip()
        if len(ans) == 1 and ans.upper() in _LETTERS:
            return ans.upper()
        for idx, choice in enumerate(choices):
            if ans.lower() == choice.lower():
                return _LETTERS[idx]
    return None

def add_prompt(
    in_path: Path,
    out_path: Path,
    overwrite: bool = False,
    normalize_answers: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)
            if not overwrite and ex.get("prompt_template"):
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                continue
            choices = _clean_choices(ex.get("choices"))
            ex["choices"] = choices or None
            if _is_mc(ex):
                ex.setdefault("context", ex.get("context") or "")
                if normalize_answers:
                    ex.setdefault("answer_info", ex.get("answer"))
                    letter = _answer_to_letter(ex.get("answer"), choices)
                    if letter:
                        ex["answer"] = letter
                ex["prompt_template"] = MC_PROMPT
            else:
                ex["prompt_template"] = TEXT_PROMPT
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="source jsonl")
    ap.add_argument("--output", required=True, help="output jsonl")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing prompt_template")
    ap.add_argument(
        "--normalize-answers",
        action="store_true",
        help="normalize MC answers to letters; keep raw in answer_info",
    )
    args = ap.parse_args()
    add_prompt(
        Path(args.input),
        Path(args.output),
        overwrite=args.overwrite,
        normalize_answers=args.normalize_answers,
    )


if __name__ == "__main__":
    main()
