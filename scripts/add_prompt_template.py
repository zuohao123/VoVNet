import argparse
import json
from pathlib import Path

MC_PROMPT = (
    "Question: {question}\nContext: {context}\nOptions:\n{choices}\nAnswer (letter only):"
)
TEXT_PROMPT = "Question: {question}\nAnswer:"


def _is_mc(example: dict) -> bool:
    choices = example.get("choices") or []
    if not isinstance(choices, list):
        return False
    valid = [c for c in choices if isinstance(c, str) and c.strip()]
    return len(valid) >= 2


def add_prompt(in_path: Path, out_path: Path, overwrite: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)
            if not overwrite and ex.get("prompt_template"):
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                continue
            if _is_mc(ex):
                ex.setdefault("context", ex.get("context") or "")
                ex["prompt_template"] = MC_PROMPT
            else:
                ex["prompt_template"] = TEXT_PROMPT
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="source jsonl")
    ap.add_argument("--output", required=True, help="output jsonl")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing prompt_template")
    args = ap.parse_args()
    add_prompt(Path(args.input), Path(args.output), args.overwrite)


if __name__ == "__main__":
    main()
