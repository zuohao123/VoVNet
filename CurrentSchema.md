# Current Training JSONL Schema (VoVNet)

This repository reads training/eval data through:

- Dataset: `src/data/adapters/jsonl.py::JsonlVQADataset`
- Collator: `src/data/collate.py::VLMDataCollator`

The JSONL file is a list of per-line JSON objects. The collator builds the
prompt from `question/context/choices` and may insert the image token
automatically if the tokenizer defines one.

## Required fields

- `question` (str): user question or instruction.
- `answer` (str or dict): supervision target. If dict, common keys are
  `text`, `aliases`, `label`, `raw`.
- `image` (optional): either a path string or an object:
  - string: local path or filename (resolved relative to JSONL directory).
  - object: `{ "source": "local" | "hf" | "url", "path": "...", "url": "...", "sha1": "..." }`

## Optional fields (supported)

- `id` (str/int): sample id; default is row index.
- `dataset` (str): dataset name for logging; fallback to `meta.dataset`.
- `context` / `hint` / `rationale` (str): optional context.
- `choices` / `options` / `candidates` (list[str]): multiple-choice options.
- `prompt_template` (str): per-sample template. If absent, uses
  `cfg.data.prompt_template`.
- `answer_info` (dict): alternative answer container.
- `meta` (dict): free-form metadata. `meta.dataset` is used if `dataset` is missing.

## Example (single JSONL line)

```json
{
  "id": "demo_0001",
  "dataset": "demo",
  "split": "train",
  "task_type": "instruction_vqa",
  "image": {
    "source": "local",
    "path": "data/images/demo/0001.jpg",
    "url": null,
    "sha1": "8d4c2c4c0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f"
  },
  "question": "What is written on the sign?",
  "context": null,
  "choices": null,
  "answer": {
    "text": "open",
    "aliases": ["OPEN"],
    "label": null,
    "raw": ["open", "OPEN"]
  },
  "meta": {
    "language": "en",
    "source_fields": {
      "question": "question",
      "answers": "answers",
      "image": "image"
    },
    "extra": {
      "modality_flags": {
        "has_image": true
      }
    }
  },
  "prompt_template": "Question: {question}\\nAnswer:"
}
```
