# Value-of-Vision (VoVNet)

Research repository for cost-aware, uncertainty-aware vision calling on top of Qwen3-VL-8B.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

JSONL entries (answer can be a string or an object with `text`):

```json
{"image": "path/to/image.jpg", "question": "...", "answer": "...", "id": "123", "meta": {"source": "toy"}}
```

Fields are configurable in `configs/*.yaml`.

## Training (ZeRO-2 / ZeRO-3)

1) Edit configs:
- `configs/base.yaml` for defaults
- `configs/train_mmbench.yaml` for overrides

2) Launch training:

```bash
python scripts/train.py --config configs/base.yaml --config configs/train_mmbench.yaml
```

DeepSpeed configs are auto-generated into `outputs/deepspeed_config.json`.
Training runs also write reproducibility metadata to:
- `outputs/run_metadata_train.json`
- `outputs/configs/` (copied config files)

For multi-GPU runs, use accelerate:

```bash
accelerate launch --num_processes 8 scripts/train.py --config configs/base.yaml --config configs/train_mmbench.yaml
```

## Running baselines

```bash
python scripts/run_baselines.py --config configs/base.yaml --config configs/eval.yaml
```

Outputs: `outputs/baselines.json`

## Evaluation & Pareto

```bash
python scripts/eval.py --config configs/base.yaml --config configs/eval.yaml
python scripts/reproduce_paper.py --config configs/base.yaml --config configs/eval.yaml
```

Pareto sweep outputs: `outputs/repro/pareto.json`
Matrix evaluation outputs:
- `outputs/summary.json`
- `outputs/results.csv`
- `outputs/eval_matrix.json`
- `outputs/eval_matrix.csv`
- `outputs/run_metadata_eval.json`

## Latency profiling

```bash
python scripts/profile_latency.py --config configs/base.yaml --config configs/eval.yaml
```

Outputs: `outputs/latency.json`

## Reproduce paper results

One command for the main table + Pareto curve:

```bash
python scripts/reproduce_paper.py --config configs/base.yaml --config configs/eval.yaml
```

Outputs:
- `outputs/repro/vovnet.json`
- `outputs/repro/baselines.json`
- `outputs/repro/pareto.json`
- `outputs/repro/tables.csv`

## Dataset preparation

One-click download + prepare (MMBench/MMBench-Lite/MMMU/TextVQA):

```bash
python scripts/download_and_prepare.py --lang en
```

Outputs go to `data/processed/<dataset>/` with optional images in `data/images/<dataset>/`.
MMBench-Lite defaults to HF id `lmms-lab/MMBench-Lite`. If you need a direct TSV URL, pass
`--mmbench_lite_url <MMBENCH_LITE_URL>` or set `VOVNET_MMBENCH_LITE_URL`.

HF-only mode (MMBench from HF; optionally override MMBench-Lite HF id or URL):

```bash
python scripts/download_and_prepare.py \
  --prepare_only \
  --mmbench_hf_id lmms-lab/MMBench_EN \
  --mmbench_splits dev,test \
  --mmbench_train_split dev
```

## Recommended dataset recipe

The project includes a recipe file at `configs/data_recipe.yaml` that specifies:
- train vs eval datasets
- splits and max sample counts
- fast_dev vs paper mode

Run fast dev:

```bash
python scripts/prepare_all.py --mode fast_dev
```

Run paper mode:

```bash
python scripts/prepare_all.py --mode paper
```

Outputs and a manifest are written to `data/processed/manifest.json`.

## Notes

- The code uses LoRA by default and freezes the vision encoder unless configured otherwise.
- Qwen3-VL loaders use conservative fallbacks with `trust_remote_code` configurable.
- The vision budget controller resizes images for coarse vs full fidelity.
- For toy data, create a small JSONL and run training with a single epoch.
