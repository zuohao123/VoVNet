# VoVNet Experiments Pipeline (Top-Conference Ready)

This document is a step-by-step experiment pipeline for VoVNet (Value-of-Vision). It is organized by priority: **must-do -> bonus -> optional**. Every block includes purpose, config, command, outputs, and sanity checks.

All experiments **must use the same evaluation pipeline** and **unified settings** (prompt template, decoding hyperparams, vision budget, cost definition).

---

## 0. Global Rules (Always Use These)

**Unified evaluation pipeline**
- Use `scripts/eval.py` (matrix runner) or wrapper sweep scripts that call the same evaluation core (`src/eval/matrix.evaluate_dataset`).
- All baselines must use the same prompt template and decoding settings from config.
- `--dataset_config` supports preset names to avoid temp YAML files:
  - `mmbench`, `mmmu`, `textvqa` (comma lists allowed, e.g. `"mmbench,mmmu,textvqa"`).

**Background execution**
- Create log directory once: `mkdir -p logs`
- For any training/eval command: use `nohup <cmd> > logs/<name>.out 2>&1 &`
- If a block contains multiple background commands, run them one-by-one to avoid GPU contention.

**Unified settings (do not change between experiments)**
- Prompt template: `data.prompt_template` in config.
- Decoding: `eval.max_new_tokens / num_beams / do_sample / temperature`.
- Vision budget: `vision_budget.*` (coarse/full long side, max pixels, patch size).
- Cost definition: `expected_cost` computed from **vision token counts**.  
  For pruning baseline, cost is computed from **remaining vision tokens** (`remaining_vision_tokens`).

**Output contract (required for all experiments)**
- Primary outputs: `results.csv` + `summary.json` in the experiment `output_dir`.
- `scripts/eval.py` already writes `results.csv` + `summary.json`.
- `results.csv` includes baseline_name, lambda_cost, threshold, pruning_ratio, action_ratio, accuracy, avg_cost, latency, mem.
- `summary.json` points to run metadata with seed/config/dataset hashes.
- Sweep/analysis scripts currently write:
  - `outputs/**/pareto_threshold.csv` + `pareto_threshold.json`
  - `outputs/**/pareto_pruning.csv` + `pareto_pruning.json`
  - `outputs/**/analysis_oracle.json`
- **TODO**: add `--output_prefix results` (default) to `scripts/pareto_threshold.py`, `scripts/pareto_pruning.py`, and `scripts/oracle_action.py` so they also emit `results.csv` + `summary.json`.

**Checkpoint loading**
- `scripts/eval.py --checkpoint` supports:
  - Accelerate state dir, or
  - DDP `.pt` checkpoints saved by `scripts/train_ddp.py` (keys: `model`, `optimizer`, `scheduler`).

**Cache reuse**
- `use_cache` is controlled internally; there is no config switch yet.  
  **TODO**: add `eval.cache_reuse` (default `true`), and log whether cache reuse is enabled.

---

## A. Main Results (Pareto Curve, VoVNet) - **Must Do**

### A0. Train VoVNet (two-stage in one command)
**Purpose**: produce a trained checkpoint for the main results.

**Config**
- Train split: `data.train_jsonl` in `configs/train_mmbench_llava_textvqa.yaml`.
- Eval split: `data.eval_jsonl` in `configs/train_mmbench_llava_textvqa.yaml`.
- Two-stage training is enabled by `training.stage1_epochs` and `training.stage1_baseline_name`.

**Command**
```bash
nohup torchrun --nproc_per_node 8 scripts/train_ddp.py \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  > logs/train_ddp_mmbench_llava_textvqa.out 2>&1 &
```

**Expected outputs**
- Checkpoints: `outputs/checkpoint-*.pt`
- Training logs under `outputs`

**Sanity checks**
- Checkpoints appear every `training.save_every` steps.
- Training log shows stage1 then stage2 (two-stage schedule).

### A1. MMBench Test Pareto (lambda_cost sweep)
**Purpose**: main accuracy-cost tradeoff curve (reviewers' core question).

**Config**
- Use `configs/base.yaml` + `configs/train_mmbench_llava_textvqa.yaml`.
- Eval split: `--dataset_config mmbench` (MMBench test).
- Baseline: `policy.baseline_name: null` (VoVNet).
- Sweep `--pareto` list for lambda_cost.

**Command**
```bash
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --output_dir outputs/pareto_vovnet \
  --pareto 0 0.01 0.02 0.05 0.1 \
  > logs/eval_pareto_vovnet.out 2>&1 &
```

**Expected outputs**
- `outputs/pareto_vovnet/eval_matrix.csv`
- `outputs/pareto_vovnet/eval_matrix.json`
- `outputs/pareto_vovnet/results.csv`
- `outputs/pareto_vovnet/summary.json`

**Sanity checks**
- `avg_cost` decreases as lambda increases.
- `action_ratio` shifts toward NO/COARSE as lambda increases.

### A2. Multi-seed stability (2-3 seeds on key lambda)
**Purpose**: variance/stability evidence.

**Config**
- Create per-seed override:
```bash
cat > configs/seed_42.yaml <<'YAML'
training:
  seed: 42
YAML
```

**Command (repeat for 2-3 seeds)**
```bash
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/seed_42.yaml \
  --output_dir outputs/pareto_vovnet_seed42 \
  --pareto 0.02 0.05 0.1 \
  > logs/eval_pareto_vovnet_seed42.out 2>&1 &
```

**Expected outputs**
- Per-seed `eval_matrix.csv/json`
- Per-seed `results.csv/summary.json`

**Sanity checks**
- Accuracy variance is small across seeds for the same lambda.

---

## B. Required Baselines (Cost-Aligned) - **Must Do**

### B1. Always-Full / Always-Coarse / No-Vision
**Purpose**: fixed-policy baselines at aligned cost.

**Config**
- Add a baseline override with `policy.baseline_name`.

**Command**
Note: run one baseline at a time.
```bash
cat > configs/baseline_always_full.yaml <<'YAML'
policy:
  baseline_name: "always_full"
YAML
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/baseline_always_full.yaml \
  --output_dir outputs/baseline_always_full \
  > logs/eval_baseline_always_full.out 2>&1 &

cat > configs/baseline_always_coarse.yaml <<'YAML'
policy:
  baseline_name: "always_coarse"
YAML
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/baseline_always_coarse.yaml \
  --output_dir outputs/baseline_always_coarse \
  > logs/eval_baseline_always_coarse.out 2>&1 &

cat > configs/baseline_no_vision.yaml <<'YAML'
policy:
  baseline_name: "no_vision"
YAML
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/baseline_no_vision.yaml \
  --output_dir outputs/baseline_no_vision \
  > logs/eval_baseline_no_vision.out 2>&1 &
```

**Expected outputs**
- Each run: `results.csv` + `summary.json`

**Sanity checks**
- Always-Full: `action_ratio` ~ [0, 0, 1], highest `avg_cost`.
- Always-Coarse: `action_ratio` ~ [0, 1, 0], mid `avg_cost`.
- No-Vision: `action_ratio` ~ [1, 0, 0], `avg_cost` ~ 0.

### B2. Uncertainty Threshold (entropy or margin)
**Purpose**: heuristic baseline using text-first uncertainty.

**Config**
- `policy.baseline_name: "uncertainty_threshold"`
- `policy.baseline_uncertainty: "entropy" | "margin"`
- `policy.baseline_threshold: <float>`
- `policy.baseline_vision: "full" | "coarse"`

**Command (single threshold, writes results.csv/summary.json)**
```bash
cat > configs/baseline_uncertainty_entropy_full.yaml <<'YAML'
policy:
  baseline_name: "uncertainty_threshold"
  baseline_uncertainty: "entropy"
  baseline_threshold: 0.50
  baseline_vision: "full"
YAML
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/baseline_uncertainty_entropy_full.yaml \
  --output_dir outputs/baseline_uncertainty_t0_50 \
  > logs/eval_baseline_uncertainty_t0_50.out 2>&1 &
```

**Command (Pareto sweep, writes pareto_threshold.csv/json)**
```bash
nohup python -m scripts.pareto_threshold \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --output_dir outputs/pareto_threshold_entropy_full \
  --thresholds 0.10 0.20 0.30 0.40 0.50 \
  --uncertainty entropy \
  --vision full \
  > logs/pareto_threshold_entropy_full.out 2>&1 &
```

**Expected outputs**
- Single threshold: `results.csv` + `summary.json`
- Sweep: `pareto_threshold.csv` + `pareto_threshold.json`
- **TODO**: add results.csv/summary.json to `scripts/pareto_threshold.py`.

**Sanity checks**
- Lower threshold -> more FULL actions -> higher `avg_cost`.

### B3. Random Policy Matched
**Purpose**: random action selection matched to VoVNet action ratios (cost-aligned).

**Config**
- `policy.baseline_name: "random_policy_matched"`
- `policy.baseline_target_ratios: [no, coarse, full]`
- `policy.baseline_seed: <int>`

**Command (generate ratios from VoVNet results)**
```bash
nohup python - <<'PY' > logs/gen_random_matched_cfg.out 2>&1 &
import csv
from pathlib import Path

target_lambda = 0.05
results_path = Path("outputs/pareto_vovnet/results.csv")
dataset_name = "mmbench"

ratio = None
with results_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("dataset") != dataset_name:
            continue
        try:
            lam = float(row.get("lambda_cost", "nan"))
        except ValueError:
            continue
        baseline = (row.get("baseline_name") or "").strip().lower()
        if baseline not in {"", "none", "null"}:
            continue
        if abs(lam - target_lambda) > 1e-9:
            continue
        ratio = [
            float(row["action_rate_no"]),
            float(row["action_rate_coarse"]),
            float(row["action_rate_full"]),
        ]
        break

if ratio is None:
    raise SystemExit("Could not find matching lambda in results.csv")

Path("configs").mkdir(parents=True, exist_ok=True)
with open("configs/baseline_random_matched.yaml", "w") as f:
    f.write("policy:\n")
    f.write("  baseline_name: \"random_policy_matched\"\n")
    f.write("  baseline_seed: 42\n")
    f.write(f"  baseline_target_ratios: [{ratio[0]:.6f}, {ratio[1]:.6f}, {ratio[2]:.6f}]\n")
print("Wrote configs/baseline_random_matched.yaml with ratios", ratio)
PY

nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/baseline_random_matched.yaml \
  --output_dir outputs/baseline_random_matched \
  > logs/eval_baseline_random_matched.out 2>&1 &
```

**Expected outputs**
- `results.csv` + `summary.json`

**Sanity checks**
- `action_ratio` close to target ratios (small random deviation).
- `avg_cost` close to the VoVNet point used for matching.

**Optional (bucketed ratios by entropy)**
- **TODO**: add a helper script to compute bucket ratios from VoVNet outputs.
- Required parameters once implemented:
  - `policy.baseline_bucket_ratios: [[no, coarse, full], ... x3 buckets]`
  - `policy.baseline_bucket_thresholds: [t1, t2]` (if omitted, auto-quantile is used)

### B4. Vision Token Pruning Proxy
**Purpose**: cost-aligned baseline that always uses vision but prunes tokens.

**Config**
- `policy.baseline_name: "vision_token_pruning_proxy"`
- `policy.baseline_pruning_ratio: 1.0/0.75/0.5/0.25`
- `policy.baseline_pruning_mode: "stride" | "topk_norm"`

**Command (single ratio, writes results.csv/summary.json)**
```bash
cat > configs/baseline_pruning_ratio_050.yaml <<'YAML'
policy:
  baseline_name: "vision_token_pruning_proxy"
  baseline_pruning_ratio: 0.50
  baseline_pruning_mode: "stride"
YAML
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/baseline_pruning_ratio_050.yaml \
  --output_dir outputs/baseline_pruning_ratio_050 \
  > logs/eval_baseline_pruning_ratio_050.out 2>&1 &
```

**Command (Pareto sweep, writes pareto_pruning.csv/json)**
```bash
nohup python -m scripts.pareto_pruning \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --output_dir outputs/pareto_pruning_stride \
  --ratios 1.0 0.75 0.50 0.25 \
  --mode stride \
  > logs/pareto_pruning_stride.out 2>&1 &
```

**Expected outputs**
- Single ratio: `results.csv` + `summary.json`
- Sweep: `pareto_pruning.csv` + `pareto_pruning.json`
- **TODO**: add results.csv/summary.json to `scripts/pareto_pruning.py`.

**Sanity checks**
- `remaining_vision_tokens` scales roughly with pruning ratio.
- Accuracy drops smoothly as pruning_ratio decreases.

---

## C. Generalization Evaluation (No Training on These) - **Must Do**

### C1. MMMU + TextVQA + MMBench (held-out)
**Purpose**: show action shifts under strong visual vs OCR-heavy tasks.

**Config**
- Use `--dataset_config "mmbench,mmmu,textvqa"` (already includes MMBench test, MMMU validation, TextVQA validation).

**Command**
```bash
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config "mmbench,mmmu,textvqa" \
  --output_dir outputs/generalization_vovnet \
  > logs/eval_generalization_vovnet.out 2>&1 &
```

**Expected outputs**
- `outputs/generalization_vovnet/results.csv`
- `outputs/generalization_vovnet/summary.json`

**Sanity checks**
- MMMU shows higher FULL action ratio than MMBench.
- TextVQA has low NO action ratio (OCR-heavy).

**Data leakage check**
- Confirm none of MMMU/TextVQA splits appear in training JSONL.

---

## D. Ablations (Answer "Is It Just a Heuristic?") - **Must Do**

### D1. Policy input ablation (hidden vs hidden+entropy vs hidden+margin)
**Purpose**: show policy signal contribution beyond text hidden states.

**Status**
- **TODO**: add policy input switches. Suggested config keys:
  - `policy.feature_mode: "hidden" | "hidden_entropy" | "hidden_margin" | "hidden_entropy_margin"`
  - Default: `"hidden"` (current behavior).

**Command (after TODO is implemented)**
```bash
cat > configs/ablate_feature_hidden_entropy.yaml <<'YAML'
policy:
  feature_mode: "hidden_entropy"
YAML
```

### D2. Training strategy ablation (Gumbel-ST vs soft mixture)
**Purpose**: show effect of discrete vs soft action selection.

**Config**
- `policy.use_straight_through: true` (Gumbel-ST)
- `policy.use_straight_through: false` (soft mixture)

**Command**
```bash
cat > configs/ablate_soft_mixture.yaml <<'YAML'
policy:
  use_straight_through: false
YAML
nohup torchrun --nproc_per_node 8 scripts/train_ddp.py \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --config configs/ablate_soft_mixture.yaml \
  > logs/train_ddp_soft_mixture.out 2>&1 &
```

**Expected outputs**
- Training checkpoints under `outputs/`
- Evaluate with `scripts/eval.py` (see A1), using the ablated checkpoint.

**Sanity checks**
- Soft mixture runs without image-token expansion error (model raises if unsupported).

### D3. Cost-term ablation (lambda_cost=0 vs >0)
**Purpose**: show cost regularization matters.

**Config**
- `policy.lambda_cost: 0.0` (no cost term)

**Command**
```bash
cat > configs/ablate_lambda0.yaml <<'YAML'
policy:
  lambda_cost: 0.0
YAML
nohup torchrun --nproc_per_node 8 scripts/train_ddp.py \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --config configs/ablate_lambda0.yaml \
  > logs/train_ddp_lambda0.out 2>&1 &
```

**Expected outputs**
- Checkpoints under `outputs/`
- Eval results via `scripts/eval.py` (A1)

**Sanity checks**
- Cost-unconstrained model tends to higher FULL ratios at eval.

### D4. Action-space ablation (2-class vs 3-class)
**Purpose**: show importance of the COARSE action.

**Status**
- **TODO**: add a binary action space option. Suggested config keys:
  - `policy.action_space: "binary" | "ternary"` (default `"ternary"`).

**Command (after TODO is implemented)**
```bash
cat > configs/ablate_binary_actions.yaml <<'YAML'
policy:
  action_space: "binary"
YAML
```

---

## E. Reliability / Safety - **Bonus**

### E1. Fallback mechanism (NO -> COARSE/FULL under high entropy)
**Purpose**: prevent unsafe NO when uncertainty is high.

**Config**
- `policy.fallback_mode: "coarse" | "full"`
- `policy.fallback_entropy_threshold: <float>` and/or `policy.fallback_margin_threshold: <float>`

**Command**
```bash
cat > configs/fallback_full.yaml <<'YAML'
policy:
  fallback_mode: "full"
  fallback_entropy_threshold: 0.50
YAML
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/fallback_full.yaml \
  --output_dir outputs/fallback_full \
  > logs/eval_fallback_full.out 2>&1 &
```

**Expected outputs**
- `results.csv` + `summary.json`

**Sanity checks**
- `action_ratio` shifts from NO toward FULL with fallback enabled.

**TODO**
- Add `fallback_trigger_rate` to eval outputs and `results.csv`.

### E2. Error decomposition (by action bucket)
**Purpose**: expose failure modes by action choice.

**Status**
- **TODO**: add an error-slicing script (e.g., `scripts/error_slices.py`) that reports:
  - per-action accuracy
  - avg entropy / avg cost
  - top error slices by dataset fields

---

## F. System / Deployment Metrics - **Bonus**

### F1. Profiling latency/memory/tokens
**Purpose**: report P50/P90 latency, tokens/s, mem_peak.

**Config**
- `eval.profile: true`

**Command (VoVNet + fixed baselines)**
Note: run one profile job at a time.
```bash
cat > configs/profile_eval.yaml <<'YAML'
eval:
  profile: true
YAML
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/profile_eval.yaml \
  --output_dir outputs/profile_vovnet \
  > logs/profile_vovnet.out 2>&1 &

nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/profile_eval.yaml \
  --config configs/baseline_always_full.yaml \
  --output_dir outputs/profile_always_full \
  > logs/profile_always_full.out 2>&1 &

nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/profile_eval.yaml \
  --config configs/baseline_always_coarse.yaml \
  --output_dir outputs/profile_always_coarse \
  > logs/profile_always_coarse.out 2>&1 &
```

**Expected outputs**
- `results.csv` includes `latency_p50_ms`, `latency_p90_ms`, `mem_peak_p50_mb`, `mem_peak_p90_mb`.
- `summary.json` includes full `profile` with `tokens_s`.

### F2. KV cache reuse on/off
**Purpose**: show end-to-end latency impact of cache reuse.

**Status**
- **TODO**: add `eval.cache_reuse: true|false` (default `true`) and log it in outputs.

---

## G. Final Tables and Plot Inputs - **Bonus**

### G1. Build main_table.csv / pareto.csv / action_ratio.csv
**Purpose**: produce paper-ready CSVs for tables and plots.

**Command**
```bash
nohup python - <<'PY' > logs/build_tables.out 2>&1 &
import csv
from pathlib import Path

out_dir = Path("outputs/plots")
out_dir.mkdir(parents=True, exist_ok=True)

def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

rows = []
for path in [
    "outputs/pareto_vovnet/results.csv",
    "outputs/baseline_always_full/results.csv",
    "outputs/baseline_always_coarse/results.csv",
    "outputs/baseline_no_vision/results.csv",
    "outputs/baseline_uncertainty_t0_50/results.csv",
    "outputs/baseline_random_matched/results.csv",
    "outputs/baseline_pruning_ratio_050/results.csv",
]:
    if Path(path).exists():
        rows.extend(load_rows(path))

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# Pareto curve (MMBench only)
pareto_rows = [r for r in rows if r.get("dataset") == "mmbench"]
write_csv(out_dir / "pareto.csv", pareto_rows)

# Action ratio summary
action_rows = [
    {
        "dataset": r.get("dataset"),
        "baseline_name": r.get("baseline_name"),
        "lambda_cost": r.get("lambda_cost"),
        "action_rate_no": r.get("action_rate_no"),
        "action_rate_coarse": r.get("action_rate_coarse"),
        "action_rate_full": r.get("action_rate_full"),
        "avg_cost": r.get("avg_cost"),
        "accuracy": r.get("accuracy"),
    }
    for r in rows
]
write_csv(out_dir / "action_ratio.csv", action_rows)

# Main table (filter MMBench and key baselines)
main_table = [
    r for r in rows
    if r.get("dataset") == "mmbench"
]
write_csv(out_dir / "main_table.csv", main_table)
print("Wrote", out_dir / "pareto.csv", out_dir / "action_ratio.csv", out_dir / "main_table.csv")
PY
```

**Expected outputs**
- `outputs/plots/main_table.csv`
- `outputs/plots/pareto.csv`
- `outputs/plots/action_ratio.csv`

**Plotting**
- Use your preferred plotting tool (e.g., matplotlib) to draw accuracy-cost curves and stacked action ratios.

---

## H. Oracle Action (Analysis Only) - **Optional**

### H1. Oracle action analysis (small subset)
**Purpose**: lower-bound cost for accuracy by selecting the cheapest correct action.

**Config**
- Use a small `max_samples` because it runs NO/COARSE/FULL per sample.

**Command**
```bash
nohup python -m scripts.oracle_action \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --output_dir outputs/oracle_action \
  --max_samples 200 \
  > logs/oracle_action.out 2>&1 &
```

**Expected outputs**
- `outputs/oracle_action/analysis_oracle.json`
- **TODO**: add `results.csv` + `summary.json` to `scripts/oracle_action.py`.

**Sanity checks**
- `oracle_accuracy` >= `vovnet_accuracy`.
- `oracle_avg_cost` <= `vovnet_avg_cost`.

---

## MVP (Minimal Reproducible Experiments)
**Goal**: reproduce the main trend with minimum runs.

1) Train (one run):
```bash
nohup torchrun --nproc_per_node 8 scripts/train_ddp.py \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  > logs/train_ddp_mvp.out 2>&1 &
```

2) VoVNet Pareto (3 lambda points):
```bash
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --output_dir outputs/pareto_vovnet_mvp \
  --pareto 0 0.05 0.1 \
  > logs/eval_pareto_vovnet_mvp.out 2>&1 &
```

3) Fixed baselines (Always-Full + No-Vision):
Note: requires `configs/baseline_always_full.yaml` and `configs/baseline_no_vision.yaml` from B1.
```bash
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/baseline_always_full.yaml \
  --output_dir outputs/baseline_always_full_mvp \
  > logs/eval_baseline_always_full_mvp.out 2>&1 &

nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --config configs/baseline_no_vision.yaml \
  --output_dir outputs/baseline_no_vision_mvp \
  > logs/eval_baseline_no_vision_mvp.out 2>&1 &
```

---

## Full Experiment Suite (Paper-Ready)
**Goal**: finalize all tables/figures with multi-seed stability and full sweeps.

1) Train 2-3 seeds (same config, different `training.seed`).
2) VoVNet Pareto sweep on MMBench test (A1) for all seeds.
3) Baseline sweeps: uncertainty thresholds + pruning ratios (B2/B4).
4) Random matched baselines for key lambda points (B3).
5) Generalization runs (C1).
6) Ablations (D1-D4) + fallback reliability (E1).
7) Profiling (F1) and cache-reuse (F2, after TODO).
8) Aggregate outputs into plots/tables (G1).
