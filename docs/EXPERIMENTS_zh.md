# VoVNet 实验 Pipeline（中文说明）

本文档是 VoVNet（Value-of-Vision）的实验执行顺序说明，按优先级组织：**必做 -> 加分 -> 可选**。每个实验块都包含目的、配置、命令、输出和关键检查点。

所有实验**必须**使用同一评测口径：相同 prompt 模板、解码超参、视觉预算与成本统计口径。

---

## 0. 全局规则（必须遵守）

**统一评测管线**
- 使用 `scripts/eval.py`（matrix runner）或调用同一评测核心 `src/eval/matrix.evaluate_dataset` 的脚本。
- 所有 baseline 必须使用相同 prompt 模板与解码设置（来自配置）。
- `--dataset_config` 支持内置数据集名（避免临时 YAML）：
  - `mmbench`、`mmmu`、`textvqa`（可用逗号列表，如 `"mmbench,mmmu,textvqa"`）。

**后台执行**
- 先创建日志目录：`mkdir -p logs`
- 训练/评测命令统一使用：`nohup <cmd> > logs/<name>.out 2>&1 &`
- 同一代码块内的后台命令建议逐条执行，避免并发占满 GPU。

**统一设置（实验间不可变）**
- Prompt 模板：`data.prompt_template`
- 解码：`eval.max_new_tokens / num_beams / do_sample / temperature`
- 视觉预算：`vision_budget.*`（coarse/full long side、max pixels、patch size）
- 成本口径：使用视觉 token 计数 `vision_token_count`
  - Pruning baseline 使用 `remaining_vision_tokens` 作为成本口径

**输出规范（所有实验必须有）**
- 主输出：`results.csv` + `summary.json`（在各实验 `output_dir` 下）
- `scripts/eval.py` 已生成 `results.csv` + `summary.json`
- `results.csv` 字段包含 baseline_name / lambda_cost / threshold / pruning_ratio / action_ratio / accuracy / avg_cost / latency / mem
- `summary.json` 记录 run metadata（seed/config/dataset）
- 目前 sweep 脚本输出：
  - `outputs/**/pareto_threshold.csv` + `pareto_threshold.json`
  - `outputs/**/pareto_pruning.csv` + `pareto_pruning.json`
  - `outputs/**/analysis_oracle.json`
- **TODO**：为 `scripts/pareto_threshold.py`、`scripts/pareto_pruning.py`、`scripts/oracle_action.py` 增加 `results.csv` + `summary.json`

**Checkpoint 加载**
- `scripts/eval.py --checkpoint` 使用 Accelerate 的 `load_state()`（要求 Accelerate state 目录）
- **TODO**：增加 `eval.ddp_checkpoint_path`（默认 null）支持加载 `scripts/train_ddp.py` 生成的 `checkpoint-*.pt`

**Cache 复用**
- `use_cache` 目前内部控制，暂无开关  
- **TODO**：新增 `eval.cache_reuse: true|false`（默认 true）并在输出中记录

---

## A. 主结果（VoVNet Pareto）- 必做

### A0. 训练 VoVNet（两阶段，一条命令）
**目的**：产出主结果所需 checkpoint。

**配置**
- 训练集：`configs/train_mmbench_llava_textvqa.yaml` 中的 `data.train_jsonl`
- 评测集：`configs/train_mmbench_llava_textvqa.yaml` 中的 `data.eval_jsonl`
- 两阶段由 `training.stage1_epochs` + `training.stage1_baseline_name` 控制

**命令**
```bash
nohup torchrun --nproc_per_node 8 scripts/train_ddp.py \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  > logs/train_ddp_mmbench_llava_textvqa.out 2>&1 &
```

**预期输出**
- `outputs/checkpoint-*.pt`
- 训练日志在 `outputs/`

**检查点**
- checkpoint 每 `training.save_every` 步保存一次
- 日志显示 stage1 -> stage2

### A1. MMBench 测试集 Pareto（lambda_cost 扫描）
**目的**：主结果 accuracy-cost 曲线。

**配置**
- 使用 `configs/base.yaml` + `configs/train_mmbench_llava_textvqa.yaml`
- 评测集：`--dataset_config mmbench`（MMBench test）
- baseline：`policy.baseline_name: null`（VoVNet）

**命令**
```bash
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --output_dir outputs/pareto_vovnet \
  --pareto 0 0.01 0.02 0.05 0.1 \
  > logs/eval_pareto_vovnet.out 2>&1 &
```

**预期输出**
- `outputs/pareto_vovnet/eval_matrix.csv`
- `outputs/pareto_vovnet/eval_matrix.json`
- `outputs/pareto_vovnet/results.csv`
- `outputs/pareto_vovnet/summary.json`

**检查点**
- `avg_cost` 随 lambda 变大而降低
- `action_ratio` 往 NO/COARSE 方向移动

### A2. 多随机种子稳定性（2-3 个 seed）
**目的**：稳定性与方差证明。

**配置**
```bash
cat > configs/seed_42.yaml <<'YAML'
training:
  seed: 42
YAML
```

**命令（对多个 seed 重复）**
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

**预期输出**
- 每个 seed 的 `results.csv` + `summary.json`

**检查点**
- 同一 lambda 的 accuracy 方差应较小

---

## B. 必要基线（成本对齐）- 必做

### B1. Always-Full / Always-Coarse / No-Vision
**目的**：固定策略基线。

**配置**
- `policy.baseline_name: "always_full" | "always_coarse" | "no_vision"`

**命令**
注：以下后台评测命令建议逐条执行。
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

**预期输出**
- 每次运行都有 `results.csv` + `summary.json`

**检查点**
- Always-Full: `action_ratio` ~ [0, 0, 1]，`avg_cost` 最高
- Always-Coarse: `action_ratio` ~ [0, 1, 0]，`avg_cost` 中等
- No-Vision: `action_ratio` ~ [1, 0, 0]，`avg_cost` ~ 0

### B2. 不确定性阈值基线（entropy / margin）
**目的**：文本不确定性驱动的启发式 baseline。

**配置**
- `policy.baseline_name: "uncertainty_threshold"`
- `policy.baseline_uncertainty: "entropy" | "margin"`
- `policy.baseline_threshold: <float>`
- `policy.baseline_vision: "full" | "coarse"`

**命令（单阈值）**
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

**命令（Pareto 扫描）**
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

**预期输出**
- 单阈值：`results.csv` + `summary.json`
- 扫描：`pareto_threshold.csv` + `pareto_threshold.json`
- **TODO**：为 `scripts/pareto_threshold.py` 增加 `results.csv/summary.json`

**检查点**
- 阈值更小 -> FULL 比例上升 -> `avg_cost` 增加

### B3. Random Policy Matched
**目的**：动作比例匹配 VoVNet 的随机策略基线。

**配置**
- `policy.baseline_name: "random_policy_matched"`
- `policy.baseline_target_ratios: [no, coarse, full]`
- `policy.baseline_seed: <int>`

**命令（从 VoVNet 结果提取 action_ratio）**
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

**预期输出**
- `results.csv` + `summary.json`

**检查点**
- `action_ratio` 与 target 接近
- `avg_cost` 接近匹配的 VoVNet 点

**可选（按 entropy 分桶）**
- **TODO**：增加从 VoVNet 统计桶比例的脚本
- 需要的参数：
  - `policy.baseline_bucket_ratios: [[no, coarse, full], ... x3]`
  - `policy.baseline_bucket_thresholds: [t1, t2]`（可选，未给则自动分位）

### B4. Vision Token Pruning Proxy
**目的**：始终有视觉但下采样视觉 token 的成本对齐基线。

**配置**
- `policy.baseline_name: "vision_token_pruning_proxy"`
- `policy.baseline_pruning_ratio: 1.0/0.75/0.5/0.25`
- `policy.baseline_pruning_mode: "stride" | "topk_norm"`

**命令（单 ratio）**
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

**命令（Pareto 扫描）**
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

**预期输出**
- 单 ratio：`results.csv` + `summary.json`
- 扫描：`pareto_pruning.csv` + `pareto_pruning.json`
- **TODO**：为 `scripts/pareto_pruning.py` 增加 `results.csv/summary.json`

**检查点**
- `remaining_vision_tokens` 随 ratio 下降
- accuracy 随 ratio 下降平滑

---

## C. 泛化评测（不在训练集上）- 必做

### C1. MMMU / TextVQA / MMBench
**目的**：验证策略在视觉依赖与 OCR 场景的泛化。

**配置**
- 使用 `--dataset_config "mmbench,mmmu,textvqa"`（已包含 MMBench、MMMU、TextVQA）

**命令**
```bash
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config "mmbench,mmmu,textvqa" \
  --output_dir outputs/generalization_vovnet \
  > logs/eval_generalization_vovnet.out 2>&1 &
```

**预期输出**
- `outputs/generalization_vovnet/results.csv`
- `outputs/generalization_vovnet/summary.json`

**检查点**
- MMMU 的 FULL 比例高于 MMBench
- TextVQA 的 NO 比例较低

**防泄露**
- 确认 MMMU / TextVQA splits 未出现在训练 JSONL

---

## D. 消融实验 - 必做

### D1. Policy 输入消融（hidden vs hidden+entropy vs hidden+margin）
**目的**：验证策略信号贡献。

**状态**
- **TODO**：新增输入消融开关，建议参数：
  - `policy.feature_mode: "hidden" | "hidden_entropy" | "hidden_margin" | "hidden_entropy_margin"`

### D2. 训练策略消融（Gumbel-ST vs soft mixture）
**目的**：离散/软混合对比。

**配置**
- `policy.use_straight_through: true/false`

**命令**
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

### D3. 成本项消融（lambda_cost=0 vs >0）
**配置**
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

### D4. 动作空间消融（二分类 vs 三分类）
**状态**
- **TODO**：增加 `policy.action_space: "binary" | "ternary"`

---

## E. 可靠性/安全性 - 加分

### E1. Fallback 机制
**目的**：NO 被高不确定性触发时升级 COARSE/FULL。

**配置**
- `policy.fallback_mode: "coarse" | "full"`
- `policy.fallback_entropy_threshold` / `policy.fallback_margin_threshold`

**命令**
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

**TODO**
- 添加 `fallback_trigger_rate` 统计并写入输出

### E2. 错误分解（按动作分桶）
**状态**
- **TODO**：增加错误切片脚本 `scripts/error_slices.py`

---

## F. 系统/部署指标 - 加分

### F1. Profiling（Latency / Mem / Tokens）
**配置**
```bash
cat > configs/profile_eval.yaml <<'YAML'
eval:
  profile: true
YAML
```

**命令**
注：以下 profile 命令建议逐条执行。
```bash
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

### F2. KV cache 复用开关
**状态**
- **TODO**：增加 `eval.cache_reuse: true|false`

---

## G. 最终表格与图数据 - 加分

### G1. 生成 main_table / pareto / action_ratio
**命令**
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

pareto_rows = [r for r in rows if r.get("dataset") == "mmbench"]
write_csv(out_dir / "pareto.csv", pareto_rows)

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

main_table = [r for r in rows if r.get("dataset") == "mmbench"]
write_csv(out_dir / "main_table.csv", main_table)
print("Wrote", out_dir / "pareto.csv", out_dir / "action_ratio.csv", out_dir / "main_table.csv")
PY
```

---

## H. Oracle Action（分析专用）- 可选

### H1. Oracle action 分析（小样本）
**目的**：给出最低成本仍答对的理论上界。

**命令**
```bash
nohup python -m scripts.oracle_action \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --output_dir outputs/oracle_action \
  --max_samples 200 \
  > logs/oracle_action.out 2>&1 &
```

**预期输出**
- `outputs/oracle_action/analysis_oracle.json`
- **TODO**：增加 `results.csv` + `summary.json`

---

## MVP（最小可复现）

1) 训练：
```bash
nohup torchrun --nproc_per_node 8 scripts/train_ddp.py \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  > logs/train_ddp_mvp.out 2>&1 &
```

2) VoVNet Pareto（3 个点）：
```bash
nohup python -m scripts.eval \
  --config configs/base.yaml \
  --config configs/train_mmbench_llava_textvqa.yaml \
  --dataset_config mmbench \
  --output_dir outputs/pareto_vovnet_mvp \
  --pareto 0 0.05 0.1 \
  > logs/eval_pareto_vovnet_mvp.out 2>&1 &
```

3) 固定基线（Always-Full + No-Vision）
注：需要先执行 B1 生成 baseline 配置文件。
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

## 完整实验（投稿用）

1) 多 seed 训练（2-3 个 seed）
2) MMBench Pareto 扫描（A1）
3) Baseline 扫描（B2/B4）
4) Random matched baseline（B3）
5) 泛化评测（C1）
6) 消融（D1-D4）+ fallback（E1）
7) Profiling（F1）+ cache_reuse（F2）
8) 汇总图表（G1）
