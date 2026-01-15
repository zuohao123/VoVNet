# 数据集准备与训练执行指南（服务器参考）

本指南用于在服务器上执行数据集下载/规范化与模型训练。命令均基于仓库内现有脚本。

## 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如服务器上没有 venv 模块或希望使用 conda，可用 Miniconda 创建隔离环境：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
/opt/miniconda/bin/conda init bash
source ~/.bashrc
conda create -n vovnet python=3.10 -y
conda activate vovnet
pip install -r requirements.txt
```

如服务器无外网或需离线：
- 将模型权重按本地路径放置（见第 2 节）
- 如需访问 HuggingFace 数据集，请确保网络权限或提前下载缓存

## 2. 本地模型权重目录（Qwen3-VL）

建议在仓库下放置模型快照，例如：

```
models/
  qwen3-vl-8b-instruct/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    preprocessor_config.json
```

在 `configs/base.yaml` 中修改：

```yaml
model:
  base_model_name: "models/qwen3-vl-8b-instruct"
  full_model_name: "models/qwen3-vl-8b-thinking"  # 可选
  use_thinking_for_full: false
  trust_remote_code: true
```

如只使用一个模型，`use_thinking_for_full` 保持 `false` 即可。

## 3. 数据集下载与规范化（统一 JSONL）

### 3.1 单个数据集

示例：VQA v2（如数据集需要配置名，再显式指定 `--subset`）

```bash
python scripts/prepare_dataset.py \
  --dataset vqa_v2 \
  --splits train,validation \
  --download-images \
  --max_samples 1000
```

输出：
- `data/processed/vqa_v2/vqa_v2_train.jsonl`
- `data/processed/vqa_v2/vqa_v2_validation.jsonl`
- 图片缓存（如果开启）：`data/images/vqa_v2/`

### 3.2 批量数据集

使用 `configs/data_recipe.yaml` 统一控制数据集列表与采样规模：

```bash
python scripts/prepare_all.py --mode fast_dev
python scripts/prepare_all.py --mode paper
```

### 3.3 常用参数说明
- `--download-images`：下载并保存到 `data/images/<dataset>/`，否则保留 HF 引用
- `--export-parquet`：额外导出 Parquet（需要 `pyarrow`）
- `--max_samples`：抽样上限（便于快速验证）
- `--subset`：HF 配置名（部分数据集必须指定）
- `--streaming`：启用 HF streaming（适合超大数据集或生成失败时）
- `VOVNET_HF_DATASET_ID_<NAME>`：覆盖 HF 数据集 ID（例如 `VOVNET_HF_DATASET_ID_VQA_V2`）

### 3.4 访问受限数据集 / HF Token
如遇到权限问题，请先登录：

```bash
huggingface-cli login
```

或设置环境变量（仅对需要权限的数据集）：

```bash
export HF_TOKEN=你的token
# 或
export HUGGINGFACE_HUB_TOKEN=你的token
```

若服务器无法联网，请在有网环境准备好缓存并拷贝到 `~/.cache/huggingface`。

## 4. 训练脚本执行

训练入口：`scripts/train.py`（配置文件在 `configs/`）

### 4.1 基本训练

```bash
python scripts/train.py --config configs/base.yaml --config configs/train_vqa.yaml
```

### 4.2 多 GPU（8x V100）

```bash
accelerate launch --num_processes 8 \
  scripts/train.py --config configs/base.yaml --config configs/train_vqa.yaml
```

### 4.3 训练数据路径
`configs/train_vqa.yaml` 默认读取：

```yaml
data:
  train_jsonl: "data/train_vqa.jsonl"
  eval_jsonl: "data/val_vqa.jsonl"
```

如果你用的是上面脚本生成的输出，请修改为：

```yaml
data:
  train_jsonl: "data/processed/vqa_v2/vqa_v2_train.jsonl"
  eval_jsonl: "data/processed/vqa_v2/vqa_v2_validation.jsonl"
```

## 5. 常见问题排查

- **模型加载失败**：检查 `model.base_model_name` 是否指向本地目录，且目录包含完整模型文件。
- **数据集加载失败**：尝试指定 `--subset`，或确认 HF 权限。
- **图片为空**：未开启 `--download-images` 时会保留 HF 引用，模型训练时需要确认是否可解析。
- **离线运行**：确保模型与数据集都已缓存到本地。

如需我按你的服务器目录结构生成具体命令，请告诉我实际路径。 
