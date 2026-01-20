# VoVNet 输入→输出流程说明（中文）

本文基于 `src/models/vovnet.py` 的实现，描述 VoVNet 从输入到输出的完整处理流程，强调训练/评估时的行为差异与关键中间量。

## 1. 输入与输出概览

### 输入
- `input_ids`: `Tensor[B, T]`，文本 token id。
- `attention_mask`: `Tensor[B, T]`，文本注意力掩码。
- `images`: `List[PIL.Image]` 或 `None`，图像列表，可为空。
- `labels`: `Tensor[B, T]` 或 `None`，用于计算任务损失。

### 输出（`dict`）
- `logits`: `Tensor[B, T, V]`，最终用于任务的语言模型 logits。
- `action_logits`: `Tensor[B, 3]`，策略头输出，动作空间为 {NO_VISION, COARSE_VISION, FULL_VISION}。
- `action_probs`: `Tensor[B, 3]`，动作概率（训练可为硬 Gumbel 或 softmax）。
- `actions`: `Tensor[B]`，最终动作（训练硬采样/评估 argmax）。
- `expected_cost`: `Tensor[B]`，期望成本（用 action_probs 与成本向量加权）。
- `uncertainty`: `Tensor[B]`，不确定性（基于文本 logits 的熵）。
- `vision_tokens`: `Tensor[B]`，视觉 token 数估计。
- `token_count_coarse`: `Tensor[B]`，COARSE 视觉 token 数（基于预处理/视觉 tokenizer）。
- `token_count_full`: `Tensor[B]`，FULL 视觉 token 数（基于预处理/视觉 tokenizer）。
- `gain_pred`: `Tensor[B, 2]`，预测的 (coarse_gain, full_gain)，用于 VoV 回归/排序训练。
- `gain_true`: `Tensor[B, 2]`，反事实 gain（loss_no - loss_vision），仅在训练/开启计算时返回。
- `actions_raw`: `Tensor[B]`，策略原始动作（回退前）。
- `fallback_mask`: `Tensor[B]`，回退触发标记。
- `fallback_entropy_trigger`: `Tensor[B]`，因熵阈值触发的样本标记。
- `fallback_margin_trigger`: `Tensor[B]`，因 margin 阈值触发的样本标记。
- `margin`: `Tensor[B]`，文本 logits 的 top1-top2 概率差。
- `text_logits`: `Tensor[B, T, V]`，文本前向 logits（用于回退前基线评估）。
- `labels`: 原样透传。

## 2. 模块结构（关键组件）

- **`base_vlm`**：基础 Qwen3-VL 模型（文本+视觉）。
- **`full_vlm`**（可选）：高预算模型（如 Thinking 版）。
- **`vow_head`**：轻量 MLP，对文本隐藏状态池化后的表示进行特征抽取。
- **`policy`**：线性层输出动作 logits。
- **`vision_budget`**：视觉预算控制器（图像缩放、token 数估计）。

## 3. 核心流程（`forward`）

### Step 1：Text-First 前向（`text_first`）
1. 调用 `base_vlm.encode_text`，仅输入文本。
2. 取最后一层隐藏状态 `hidden_states[-1]`，按 `attention_mask` 做均值池化。
3. 经过 `vow_head` 得到 `vow_features`。
4. 通过 `policy` 或 `gain_head` 输出 `action_logits`（支持 gain→logits 映射）。

### Step 2：动作选择（`_select_actions`）
- **训练时**：
  - 若 `use_straight_through=True`：使用 Gumbel-Softmax，`hard=True`，得到 one-hot 近似动作。
  - 否则使用 softmax 概率。
- **评估时**：
  - `eval_sample=True` 时按概率采样；否则取 argmax。

输出 `action_probs` 和 `actions`。

### Step 2.5：评估回退（Fallback）
- 若策略输出 **NO_VISION** 且文本不确定性过高：
  - `entropy > t` 或 `margin < t2`
  - 按配置自动升级为 **COARSE** 或 **FULL**
- 输出 `actions_raw`（回退前）与 `actions`（回退后），并记录触发条件统计。

### Step 3：不确定性与成本估计
- **不确定性**：使用文本 logits 最后一 token 的熵 `entropy_from_logits`。
- **期望成本**：`expected_cost = cost_scale * (p1*token_count_coarse + p2*token_count_full)`。

### Step 4：条件视觉调用
- 若 **训练且 `use_straight_through=False`**：走软混合（`_forward_soft_mixture`）。
- 否则走硬动作分支（`_forward_hard_actions`）。

#### 4.1 硬动作分支（`_forward_hard_actions`）
对 batch 中每个样本：
- **NO_VISION**：直接使用 text-first 的 logits。
- **COARSE_VISION / FULL_VISION**：
  1. 通过 `vision_budget.prepare_image` 进行缩放（低/高分辨率）。
  2. 用 processor 或 numpy 转成 `pixel_values`。
  3. 调用 `base_vlm` 或 `full_vlm` 的 `forward_with_vision`。
  4. 估计视觉 token 数量（patch-size 近似）。

> 若 batch 中所有动作一致且非 NO_VISION，会走 `_forward_mode` 的批量路径以减少开销。

#### 4.2 软混合分支（`_forward_soft_mixture`）
- 分别计算 **COARSE** 与 **FULL** 的视觉 logits。
- 使用 action_probs 对三种 logits 加权：
  - `logits = p0 * text_logits + p1 * coarse_logits + p2 * full_logits`
- 视觉 token 期望：`p1 * coarse_tokens + p2 * full_tokens`。

### Step 5：输出聚合
返回包含 `logits`、动作相关信息、不确定性、成本、视觉 token 数等字段。

## 4. 关键细节与注意事项

### 4.1 视觉预算控制
- `VisionBudgetController` 通过 **长边限制**与**最大像素数**控制预算。
- `estimate_visual_tokens` 使用 patch-size 启发式，留有 TODO 以适配真实视觉 tokenizer。

### 4.2 KV Cache 重用（尽力而为）
- `_forward_mode` 默认传入 `text_outputs.past_key_values`。
- 若模型不支持 cache 或接口不一致，`BaseVLM._safe_forward` 会过滤不支持的参数，避免崩溃。

### 4.3 处理器与像素输入
- 优先使用 `base_vlm.processor` 转换图片到 `pixel_values`。
- 若 processor 不可用，则退化为 numpy + torch 处理（`HWC -> CHW`）。

## 5. 伪代码总结

```text
input_ids, attention_mask, images

# text-first
text_outputs, action_logits = text_first(input_ids, attention_mask)

# action selection
action_probs, actions = select_actions(action_logits)

# uncertainty + cost
uncertainty = entropy(text_outputs.logits[:, -1])
expected_cost = sum(action_probs * costs)

# conditional vision
if training and not straight_through:
    logits, vision_tokens = soft_mixture(...)
else:
    logits, vision_tokens = hard_actions(...)

return {logits, action_logits, action_probs, actions,
        expected_cost, uncertainty, vision_tokens, labels}
```

## 6. 动作空间与成本

- `a0 = NO_VISION`，成本 0
- `a1 = COARSE_VISION`，成本 `c1`
- `a2 = FULL_VISION`，成本 `c2`

成本通过 `action_probs` 与真实视觉 token 数加权形成 `expected_cost`，作为训练的 cost loss 组成部分。

## 7. 与训练/评估的交互关系（简述）

- 训练时：`Trainer` 会将 `logits` 与 `labels` 计算任务损失，再加成本损失。
- 评估时：`Trainer.evaluate` 仅计算任务指标，但仍可读取动作分布与成本统计。

## 8. 训练日志字段说明（中文）

训练日志每隔 `training.log_every` 步打印一次（0-based 触发，显示为 `global_step+1`），字段含义如下：

- `stage`：当前训练阶段（如 `stage1_full` 或 `stage2_policy`）。
- `epoch/step/global_step`：当前轮次、当前轮内步、全局步数。
- `progress/eta/elapsed`：训练进度百分比、预计剩余时间、已用时间。
- `window_samples/window_samples_s/window_tokens_s`：当前日志窗口内样本数、样本/秒、token/秒。
- `avg_total/avg_task/avg_cost/avg_cal/avg_gain`：窗口内平均总损失、任务损失、成本损失、校准损失、增益损失。
- `lr`：当前学习率。
- `budget=coarse_long/full_long/coarse_pixels/full_pixels/patch`：视觉预算配置（coarse/full 的长边、像素上限、patch 大小）。
- `token_cap`：视觉 token 上限（`none` 表示未启用）。
- `lambda_cost`：成本损失权重（stage1 通常为 0）。
- `action_entropy`：动作概率平均熵，越大表示策略越不确定。
- `action_ratio`：动作比例，顺序为 `[NO_VISION, COARSE_VISION, FULL_VISION]`。
- `vision_tokens`：本批次平均视觉 token 数估计。
- `token_count_coarse/full`：本批次平均 coarse/full 视觉 token 数估计。
- `expected_cost`：期望成本（基于动作概率与 token 估计）。
- `flops_proxy`：粗略算力代理指标（`vision_tokens × seq_len`）。
- `ece`：token 级别校准误差（越小越好）。
- `latency_ms/mem_peak_mb/tokens_s`：单步延迟、显存峰值、token 吞吐（需开启 profiling）。
- `gain_corr_*`：增益预测与真实增益的相关性（仅在启用 gain supervision 时出现）。

---

如需我补充 **更具体的 tensor shape、模型前向/生成接口兼容性** 或 **将流程与损失函数/日志指标对齐的图示**，告诉我即可。
