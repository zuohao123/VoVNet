# VoVNet 研究动机与方法总结

本文基于当前代码实现与已有说明，概括 Value-of-Vision（VoVNet）的研究动机、核心问题与方法设计，供研究阅读与复现参考。

## 1. 研究动机

视觉语言模型（VLM）在推理时通常默认调用视觉编码器，但在很多场景中：
- 问题可由文本本身直接回答（无需视觉信息）。
- 视觉输入的成本高（显存/延迟/吞吐），在实际部署中不可忽略。
- 不同样本对视觉信息的“价值”不同，统一调用会造成资源浪费。

因此，需要一种 **成本可控、可解释、可训练** 的视觉调用策略：在保证准确率的前提下，尽量减少不必要的视觉开销。

## 2. 研究问题与目标

**核心问题**：对每个输入样本，是否需要视觉？如果需要，应使用哪种视觉预算？

**目标**：
- 在任务准确率与视觉成本之间取得更优的 Pareto 权衡。
- 让策略能够根据文本不确定性与上下文结构，动态选择视觉预算。

## 3. 方法概述（Value-of-Vision）

VoVNet 将“是否调用视觉”转化为一个 **三分类决策**：
- a0 = NO_VISION（不调用视觉）
- a1 = COARSE_VISION（低预算视觉）
- a2 = FULL_VISION（高预算视觉）

方法关键点：
1. **Text-first pass**：先只用文本前向，得到文本隐藏表示与不确定性。
2. **Value-of-Vision 预测**：使用轻量 MLP（vow_head）对文本表征进行特征提取，线性层输出动作 logits。
3. **成本约束的动作选择**：训练时用 Gumbel-Softmax 或软混合，评估时取 argmax（或采样）。
4. **条件视觉调用**：根据动作选择调用视觉编码器，分别走 COARSE 或 FULL 分支。

## 4. 模型结构与关键组件

- **base_vlm**：Qwen3-VL-8B-Instruct（主干模型）。
- **full_vlm（可选）**：Qwen3-VL-8B-Thinking 或更高预算版本。
- **vow_head**：文本隐藏状态池化后的轻量 MLP。
- **policy**：输出动作 logits，决定调用策略。
- **vision_budget**：视觉预算控制器，负责图像缩放与 token 估计。

## 5. 训练流程（概念版）

1) **文本前向**：
   - 仅输入文本，得到隐藏状态与文本 logits。
   - 计算文本不确定性（entropy）。

2) **动作选择**：
   - 训练：Gumbel-Softmax（硬采样）或 softmax（软混合）。
   - 评估：argmax（可选采样）。

3) **条件视觉调用**：
   - NO_VISION：直接使用文本 logits。
   - COARSE_VISION：低分辨率/低 token 预算。
   - FULL_VISION：高分辨率或更高预算模型。

4) **损失函数**：
   - `task_loss`：语言模型任务损失。
   - `cost_loss`：期望视觉成本惩罚（与动作概率加权）。
   - `calibration_loss`：可选的校准损失（当前为占位，便于拓展）。

总损失形式：
`total_loss = task_loss + lambda_cost * expected_cost + lambda_cal * calibration_loss`

## 6. 视觉预算与成本建模

视觉预算由 `VisionBudgetController` 控制：
- **coarse**：小长边与最大像素限制。
- **full**：更高分辨率预算。
- token 数用 patch-size 启发式估计（留有 TODO 适配真实 tokenizer）。

成本模型：
- cost(a0)=0
- cost(a1)=c1
- cost(a2)=c2

训练时对 `action_probs` 做期望成本加权，形成 `expected_cost`。

## 7. 不确定性与策略信号

当前实现主要使用 **文本 logits 的熵** 作为不确定性指标：
`entropy_from_logits(logits_last_token)`

代码中还提供：
- margin（top-1 与 top-2 概率差）
- ECE（Expected Calibration Error）

这些可作为未来策略输入或分析指标扩展。

## 8. 评估与基线

评估侧重点：
- 任务准确率 / EM
- 各动作比例
- 成本与延迟
- Pareto 曲线（准确率 vs 成本）
- 校准指标（ECE）

基线包含：
- Always Full / Always Coarse / No Vision
- Random Policy
- Uncertainty Threshold

## 9. 实践边界与可扩展点

- **KV cache 复用**为“尽力而为”机制：模型不支持则回退重算。
- 视觉 token 估计为启发式，后续可接入真实视觉 tokenizer。
- 可替换不确定性指标或引入更强的价值估计器。
- 支持多模型（full_vlm）提升高预算视觉路径的上限。

---

如果需要更具体的实现级解读（例如 `src/models/vovnet.py` 的张量形状、训练日志字段、不同分支的显存开销对比），告诉我即可补充。
