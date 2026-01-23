"""Configuration utilities for VoVNet."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ModelConfig:
    """Model and LoRA settings."""

    base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    full_model_name: str = "Qwen/Qwen3-VL-8B-Thinking"
    use_thinking_for_full: bool = False
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    freeze_vision_encoder: bool = True


@dataclass
class PolicyConfig:
    """VoV policy settings."""

    vow_hidden_dim: int = 256
    baseline_name: str | None = None
    baseline_uncertainty: str = "entropy"
    baseline_threshold: float = 0.5
    baseline_vision: str = "full"
    baseline_seed: int | None = None
    baseline_target_ratios: List[float] | None = None
    baseline_bucket_ratios: List[List[float]] | None = None
    baseline_bucket_thresholds: List[float] | None = None
    baseline_pruning_ratio: float = 1.0
    baseline_pruning_mode: str = "stride"
    finetune_pruning: bool = False
    baseline_merge_ratio: float = 1.0
    baseline_merge_mode: str = "cosine"
    baseline_merge_weight: str = "norm"
    baseline_enable_prune: bool = False
    baseline_prune_ratio: float = 1.0
    baseline_prune_mode: str = "topk_norm"
    baseline_pool_factor: int = 1
    gumbel_tau: float = 1.0
    use_straight_through: bool = True
    eval_sample: bool = False
    policy_mode: str = "logits"
    fallback_mode: str = "none"
    fallback_entropy_threshold: float | None = None
    fallback_margin_threshold: float | None = None
    cost_scale: float = 1.0
    cost_c1: float = 1.0
    cost_c2: float = 4.0
    lambda_cost: float = 0.1
    cost_warmup_steps: int = 0
    calibration_lambda: float = 0.0
    entropy_weight: float = 0.0
    gain_supervision: bool = False
    gain_loss_type: str = "mse"
    gain_loss_weight: float = 0.0
    gain_margin: float = 0.0
    explore_prob: float = 0.0


@dataclass
class VisionBudgetConfig:
    """Vision budget control for coarse/full settings."""

    coarse_long_side: int = 336
    full_long_side: int = 672
    coarse_max_pixels: int = 336 * 336
    full_max_pixels: int = 672 * 672
    patch_size: int = 14
    token_cap: int | None = None


@dataclass
class DataConfig:
    """Dataset and prompt settings."""

    train_jsonl: str | None = None
    eval_jsonl: str | None = None
    hf_dataset_name: str | None = None
    hf_dataset_split: str = "train"
    text_field: str = "question"
    answer_field: str = "answer"
    image_field: str = "image"
    prompt_template: str = "Question: {question}\nAnswer:"
    max_samples: int | None = None


@dataclass
class TrainingConfig:
    """Training hyperparameters and runtime config."""

    output_dir: str = "outputs"
    per_device_batch_size: int = 1
    gradient_accumulation: int = 4
    epochs: int = 1
    stage1_epochs: int = 0
    stage1_max_steps: int | None = None
    stage1_baseline_name: str | None = None
    stage1_lambda_cost: float | None = None
    stage2_max_steps: int | None = None
    lr: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    log_every: int = 10
    save_every: int = 500
    deepspeed_stage: int = 2
    use_fsdp: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42
    profile: bool = False


@dataclass
class EvalConfig:
    """Evaluation settings."""

    batch_size: int = 1
    max_new_tokens: int = 32
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    profile: bool = False
    log_pred_interval: int = 0
    log_pred_examples: int = 1
    log_pred_max_chars: int = 200


@dataclass
class Config:
    """Top-level configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    vision_budget: VisionBudgetConfig = field(default_factory=VisionBudgetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep dict copy of the config."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls()
        _update_dataclass(cfg, data)
        cfg._validate()
        return cfg

    def update_from_yaml(self, path: str | Path) -> None:
        """Update the config in-place from a YAML file."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        _update_dataclass(self, data)
        self._validate()

    def _validate(self) -> None:
        _coerce_types(self)
        if self.training.deepspeed_stage not in (0, 2, 3):
            raise ValueError("deepspeed_stage must be 0, 2, or 3")
        if self.training.stage1_epochs < 0:
            raise ValueError("stage1_epochs must be >= 0")
        if self.training.stage1_epochs > self.training.epochs:
            raise ValueError("stage1_epochs must be <= total epochs")
        if self.training.stage1_max_steps is not None and self.training.stage1_max_steps <= 0:
            raise ValueError("stage1_max_steps must be > 0 when set")
        if self.training.stage2_max_steps is not None and self.training.stage2_max_steps <= 0:
            raise ValueError("stage2_max_steps must be > 0 when set")
        if self.training.stage1_baseline_name:
            stage1 = self.training.stage1_baseline_name.strip().lower()
            if stage1 not in {
                "always_full",
                "full",
                "always_coarse",
                "coarse",
                "no_vision",
                "no_vision_only",
                "no",
                "none",
                "null",
            }:
                raise ValueError(
                    "stage1_baseline_name must be always_full, always_coarse, no_vision, or null"
                )
        if self.policy.policy_mode not in ("logits", "gain"):
            raise ValueError("policy_mode must be logits or gain")
        if self.policy.fallback_mode not in ("none", "coarse", "full"):
            raise ValueError("fallback_mode must be none, coarse, or full")
        if self.policy.cost_warmup_steps < 0:
            raise ValueError("policy.cost_warmup_steps must be >= 0")
        if self.policy.entropy_weight < 0:
            raise ValueError("policy.entropy_weight must be >= 0")
        if not 0.0 <= self.policy.explore_prob <= 1.0:
            raise ValueError("policy.explore_prob must be between 0 and 1")
        baseline = self.policy.baseline_name
        if baseline is not None:
            normalized = baseline.strip().lower()
            if normalized and normalized not in {
                "always_full",
                "full",
                "always_coarse",
                "coarse",
                "no_vision",
                "no_vision_only",
                "no",
                "uncertainty_threshold",
                "uncertainty",
                "threshold",
                "random_policy_matched",
                "random_matched",
                "vision_token_pruning_proxy",
                "pruning_proxy",
                "vision_pruning",
                "token_merge_prune_proxy",
                "token_merge",
                "multi_granularity_proxy",
                "resolution_scaling",
                "none",
                "null",
            }:
                raise ValueError(
                    "baseline_name must be always_full, always_coarse, no_vision, "
                    "uncertainty_threshold, random_policy_matched, vision_token_pruning_proxy, "
                    "token_merge_prune_proxy, multi_granularity_proxy, resolution_scaling, or null"
                )
        uncertainty = self.policy.baseline_uncertainty.strip().lower()
        if uncertainty not in {"entropy", "margin"}:
            raise ValueError("baseline_uncertainty must be entropy or margin")
        baseline_vision = self.policy.baseline_vision.strip().lower()
        if baseline_vision not in {"full", "coarse"}:
            raise ValueError("baseline_vision must be full or coarse")
        if baseline is not None:
            normalized = baseline.strip().lower()
            if normalized in {"random_policy_matched", "random_matched"}:
                ratios = self.policy.baseline_target_ratios
                bucket_ratios = self.policy.baseline_bucket_ratios
                if not bucket_ratios and not ratios:
                    raise ValueError(
                        "random_policy_matched requires baseline_target_ratios or baseline_bucket_ratios"
                    )
                if ratios is not None and len(ratios) != 3:
                    raise ValueError("baseline_target_ratios must have 3 values")
                if bucket_ratios is not None:
                    if len(bucket_ratios) != 3:
                        raise ValueError("baseline_bucket_ratios must have 3 buckets")
                    for bucket in bucket_ratios:
                        if len(bucket) != 3:
                            raise ValueError("each bucket ratio must have 3 values")
                thresholds = self.policy.baseline_bucket_thresholds
                if thresholds is not None:
                    if len(thresholds) != 2:
                        raise ValueError("baseline_bucket_thresholds must have 2 values")
                    if thresholds[0] >= thresholds[1]:
                        raise ValueError("baseline_bucket_thresholds must be increasing")
            if normalized in {"vision_token_pruning_proxy", "pruning_proxy", "vision_pruning"}:
                ratio = float(self.policy.baseline_pruning_ratio)
                if ratio <= 0.0 or ratio > 1.0:
                    raise ValueError("baseline_pruning_ratio must be in (0, 1]")
                mode = self.policy.baseline_pruning_mode.strip().lower()
                if mode not in {"stride", "topk_norm", "topk"}:
                    raise ValueError("baseline_pruning_mode must be stride or topk_norm")
            if normalized in {"token_merge_prune_proxy", "token_merge"}:
                merge_ratio = float(self.policy.baseline_merge_ratio)
                if merge_ratio <= 0.0 or merge_ratio > 1.0:
                    raise ValueError("baseline_merge_ratio must be in (0, 1]")
                merge_mode = self.policy.baseline_merge_mode.strip().lower()
                if merge_mode not in {"cosine", "l2"}:
                    raise ValueError("baseline_merge_mode must be cosine or l2")
                merge_weight = self.policy.baseline_merge_weight.strip().lower()
                if merge_weight not in {"norm", "mean"}:
                    raise ValueError("baseline_merge_weight must be norm or mean")
                if self.policy.baseline_enable_prune:
                    prune_ratio = float(self.policy.baseline_prune_ratio)
                    if prune_ratio <= 0.0 or prune_ratio > 1.0:
                        raise ValueError("baseline_prune_ratio must be in (0, 1]")
                    prune_mode = self.policy.baseline_prune_mode.strip().lower()
                    if prune_mode not in {"stride", "topk_norm", "topk"}:
                        raise ValueError("baseline_prune_mode must be stride or topk_norm")
            if normalized in {"multi_granularity_proxy"}:
                pool_factor = int(self.policy.baseline_pool_factor)
                if pool_factor < 1:
                    raise ValueError("baseline_pool_factor must be >= 1")
        if self.policy.gain_loss_type not in (
            "mse",
            "huber",
            "rank_hinge",
            "rank_logistic",
        ):
            raise ValueError("gain_loss_type must be mse, huber, rank_hinge, rank_logistic")


def _update_dataclass(obj: Any, updates: Dict[str, Any]) -> None:
    """Recursively update a dataclass instance from a dict."""
    for key, value in updates.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Expected dict for {key}, got {type(value)}")
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _to_float(value: Any, field: str, allow_none: bool = False) -> Any:
    if value is None:
        return None if allow_none else value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"none", "null", ""}:
            return None if allow_none else value
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{field} must be a float") from exc
    return value


def _to_int(value: Any, field: str, allow_none: bool = False) -> Any:
    if value is None:
        return None if allow_none else value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"none", "null", ""}:
            return None if allow_none else value
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError as exc:
                raise ValueError(f"{field} must be an int") from exc
    return value


def _to_float_list(value: Any, field: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of floats")
    return [float(item) for item in value]


def _to_float_list_list(value: Any, field: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of float lists")
    return [_to_float_list(item, field) for item in value]


def _coerce_types(cfg: Config) -> None:
    cfg.training.lr = _to_float(cfg.training.lr, "training.lr")
    cfg.training.weight_decay = _to_float(cfg.training.weight_decay, "training.weight_decay")
    cfg.training.warmup_steps = _to_int(cfg.training.warmup_steps, "training.warmup_steps")
    cfg.training.max_grad_norm = _to_float(cfg.training.max_grad_norm, "training.max_grad_norm")
    cfg.training.per_device_batch_size = _to_int(
        cfg.training.per_device_batch_size, "training.per_device_batch_size"
    )
    cfg.training.gradient_accumulation = _to_int(
        cfg.training.gradient_accumulation, "training.gradient_accumulation"
    )
    cfg.training.epochs = _to_int(cfg.training.epochs, "training.epochs")
    cfg.training.stage1_epochs = _to_int(cfg.training.stage1_epochs, "training.stage1_epochs")
    cfg.training.stage1_max_steps = _to_int(
        cfg.training.stage1_max_steps, "training.stage1_max_steps", allow_none=True
    )
    cfg.training.stage1_lambda_cost = _to_float(
        cfg.training.stage1_lambda_cost, "training.stage1_lambda_cost", allow_none=True
    )
    cfg.training.stage2_max_steps = _to_int(
        cfg.training.stage2_max_steps, "training.stage2_max_steps", allow_none=True
    )
    cfg.training.log_every = _to_int(cfg.training.log_every, "training.log_every")
    cfg.training.save_every = _to_int(cfg.training.save_every, "training.save_every")
    cfg.training.deepspeed_stage = _to_int(cfg.training.deepspeed_stage, "training.deepspeed_stage")
    cfg.training.seed = _to_int(cfg.training.seed, "training.seed")

    cfg.eval.batch_size = _to_int(cfg.eval.batch_size, "eval.batch_size")
    cfg.eval.max_new_tokens = _to_int(cfg.eval.max_new_tokens, "eval.max_new_tokens")
    cfg.eval.num_beams = _to_int(cfg.eval.num_beams, "eval.num_beams")
    cfg.eval.temperature = _to_float(cfg.eval.temperature, "eval.temperature")
    cfg.eval.log_pred_interval = _to_int(cfg.eval.log_pred_interval, "eval.log_pred_interval")
    cfg.eval.log_pred_examples = _to_int(cfg.eval.log_pred_examples, "eval.log_pred_examples")
    cfg.eval.log_pred_max_chars = _to_int(cfg.eval.log_pred_max_chars, "eval.log_pred_max_chars")

    cfg.policy.gumbel_tau = _to_float(cfg.policy.gumbel_tau, "policy.gumbel_tau")
    cfg.policy.cost_scale = _to_float(cfg.policy.cost_scale, "policy.cost_scale")
    cfg.policy.cost_c1 = _to_float(cfg.policy.cost_c1, "policy.cost_c1")
    cfg.policy.cost_c2 = _to_float(cfg.policy.cost_c2, "policy.cost_c2")
    cfg.policy.lambda_cost = _to_float(cfg.policy.lambda_cost, "policy.lambda_cost")
    cfg.policy.cost_warmup_steps = _to_int(
        cfg.policy.cost_warmup_steps, "policy.cost_warmup_steps"
    )
    cfg.policy.calibration_lambda = _to_float(
        cfg.policy.calibration_lambda, "policy.calibration_lambda"
    )
    cfg.policy.entropy_weight = _to_float(
        cfg.policy.entropy_weight, "policy.entropy_weight"
    )
    cfg.policy.gain_loss_weight = _to_float(
        cfg.policy.gain_loss_weight, "policy.gain_loss_weight"
    )
    cfg.policy.gain_margin = _to_float(cfg.policy.gain_margin, "policy.gain_margin")
    cfg.policy.explore_prob = _to_float(
        cfg.policy.explore_prob, "policy.explore_prob"
    )
    cfg.policy.baseline_threshold = _to_float(
        cfg.policy.baseline_threshold, "policy.baseline_threshold"
    )
    cfg.policy.baseline_seed = _to_int(
        cfg.policy.baseline_seed, "policy.baseline_seed", allow_none=True
    )
    cfg.policy.baseline_target_ratios = _to_float_list(
        cfg.policy.baseline_target_ratios, "policy.baseline_target_ratios"
    )
    cfg.policy.baseline_bucket_ratios = _to_float_list_list(
        cfg.policy.baseline_bucket_ratios, "policy.baseline_bucket_ratios"
    )
    cfg.policy.baseline_bucket_thresholds = _to_float_list(
        cfg.policy.baseline_bucket_thresholds, "policy.baseline_bucket_thresholds"
    )
    cfg.policy.baseline_pruning_ratio = _to_float(
        cfg.policy.baseline_pruning_ratio, "policy.baseline_pruning_ratio"
    )
    if isinstance(cfg.policy.baseline_pruning_mode, str):
        cfg.policy.baseline_pruning_mode = cfg.policy.baseline_pruning_mode.strip().lower()
    cfg.policy.baseline_merge_ratio = _to_float(
        cfg.policy.baseline_merge_ratio, "policy.baseline_merge_ratio"
    )
    if isinstance(cfg.policy.baseline_merge_mode, str):
        cfg.policy.baseline_merge_mode = cfg.policy.baseline_merge_mode.strip().lower()
    if isinstance(cfg.policy.baseline_merge_weight, str):
        cfg.policy.baseline_merge_weight = cfg.policy.baseline_merge_weight.strip().lower()
    if isinstance(cfg.policy.baseline_enable_prune, str):
        cfg.policy.baseline_enable_prune = (
            cfg.policy.baseline_enable_prune.strip().lower() in {"1", "true", "yes"}
        )
    cfg.policy.baseline_prune_ratio = _to_float(
        cfg.policy.baseline_prune_ratio, "policy.baseline_prune_ratio"
    )
    if isinstance(cfg.policy.baseline_prune_mode, str):
        cfg.policy.baseline_prune_mode = cfg.policy.baseline_prune_mode.strip().lower()
    cfg.policy.baseline_pool_factor = _to_int(
        cfg.policy.baseline_pool_factor, "policy.baseline_pool_factor"
    )

    cfg.vision_budget.coarse_long_side = _to_int(
        cfg.vision_budget.coarse_long_side, "vision_budget.coarse_long_side"
    )
    cfg.vision_budget.full_long_side = _to_int(
        cfg.vision_budget.full_long_side, "vision_budget.full_long_side"
    )
    cfg.vision_budget.coarse_max_pixels = _to_int(
        cfg.vision_budget.coarse_max_pixels, "vision_budget.coarse_max_pixels"
    )
    cfg.vision_budget.full_max_pixels = _to_int(
        cfg.vision_budget.full_max_pixels, "vision_budget.full_max_pixels"
    )
    cfg.vision_budget.patch_size = _to_int(
        cfg.vision_budget.patch_size, "vision_budget.patch_size"
    )
    cfg.vision_budget.token_cap = _to_int(
        cfg.vision_budget.token_cap, "vision_budget.token_cap", allow_none=True
    )
