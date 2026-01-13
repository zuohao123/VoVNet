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
    calibration_lambda: float = 0.0
    gain_supervision: bool = False
    gain_loss_type: str = "mse"
    gain_loss_weight: float = 0.0
    gain_margin: float = 0.0


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
        if self.training.deepspeed_stage not in (0, 2, 3):
            raise ValueError("deepspeed_stage must be 0, 2, or 3")
        if self.policy.policy_mode not in ("logits", "gain"):
            raise ValueError("policy_mode must be logits or gain")
        if self.policy.fallback_mode not in ("none", "coarse", "full"):
            raise ValueError("fallback_mode must be none, coarse, or full")
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
