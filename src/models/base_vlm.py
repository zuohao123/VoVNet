"""Base VLM loader with safe fallbacks."""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

def _ensure_torch_compiler_compat() -> None:
    """Backfill torch.compiler.is_compiling for older torch releases."""
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        class _CompilerShim:
            @staticmethod
            def is_compiling() -> bool:
                return False

        torch.compiler = _CompilerShim()  # type: ignore[attr-defined]
        return
    if not hasattr(compiler, "is_compiling"):
        def _is_compiling() -> bool:
            return False

        setattr(compiler, "is_compiling", _is_compiling)


_ensure_torch_compiler_compat()

import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


@dataclass
class VLMOutputs:
    """Container for model outputs used by VoVNet."""

    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    past_key_values: Optional[Any]


class BaseVLM(nn.Module):
    """Load Qwen3-VL models with conservative defaults."""

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        torch_dtype: str = "bfloat16",
        device_map: Optional[str | Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = getattr(torch, torch_dtype)
        self.device_map = device_map

        self.processor = self._load_processor()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_processor(self) -> Optional[Any]:
        try:
            return AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
        except Exception:
            return None

    def _load_tokenizer(self) -> Any:
        try:
            return AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
        except Exception:
            return None

    def _load_model(self) -> nn.Module:
        auto_vision_cls = getattr(transformers, "AutoModelForVision2Seq", None)
        if auto_vision_cls is not None:
            try:
                return auto_vision_cls.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                )
            except Exception:
                pass
        try:
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
        except Exception:
            return AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )

    def supports_vision(self) -> bool:
        """Check whether the model forward supports pixel_values."""
        signature = inspect.signature(self.model.forward)
        if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
            return True
        return "pixel_values" in signature.parameters

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing if available."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def apply_lora(
        self,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: list[str],
    ) -> None:
        """Attach LoRA adapters to the base model."""
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("peft is required for LoRA training") from exc

        logger = logging.getLogger(__name__)
        selected_targets = self._resolve_lora_targets(target_modules)
        if selected_targets != target_modules:
            logger.warning(
                "LoRA target_modules not found; falling back to %s",
                selected_targets,
            )

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=selected_targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

    def _resolve_lora_targets(self, target_modules: list[str]) -> list[str]:
        if not target_modules:
            return target_modules
        linear_names = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        for target in target_modules:
            if any(name.endswith(target) for name in linear_names):
                return target_modules

        common = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "proj",
            "wq",
            "wk",
            "wv",
            "wo",
        ]
        fallback = {name.split(".")[-1] for name in linear_names if name.split(".")[-1] in common}
        if fallback:
            return sorted(fallback)
        return target_modules

    def freeze_vision_encoder(self) -> None:
        """Freeze vision-related parameters by name heuristic."""
        for name, param in self.model.named_parameters():
            if "vision" in name or "visual" in name:
                param.requires_grad = False

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = True,
    ) -> VLMOutputs:
        """Text-only forward pass."""
        outputs = self._safe_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=use_cache,
        )
        return VLMOutputs(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            past_key_values=getattr(outputs, "past_key_values", None),
        )

    def forward_with_vision(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
    ) -> Any:
        """Forward pass that may include vision inputs."""
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": use_cache,
        }
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        return self._safe_forward(**kwargs)

    def generate(self, **kwargs: Any) -> Any:
        """Generate tokens if the model supports it."""
        if not hasattr(self.model, "generate"):
            raise RuntimeError("Underlying model does not implement generate")
        return self.model.generate(**kwargs)

    def _safe_forward(self, **kwargs: Any) -> Any:
        """Call model.forward with only supported keyword arguments."""
        signature = inspect.signature(self.model.forward)
        if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
            return self.model(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return self.model(**filtered)
