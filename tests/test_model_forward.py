"""VoVNet forward smoke test with a dummy VLM."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch
from PIL import Image

from src.models.base_vlm import VLMOutputs
from src.models.vovnet import VoVNet
from src.models.vision_budget import VisionBudgetController


class DummyHFModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 8, vocab_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
        pixel_values: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
    ) -> Any:
        batch, seq = input_ids.shape
        hidden = torch.randn(batch, seq, self.config.hidden_size)
        logits = torch.randn(batch, seq, self.vocab_size)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,), past_key_values=None)


class DummyVLM:
    def __init__(self) -> None:
        self.model = DummyHFModel()
        self.processor = None
        self.tokenizer = None

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = True
    ) -> VLMOutputs:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=use_cache,
        )
        return VLMOutputs(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            past_key_values=None,
        )

    def forward_with_vision(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
    ) -> Any:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=use_cache,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
        )


def test_vovnet_forward() -> None:
    vlm = DummyVLM()
    budget = VisionBudgetController(
        coarse_long_side=32,
        full_long_side=64,
        coarse_max_pixels=1024,
        full_max_pixels=4096,
    )
    model = VoVNet(
        base_vlm=vlm,
        full_vlm=None,
        vision_budget=budget,
        vow_hidden_dim=16,
        gumbel_tau=1.0,
        use_straight_through=True,
        eval_sample=False,
        cost_c1=1.0,
        cost_c2=2.0,
    )

    input_ids = torch.randint(0, 10, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    images = [Image.new("RGB", (64, 64), color=0) for _ in range(2)]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
    assert "logits" in outputs
    assert outputs["logits"].shape[0] == 2
    assert outputs["action_probs"].shape[-1] == 3
