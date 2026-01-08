"""Vision transforms for different budgets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from PIL import Image

from src.models.vision_budget import VisionBudgetController


@dataclass
class VisionTransform:
    """Resize images according to a vision budget mode."""

    budget: VisionBudgetController
    mode: str

    def __call__(self, images: List[Image.Image]) -> List[Image.Image]:
        return [self.budget.prepare_image(img, self.mode) for img in images]
