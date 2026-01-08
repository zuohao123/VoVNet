"""Vision budget controls for coarse and full modes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image


@dataclass
class VisionBudgetController:
    """Resize and estimate visual token usage for different budgets."""

    coarse_long_side: int
    full_long_side: int
    coarse_max_pixels: int
    full_max_pixels: int
    patch_size: int = 14
    token_cap: Optional[int] = None

    def prepare_image(self, image: Image.Image, mode: str) -> Image.Image:
        """Resize the image according to the chosen mode."""
        if mode not in {"coarse", "full"}:
            return image
        if mode == "coarse":
            long_side = self.coarse_long_side
            max_pixels = self.coarse_max_pixels
        else:
            long_side = self.full_long_side
            max_pixels = self.full_max_pixels
        return _resize_image(image, long_side=long_side, max_pixels=max_pixels)

    def estimate_visual_tokens(self, image: Image.Image) -> int:
        """Estimate the number of visual tokens for the image.

        This uses a patch-size heuristic. TODO: align with model-specific visual tokenizer.
        """
        width, height = image.size
        tokens = int((width * height + self.patch_size**2 - 1) // (self.patch_size**2))
        if self.token_cap is not None:
            tokens = min(tokens, self.token_cap)
        return tokens


def _resize_image(image: Image.Image, long_side: int, max_pixels: int) -> Image.Image:
    """Resize image to satisfy long-side and max pixel constraints."""
    width, height = image.size
    if width == 0 or height == 0:
        return image

    scale_long = long_side / max(width, height)
    scale_pixels = (max_pixels / (width * height)) ** 0.5
    scale = min(1.0, scale_long, scale_pixels)

    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    if new_w == width and new_h == height:
        return image
    return image.resize((new_w, new_h), resample=Image.BICUBIC)
