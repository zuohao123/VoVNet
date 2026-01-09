"""Image utilities for dataset preparation."""
from __future__ import annotations

import hashlib
import io
import logging
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def image_to_jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    """Convert PIL image to JPEG bytes deterministically."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def save_image(image: Image.Image, path: Path, quality: int = 95) -> str:
    """Save image as JPEG and return SHA1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = image_to_jpeg_bytes(image, quality=quality)
    path.write_bytes(data)
    return sha1_bytes(data)


def sha1_bytes(data: bytes) -> str:
    """Compute SHA1 hash for bytes."""
    return hashlib.sha1(data).hexdigest()


def sha1_path(path: Path) -> str:
    """Compute SHA1 for a file path."""
    return sha1_bytes(path.read_bytes())


def load_image_from_url(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """Load image from a URL using urllib."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # pragma: no cover - network dependency
        logger.warning("Failed to load image from url %s: %s", url, exc)
        return None


def load_image_from_path(path: Path) -> Optional[Image.Image]:
    """Load image from a local path."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to load image from path %s: %s", path, exc)
        return None
