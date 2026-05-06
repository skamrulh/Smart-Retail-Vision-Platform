"""Utility helpers for the Smart Retail API."""

import base64
import io
from typing import List

import numpy as np
from PIL import Image


def pil_from_base64(b64: str) -> Image.Image:
    """Decode a base64 string to a PIL RGB image."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def bbox_clamp(bbox: List, width: int, height: int) -> List[int]:
    """
    Clamp a bounding box to image dimensions.

    Args:
        bbox:   [x1, y1, x2, y2] (may be float or out-of-bounds)
        width:  Image width in pixels.
        height: Image height in pixels.

    Returns:
        [x1, y1, x2, y2] as ints, clamped to [0, width] × [0, height].
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width,  int(x2))
    y2 = min(height, int(y2))   # was: x3 = min(height, int(y2)) — wrong variable name
    return [x1, y1, x2, y2]


def np_from_base64(b64: str) -> np.ndarray:
    """Decode a base64 string directly to a numpy RGB array."""
    return np.array(pil_from_base64(b64))
