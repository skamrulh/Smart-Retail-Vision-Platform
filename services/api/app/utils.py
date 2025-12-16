from typing import Tuple
from PIL import Image
import numpy as np

def pil_from_base64(b64: str) -> Image.Image:
    import base64, io
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def bbox_clamp(bbox, width: int, height: int):
    x1,y1,x2,y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(width, int(x2)); x3 = min(height, int(y2))
    return [x1,y1,x2,x3]
