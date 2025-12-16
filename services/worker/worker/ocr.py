"""
OCR wrapper with fallback.

API:
- OCR(model_path=None)
- read_pil_image(pil_image) -> str
"""

from typing import Optional
from PIL import Image
import numpy as np

class OCRStub:
    def read_pil_image(self, pil):
        return ""

class OCR:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._backend = None
        self._reader = None
        self._init_backend()

    def _init_backend(self):
        try:
            import easyocr  # type: ignore
            self._reader = easyocr.Reader(['en'], gpu=False)
            self._backend = "easyocr"
            return
        except Exception:
            pass
        # (Optional) ONNX OCR integration could go here

        self._backend = "stub"
        self._reader = OCRStub()

    def read_pil_image(self, pil_image: Image.Image) -> str:
        try:
            if self._backend == "easyocr":
                arr = np.array(pil_image)
                data = self._reader.readtext(arr)
                return " ".join([d[1] for d in data]) if data else ""
            else:
                return ""
        except Exception:
            return ""
