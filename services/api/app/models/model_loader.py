# abstraction layer for detection, ocr, and reporting
from typing import List, Dict, Any
import numpy as np

# Using lazy imports to avoid heavy imports at import time (makes tests easier).
class DetectionModel:
    @staticmethod
    def load_default():
        return DetectionModel()

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self._model = None
        self._names = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path or "yolov8n.pt")
            self._names = self._model.names
        except Exception:
            # fallback stub for environments without ultralytics
            self._model = None
            self._names = {0: "person", 1: "shelf", 2: "product_label"}

    def predict(self, img: np.ndarray) -> List[Dict[str, Any]]:
        self._load_model()
        if self._model is None:
            # return empty or synthetic detection for local dev
            h, w = img.shape[:2]
            return []  
        results = self._model(img, imgsz=640)[0]
        out = []
        boxes = results.boxes
        for box, score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            x1,y1,x2,y2 = map(int, box)
            label = self._names[int(cls)] if self._names else str(int(cls))
            out.append({"label": label, "score": float(score), "bbox": [x1,y1,x2,y2]})
        return out

class OCRModel:
    @staticmethod
    def load_default():
        return OCRModel()

    def __init__(self):
        self._reader = None

    def _load(self):
        if self._reader is not None:
            return
        try:
            import easyocr
            self._reader = easyocr.Reader(['en'], gpu=False)
        except Exception:
            self._reader = None

    def read_pil_image(self, pil_image) -> str:
        self._load()
        import numpy as _np
        if self._reader is None:
            return ""
        data = self._reader.readtext(_np.array(pil_image))
        return " ".join([d[1] for d in data])

class ReportGenerator:
    @staticmethod
    def load_default():
        return ReportGenerator()

    def __init__(self):
        self._nlp = None

    def _load(self):
        if self._nlp is not None:
            return
        try:
            from transformers import pipeline
            self._nlp = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            self._nlp = None

    def summarize_quick(self, detections, ocr):
        counts = {}
        for d in detections:
            counts[d.get("label","unknown")] = counts.get(d.get("label","unknown"), 0) + 1
        text = "Detected counts: " + ", ".join(f"{k}:{v}" for k,v in counts.items())
        self._load()
        if self._nlp is None:
            return text
        try:
            out = self._nlp(text, max_length=50, min_length=10, truncation=True)
            return out[0].get("summary_text", text)
        except Exception:
            return text
