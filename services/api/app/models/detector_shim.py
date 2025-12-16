"""
Production-aware detector shim.

Behavior:
- If ultralytics YOLO is installed and a YOLO model path is provided, please use it.
- Else if onnxruntime is available and an ONNX path is provided, please use ONNX runtime.
- Else fallback to a no-op stub (returns []) so tests & CI work.

API:
- Detector(model_path=None, onnx_path=None)
- predict(np_img) -> list of dicts: {"label":str,"score":float,"bbox":[x1,y1,x2,y2]}

"""

from typing import List, Dict, Optional
import numpy as np

class DetectorStub:
    def __init__(self, *args, **kwargs):
        self.names = {0: "person", 1: "shelf", 2: "product_label"}

    def predict(self, img: np.ndarray) -> List[Dict]:
        return []

class Detector:
    def __init__(self, model_path: Optional[str] = None, onnx_path: Optional[str] = None):
        self.model_path = model_path
        self.onnx_path = onnx_path
        self._backend = None
        self._names = None
        self._init_backend()

    def _init_backend(self):
        # Tring ultralytics first
        try:
            from ultralytics import YOLO  # type: ignore
            self._backend = "yolo"
            if self.model_path:
                self.model = YOLO(self.model_path)
            else:
                # default small model
                self.model = YOLO("yolov8n.pt")
            self._names = getattr(self.model, "names", None)
            return
        except Exception:
            pass

        # Tring ONNX
        try:
            import onnxruntime as ort  
            if self.onnx_path:
                self._backend = "onnx"
                self.ort_sess = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
                self._names = {0: "person", 1: "shelf", 2: "product_label"}
                return
        except Exception:
            pass

        # fallback
        self._backend = "stub"
        self.model = DetectorStub()
        self._names = self.model.names

    def _postprocess_yolo(self, results) -> List[Dict]:
        out = []
        boxes = results.boxes
        # boxes.xyxy, boxes.conf, boxes.cls
        try:
            xy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            clss = boxes.cls.tolist()
            for box, score, cls in zip(xy, confs, clss):
                x1,y1,x2,y2 = map(int, box)
                label = self._names.get(int(cls), str(int(cls))) if isinstance(self._names, dict) else str(int(cls))
                out.append({"label": label, "score": float(score), "bbox": [x1,y1,x2,y2]})
        except Exception:
            # safe fallback
            pass
        return out

    def predict(self, img: np.ndarray) -> List[Dict]:
        if self._backend == "yolo":
            try:
                results = self.model(img, imgsz=640)[0]
                return self._postprocess_yolo(results)
            except Exception:
                return []
        elif self._backend == "onnx":
            # Basic ONNX runtime inference skeleton (user must adapt input preprocessing)
            try:
                import cv2
                h, w = img.shape[:2]
                # preprocess to 640x640, normalize
                inp = cv2.resize(img, (640, 640))
                inp = inp[:, :, ::-1]  # BGR->RGB if needed; best-effort
                inp = inp.astype("float32") / 255.0
                inp = np.transpose(inp, (2,0,1))[np.newaxis, :].astype("float32")
                inputs = {self.ort_sess.get_inputs()[0].name: inp}
                out = self.ort_sess.run(None, inputs)
                # postprocessing is model-specific
                return []
            except Exception:
                return []
        else:
            return [] 