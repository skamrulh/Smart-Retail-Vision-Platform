import base64
import io
import asyncio
import json
import numpy as np
from PIL import Image
from .models.model_loader import DetectionModel, OCRModel, ReportGenerator
from .redis_client import RedisClient

class Predictor:
    def __init__(self):
        self.detector = DetectionModel.load_default()
        self.ocr = OCRModel.load_default()
        self.reporter = ReportGenerator.load_default()
        self.redis = RedisClient.from_env()

    async def process_frame(self, frame_base64: str):
        # decoding
        image_bytes = base64.b64decode(frame_base64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(img)

        # running detection
        dets = self.detector.predict(np_img)

        # OCR for shelf ROIs (if any)
        ocr_results = {}
        shelf_rois = [d for d in dets if d["label"] in ["shelf", "product_label", "price_tag"]]
        for i, roi in enumerate(shelf_rois):
            x1,y1,x2,y2 = roi["bbox"]
            # clamp coords
            x1,y1,x2,y2 = max(0,int(x1)), max(0,int(y1)), int(x2), int(y2)
            crop = img.crop((x1,y1,x2,y2))
            text = self.ocr.read_pil_image(crop)
            ocr_results[f"roi_{i}"] = text

        # publishing events
        event = {
            "objects": dets,
            "ocr": ocr_results,
            "timestamp": asyncio.get_event_loop().time()
        }
        # fire-and-forget publishing
        try:
            await self.redis.publish_event("frames", event)
        except Exception:
            # prevent failure the request if redis is unavailable
            pass

        insight = self.reporter.summarize_quick(dets, ocr_results)

        return {
            "objects": dets,
            "ocr_text": ocr_results,
            "meta": {"insight": insight}
        }
