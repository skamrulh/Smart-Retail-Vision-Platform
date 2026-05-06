"""
Inference pipeline: decode frame → detect → OCR → generate insight.

The Predictor is instantiated once (via FastAPI lifespan) and stored on
app.state — not at module level — so the API can start cleanly in
environments without Redis or ML weights (e.g. CI).
"""
import base64
import io
import time
import logging
import numpy as np
from PIL import Image

from .models.model_loader import DetectionModel, OCRModel, ReportGenerator
from .redis_client import RedisClient
from .health import FRAMES_PROCESSED

logger = logging.getLogger(__name__)


class Predictor:
    """
    Stateful inference object held on app.state.

    All three ML models use lazy loading: weights are only downloaded /
    loaded on the first call, not at construction time.
    """

    def __init__(self):
        self.detector  = DetectionModel.load_default()
        self.ocr       = OCRModel.load_default()
        self.reporter  = ReportGenerator.load_default()
        self.redis     = RedisClient.from_env()

    async def process_frame(self, frame_base64: str) -> dict:
        """
        Full per-frame pipeline: decode → detect → OCR → insight → publish.

        Args:
            frame_base64: Base64-encoded JPEG frame.

        Returns:
            Dict with 'objects', 'ocr_text', and 'meta' keys.
        """
        # ── Decode ────────────────────────────────────────────────────────────
        image_bytes = base64.b64decode(frame_base64)
        img   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(img)

        # ── Detection ─────────────────────────────────────────────────────────
        dets = self.detector.predict(np_img)

        # ── OCR for shelf / label ROIs ─────────────────────────────────────────
        ocr_results: dict = {}
        for i, roi in enumerate(dets):
            if roi.get("label") in ("shelf", "product_label", "price_tag"):
                x1, y1, x2, y2 = [max(0, int(v)) for v in roi["bbox"]]
                if x2 > x1 and y2 > y1:
                    crop = img.crop((x1, y1, x2, y2))
                    ocr_results[f"roi_{i}"] = self.ocr.read_pil_image(crop)

        # ── Publish event to Redis Stream ──────────────────────────────────────
        event = {
            "objects":   dets,
            "ocr":       ocr_results,
            "timestamp": time.time(),   # was asyncio.get_event_loop().time() — deprecated
        }
        try:
            await self.redis.publish_event("frames", event)
        except Exception as e:
            logger.debug(f"Redis publish skipped (offline mode): {e}")

        # ── Generate NL insight ───────────────────────────────────────────────
        insight = self.reporter.summarize_quick(dets, ocr_results)
        FRAMES_PROCESSED.inc()

        return {
            "objects":  dets,
            "ocr_text": ocr_results,
            "meta":     {"insight": insight, "frame_timestamp": event["timestamp"]},
        }
