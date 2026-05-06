"""
Frame processing worker.

Reads base64-encoded frames, runs detection + OCR, and publishes
structured events to the Redis Stream for downstream consumers.

Key design decisions:
- _detector and _ocr are module-level singletons: models are loaded once
  on first use, not recreated for every frame (avoids catastrophic startup
  cost on each call if YOLO weights are present).
- redis is also a module-level singleton with async methods so coroutines
  can await it correctly.
"""
import json
import base64
import io
import logging
import numpy as np
from PIL import Image

from .detector import DetectionModel
from .ocr import OCR
from .redis_client import RedisClient

logger = logging.getLogger(__name__)

# ── Module-level singletons — loaded once, reused every frame ─────────────────
_detector = DetectionModel()
_ocr      = OCR()
_redis    = RedisClient()        # async-safe: connection is lazy


async def process_frame_base64(frame_b64: str) -> dict:
    """
    Decode a base64 JPEG frame, run detection + OCR, publish results.

    Args:
        frame_b64: Base64-encoded JPEG bytes.

    Returns:
        Dict with 'detections' and 'ocr' keys.
    """
    # ── Decode ────────────────────────────────────────────────────────────────
    try:
        image_bytes = base64.b64decode(frame_b64)
        img   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        npimg = np.array(img)
    except Exception as e:
        logger.warning(f"Frame decode failed: {e}")
        return {"detections": [], "ocr": {}}

    # ── Detection ─────────────────────────────────────────────────────────────
    dets = _detector.predict(npimg)

    # ── OCR for shelf / label ROIs ────────────────────────────────────────────
    ocr_results: dict = {}
    for i, d in enumerate(dets):
        if d.get("label") in ("product_label", "price_tag", "shelf"):
            x1, y1, x2, y2 = [max(0, int(v)) for v in d.get("bbox", [0, 0, 0, 0])]
            if x2 > x1 and y2 > y1:         # skip degenerate boxes
                crop = img.crop((x1, y1, x2, y2))
                ocr_results[f"roi_{i}"] = _ocr.read_pil_image(crop)

    # ── Publish to Redis Stream ───────────────────────────────────────────────
    event = {"detections": dets, "ocr": ocr_results}
    await _redis.xadd("processing:events", {"data": json.dumps(event)})

    return event
