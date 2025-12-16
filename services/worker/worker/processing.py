import json
import base64
import io
from PIL import Image
from .detector import DetectionModel
from .ocr import OCRModel
from .redis_client import RedisClient

redis = RedisClient()

async def process_frame_base64(frame_b64):
    image_bytes = base64.b64decode(frame_b64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    npimg = None
    try:
        import numpy as np
        npimg = np.array(img)
    except Exception:
        npimg = None

    dets = DetectionModel().predict(npimg) if npimg is not None else []
    ocr_results = {}
    for i, d in enumerate(dets):
        if d.get("label") in ("product_label", "price_tag", "shelf"):
            x1,y1,x2,y2 = d.get("bbox", [0,0,0,0])
            crop = img.crop((x1,y1,x2,y2))
            txt = OCRModel().read_pil_image(crop)
            ocr_results[f"roi_{i}"] = txt

    event = {"detections": dets, "ocr": ocr_results}
    try:
        await redis.xadd("processing:events", {"data": json.dumps(event)})
    except Exception:
        pass
