from pydantic import BaseModel
from typing import List, Dict, Any

class InferenceRequest(BaseModel):
    frame_base64: str  # base64 encoded image

class DetectedObject(BaseModel):
    label: str
    score: float
    bbox: List[int]

class InferenceResponse(BaseModel):
    objects: List[DetectedObject]
    ocr_text: Dict[str, str]
    meta: Dict[str, Any]
