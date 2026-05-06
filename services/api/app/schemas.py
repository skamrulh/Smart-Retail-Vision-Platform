"""Pydantic request/response schemas for the Smart Retail Vision API."""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    frame_base64: str = Field(..., description="Base64-encoded JPEG frame")


class DetectedObject(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    bbox:  List[int] = Field(..., min_length=4, max_length=4,
                             description="[x1, y1, x2, y2] in pixels")


class InferenceResponse(BaseModel):
    objects:  List[DetectedObject]
    ocr_text: Dict[str, str]
    meta:     Dict[str, Any]


class AnalyticsResponse(BaseModel):
    label_counts:    Dict[str, int]
    foot_traffic:    int = Field(..., description="Number of 'person' detections")
    shelf_occupancy: Optional[float] = Field(
        None, description="Fraction of shelves with readable OCR text (0–1)"
    )
    restock_alert:   bool = Field(..., description="True if any shelf appears empty")
    ocr_text:        Dict[str, str]
    insight:         str  = Field(..., description="Natural-language summary")
    raw_detections:  List[DetectedObject]
