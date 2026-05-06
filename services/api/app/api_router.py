"""
API routes for the Smart Retail Vision Platform.

All routes access the Predictor via request.app.state.predictor so
the module can be imported in tests without triggering Redis/model connections.
"""
import time
import logging
from collections import Counter as PyCounter
from typing import Dict, List, Any

from fastapi import APIRouter, HTTPException, Request

from .schemas import (
    InferenceRequest,
    InferenceResponse,
    DetectedObject,
    AnalyticsResponse,
)
from .health import INFERENCE_LATENCY

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/infer", response_model=InferenceResponse, tags=["inference"])
async def infer(req: InferenceRequest, request: Request):
    """
    Process a single video frame.

    Send a base64-encoded JPEG and receive detected objects, OCR text
    extracted from shelf/label regions, and a natural-language insight.
    """
    predictor = request.app.state.predictor
    t0 = time.perf_counter()
    try:
        result = await predictor.process_frame(req.frame_base64)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    INFERENCE_LATENCY.observe(time.perf_counter() - t0)

    objects = [
        DetectedObject(
            label=d.get("label", ""),
            score=float(d.get("score", 0.0)),
            bbox=d.get("bbox", [0, 0, 0, 0]),
        )
        for d in result.get("objects", [])
    ]
    return InferenceResponse(
        objects=objects,
        ocr_text=result.get("ocr_text", {}),
        meta=result.get("meta", {}),
    )


@router.post("/analytics", response_model=AnalyticsResponse, tags=["analytics"])
async def analytics(req: InferenceRequest, request: Request):
    """
    Run inference and return enriched analytics summary.

    In addition to raw detections, returns:
    - per-label object counts
    - shelf occupancy estimate (ratio of shelf detections with OCR text)
    - foot-traffic count (number of 'person' detections)
    - restock alert flag (any shelf with no OCR text)
    """
    predictor = request.app.state.predictor
    try:
        result = await predictor.process_frame(req.frame_base64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    dets: List[Dict[str, Any]] = result.get("objects", [])
    ocr:  Dict[str, str]       = result.get("ocr_text", {})

    label_counts = dict(PyCounter(d.get("label", "unknown") for d in dets))

    shelf_dets = [d for d in dets if d.get("label") == "shelf"]
    shelves_with_text = sum(1 for k, v in ocr.items() if v.strip())
    shelf_occupancy = (
        round(shelves_with_text / len(shelf_dets), 2) if shelf_dets else None
    )

    restock_alert = bool(shelf_dets) and shelves_with_text < len(shelf_dets)

    return AnalyticsResponse(
        label_counts=label_counts,
        foot_traffic=label_counts.get("person", 0),
        shelf_occupancy=shelf_occupancy,
        restock_alert=restock_alert,
        ocr_text=ocr,
        insight=result.get("meta", {}).get("insight", ""),
        raw_detections=[
            DetectedObject(
                label=d.get("label", ""),
                score=float(d.get("score", 0.0)),
                bbox=d.get("bbox", [0, 0, 0, 0]),
            )
            for d in dets
        ],
    )
