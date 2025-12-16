from fastapi import APIRouter, HTTPException
from .schemas import InferenceRequest, InferenceResponse, DetectedObject
from .inference import Predictor

router = APIRouter()
predictor = Predictor()

@router.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    try:
        result = await predictor.process_frame(req.frame_base64)
        
        objs = []
        for d in result.get("objects", []):
            objs.append(DetectedObject(label=d.get("label",""), score=float(d.get("score",0.0)), bbox=d.get("bbox",[0,0,0,0])))
        return InferenceResponse(objects=objs, ocr_text=result.get("ocr_text",{}), meta=result.get("meta",{}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
