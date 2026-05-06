"""
Integration tests for FastAPI endpoints.

All ML models and Redis are mocked so tests run without any infrastructure.
"""
import sys, os, base64, io, json, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "api"))

from PIL import Image
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient


def _make_b64_frame(w=32, h=32):
    img = Image.new("RGB", (w, h), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Build app with mocked predictor ──────────────────────────────────────────

def _build_client(detections=None, ocr=None, insight="test insight"):
    """Create TestClient with a fully mocked Predictor."""
    from app.main import app

    mock_predictor = MagicMock()
    mock_predictor.process_frame = AsyncMock(return_value={
        "objects":  detections or [],
        "ocr_text": ocr or {},
        "meta":     {"insight": insight, "frame_timestamp": 1234567890.0},
    })
    app.state.predictor = mock_predictor
    return TestClient(app, raise_server_exceptions=False)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200():
    c = _build_client()
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_health_reports_predictor_loaded():
    c = _build_client()
    r = c.get("/health")
    assert r.json()["predictor_loaded"] is True

# ── /metrics ──────────────────────────────────────────────────────────────────

def test_metrics_returns_200():
    c = _build_client()
    r = c.get("/metrics")
    assert r.status_code == 200

def test_metrics_contains_prometheus_format():
    c = _build_client()
    r = c.get("/metrics")
    assert "HELP" in r.text or "api_requests" in r.text

# ── / ─────────────────────────────────────────────────────────────────────────

def test_root_returns_service_name():
    c = _build_client()
    r = c.get("/")
    assert r.status_code == 200
    assert "smart-retail" in r.json()["service"]

# ── /api/infer ────────────────────────────────────────────────────────────────

def test_infer_empty_detections():
    c = _build_client()
    r = c.post("/api/infer", json={"frame_base64": _make_b64_frame()})
    assert r.status_code == 200
    data = r.json()
    assert "objects" in data and "ocr_text" in data and "meta" in data
    assert data["objects"] == []

def test_infer_with_detections():
    dets = [{"label":"person","score":0.92,"bbox":[10,20,50,80]},
            {"label":"shelf", "score":0.88,"bbox":[0,100,640,300]}]
    c = _build_client(detections=dets)
    r = c.post("/api/infer", json={"frame_base64": _make_b64_frame()})
    assert r.status_code == 200
    objs = r.json()["objects"]
    assert len(objs) == 2
    assert objs[0]["label"] == "person"
    assert objs[0]["score"] == pytest.approx(0.92, abs=0.01)

def test_infer_returns_insight_in_meta():
    c = _build_client(insight="2 people near the shelf.")
    r = c.post("/api/infer", json={"frame_base64": _make_b64_frame()})
    assert "insight" in r.json()["meta"]

def test_infer_missing_body_returns_422():
    c = _build_client()
    r = c.post("/api/infer", json={})
    assert r.status_code == 422

# ── /api/analytics ────────────────────────────────────────────────────────────

def test_analytics_returns_label_counts():
    dets = [{"label":"person","score":0.9,"bbox":[0,0,10,10]},
            {"label":"person","score":0.8,"bbox":[20,0,30,10]},
            {"label":"shelf", "score":0.7,"bbox":[0,50,100,200]}]
    c = _build_client(detections=dets)
    r = c.post("/api/analytics", json={"frame_base64": _make_b64_frame()})
    assert r.status_code == 200
    data = r.json()
    assert data["label_counts"]["person"] == 2
    assert data["label_counts"]["shelf"]  == 1
    assert data["foot_traffic"] == 2

def test_analytics_restock_alert_when_shelf_no_ocr():
    dets = [{"label":"shelf","score":0.9,"bbox":[0,0,100,200]}]
    c = _build_client(detections=dets, ocr={})   # shelf detected but no OCR
    r = c.post("/api/analytics", json={"frame_base64": _make_b64_frame()})
    assert r.json()["restock_alert"] is True

def test_analytics_no_restock_alert_with_ocr():
    dets = [{"label":"shelf","score":0.9,"bbox":[0,0,100,200]}]
    c = _build_client(detections=dets, ocr={"roi_0": "BREAD 2.99"})
    r = c.post("/api/analytics", json={"frame_base64": _make_b64_frame()})
    assert r.json()["restock_alert"] is False

def test_analytics_no_detections_no_alert():
    c = _build_client(detections=[])
    r = c.post("/api/analytics", json={"frame_base64": _make_b64_frame()})
    assert r.status_code == 200
    assert r.json()["restock_alert"] is False
    assert r.json()["foot_traffic"] == 0

def test_analytics_shelf_occupancy_calculated():
    dets = [{"label":"shelf","score":0.9,"bbox":[0,0,100,200]},
            {"label":"shelf","score":0.85,"bbox":[200,0,300,200]}]
    c = _build_client(detections=dets, ocr={"roi_0": "MILK"})
    r = c.post("/api/analytics", json={"frame_base64": _make_b64_frame()})
    # 1 out of 2 shelves has OCR text
    assert r.json()["shelf_occupancy"] == pytest.approx(0.5, abs=0.01)

# ── Request counter middleware ────────────────────────────────────────────────

def test_prometheus_counter_increments():
    """After N requests, api_requests_total should be > 0."""
    from app.health import REQUESTS
    before = sum(REQUESTS.labels(method=m, path=p, status_code=s)._value.get()
                 for m, p, s in [("GET","/health","200")])
    c = _build_client()
    c.get("/health")
    after = REQUESTS.labels(method="GET", path="/health", status_code="200")._value.get()
    assert after > before
