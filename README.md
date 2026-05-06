# AI-Powered Smart Retail Vision Platform

![CI](https://github.com/<your-username>/Smart-Retail-Vision-Platform/actions/workflows/deploy.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Docker](https://img.shields.io/badge/deploy-Docker-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade, distributed computer vision platform for retail analytics. Ingests live camera feeds, runs real-time object detection + OCR, generates natural-language shelf insights, and delivers structured analytics to retail operators — enabling smarter merchandising, automated restock alerts, and data-driven store optimisation.

---

## System Architecture

```
[Camera Feeds]
      │  JPEG frames @ ~10fps
      ▼
[Worker Service]  ─── frame_b64 → Redis Stream "camera:frames"
  camera_consumer.py   (camera → Redis producer)
  processing.py        (Redis consumer → detection → OCR → event)
      │
      │  structured events → Redis Stream "processing:events"
      ▼
[API Service]                      port 8000
  POST /api/infer      ← raw detection + OCR per frame
  POST /api/analytics  ← enriched analytics (counts, occupancy, alerts)
  GET  /health         ← liveness / readiness probe
  GET  /metrics        ← Prometheus metrics
      │
      ▼
[ML Pipeline]
  ┌─────────────────────────────────────────────────────────────┐
  │ DetectionModel  — YOLOv8n (ultralytics) or ONNX Runtime    │
  │   Detects: person, shelf, product_label, price_tag         │
  │                                                             │
  │ OCRModel        — EasyOCR (English)                        │
  │   Reads text from cropped shelf/label ROIs                 │
  │                                                             │
  │ ReportGenerator — DistilBART (HuggingFace Transformers)    │
  │   Summarises detection counts into natural-language text   │
  └─────────────────────────────────────────────────────────────┘
      │
      ▼
[Infrastructure]
  Redis    — frame streams + result buffering
  MinIO    — model artefact & frame storage
  Postgres — metadata persistence
  Prometheus + Grafana — observability
```

---

## Services

| Service | Path | Purpose |
|---|---|---|
| **API** | `services/api/` | FastAPI inference & analytics endpoints |
| **Worker** | `services/worker/` | Camera producer + async frame processor |
| **Trainer** | `services/trainer/` | YOLOv8 fine-tuning + ONNX export pipeline |

---

## Quick Start

### Docker Compose (full stack)

```bash
git clone https://github.com/<your-username>/Smart-Retail-Vision-Platform.git
cd Smart-Retail-Vision-Platform
docker compose up --build
```

| Service | URL |
|---|---|
| API + Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| MinIO console | http://localhost:9000 |

### Local (API only)

```bash
cd services/api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

## API Usage

### Process a frame — raw detections

```bash
# Encode any image to base64
FRAME=$(base64 -i test_frame.jpg | tr -d '\n')

curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -d "{\"frame_base64\": \"$FRAME\"}"
```

```json
{
  "objects": [
    {"label": "person",        "score": 0.93, "bbox": [120, 80, 250, 400]},
    {"label": "shelf",         "score": 0.89, "bbox": [0, 150, 640, 380]},
    {"label": "product_label", "score": 0.81, "bbox": [45, 200, 180, 260]}
  ],
  "ocr_text": {
    "roi_1": "ORGANIC OATS 500g",
    "roi_2": "£2.49"
  },
  "meta": {
    "insight": "1 customer near oats shelf. Label and price tag visible.",
    "frame_timestamp": 1712345678.4
  }
}
```

### Get enriched analytics

```bash
curl -X POST http://localhost:8000/api/analytics \
  -H "Content-Type: application/json" \
  -d "{\"frame_base64\": \"$FRAME\"}"
```

```json
{
  "label_counts":    {"person": 1, "shelf": 2, "product_label": 3},
  "foot_traffic":    1,
  "shelf_occupancy": 0.67,
  "restock_alert":   true,
  "ocr_text":        {"roi_1": "BREAD", "roi_2": "MILK"},
  "insight":         "2 shelves detected. 1 shelf appears empty — restock recommended.",
  "raw_detections":  [...]
}
```

### Health check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "predictor_loaded": true}
```

---

## Analytics Explained

| Field | Type | Description |
|---|---|---|
| `label_counts` | dict | Count of each detected object class |
| `foot_traffic` | int | Number of `person` detections in the frame |
| `shelf_occupancy` | float 0–1 | Fraction of shelves with readable OCR text |
| `restock_alert` | bool | `true` if any shelf has no OCR text (appears empty) |
| `insight` | str | Natural-language summary from DistilBART |

---

## Running Tests

```bash
# Install test dependencies (no GPU required — ML mocked via conftest.py)
pip install fastapi httpx pytest pydantic pydantic-settings prometheus-client Pillow numpy redis

# Run 47 tests across API, models, utils, and worker
pytest tests/ -v
# 47 passed in ~1.5s
```

---

## Training a Custom Detector

```bash
docker compose run trainer

# Or manually:
cd services/trainer
pip install -r requirements.txt
DATA_YAML=datasets/retail.yaml EPOCHS=50 MODEL_OUT=./out python train/train.py
# Exports: ./out/yolov8_retail.onnx
```

Mount the exported ONNX model and set `MODEL_PATH=/app/models/yolov8_retail.onnx` in the API container.

---

## Environment Variables

| Variable | Default | Service | Description |
|---|---|---|---|
| `REDIS_URL` | `redis://redis:6379` | api, worker | Redis connection string |
| `MODEL_PATH` | `models/detector.onnx` | api | Path to ONNX/YOLO model |
| `CAM_SOURCE` | `0` | worker | Camera index or RTSP URL |
| `CAM_SLEEP` | `0.1` | worker | Seconds between frames |
| `DATA_YAML` | `datasets/retail.yaml` | trainer | YOLOv8 dataset config |
| `EPOCHS` | `5` | trainer | Training epochs |
| `MODEL_OUT` | `/out` | trainer | ONNX export directory |

---

## Observability

`GET /metrics` exposes Prometheus metrics:
- `api_requests_total` — requests by method, path, status code
- `inference_duration_seconds` — end-to-end latency histogram
- `frames_processed_total` — total frames processed through detection

---

## Tech Stack

`Python 3.11` · `FastAPI` · `YOLOv8 (Ultralytics)` · `EasyOCR` · `HuggingFace Transformers` · `Redis Streams` · `MinIO` · `PostgreSQL` · `Prometheus` · `Grafana` · `Docker` · `Kubernetes` · `GitHub Actions`
