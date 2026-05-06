"""
Prometheus metrics and health state for the Smart Retail API.

REQUESTS counter is incremented by a FastAPI middleware in main.py
so every endpoint's traffic is captured automatically.
"""
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# ── Counters (incremented in main.py middleware) ──────────────────────────────
REQUESTS = Counter(
    "api_requests_total",
    "Total HTTP requests by method, path, and status code",
    ["method", "path", "status_code"],
)
INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "End-to-end inference latency in seconds",
)
FRAMES_PROCESSED = Counter(
    "frames_processed_total",
    "Total video frames processed through the detection pipeline",
)


def metrics_response() -> Response:
    """Return a Prometheus-format metrics response."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
