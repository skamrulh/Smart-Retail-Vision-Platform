"""
Smart Retail Vision Platform — FastAPI application entry point.

Lifecycle:
  - lifespan() creates a single Predictor on startup and stores it on
    app.state so api_router can access it via request.app.state.predictor.
  - A middleware increments the REQUESTS Prometheus counter on every response.
"""
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .api_router import router
from .health import REQUESTS, metrics_response
from .inference import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan — single Predictor instance for the process lifetime ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising Predictor…")
    app.state.predictor = Predictor()
    logger.info("Predictor ready.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Retail Vision API",
    version="1.0.0",
    description=(
        "Real-time computer vision API for retail analytics: "
        "customer detection, shelf monitoring, OCR, and NL insights."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Prometheus request counter middleware ─────────────────────────────────────
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    response = await call_next(request)
    REQUESTS.labels(
        method=request.method,
        path=request.url.path,
        status_code=str(response.status_code),
    ).inc()
    return response


# ── Routers & built-in endpoints ──────────────────────────────────────────────
app.include_router(router, prefix="/api")


@app.get("/", tags=["meta"])
async def root():
    return {"service": "smart-retail-api", "version": "1.0.0", "status": "ok"}


@app.get("/health", tags=["meta"])
async def health(request: Request):
    """Liveness and readiness probe — returns 200 when the service is ready."""
    return {
        "status":           "healthy",
        "predictor_loaded": hasattr(request.app.state, "predictor"),
    }


@app.get("/metrics", tags=["meta"])
async def metrics():
    """Prometheus metrics endpoint."""
    return metrics_response()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
