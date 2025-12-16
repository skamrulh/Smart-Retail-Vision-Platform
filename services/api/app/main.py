from fastapi import FastAPI
from .api_router import router
from .health import metrics_endpoint

app = FastAPI(title="Smart Retail API", version="0.1.0")
app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"service": "smart-retail-api", "status": "ok"}

@app.get("/metrics")
async def metrics():
    return metrics_endpoint()
