from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

REQUESTS = Counter("api_requests_total", "Total API requests")

def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
