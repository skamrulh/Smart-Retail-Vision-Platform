"""
Stub out heavy optional dependencies (ultralytics, easyocr, transformers,
redis, PIL) before any test module is imported so tests run fast in CI
without GPU, model weights, or a running Redis instance.
"""
import sys
from unittest.mock import MagicMock
import numpy as np

# ── Stub ultralytics ──────────────────────────────────────────────────────────
ul = MagicMock()
sys.modules.setdefault("ultralytics", ul)

# ── Stub easyocr ─────────────────────────────────────────────────────────────
sys.modules.setdefault("easyocr", MagicMock())

# ── Stub transformers ─────────────────────────────────────────────────────────
tf = MagicMock()
sys.modules.setdefault("transformers", tf)

# ── Stub torch ────────────────────────────────────────────────────────────────
sys.modules.setdefault("torch", MagicMock())

# ── Stub redis.asyncio ────────────────────────────────────────────────────────
import redis as _redis_pkg
aio_mock = MagicMock()
_redis_pkg.asyncio = aio_mock
sys.modules["redis.asyncio"] = aio_mock

# ── Stub cv2 ─────────────────────────────────────────────────────────────────
cv2_mock = MagicMock()
cv2_mock.VideoCapture.return_value.isOpened.return_value = False
sys.modules.setdefault("cv2", cv2_mock)

# ── Stub pydantic_settings ────────────────────────────────────────────────────
try:
    import pydantic_settings  # noqa
except ImportError:
    from pydantic import BaseModel
    mock_ps = MagicMock()
    mock_ps.BaseSettings = BaseModel
    sys.modules["pydantic_settings"] = mock_ps
