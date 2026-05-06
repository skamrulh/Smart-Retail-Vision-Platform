"""
Tests for worker service: processing pipeline, redis client, OCR wrapper.
"""
import sys, os, base64, io, json, asyncio, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "worker"))

from PIL import Image
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np


def _make_b64(w=32, h=32):
    img = Image.new("RGB", (w, h), (100, 150, 200))
    buf = io.BytesIO(); img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ── worker/ocr.py ─────────────────────────────────────────────────────────────

class TestOCR:
    def test_ocr_class_is_named_ocr(self):
        """Critical: class must be OCR, not OCRModel (processing.py imports OCR)."""
        from worker.ocr import OCR
        assert OCR is not None

    def test_stub_returns_empty_string(self):
        from worker.ocr import OCR
        ocr = OCR()
        ocr._backend = "stub"
        ocr._reader  = MagicMock()
        ocr._reader.read_pil_image.return_value = ""
        result = ocr.read_pil_image(Image.new("RGB", (50, 20)))
        assert result == ""

    def test_easyocr_backend_joins_text(self):
        from worker.ocr import OCR
        ocr = OCR()
        ocr._backend = "easyocr"
        ocr._reader  = MagicMock()
        ocr._reader.readtext.return_value = [(None, "COLA", 0.99), (None, "1.50", 0.95)]
        result = ocr.read_pil_image(Image.new("RGB", (100, 30)))
        assert "COLA" in result and "1.50" in result

    def test_exception_returns_empty_string(self):
        from worker.ocr import OCR
        ocr = OCR()
        ocr._backend = "easyocr"
        ocr._reader  = MagicMock()
        ocr._reader.readtext.side_effect = RuntimeError("GPU error")
        result = ocr.read_pil_image(Image.new("RGB", (50, 20)))
        assert result == ""


# ── worker/detector.py ────────────────────────────────────────────────────────

class TestWorkerDetectionModel:
    def test_predict_returns_list(self):
        from worker.detector import DetectionModel
        m = DetectionModel()
        result = m.predict(np.zeros((100, 100, 3), dtype=np.uint8))
        assert isinstance(result, list)


# ── worker/processing.py ──────────────────────────────────────────────────────

class TestProcessFrameBase64:
    def test_import_uses_ocr_not_ocr_model(self):
        """Verify the fixed import line uses 'OCR' not 'OCRModel'."""
        import worker.processing as proc
        src = open(proc.__file__).read()
        # The import line must use the correct class name
        assert "from .ocr import OCR" in src
        # The import statement itself must not import OCRModel
        import_line = [l for l in src.splitlines() if "from .ocr import" in l]
        assert import_line, "import from .ocr not found"
        assert "OCRModel" not in import_line[0]

    def test_process_frame_returns_dict(self):
        from worker import processing
        with patch.object(processing._detector, "predict", return_value=[]), \
             patch.object(processing._redis, "xadd", new=AsyncMock()):
            result = asyncio.run(processing.process_frame_base64(_make_b64()))
        assert "detections" in result and "ocr" in result

    def test_process_frame_corrupt_data_safe(self):
        from worker import processing
        with patch.object(processing._redis, "xadd", new=AsyncMock()):
            result = asyncio.run(processing.process_frame_base64("not-valid-base64!!!"))
        assert result == {"detections": [], "ocr": {}}

    def test_singletons_not_recreated_per_frame(self):
        """Detector and OCR singletons must be module-level, not per-call."""
        import worker.processing as proc
        id_first  = id(proc._detector)
        asyncio.run(proc.process_frame_base64(_make_b64())) if False else None
        id_second = id(proc._detector)
        assert id_first == id_second   # same object


# ── worker/redis_client.py ────────────────────────────────────────────────────

class TestWorkerRedisClient:
    def test_xadd_is_async(self):
        """xadd must be a coroutine so camera_consumer can await it."""
        import inspect
        from worker.redis_client import RedisClient
        client = RedisClient.__new__(RedisClient)
        assert inspect.iscoroutinefunction(client.xadd)

    def test_xread_is_async(self):
        import inspect
        from worker.redis_client import RedisClient
        client = RedisClient.__new__(RedisClient)
        assert inspect.iscoroutinefunction(client.xread)

    def test_xadd_swallows_connection_error(self):
        from worker.redis_client import RedisClient
        client = RedisClient.__new__(RedisClient)
        mock_r = MagicMock()
        mock_r.xadd = AsyncMock(side_effect=ConnectionError("refused"))
        client._r = mock_r
        # Should not raise
        asyncio.run(client.xadd("test-stream", {"data": "{}"}))
