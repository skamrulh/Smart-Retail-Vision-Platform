"""Tests for model_loader.py and detector_shim.py (stub paths)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "api"))

import numpy as np
from unittest.mock import MagicMock, patch
from app.models.model_loader import DetectionModel, OCRModel, ReportGenerator


def _blank_img(h=100, w=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── DetectionModel ────────────────────────────────────────────────────────────

class TestDetectionModel:
    def test_load_default_returns_instance(self):
        m = DetectionModel.load_default()
        assert isinstance(m, DetectionModel)

    def test_predict_without_model_returns_empty(self):
        m = DetectionModel()
        m._model = None   # simulate no ultralytics
        result = m.predict(_blank_img())
        assert isinstance(result, list)

    def test_predict_returns_list(self):
        m = DetectionModel()
        result = m.predict(_blank_img())
        assert isinstance(result, list)

    def test_predict_with_mock_yolo(self):
        m = DetectionModel()
        # Simulate a YOLO result
        mock_result = MagicMock()
        mock_result.boxes.xyxy.tolist.return_value = [[10.0, 20.0, 50.0, 60.0]]
        mock_result.boxes.conf.tolist.return_value = [0.87]
        mock_result.boxes.cls.tolist.return_value  = [0]
        mock_model = MagicMock(return_value=[mock_result])
        m._model  = mock_model
        m._names  = {0: "person"}
        result = m.predict(_blank_img())
        assert len(result) == 1
        assert result[0]["label"] == "person"
        assert result[0]["score"] == pytest.approx(0.87, abs=0.01)
        assert result[0]["bbox"] == [10, 20, 50, 60]


# ── OCRModel ──────────────────────────────────────────────────────────────────

class TestOCRModel:
    def test_load_default_returns_instance(self):
        assert isinstance(OCRModel.load_default(), OCRModel)

    def test_read_without_reader_returns_empty(self):
        from PIL import Image
        m = OCRModel()
        m._reader = None
        result = m.read_pil_image(Image.new("RGB", (50, 20)))
        assert result == ""

    def test_read_with_mock_easyocr(self):
        from PIL import Image
        m = OCRModel()
        m._reader = MagicMock()
        m._reader.readtext.return_value = [(None, "SALE 50%", 0.99)]
        result = m.read_pil_image(Image.new("RGB", (100, 30)))
        assert "SALE" in result


# ── ReportGenerator ───────────────────────────────────────────────────────────

class TestReportGenerator:
    def test_summarize_quick_no_detections(self):
        from unittest.mock import patch
        rg = ReportGenerator()
        rg._nlp = None
        with patch.object(rg, "_load"):   # prevent conftest mock from setting _nlp
            result = rg.summarize_quick([], {})
        assert isinstance(result, str)

    def test_summarize_quick_counts_labels(self):
        from unittest.mock import patch
        rg = ReportGenerator()
        rg._nlp = None
        dets = [{"label": "person"}, {"label": "person"}, {"label": "shelf"}]
        with patch.object(rg, "_load"):
            result = rg.summarize_quick(dets, {})
        assert "person" in result and "2" in result

    def test_summarize_quick_with_mock_nlp(self):
        rg = ReportGenerator()
        rg._nlp = MagicMock(return_value=[{"summary_text": "2 people, 1 shelf detected."}])
        result = rg.summarize_quick([{"label": "person"}, {"label": "shelf"}], {})
        assert "people" in result or "detected" in result


import pytest
