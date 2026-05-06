"""Tests for services/api/app/utils.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "api"))

import pytest
from app.utils import bbox_clamp, pil_from_base64, np_from_base64
import base64, io, numpy as np
from PIL import Image


# ── bbox_clamp ────────────────────────────────────────────────────────────────

def test_bbox_clamp_within_bounds():
    result = bbox_clamp([10, 20, 100, 200], width=640, height=480)
    assert result == [10, 20, 100, 200]

def test_bbox_clamp_negative_origin():
    result = bbox_clamp([-5, -10, 100, 200], width=640, height=480)
    assert result[0] == 0 and result[1] == 0

def test_bbox_clamp_exceeds_width():
    result = bbox_clamp([0, 0, 700, 200], width=640, height=480)
    assert result[2] == 640

def test_bbox_clamp_exceeds_height():
    result = bbox_clamp([0, 0, 100, 600], width=640, height=480)
    assert result[3] == 480

def test_bbox_clamp_returns_four_ints():
    result = bbox_clamp([1.5, 2.7, 300.9, 400.1], width=640, height=480)
    assert len(result) == 4
    assert all(isinstance(v, int) for v in result)

def test_bbox_clamp_x3_bug_fixed():
    """The original code named the clamped y2 as 'x3' causing confusion.
    This test verifies the 4th element is clamped by HEIGHT not WIDTH."""
    result = bbox_clamp([0, 0, 100, 600], width=640, height=300)
    # 4th element must be clamped to height=300, not width=640
    assert result[3] == 300
    assert result[2] == 100  # x2 unchanged

def test_bbox_clamp_float_inputs():
    result = bbox_clamp([0.0, 0.0, 640.0, 480.0], 640, 480)
    assert result == [0, 0, 640, 480]


# ── pil_from_base64 ───────────────────────────────────────────────────────────

def _make_b64_image(w=10, h=10):
    img = Image.new("RGB", (w, h), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def test_pil_from_base64_returns_rgb():
    img = pil_from_base64(_make_b64_image())
    assert img.mode == "RGB"

def test_pil_from_base64_correct_size():
    img = pil_from_base64(_make_b64_image(20, 15))
    assert img.size == (20, 15)

def test_np_from_base64_shape():
    arr = np_from_base64(_make_b64_image(8, 8))
    assert arr.shape == (8, 8, 3)
    assert arr.dtype == np.uint8
