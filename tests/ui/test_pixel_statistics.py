"""Tests for visionleaf_ai.ui.components.pixel_statistics.

Purpose:
    Verify the pure `compute_pixel_statistics` function — the
    Streamlit-rendering half is exercised via the full-page AppTest in
    test_processing_lab_page.py.

Dependencies:
    pytest; numpy; visionleaf_ai.ui.components.pixel_statistics.
"""

from __future__ import annotations

import numpy as np

from visionleaf_ai.ui.components.pixel_statistics import compute_pixel_statistics


def test_stats_on_flat_image():
    flat = np.full((10, 20, 3), 100, dtype=np.uint8)
    stats = compute_pixel_statistics(flat)
    assert stats["resolution"] == (20, 10)
    assert stats["mean"] == 100.0
    assert stats["std"] == 0.0
    assert stats["min"] == 100.0
    assert stats["max"] == 100.0
    assert stats["dynamic_range"] == 0.0


def test_stats_on_gradient_image():
    gradient = np.linspace(0, 255, 256, dtype=np.uint8).reshape(1, 256)
    stats = compute_pixel_statistics(gradient)
    assert stats["min"] == 0.0
    assert stats["max"] == 255.0
    assert stats["dynamic_range"] == 255.0


def test_stats_resolution_is_width_then_height():
    image = np.zeros((30, 50), dtype=np.uint8)  # height=30, width=50
    stats = compute_pixel_statistics(image)
    assert stats["resolution"] == (50, 30)


def test_stats_works_on_grayscale():
    image = np.random.randint(0, 256, size=(16, 16), dtype=np.uint8)
    stats = compute_pixel_statistics(image)
    assert isinstance(stats["mean"], float)
    assert isinstance(stats["std"], float)
