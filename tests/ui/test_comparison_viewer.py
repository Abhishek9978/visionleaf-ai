"""Tests for visionleaf_ai.ui.components.comparison_viewer.

Purpose:
    Verify the pure `_build_before_after_composite` function — the
    Streamlit-rendering half is exercised via the full-page AppTest.

Dependencies:
    pytest; numpy; visionleaf_ai.ui.components.comparison_viewer.
"""

from __future__ import annotations

import numpy as np

from visionleaf_ai.ui.components.comparison_viewer import _build_before_after_composite


def test_composite_at_zero_percent_is_entirely_processed():
    original = np.zeros((10, 10, 3), dtype=np.uint8)
    processed = np.full((10, 10, 3), 200, dtype=np.uint8)
    composite = _build_before_after_composite(original, processed, split_percent=0)
    # Everything except the thin divider line should be "processed".
    assert (composite[:, 5] == 200).all()


def test_composite_at_hundred_percent_is_entirely_original():
    original = np.full((10, 10, 3), 50, dtype=np.uint8)
    processed = np.full((10, 10, 3), 200, dtype=np.uint8)
    composite = _build_before_after_composite(original, processed, split_percent=100)
    assert (composite[:, 0] == 50).all()


def test_composite_at_fifty_percent_splits_in_half():
    original = np.zeros((10, 10, 3), dtype=np.uint8)
    processed = np.full((10, 10, 3), 200, dtype=np.uint8)
    composite = _build_before_after_composite(original, processed, split_percent=50)
    assert (composite[:, 0] == 0).all()  # left half from original
    assert (composite[:, 9] == 200).all()  # right half from processed


def test_composite_preserves_shape():
    original = np.zeros((16, 24, 3), dtype=np.uint8)
    processed = np.full((16, 24, 3), 128, dtype=np.uint8)
    composite = _build_before_after_composite(original, processed, split_percent=30)
    assert composite.shape == original.shape


def test_composite_works_on_grayscale():
    original = np.zeros((10, 10), dtype=np.uint8)
    processed = np.full((10, 10), 200, dtype=np.uint8)
    composite = _build_before_after_composite(original, processed, split_percent=50)
    assert composite.shape == (10, 10)
