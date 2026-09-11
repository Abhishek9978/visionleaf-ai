"""Tests for visionleaf_ai.processing.enhancement.histogram_equalization.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.enhancement.histogram_equalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.enhancement.histogram_equalization import (
    HistogramEqualization,
)


def test_histogram_equalization_increases_spread_on_low_contrast_image():
    # A genuinely low-contrast image: all values crammed into [100, 110].
    low_contrast = np.random.randint(100, 111, size=(32, 32), dtype=np.uint8)
    session = ImageSession(original_image=low_contrast.copy(), current_image=low_contrast.copy())

    stage = HistogramEqualization()
    result = stage.process(session)

    assert result.current_image.astype(float).std() > low_contrast.astype(float).std()


def test_histogram_equalization_works_on_rgb(loaded_session):
    result = HistogramEqualization().process(loaded_session)
    assert result.current_image.shape == loaded_session.original_image.shape


def test_histogram_equalization_preserves_hue_direction_on_rgb(loaded_session):
    # Since only luminance is equalized (YCrCb), a pure grayscale-like
    # RGB image (R==G==B everywhere) should stay grayscale-like after
    # equalization — a sanity check that color channels aren't equalized
    # independently (which would shift color balance).
    result = HistogramEqualization().process(loaded_session)
    r, g, b = result.current_image[..., 0], result.current_image[..., 1], result.current_image[..., 2]
    # Allow small rounding differences from the YCrCb round-trip.
    assert np.abs(r.astype(int) - g.astype(int)).max() <= 2
    assert np.abs(g.astype(int) - b.astype(int)).max() <= 2


def test_histogram_equalization_records_history_event(loaded_session):
    result = HistogramEqualization().process(loaded_session)
    assert result.processing_history[0].label == "Histogram Equalization"


def test_histogram_equalization_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        HistogramEqualization().process(empty_session)


def test_histogram_equalization_rejects_non_uint8_image():
    float_image = np.random.rand(16, 16, 3).astype(np.float32)
    session = ImageSession(original_image=float_image, current_image=float_image)
    with pytest.raises(ValidationError):
        HistogramEqualization().process(session)


def test_histogram_equalization_works_on_grayscale(loaded_grayscale_session):
    result = HistogramEqualization().process(loaded_grayscale_session)
    assert result.current_image.ndim == 2


def test_histogram_equalization_get_info_structure():
    info = HistogramEqualization().get_info()
    assert info.name == "Histogram Equalization"
    assert "CLAHE" in info.limitations[-1] or any("CLAHE" in lim for lim in info.limitations)
