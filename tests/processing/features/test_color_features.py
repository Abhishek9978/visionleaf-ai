"""Tests for color_features.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.features.color_features.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.features.color_features import (
    ColorFeatures,
    ColorFeaturesParams,
)


def test_color_features_produces_expected_keys(fully_prepared_session):
    result = ColorFeatures().process(fully_prepared_session)
    keys = set(result.color_features.keys())
    assert keys == {"rgb_mean", "rgb_std", "hsv_mean", "hsv_std", "histogram_normalized", "bins"}


def test_color_features_rgb_mean_has_three_channels(fully_prepared_session):
    result = ColorFeatures().process(fully_prepared_session)
    assert len(result.color_features["rgb_mean"]) == 3
    assert len(result.color_features["rgb_std"]) == 3


def test_color_features_histogram_bins_match_param(fully_prepared_session):
    result = ColorFeatures(ColorFeaturesParams(bins=16)).process(fully_prepared_session)
    for channel_hist in result.color_features["histogram_normalized"]:
        assert len(channel_hist) == 16


def test_color_features_histogram_sums_to_one_per_channel(fully_prepared_session):
    result = ColorFeatures().process(fully_prepared_session)
    for channel_hist in result.color_features["histogram_normalized"]:
        assert abs(sum(channel_hist) - 1.0) < 1e-6


def test_color_features_uses_only_foreground_pixels_when_masked():
    # An image where the background is pure black (0,0,0) and the
    # foreground is pure white — if background pixels leaked into the
    # stats, the RGB mean would be pulled toward 0.
    from visionleaf_ai.core.image_session import ImageSession

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = 255
    session = ImageSession(
        original_image=image.copy(), current_image=image.copy(), segmented_image=image.copy()
    )
    result = ColorFeatures().process(session)
    # Foreground-only mean should be exactly 255, not pulled down by background zeros.
    assert all(abs(v - 255.0) < 1e-6 for v in result.color_features["rgb_mean"])


def test_color_features_falls_back_to_roi_image(roi_only_session):
    result = ColorFeatures().process(roi_only_session)
    assert result.color_features is not None


def test_color_features_records_history(fully_prepared_session):
    result = ColorFeatures().process(fully_prepared_session)
    assert result.processing_history[-1].label == "Color Features"


def test_color_features_rejects_missing_segmentation(session_without_segmentation):
    with pytest.raises(ValidationError):
        ColorFeatures().process(session_without_segmentation)


def test_color_features_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        ColorFeatures().process(empty_session)


def test_color_features_rejects_invalid_bins(fully_prepared_session):
    with pytest.raises(ValidationError):
        ColorFeatures(ColorFeaturesParams(bins=0)).process(fully_prepared_session)


def test_color_features_get_info_structure():
    info = ColorFeatures().get_info()
    assert info.name == "Color Features"
    assert info.category == "features"
    assert info.typical_applications
