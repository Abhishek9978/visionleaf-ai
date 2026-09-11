"""Tests for edge_features.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.features.edge_features.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.features.edge_features import EdgeFeatures, EdgeFeaturesParams


def test_edge_features_produces_expected_keys(fully_prepared_session):
    result = EdgeFeatures().process(fully_prepared_session)
    keys = set(result.edge_features.keys())
    assert keys == {
        "canny_edge_density",
        "sobel_gradient_magnitude_mean",
        "laplacian_response_variance",
    }


def test_edge_features_flat_image_has_zero_edge_density():
    flat = np.full((30, 30, 3), 128, dtype=np.uint8)
    session = ImageSession(original_image=flat.copy(), current_image=flat.copy(), roi_image=flat.copy())
    result = EdgeFeatures().process(session)
    assert result.edge_features["canny_edge_density"] == pytest.approx(0.0, abs=1e-6)
    assert result.edge_features["laplacian_response_variance"] == pytest.approx(0.0, abs=1e-6)


def test_edge_features_sharp_edge_has_nonzero_density():
    half_black_half_white = np.zeros((30, 30, 3), dtype=np.uint8)
    half_black_half_white[:, 15:] = 255
    session = ImageSession(
        original_image=half_black_half_white.copy(),
        current_image=half_black_half_white.copy(),
        roi_image=half_black_half_white.copy(),
    )
    result = EdgeFeatures().process(session)
    assert result.edge_features["canny_edge_density"] > 0
    assert result.edge_features["sobel_gradient_magnitude_mean"] > 0


def test_edge_features_records_history(fully_prepared_session):
    result = EdgeFeatures().process(fully_prepared_session)
    assert result.processing_history[-1].label == "Edge Features"


def test_edge_features_rejects_missing_segmentation(session_without_segmentation):
    with pytest.raises(ValidationError):
        EdgeFeatures().process(session_without_segmentation)


def test_edge_features_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        EdgeFeatures().process(empty_session)


def test_edge_features_rejects_high_threshold_not_greater_than_low(fully_prepared_session):
    with pytest.raises(ValidationError):
        EdgeFeatures(
            EdgeFeaturesParams(canny_low_threshold=150, canny_high_threshold=50)
        ).process(fully_prepared_session)


def test_edge_features_rejects_out_of_range_threshold(fully_prepared_session):
    with pytest.raises(ValidationError):
        EdgeFeatures(EdgeFeaturesParams(canny_low_threshold=-5)).process(fully_prepared_session)


def test_edge_features_rejects_even_sobel_kernel(fully_prepared_session):
    with pytest.raises(ValidationError):
        EdgeFeatures(EdgeFeaturesParams(sobel_kernel_size=4)).process(fully_prepared_session)


def test_edge_features_get_info_structure():
    info = EdgeFeatures().get_info()
    assert info.name == "Edge Features"
    assert info.category == "features"
