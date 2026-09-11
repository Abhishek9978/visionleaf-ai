"""Tests for processing.features._common.

Purpose:
    Directly test the shared helper functions (input resolution,
    foreground masking, flattening) rather than only exercising them
    indirectly through the extractor stages.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.features._common.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.features._common import (
    flatten_color_features,
    flatten_edge_features,
    flatten_shape_features,
    flatten_texture_features,
    require_feature_input_image,
    require_grayscale_input,
)


def test_require_feature_input_image_prefers_segmented_over_roi():
    segmented = np.full((10, 10, 3), 1, dtype=np.uint8)
    roi = np.full((10, 10, 3), 2, dtype=np.uint8)
    session = ImageSession(
        original_image=np.zeros((10, 10, 3), dtype=np.uint8),
        segmented_image=segmented,
        roi_image=roi,
    )
    image, mask = require_feature_input_image(session)
    assert image is segmented
    assert mask is not None


def test_require_feature_input_image_falls_back_to_roi_with_no_mask():
    roi = np.full((10, 10, 3), 2, dtype=np.uint8)
    session = ImageSession(original_image=np.zeros((10, 10, 3), dtype=np.uint8), roi_image=roi)
    image, mask = require_feature_input_image(session)
    assert image is roi
    assert mask is None


def test_require_feature_input_image_raises_when_neither_exists():
    session = ImageSession(original_image=np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(ValidationError):
        require_feature_input_image(session)


def test_require_feature_input_image_raises_on_all_zero_segmented_image():
    all_black = np.zeros((10, 10, 3), dtype=np.uint8)
    session = ImageSession(
        original_image=all_black.copy(), segmented_image=all_black.copy()
    )
    with pytest.raises(ValidationError):
        require_feature_input_image(session)


def test_require_grayscale_input_converts_color_image():
    color = np.full((10, 10, 3), (10, 20, 30), dtype=np.uint8)
    session = ImageSession(original_image=color.copy(), roi_image=color.copy())
    gray = require_grayscale_input(session)
    assert gray.ndim == 2


def test_flatten_color_features_fixed_order():
    color_features = {
        "rgb_mean": [1.0, 2.0, 3.0],
        "rgb_std": [4.0, 5.0, 6.0],
        "hsv_mean": [7.0, 8.0, 9.0],
        "hsv_std": [10.0, 11.0, 12.0],
        "histogram_normalized": [[0.5, 0.5], [0.25, 0.75], [1.0, 0.0]],
    }
    flat = flatten_color_features(color_features)
    assert flat == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.5, 0.5, 0.25, 0.75, 1.0, 0.0]


def test_flatten_texture_features_includes_only_present_subkeys():
    only_glcm = {"glcm": {"contrast": 1.0, "energy": 2.0}}
    flat = flatten_texture_features(only_glcm)
    assert flat == [1.0, 2.0]  # contrast, then energy, per the fixed property order


def test_flatten_texture_features_combines_glcm_and_lbp():
    both = {
        "glcm": {"contrast": 1.0},
        "lbp": {"histogram_normalized": [0.5, 0.5]},
    }
    flat = flatten_texture_features(both)
    assert flat == [1.0, 0.5, 0.5]


def test_flatten_shape_features_fixed_order_and_length():
    shape_features = {
        "area": 100.0,
        "perimeter": 40.0,
        "aspect_ratio": 1.0,
        "extent": 0.9,
        "solidity": 1.0,
        "circularity": 0.78,
        "equivalent_diameter": 11.3,
        "hu_moments": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
    flat = flatten_shape_features(shape_features)
    assert len(flat) == 14
    assert flat[:7] == [100.0, 40.0, 1.0, 0.9, 1.0, 0.78, 11.3]


def test_flatten_edge_features_fixed_order():
    edge_features = {
        "canny_edge_density": 0.1,
        "sobel_gradient_magnitude_mean": 12.5,
        "laplacian_response_variance": 3.3,
    }
    assert flatten_edge_features(edge_features) == [0.1, 12.5, 3.3]
