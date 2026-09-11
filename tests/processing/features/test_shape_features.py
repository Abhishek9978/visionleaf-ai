"""Tests for shape_features.py.

Dependencies:
    pytest; numpy; opencv-python (cv2); visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.features.shape_features.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.features.shape_features import ShapeFeatures


def _square_contour_session(side: int = 40) -> ImageSession:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20 : 20 + side, 20 : 20 + side] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = np.stack([mask] * 3, axis=-1)
    return ImageSession(
        original_image=image.copy(), current_image=image.copy(), largest_contour=contours[0]
    )


def test_shape_features_square_has_expected_aspect_ratio_and_solidity():
    session = _square_contour_session()
    result = ShapeFeatures().process(session)
    assert result.shape_features["aspect_ratio"] == pytest.approx(1.0, abs=0.02)
    assert result.shape_features["solidity"] == pytest.approx(1.0, abs=0.02)


def test_shape_features_square_circularity_matches_theory():
    # A square's circularity (4*pi*area/perimeter^2) should be pi/4 ~= 0.785.
    session = _square_contour_session()
    result = ShapeFeatures().process(session)
    assert result.shape_features["circularity"] == pytest.approx(np.pi / 4, abs=0.03)


def test_shape_features_produces_seven_hu_moments():
    session = _square_contour_session()
    result = ShapeFeatures().process(session)
    assert len(result.shape_features["hu_moments"]) == 7


def test_shape_features_records_history():
    session = _square_contour_session()
    result = ShapeFeatures().process(session)
    assert result.processing_history[-1].label == "Shape Features"


def test_shape_features_rejects_missing_contour(session_without_contour):
    with pytest.raises(ValidationError):
        ShapeFeatures().process(session_without_contour)


def test_shape_features_rejects_zero_area_contour():
    degenerate_contour = np.array([[[5, 5]], [[5, 5]], [[5, 5]]])
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    session = ImageSession(
        original_image=image.copy(), current_image=image.copy(), largest_contour=degenerate_contour
    )
    with pytest.raises(ValidationError):
        ShapeFeatures().process(session)


def test_shape_features_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        ShapeFeatures().process(empty_session)


def test_shape_features_larger_square_has_larger_area():
    small = ShapeFeatures().process(_square_contour_session(side=20))
    large = ShapeFeatures().process(_square_contour_session(side=60))
    assert large.shape_features["area"] > small.shape_features["area"]


def test_shape_features_get_info_structure():
    info = ShapeFeatures().get_info()
    assert info.name == "Shape Features"
    assert info.category == "features"
    assert "Hu" in info.theory or "Hu" in info.purpose
