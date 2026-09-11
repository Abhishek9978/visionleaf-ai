"""Tests for roi_crop.py, roi_mask.py, and bounding_box_visualization.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.roi.{roi_crop,roi_mask,bounding_box_visualization}.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.roi.bounding_box_visualization import (
    BoundingBoxVisualization,
    BoundingBoxVisualizationParams,
)
from visionleaf_ai.processing.roi.roi_crop import ROICrop
from visionleaf_ai.processing.roi.min_area_rectangle import MinAreaRectangle
from visionleaf_ai.processing.roi.roi_mask import ROIMask


def test_roi_crop_produces_correctly_sized_crop(session_with_bounding_box):
    result = ROICrop().process(session_with_bounding_box)
    assert result.roi_image.shape[:2] == (40, 40)


def test_roi_crop_records_history(session_with_bounding_box):
    result = ROICrop().process(session_with_bounding_box)
    assert result.processing_history[-1].label == "ROI Cropping"


def test_roi_crop_falls_back_to_axis_aligned_bounds_for_rotated_box(session_with_contours):
    session = MinAreaRectangle().process(session_with_contours)
    result = ROICrop().process(session)
    assert result.roi_image is not None
    assert result.roi_image.shape[0] > 0 and result.roi_image.shape[1] > 0


def test_roi_crop_rejects_missing_bounding_box(session_with_contours):
    with pytest.raises(ValidationError):
        ROICrop().process(session_with_contours)  # Bounding Rectangle never ran


def test_roi_crop_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        ROICrop().process(empty_session)


def test_roi_mask_zeroes_background_using_largest_contour(session_with_contours):
    result = ROIMask().process(session_with_contours)
    # Outside the square (corner) should be zeroed; inside should be preserved.
    assert result.segmented_image[5, 5].sum() == 0
    assert result.segmented_image[40, 50].sum() > 0


def test_roi_mask_falls_back_to_active_mask_without_contours(session_with_mask):
    result = ROIMask().process(session_with_mask)
    assert result.segmented_image[40, 50].sum() > 0


def test_roi_mask_records_mask_source(session_with_contours):
    result = ROIMask().process(session_with_contours)
    assert "largest_contour" in result.processing_history[-1].details


def test_roi_mask_rejects_missing_mask_and_contour(session_without_mask):
    with pytest.raises(ValidationError):
        ROIMask().process(session_without_mask)


def test_roi_mask_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        ROIMask().process(empty_session)


def test_bounding_box_visualization_draws_onto_current_image(session_with_bounding_box):
    original_current = session_with_bounding_box.current_image.copy()
    result = BoundingBoxVisualization().process(session_with_bounding_box)
    assert not np.array_equal(result.current_image, original_current)
    assert result.current_image.shape == original_current.shape


def test_bounding_box_visualization_works_with_rotated_box(session_with_contours):
    session = MinAreaRectangle().process(session_with_contours)
    result = BoundingBoxVisualization().process(session)
    assert result.current_image is not None


def test_bounding_box_visualization_rejects_non_positive_thickness(session_with_bounding_box):
    with pytest.raises(ValidationError):
        BoundingBoxVisualization(BoundingBoxVisualizationParams(thickness=0)).process(
            session_with_bounding_box
        )


def test_bounding_box_visualization_rejects_missing_bounding_box(session_with_contours):
    with pytest.raises(ValidationError):
        BoundingBoxVisualization().process(session_with_contours)


def test_bounding_box_visualization_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        BoundingBoxVisualization().process(empty_session)


def test_roi_crop_get_info_structure():
    info = ROICrop().get_info()
    assert info.name == "ROI Cropping"
    assert info.category == "roi"
