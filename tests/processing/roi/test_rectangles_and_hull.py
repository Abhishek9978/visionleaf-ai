"""Tests for bounding_rectangle.py, min_area_rectangle.py, and convex_hull.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.roi.{bounding_rectangle,min_area_rectangle,convex_hull}.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.roi.bounding_rectangle import BoundingRectangle
from visionleaf_ai.processing.roi.convex_hull import ConvexHull
from visionleaf_ai.processing.roi.min_area_rectangle import MinAreaRectangle


def test_bounding_rectangle_matches_the_known_square(session_with_contours):
    result = BoundingRectangle().process(session_with_contours)
    box = result.bounding_box
    assert box["type"] == "axis_aligned"
    assert box["x"] == 30
    assert box["y"] == 20
    assert box["w"] == 40
    assert box["h"] == 40


def test_bounding_rectangle_clears_old_roi_image(session_with_bounding_box):
    session_with_bounding_box.roi_image = "placeholder"
    result = BoundingRectangle().process(session_with_bounding_box)
    assert result.roi_image is None


def test_bounding_rectangle_rejects_missing_contour(session_with_mask):
    with pytest.raises(ValidationError):
        BoundingRectangle().process(session_with_mask)  # Find Contours never ran


def test_bounding_rectangle_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        BoundingRectangle().process(empty_session)


def test_min_area_rectangle_matches_known_square_size(session_with_contours):
    result = MinAreaRectangle().process(session_with_contours)
    box = result.bounding_box
    assert box["type"] == "min_area_rect"
    width, height = box["size"]
    assert 38 <= width <= 40
    assert 38 <= height <= 40
    assert len(box["box_points"]) == 4


def test_min_area_rectangle_rejects_missing_contour(session_with_mask):
    with pytest.raises(ValidationError):
        MinAreaRectangle().process(session_with_mask)


def test_min_area_rectangle_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        MinAreaRectangle().process(empty_session)


def test_convex_hull_reduces_or_preserves_point_count(session_with_contours):
    original_count = len(session_with_contours.largest_contour)
    result = ConvexHull().process(session_with_contours)
    assert len(result.largest_contour) <= original_count


def test_convex_hull_records_point_count_change(session_with_contours):
    result = ConvexHull().process(session_with_contours)
    assert "points" in result.processing_history[-1].details


def test_convex_hull_clears_roi_results(session_with_bounding_box):
    result = ConvexHull().process(session_with_bounding_box)
    assert result.bounding_box is None


def test_convex_hull_rejects_missing_contour(session_with_mask):
    with pytest.raises(ValidationError):
        ConvexHull().process(session_with_mask)


def test_convex_hull_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        ConvexHull().process(empty_session)


def test_min_area_rectangle_get_info_structure():
    info = MinAreaRectangle().get_info()
    assert info.name == "Minimum Area Rectangle"
    assert "rotating calipers" in info.math_intuition.lower()
