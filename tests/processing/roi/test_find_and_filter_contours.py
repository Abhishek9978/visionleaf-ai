"""Tests for find_contours.py and filter_contours.py.

Dependencies:
    pytest; numpy; opencv-python (cv2); visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.roi.{find_contours,filter_contours}.
"""

from __future__ import annotations

import cv2
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.roi.filter_contours import (
    ContourAreaFilter,
    ContourAreaFilterParams,
)
from visionleaf_ai.processing.roi.find_contours import FindContours


def test_find_contours_finds_the_square(session_with_mask):
    result = FindContours().process(session_with_mask)
    assert len(result.contours) == 1
    assert result.largest_contour is not None


def test_find_contours_largest_contour_area_matches_square(session_with_mask):
    result = FindContours().process(session_with_mask)
    area = cv2.contourArea(result.largest_contour)
    assert 1500 <= area <= 1600  # a 40x40 square, area ~1600, minus boundary effects


def test_find_contours_on_empty_mask_finds_nothing(session_with_empty_mask):
    result = FindContours().process(session_with_empty_mask)
    assert result.contours == []
    assert result.largest_contour is None


def test_find_contours_records_history(session_with_mask):
    result = FindContours().process(session_with_mask)
    assert "contours_found=1" in result.processing_history[-1].details


def test_find_contours_clears_downstream_roi_results(session_with_bounding_box):
    result = FindContours().process(session_with_bounding_box)
    assert result.bounding_box is None
    assert result.roi_image is None


def test_find_contours_rejects_missing_mask(session_without_mask):
    with pytest.raises(ValidationError):
        FindContours().process(session_without_mask)


def test_find_contours_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        FindContours().process(empty_session)


def test_filter_contours_keeps_contour_within_range(session_with_contours):
    result = ContourAreaFilter(
        ContourAreaFilterParams(min_area=1000, max_area=2000)
    ).process(session_with_contours)
    assert len(result.contours) == 1


def test_filter_contours_removes_contour_outside_range(session_with_contours):
    result = ContourAreaFilter(
        ContourAreaFilterParams(min_area=5000, max_area=10000)
    ).process(session_with_contours)
    assert len(result.contours) == 0
    assert result.largest_contour is None


def test_filter_contours_rejects_no_contours_found(session_with_empty_mask):
    session = FindContours().process(session_with_empty_mask)
    with pytest.raises(ValidationError):
        ContourAreaFilter().process(session)


def test_filter_contours_rejects_max_less_than_min(session_with_contours):
    with pytest.raises(ValidationError):
        ContourAreaFilter(ContourAreaFilterParams(min_area=100, max_area=50)).process(
            session_with_contours
        )


def test_filter_contours_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        ContourAreaFilter().process(empty_session)


def test_find_contours_get_info_structure():
    info = FindContours().get_info()
    assert info.name == "Find Contours"
    assert info.category == "roi"
    assert info.typical_applications
