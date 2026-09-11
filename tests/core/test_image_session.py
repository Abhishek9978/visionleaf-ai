"""Tests for visionleaf_ai.core.image_session.

Purpose:
    Verify ImageSession's lifecycle behavior: empty-by-default state,
    the active_image fallback logic, event recording, and that a fresh
    `.empty()` session has no data (used for resets).

Dependencies:
    pytest; numpy; visionleaf_ai.core.image_session.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageMetadata, ImageSession


def test_empty_session_is_not_loaded():
    session = ImageSession.empty()
    assert session.is_loaded() is False
    assert session.active_image is None
    assert session.processing_history == []


def test_is_loaded_true_once_original_image_set():
    session = ImageSession.empty()
    session.original_image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert session.is_loaded() is True


def test_active_image_prefers_current_over_original():
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    current = np.ones((4, 4, 3), dtype=np.uint8)
    session = ImageSession(original_image=original, current_image=current)
    assert session.active_image is current


def test_active_image_falls_back_to_original_when_no_current():
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    session = ImageSession(original_image=original, current_image=None)
    assert session.active_image is original


def test_record_event_appends_in_order():
    session = ImageSession.empty()
    session.record_event("acquisition", "Image Loaded")
    session.record_event("enhancement", "Enhanced", details="CLAHE applied")

    assert len(session.processing_history) == 2
    assert session.processing_history[0].label == "Image Loaded"
    assert session.processing_history[1].label == "Enhanced"
    assert session.processing_history[1].details == "CLAHE applied"


def test_metadata_is_frozen():
    metadata = ImageMetadata(
        filename="leaf.png",
        file_format="PNG",
        width=100,
        height=80,
        channels=3,
        file_size_bytes=2048,
        uploaded_at=datetime.now(),
        was_resized=False,
        source="upload",
    )
    with pytest.raises(AttributeError):
        metadata.filename = "changed.png"  # type: ignore[misc]


def test_two_sessions_do_not_share_history_list():
    """Guards against a dataclass mutable-default-sharing bug."""
    session_a = ImageSession.empty()
    session_b = ImageSession.empty()
    session_a.record_event("acquisition", "Image Loaded")
    assert session_b.processing_history == []


def test_reset_processing_restores_current_image_from_original():
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    modified = np.full((4, 4, 3), 255, dtype=np.uint8)
    session = ImageSession(original_image=original, current_image=modified)

    session.reset_processing()

    np.testing.assert_array_equal(session.current_image, original)
    assert session.current_image is not original  # a copy, not the same object


def test_reset_processing_clears_derived_fields():
    session = ImageSession(
        original_image=np.zeros((4, 4, 3), dtype=np.uint8),
        current_image=np.zeros((4, 4, 3), dtype=np.uint8),
        grayscale_image=np.zeros((4, 4), dtype=np.uint8),
        binary_image=np.zeros((4, 4), dtype=np.uint8),
        segmented_image=np.zeros((4, 4, 3), dtype=np.uint8),
        roi_image=np.zeros((2, 2, 3), dtype=np.uint8),
        feature_vector=np.array([1.0, 2.0]),
        prediction={"label": "Healthy"},
    )

    session.reset_processing()

    assert session.grayscale_image is None
    assert session.binary_image is None
    assert session.segmented_image is None
    assert session.roi_image is None
    assert session.feature_vector is None
    assert session.prediction is None


def test_reset_processing_trims_history_to_acquisition_only():
    session = ImageSession(original_image=np.zeros((4, 4, 3), dtype=np.uint8))
    session.record_event("acquisition", "Image Loaded")
    session.record_event("enhancement", "Brightness Adjustment")
    session.record_event("restoration", "Median Filter")

    session.reset_processing()

    assert len(session.processing_history) == 1
    assert session.processing_history[0].label == "Image Loaded"


def test_reset_processing_on_unloaded_session_does_not_raise():
    session = ImageSession.empty()
    session.reset_processing()  # should not raise even with no original_image
    assert session.current_image is None


def test_active_mask_falls_back_to_binary_image():
    session = ImageSession(
        original_image=np.zeros((4, 4, 3), dtype=np.uint8),
        binary_image=np.full((4, 4), 255, dtype=np.uint8),
    )
    np.testing.assert_array_equal(session.active_mask, session.binary_image)


def test_active_mask_prefers_segmentation_mask_over_binary_image():
    session = ImageSession(
        original_image=np.zeros((4, 4, 3), dtype=np.uint8),
        binary_image=np.zeros((4, 4), dtype=np.uint8),
        segmentation_mask=np.full((4, 4), 255, dtype=np.uint8),
    )
    assert session.active_mask is session.segmentation_mask


def test_active_mask_is_none_when_neither_field_set():
    session = ImageSession.empty()
    assert session.active_mask is None


def test_clear_mask_results_clears_everything_downstream_of_a_threshold():
    session = ImageSession(
        original_image=np.zeros((4, 4, 3), dtype=np.uint8),
        segmentation_mask=np.zeros((4, 4), dtype=np.uint8),
        contours=[np.array([[0, 0]])],
        largest_contour=np.array([[0, 0]]),
        bounding_box={"type": "axis_aligned", "x": 0, "y": 0, "w": 1, "h": 1},
        roi_image=np.zeros((2, 2, 3), dtype=np.uint8),
        segmented_image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    session.clear_mask_results()
    assert session.segmentation_mask is None
    assert session.contours is None
    assert session.largest_contour is None
    assert session.bounding_box is None
    assert session.roi_image is None
    assert session.segmented_image is None


def test_clear_contour_results_does_not_touch_binary_image_or_mask():
    binary = np.full((4, 4), 128, dtype=np.uint8)
    mask = np.full((4, 4), 200, dtype=np.uint8)
    session = ImageSession(
        original_image=np.zeros((4, 4, 3), dtype=np.uint8),
        binary_image=binary,
        segmentation_mask=mask,
        contours=[np.array([[0, 0]])],
        largest_contour=np.array([[0, 0]]),
        bounding_box={"type": "axis_aligned", "x": 0, "y": 0, "w": 1, "h": 1},
        roi_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    session.clear_contour_results()
    assert session.binary_image is binary
    assert session.segmentation_mask is mask
    assert session.contours is None
    assert session.largest_contour is None
    assert session.bounding_box is None
    assert session.roi_image is None


def test_clear_roi_results_does_not_touch_contours():
    contours = [np.array([[0, 0]])]
    largest = np.array([[0, 0]])
    session = ImageSession(
        original_image=np.zeros((4, 4, 3), dtype=np.uint8),
        contours=contours,
        largest_contour=largest,
        bounding_box={"type": "axis_aligned", "x": 0, "y": 0, "w": 1, "h": 1},
        roi_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    session.clear_roi_results()
    assert session.contours is contours
    assert session.largest_contour is largest
    assert session.bounding_box is None
    assert session.roi_image is None


def test_reset_processing_clears_segmentation_fields_too():
    session = ImageSession(
        original_image=np.zeros((4, 4, 3), dtype=np.uint8),
        current_image=np.zeros((4, 4, 3), dtype=np.uint8),
        binary_image=np.zeros((4, 4), dtype=np.uint8),
        segmentation_mask=np.zeros((4, 4), dtype=np.uint8),
        contours=[np.array([[0, 0]])],
        largest_contour=np.array([[0, 0]]),
        bounding_box={"type": "axis_aligned", "x": 0, "y": 0, "w": 1, "h": 1},
    )
    session.reset_processing()
    assert session.binary_image is None
    assert session.segmentation_mask is None
    assert session.contours is None
    assert session.largest_contour is None
    assert session.bounding_box is None


def test_require_active_image_returns_narrowed_image():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    session = ImageSession(original_image=image, current_image=image)
    assert session.require_active_image() is image


def test_require_active_image_raises_when_not_loaded():
    session = ImageSession.empty()
    with pytest.raises(ValidationError):
        session.require_active_image()


def test_require_active_mask_returns_narrowed_mask():
    mask = np.full((4, 4), 255, dtype=np.uint8)
    session = ImageSession(original_image=np.zeros((4, 4, 3), dtype=np.uint8), binary_image=mask)
    assert session.require_active_mask() is mask


def test_require_active_mask_raises_when_no_mask_exists():
    session = ImageSession(original_image=np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValidationError):
        session.require_active_mask()


def test_require_original_image_returns_narrowed_image():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    session = ImageSession(original_image=image)
    assert session.require_original_image() is image


def test_require_original_image_raises_when_not_loaded():
    session = ImageSession.empty()
    with pytest.raises(ValidationError):
        session.require_original_image()


def test_require_metadata_returns_narrowed_metadata():
    metadata = ImageMetadata(
        filename="leaf.png",
        file_format="PNG",
        width=10,
        height=10,
        channels=3,
        file_size_bytes=100,
        uploaded_at=datetime.now(),
        was_resized=False,
        source="upload",
    )
    session = ImageSession(original_image=np.zeros((4, 4, 3), dtype=np.uint8), metadata=metadata)
    assert session.require_metadata() is metadata


def test_require_metadata_raises_when_none_set():
    session = ImageSession(original_image=np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValidationError):
        session.require_metadata()


def test_reset_processing_clears_feature_extraction_fields():
    session = ImageSession(
        original_image=np.zeros((4, 4, 3), dtype=np.uint8),
        current_image=np.zeros((4, 4, 3), dtype=np.uint8),
        color_features={"rgb_mean": [1, 2, 3]},
        texture_features={"glcm": {"contrast": 1.0}},
        shape_features={"area": 100.0},
        edge_features={"canny_edge_density": 0.1},
        feature_vector=np.array([0.1, 0.2, 0.3]),
    )
    session.reset_processing()
    assert session.color_features is None
    assert session.texture_features is None
    assert session.shape_features is None
    assert session.edge_features is None
    assert session.feature_vector is None
