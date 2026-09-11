"""Tests for binary_threshold.py and binary_inverse_threshold.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.segmentation.{binary_threshold,binary_inverse_threshold}.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.segmentation.binary_inverse_threshold import (
    BinaryInverseThreshold,
    BinaryInverseThresholdParams,
)
from visionleaf_ai.processing.segmentation.binary_threshold import (
    BinaryThreshold,
    BinaryThresholdParams,
)


def test_binary_threshold_produces_two_level_output(loaded_session):
    result = BinaryThreshold(BinaryThresholdParams(threshold=127)).process(loaded_session)
    unique_values = set(np.unique(result.binary_image).tolist())
    assert unique_values <= {0, 255}


def test_binary_threshold_bright_region_becomes_white(loaded_session):
    result = BinaryThreshold(BinaryThresholdParams(threshold=127)).process(loaded_session)
    assert result.binary_image[20, 30] == 255  # inside the bright square
    assert result.binary_image[2, 2] == 0  # outside, in the dark field


def test_binary_threshold_records_history_and_clears_downstream(loaded_session):
    loaded_session.segmentation_mask = np.zeros((48, 64), dtype=np.uint8)
    loaded_session.contours = [np.array([[0, 0]])]
    result = BinaryThreshold(BinaryThresholdParams()).process(loaded_session)
    assert result.processing_history[-1].label == "Binary Threshold"
    assert result.segmentation_mask is None
    assert result.contours is None


def test_binary_threshold_rejects_out_of_range_threshold(loaded_session):
    with pytest.raises(ValidationError):
        BinaryThreshold(BinaryThresholdParams(threshold=300)).process(loaded_session)


def test_binary_threshold_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        BinaryThreshold().process(empty_session)


def test_binary_threshold_works_on_grayscale(loaded_grayscale_session):
    result = BinaryThreshold().process(loaded_grayscale_session)
    assert result.binary_image.ndim == 2


def test_binary_inverse_threshold_is_the_inverse_of_binary(loaded_session):
    import copy

    session_a = copy.deepcopy(loaded_session)
    session_b = copy.deepcopy(loaded_session)
    binary = BinaryThreshold(BinaryThresholdParams(threshold=127)).process(session_a).binary_image
    inverse = BinaryInverseThreshold(
        BinaryInverseThresholdParams(threshold=127)
    ).process(session_b).binary_image
    np.testing.assert_array_equal(binary, 255 - inverse)


def test_binary_inverse_threshold_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        BinaryInverseThreshold().process(empty_session)


def test_binary_inverse_threshold_get_info_structure():
    info = BinaryInverseThreshold().get_info()
    assert info.name == "Binary Inverse Threshold"
    assert info.category == "segmentation"
    assert info.opencv_reference is not None
