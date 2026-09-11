"""Tests for truncate_threshold.py and to_zero_threshold.py.

Dependencies:
    pytest; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.segmentation.{truncate_threshold,to_zero_threshold}.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.segmentation.to_zero_threshold import (
    ToZeroThreshold,
    ToZeroThresholdParams,
)
from visionleaf_ai.processing.segmentation.truncate_threshold import (
    TruncateThreshold,
    TruncateThresholdParams,
)


def test_truncate_clamps_bright_pixels_to_threshold(loaded_session):
    result = TruncateThreshold(TruncateThresholdParams(threshold=100)).process(loaded_session)
    assert result.binary_image.max() <= 100
    assert result.binary_image[20, 30] == 100  # the bright square, clamped


def test_truncate_leaves_dark_pixels_unchanged(loaded_session):
    result = TruncateThreshold(TruncateThresholdParams(threshold=100)).process(loaded_session)
    assert result.binary_image[2, 2] == 30  # dark field, below threshold, unchanged


def test_truncate_rejects_out_of_range_threshold(loaded_session):
    with pytest.raises(ValidationError):
        TruncateThreshold(TruncateThresholdParams(threshold=-5)).process(loaded_session)


def test_truncate_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        TruncateThreshold().process(empty_session)


def test_to_zero_zeroes_dark_pixels(loaded_session):
    result = ToZeroThreshold(ToZeroThresholdParams(threshold=100)).process(loaded_session)
    assert result.binary_image[2, 2] == 0  # dark field, at/below threshold, zeroed


def test_to_zero_leaves_bright_pixels_unchanged(loaded_session):
    result = ToZeroThreshold(ToZeroThresholdParams(threshold=100)).process(loaded_session)
    assert result.binary_image[20, 30] == 220  # bright square, above threshold, unchanged


def test_to_zero_rejects_out_of_range_threshold(loaded_session):
    with pytest.raises(ValidationError):
        ToZeroThreshold(ToZeroThresholdParams(threshold=999)).process(loaded_session)


def test_to_zero_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        ToZeroThreshold().process(empty_session)


def test_to_zero_get_info_structure():
    info = ToZeroThreshold().get_info()
    assert info.name == "To Zero Threshold"
    assert info.working_principle
