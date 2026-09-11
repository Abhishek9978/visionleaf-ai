"""Tests for visionleaf_ai.processing.restoration.median_filter.

Dependencies:
    pytest; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.restoration.median_filter.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.restoration.median_filter import (
    MedianFilter,
    MedianFilterParams,
)


def test_median_filter_removes_salt_and_pepper_noise(loaded_session):
    original = loaded_session.active_image.copy()
    stage = MedianFilter(MedianFilterParams(kernel_size=5))
    result = stage.process(loaded_session)
    # The noisy image has extreme (0/255) outlier pixels; after a median
    # filter the standard deviation should drop substantially since those
    # outliers get replaced by the surrounding flat-gray value.
    assert result.current_image.astype(float).std() < original.astype(float).std()


def test_median_filter_preserves_image_shape(loaded_session):
    original_shape = loaded_session.active_image.shape
    result = MedianFilter(MedianFilterParams(kernel_size=3)).process(loaded_session)
    assert result.current_image.shape == original_shape


def test_median_filter_records_history_event(loaded_session):
    result = MedianFilter(MedianFilterParams(kernel_size=5)).process(loaded_session)
    assert result.processing_history[0].label == "Median Filter"
    assert "kernel_size=5" in result.processing_history[0].details


def test_median_filter_rejects_even_kernel_size(loaded_session):
    stage = MedianFilter(MedianFilterParams(kernel_size=4))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_median_filter_rejects_kernel_size_one(loaded_session):
    stage = MedianFilter(MedianFilterParams(kernel_size=1))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_median_filter_rejects_kernel_larger_than_image(loaded_session):
    stage = MedianFilter(MedianFilterParams(kernel_size=999))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_median_filter_rejects_missing_image(empty_session):
    stage = MedianFilter(MedianFilterParams())
    with pytest.raises(ValidationError):
        stage.process(empty_session)


def test_median_filter_works_on_grayscale(loaded_grayscale_session):
    result = MedianFilter(MedianFilterParams()).process(loaded_grayscale_session)
    assert result.current_image.ndim == 2


def test_median_filter_get_info_structure():
    info = MedianFilter().get_info()
    assert info.name == "Median Filter"
    assert info.category == "restoration"
