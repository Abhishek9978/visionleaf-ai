"""Tests for visionleaf_ai.processing.restoration.gaussian_blur.

Dependencies:
    pytest; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.restoration.gaussian_blur.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.restoration.gaussian_blur import (
    GaussianBlur,
    GaussianBlurParams,
)


def test_gaussian_blur_reduces_noise_variance(loaded_session):
    original = loaded_session.active_image.copy()
    stage = GaussianBlur(GaussianBlurParams(kernel_size=5, sigma=1.5))
    result = stage.process(loaded_session)
    assert result.current_image.astype(float).std() < original.astype(float).std()


def test_gaussian_blur_preserves_image_shape(loaded_session):
    original_shape = loaded_session.active_image.shape
    result = GaussianBlur(GaussianBlurParams()).process(loaded_session)
    assert result.current_image.shape == original_shape


def test_gaussian_blur_records_history_event(loaded_session):
    result = GaussianBlur(GaussianBlurParams(kernel_size=3)).process(loaded_session)
    assert result.processing_history[0].label == "Gaussian Blur"
    assert "kernel_size=3" in result.processing_history[0].details


def test_gaussian_blur_rejects_even_kernel_size(loaded_session):
    stage = GaussianBlur(GaussianBlurParams(kernel_size=4))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_gaussian_blur_rejects_negative_kernel_size(loaded_session):
    stage = GaussianBlur(GaussianBlurParams(kernel_size=-3))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_gaussian_blur_rejects_negative_sigma(loaded_session):
    stage = GaussianBlur(GaussianBlurParams(kernel_size=5, sigma=-1.0))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_gaussian_blur_rejects_kernel_larger_than_image(loaded_session):
    stage = GaussianBlur(GaussianBlurParams(kernel_size=999))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_gaussian_blur_rejects_missing_image(empty_session):
    stage = GaussianBlur(GaussianBlurParams())
    with pytest.raises(ValidationError):
        stage.process(empty_session)


def test_gaussian_blur_works_on_grayscale(loaded_grayscale_session):
    result = GaussianBlur(GaussianBlurParams()).process(loaded_grayscale_session)
    assert result.current_image.ndim == 2


def test_gaussian_blur_get_info_structure():
    info = GaussianBlur().get_info()
    assert info.name == "Gaussian Blur"
    assert info.category == "restoration"
