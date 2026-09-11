"""Tests for visionleaf_ai.processing.restoration.bilateral_filter.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.restoration.bilateral_filter.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.restoration.bilateral_filter import (
    BilateralFilter,
    BilateralFilterParams,
)


def test_bilateral_filter_reduces_gaussian_noise_in_flat_region():
    # Bilateral filter is edge-preserving, which means it's specifically
    # weak against impulse (salt-and-pepper) noise — an outlier pixel's
    # own zero-distance weight dominates its neighbors' contributions.
    # It IS effective against mild Gaussian noise, which this test uses.
    rng = np.random.default_rng(3)
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    gaussian_noise = rng.normal(0, 15, flat.shape)
    noisy = np.clip(flat.astype(float) + gaussian_noise, 0, 255).astype(np.uint8)
    session = ImageSession(original_image=noisy.copy(), current_image=noisy.copy())

    stage = BilateralFilter(BilateralFilterParams(diameter=9, sigma_color=75, sigma_space=75))
    result = stage.process(session)

    assert result.current_image.astype(float).std() < noisy.astype(float).std()


def test_bilateral_filter_preserves_a_sharp_edge():
    # Half-black, half-white image: a genuine edge the filter should
    # preserve, unlike a naive average that would blur it away.
    half_edge_image = np.zeros((32, 32, 3), dtype=np.uint8)
    half_edge_image[:, 16:] = 255
    session = ImageSession(
        original_image=half_edge_image.copy(), current_image=half_edge_image.copy()
    )

    result = BilateralFilter(BilateralFilterParams()).process(session)

    # The edge column should still show a large jump, not a smooth ramp.
    left_of_edge = result.current_image[:, 15, 0].astype(int)
    right_of_edge = result.current_image[:, 16, 0].astype(int)
    assert (right_of_edge - left_of_edge).mean() > 100


def test_bilateral_filter_preserves_image_shape(loaded_session):
    original_shape = loaded_session.active_image.shape
    result = BilateralFilter(BilateralFilterParams()).process(loaded_session)
    assert result.current_image.shape == original_shape


def test_bilateral_filter_records_history_event(loaded_session):
    result = BilateralFilter(BilateralFilterParams(diameter=9)).process(loaded_session)
    assert result.processing_history[0].label == "Bilateral Filter"
    assert "diameter=9" in result.processing_history[0].details


def test_bilateral_filter_rejects_non_positive_diameter(loaded_session):
    stage = BilateralFilter(BilateralFilterParams(diameter=0))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_bilateral_filter_rejects_non_positive_sigma_color(loaded_session):
    stage = BilateralFilter(BilateralFilterParams(sigma_color=0))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_bilateral_filter_rejects_non_positive_sigma_space(loaded_session):
    stage = BilateralFilter(BilateralFilterParams(sigma_space=-5))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_bilateral_filter_rejects_diameter_larger_than_image(loaded_session):
    stage = BilateralFilter(BilateralFilterParams(diameter=999))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_bilateral_filter_rejects_missing_image(empty_session):
    stage = BilateralFilter(BilateralFilterParams())
    with pytest.raises(ValidationError):
        stage.process(empty_session)


def test_bilateral_filter_works_on_grayscale(loaded_grayscale_session):
    result = BilateralFilter(BilateralFilterParams()).process(loaded_grayscale_session)
    assert result.current_image.ndim == 2


def test_bilateral_filter_get_info_structure():
    info = BilateralFilter().get_info()
    assert info.name == "Bilateral Filter"
    assert info.category == "restoration"
