"""Shared fixtures for thresholding algorithm tests.

Purpose:
    Provide a synthetic RGB gradient image (bimodal-ish) and loaded/
    empty sessions.

Dependencies:
    pytest; numpy; visionleaf_ai.core.image_session.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.image_session import ImageSession


@pytest.fixture
def bimodal_gray_image() -> np.ndarray:
    """A clearly bimodal grayscale image: a bright square on a dark field."""
    image = np.full((48, 64), 30, dtype=np.uint8)
    image[10:38, 16:48] = 220
    return image


@pytest.fixture
def bimodal_rgb_image(bimodal_gray_image) -> np.ndarray:
    return np.stack([bimodal_gray_image] * 3, axis=-1)


@pytest.fixture
def loaded_session(bimodal_rgb_image) -> ImageSession:
    return ImageSession(
        original_image=bimodal_rgb_image.copy(), current_image=bimodal_rgb_image.copy()
    )


@pytest.fixture
def loaded_grayscale_session(bimodal_gray_image) -> ImageSession:
    return ImageSession(
        original_image=bimodal_gray_image.copy(), current_image=bimodal_gray_image.copy()
    )


@pytest.fixture
def empty_session() -> ImageSession:
    return ImageSession.empty()
