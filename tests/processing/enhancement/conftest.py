"""Shared fixtures for enhancement algorithm tests.

Purpose:
    Provide small synthetic RGB/grayscale images and empty/loaded
    sessions so each algorithm's test module doesn't redefine the same
    boilerplate.

Dependencies:
    pytest; numpy; visionleaf_ai.core.image_session.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.image_session import ImageSession


@pytest.fixture
def rgb_image() -> np.ndarray:
    """A small, non-trivial RGB test image (gradient, not flat)."""
    gradient = np.linspace(0, 255, 32, dtype=np.uint8)
    image = np.tile(gradient, (24, 1))
    return np.stack([image, image, image], axis=-1)


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """A small, non-trivial single-channel test image."""
    gradient = np.linspace(0, 255, 32, dtype=np.uint8)
    return np.tile(gradient, (24, 1))


@pytest.fixture
def loaded_session(rgb_image) -> ImageSession:
    return ImageSession(original_image=rgb_image.copy(), current_image=rgb_image.copy())


@pytest.fixture
def loaded_grayscale_session(grayscale_image) -> ImageSession:
    return ImageSession(
        original_image=grayscale_image.copy(), current_image=grayscale_image.copy()
    )


@pytest.fixture
def empty_session() -> ImageSession:
    return ImageSession.empty()
