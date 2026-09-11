"""Shared fixtures for restoration algorithm tests.

Purpose:
    Provide a small noisy test image (so denoising algorithms have
    something meaningful to remove) plus loaded/empty sessions.

Dependencies:
    pytest; numpy; visionleaf_ai.core.image_session.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.image_session import ImageSession


@pytest.fixture
def noisy_rgb_image() -> np.ndarray:
    """A flat mid-gray RGB image with salt-and-pepper noise sprinkled in."""
    rng = np.random.default_rng(42)
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    noise_mask = rng.random((32, 32)) < 0.1
    image[noise_mask] = rng.choice([0, 255], size=noise_mask.sum())[:, None]
    return image


@pytest.fixture
def loaded_session(noisy_rgb_image) -> ImageSession:
    return ImageSession(
        original_image=noisy_rgb_image.copy(), current_image=noisy_rgb_image.copy()
    )


@pytest.fixture
def loaded_grayscale_session() -> ImageSession:
    rng = np.random.default_rng(7)
    image = np.full((32, 32), 128, dtype=np.uint8)
    noise_mask = rng.random((32, 32)) < 0.1
    image[noise_mask] = rng.choice([0, 255], size=noise_mask.sum())
    return ImageSession(original_image=image.copy(), current_image=image.copy())


@pytest.fixture
def empty_session() -> ImageSession:
    return ImageSession.empty()
