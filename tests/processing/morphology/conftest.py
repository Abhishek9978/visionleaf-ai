"""Shared fixtures for morphology algorithm tests.

Purpose:
    Provide a synthetic binary mask (a solid square with a small hole
    and small isolated noise specks) so Opening/Closing/TopHat/
    BlackHat all have something meaningful to operate on.

Dependencies:
    pytest; numpy; visionleaf_ai.core.image_session.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.image_session import ImageSession


@pytest.fixture
def solid_mask() -> np.ndarray:
    """A clean solid square mask, no noise or holes."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    return mask


@pytest.fixture
def noisy_mask() -> np.ndarray:
    """A square mask with a small internal hole and small external specks."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    mask[28:32, 28:32] = 0  # a small hole inside the square
    mask[4:7, 4:7] = 255  # a small isolated noise speck outside
    return mask


@pytest.fixture
def loaded_session_with_binary(solid_mask) -> ImageSession:
    image = np.stack([solid_mask] * 3, axis=-1)
    return ImageSession(
        original_image=image.copy(), current_image=image.copy(), binary_image=solid_mask.copy()
    )


@pytest.fixture
def loaded_session_with_noisy_binary(noisy_mask) -> ImageSession:
    image = np.stack([noisy_mask] * 3, axis=-1)
    return ImageSession(
        original_image=image.copy(), current_image=image.copy(), binary_image=noisy_mask.copy()
    )


@pytest.fixture
def loaded_session_without_mask() -> ImageSession:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    return ImageSession(original_image=image.copy(), current_image=image.copy())


@pytest.fixture
def empty_session() -> ImageSession:
    return ImageSession.empty()
