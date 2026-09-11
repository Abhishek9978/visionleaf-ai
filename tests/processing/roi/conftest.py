"""Shared fixtures for contour/ROI algorithm tests.

Purpose:
    Provide sessions at various stages of the segmentation pipeline —
    with a mask only, with contours found, and fully through to a
    bounding box — so each ROI stage's test module can start from
    whatever precondition it actually needs.

Dependencies:
    pytest; numpy; opencv-python (cv2);
    visionleaf_ai.core.image_session; visionleaf_ai.processing.roi.find_contours;
    visionleaf_ai.processing.roi.bounding_rectangle.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.roi.bounding_rectangle import BoundingRectangle
from visionleaf_ai.processing.roi.find_contours import FindContours


@pytest.fixture
def square_mask() -> np.ndarray:
    """A solid square foreground on a black background."""
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:60, 30:70] = 255
    return mask


@pytest.fixture
def empty_mask() -> np.ndarray:
    """An all-black mask with no foreground at all."""
    return np.zeros((80, 100), dtype=np.uint8)


@pytest.fixture
def session_with_mask(square_mask) -> ImageSession:
    image = np.stack([square_mask] * 3, axis=-1)
    return ImageSession(
        original_image=image.copy(), current_image=image.copy(), binary_image=square_mask.copy()
    )


@pytest.fixture
def session_with_empty_mask(empty_mask) -> ImageSession:
    image = np.stack([empty_mask] * 3, axis=-1)
    return ImageSession(
        original_image=image.copy(), current_image=image.copy(), binary_image=empty_mask.copy()
    )


@pytest.fixture
def session_with_contours(session_with_mask) -> ImageSession:
    return FindContours().process(session_with_mask)


@pytest.fixture
def session_with_bounding_box(session_with_contours) -> ImageSession:
    return BoundingRectangle().process(session_with_contours)


@pytest.fixture
def session_without_mask() -> ImageSession:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    return ImageSession(original_image=image.copy(), current_image=image.copy())


@pytest.fixture
def empty_session() -> ImageSession:
    return ImageSession.empty()
