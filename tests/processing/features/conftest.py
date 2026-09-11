"""Shared fixtures for feature extraction algorithm tests.

Purpose:
    Provide sessions at various stages of readiness for feature
    extraction: a fully segmented+cropped+masked session (the normal
    case), a session with only roi_image (no mask), and sessions
    missing prerequisites (no segmentation, no contour) for testing
    validation failures.

Dependencies:
    pytest; numpy; opencv-python (cv2); visionleaf_ai.core.image_session;
    visionleaf_ai.processing.roi.{find_contours,bounding_rectangle,roi_crop,roi_mask}.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.roi.bounding_rectangle import BoundingRectangle
from visionleaf_ai.processing.roi.find_contours import FindContours
from visionleaf_ai.processing.roi.roi_crop import ROICrop
from visionleaf_ai.processing.roi.roi_mask import ROIMask


@pytest.fixture
def leaf_like_image() -> np.ndarray:
    """A synthetic RGB image: a mottled green/brown square on a dark field."""
    rng = np.random.default_rng(42)
    image = np.full((100, 100, 3), 30, dtype=np.uint8)
    region = np.full((60, 60, 3), (60, 140, 50), dtype=np.uint8)
    noise = rng.integers(0, 40, size=region.shape, dtype=np.uint8)
    region = np.clip(region.astype(int) + noise, 0, 255).astype(np.uint8)
    image[20:80, 20:80] = region
    return image


@pytest.fixture
def fully_prepared_session(leaf_like_image) -> ImageSession:
    """A session with segmentation, contour, bounding box, crop, AND mask —
    the normal precondition for any feature extractor."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    session = ImageSession(
        original_image=leaf_like_image.copy(),
        current_image=leaf_like_image.copy(),
        binary_image=mask,
    )
    session = FindContours().process(session)
    session = BoundingRectangle().process(session)
    session = ROICrop().process(session)
    session = ROIMask().process(session)
    return session


@pytest.fixture
def roi_only_session(leaf_like_image) -> ImageSession:
    """A session with only roi_image (no ROI Masking run) — the fallback path."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    session = ImageSession(
        original_image=leaf_like_image.copy(),
        current_image=leaf_like_image.copy(),
        binary_image=mask,
    )
    session = FindContours().process(session)
    session = BoundingRectangle().process(session)
    session = ROICrop().process(session)
    return session


@pytest.fixture
def session_without_segmentation(leaf_like_image) -> ImageSession:
    """A loaded session with no segmentation/ROI work done at all."""
    return ImageSession(original_image=leaf_like_image.copy(), current_image=leaf_like_image.copy())


@pytest.fixture
def session_without_contour(leaf_like_image) -> ImageSession:
    """A session with a mask but Find Contours never run."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    return ImageSession(
        original_image=leaf_like_image.copy(),
        current_image=leaf_like_image.copy(),
        binary_image=mask,
    )


@pytest.fixture
def empty_session() -> ImageSession:
    return ImageSession.empty()
