"""Shared internal helpers for thresholding stages.

Purpose:
    All seven thresholding algorithms need the same first step:
    convert the active image to grayscale (or use it directly if
    already grayscale). The two adaptive algorithms additionally share
    identical block_size/max_value validation. Centralizing both here
    avoids duplicating the same logic across sibling modules.

Description:
    Not a `PipelineStage` itself, not registered with
    `AlgorithmRegistry` — this is a private module (leading underscore)
    used only by sibling modules in `processing.segmentation`.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.utils.validators.

Public functions:
    ensure_grayscale(session) -> np.ndarray
        Return a grayscale version of the session's active image,
        also storing it into `session.grayscale_image` as a side effect.
    validate_adaptive_threshold_params(max_value, block_size) -> None
        Run the standard max_value/block_size validation both adaptive
        thresholding algorithms share.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.utils.validators import ensure_odd_positive_int, ensure_threshold_in_range


def ensure_grayscale(session: ImageSession) -> np.ndarray:
    """Return (and cache) a grayscale version of the session's active image.

    Args:
        session: The `ImageSession` to read `active_image` from.

    Returns:
        A single-channel `np.ndarray`. If `active_image` is already
        grayscale, it's returned as-is (not copied). Otherwise it's
        converted from RGB and the result is stored in
        `session.grayscale_image` for reuse by later stages/display.
    """
    image = session.require_active_image()
    if image.ndim == 2:
        return image
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    session.grayscale_image = grayscale
    return grayscale


def validate_adaptive_threshold_params(max_value: int, block_size: int) -> None:
    """Run the standard validation both adaptive thresholding algorithms share.

    Args:
        max_value: The proposed output value for foreground pixels.
        block_size: The proposed local neighborhood window size.

    Raises:
        ValidationError: If `max_value` isn't in [0, 255], `block_size`
            isn't a positive odd integer, or `block_size` is smaller
            than 3 (the minimum OpenCV accepts).
    """
    ensure_threshold_in_range(max_value, "max_value")
    ensure_odd_positive_int(block_size, "block_size")
    if block_size < 3:
        raise ValidationError(
            "block_size must be at least 3", details=f"got {block_size!r}"
        )
