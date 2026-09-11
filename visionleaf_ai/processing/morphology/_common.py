"""Shared internal helpers for morphological operation stages.

Purpose:
    All seven morphological operations need the same setup: a
    structuring element built from `kernel_size`, and a mask to
    operate on (`session.active_mask`). Centralizing this avoids seven
    copies of the same boilerplate and validation.

Description:
    Not a `PipelineStage` itself, not registered with
    `AlgorithmRegistry` — this is a private module (leading underscore)
    used only by sibling modules in `processing.morphology`.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.utils.validators.

Public functions:
    require_mask(session) -> np.ndarray
        Return the session's active mask, raising ValidationError if
        none exists yet.
    build_kernel(kernel_size) -> np.ndarray
        Build a square structuring element for morphological operations.
    validate_morphology_params(session, kernel_size, iterations) -> None
        Run the standard mask/kernel_size/iterations validation every
        morphological operation shares.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.utils.validators import ensure_kernel_fits_image, ensure_odd_positive_int


def require_mask(session: ImageSession) -> np.ndarray:
    """Return the session's active mask, or raise if none exists.

    Args:
        session: The `ImageSession` to read `active_mask` from.

    Returns:
        The active binary mask (`segmentation_mask` if set, else
        `binary_image`).

    Raises:
        ValidationError: If neither field is set — i.e. no
            thresholding stage has run yet.
    """
    mask = session.active_mask
    if mask is None:
        raise ValidationError(
            "No segmentation mask available",
            details="run a thresholding stage before morphological operations",
        )
    return mask


def build_kernel(kernel_size: int) -> np.ndarray:
    """Build a square rectangular structuring element.

    Args:
        kernel_size: Side length of the (square) structuring element.

    Returns:
        A `kernel_size` x `kernel_size` `np.ndarray` of ones (uint8),
        as `cv2.getStructuringElement(cv2.MORPH_RECT, ...)` produces.
    """
    return cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))


def validate_morphology_params(session: ImageSession, kernel_size: int, iterations: int) -> None:
    """Run the standard validation every morphological operation shares.

    Every one of the seven morphology algorithms (Erosion, Dilation,
    Opening, Closing, Morphological Gradient, Top Hat, Black Hat)
    needs exactly these four checks in exactly this order — extracted
    here so each algorithm's own `validate()` is a one-line call
    instead of eight duplicated lines.

    Args:
        session: The `ImageSession` to validate against (a mask must
            already exist — see `require_mask`).
        kernel_size: The proposed structuring-element size.
        iterations: The proposed iteration count.

    Raises:
        ValidationError: If no mask exists, `kernel_size` isn't a
            positive odd integer, `iterations` isn't a positive
            integer, or `kernel_size` exceeds the mask's dimensions.
    """
    mask = require_mask(session)
    ensure_odd_positive_int(kernel_size, "kernel_size")
    if iterations < 1:
        raise ValidationError(
            "iterations must be a positive integer", details=f"got {iterations!r}"
        )
    ensure_kernel_fits_image(kernel_size, mask.shape)
