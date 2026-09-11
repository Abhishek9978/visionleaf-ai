"""Shared internal helpers for contour and ROI stages.

Purpose:
    Contour and ROI stages share a few common needs: requiring a mask
    or contour to already exist, and computing an axis-aligned
    bounding box from whatever `bounding_box` dict is currently stored
    (which may be either axis-aligned or a rotated min-area rect).

Description:
    Not `PipelineStage`s themselves, not registered with
    `AlgorithmRegistry` — private helpers used only by sibling modules
    in `processing.roi`.

Dependencies:
    numpy; visionleaf_ai.core.

Public functions:
    require_mask(session) -> np.ndarray
    require_contours(session) -> list[np.ndarray]
    require_largest_contour(session) -> np.ndarray
    require_bounding_box(session) -> dict
    axis_aligned_rect_from_bounding_box(bounding_box) -> tuple[int, int, int, int]
"""

from __future__ import annotations

import numpy as np

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession


def require_mask(session: ImageSession) -> np.ndarray:
    """Return the session's active mask, or raise if none exists."""
    mask = session.active_mask
    if mask is None:
        raise ValidationError(
            "No segmentation mask available",
            details="run a thresholding stage before contour/ROI operations",
        )
    return mask


def require_contours(session: ImageSession) -> list[np.ndarray]:
    """Return the session's contours, or raise if none exist.

    Raises:
        ValidationError: If `session.contours` is `None` or empty
            (covers both "Find Contours never ran" and "Find Contours
            ran but found nothing").
    """
    if not session.contours:
        raise ValidationError(
            "No contours found",
            details="run Find Contours on a non-empty mask first",
        )
    return session.contours


def require_largest_contour(session: ImageSession) -> np.ndarray:
    """Return the session's largest contour, or raise if none exists."""
    if session.largest_contour is None:
        raise ValidationError(
            "No largest contour available",
            details="run Find Contours first",
        )
    return session.largest_contour


def require_bounding_box(session: ImageSession) -> dict:
    """Return the session's bounding box dict, or raise if none exists."""
    if session.bounding_box is None:
        raise ValidationError(
            "No bounding box available",
            details="run Bounding Rectangle or Minimum Area Rectangle first",
        )
    return session.bounding_box


def axis_aligned_rect_from_bounding_box(bounding_box: dict) -> tuple[int, int, int, int]:
    """Derive an axis-aligned (x, y, w, h) rectangle from a bounding_box dict.

    Args:
        bounding_box: Either an axis-aligned dict (returned as-is) or a
            min-area-rect dict (its rotated `box_points` are reduced to
            their axis-aligned bounding rectangle).

    Returns:
        `(x, y, w, h)` as plain Python ints.
    """
    if bounding_box["type"] == "axis_aligned":
        return (
            int(bounding_box["x"]),
            int(bounding_box["y"]),
            int(bounding_box["w"]),
            int(bounding_box["h"]),
        )

    points = np.array(bounding_box["box_points"])
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
