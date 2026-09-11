"""Bounding Rectangle.

Purpose:
    Compute the smallest axis-aligned rectangle that fully contains
    the largest contour — the simplest possible region-of-interest
    boundary, and the input `ROI Cropping` uses to crop the image.

Theory:
    Finds the minimum and maximum x/y coordinates among the contour's
    points and uses them as the rectangle's corners. Always aligned to
    the image's horizontal/vertical axes, regardless of the contour's
    actual orientation.

Working Principle:
    1. Require a largest contour (`session.largest_contour`).
    2. Find the min/max x and y among its points.
    3. Store `{x, y, w, h}` in `session.bounding_box`.

Math intuition:
    x = min(contour_x), y = min(contour_y)
    w = max(contour_x) - x, h = max(contour_y) - y

Advantages:
    - Simplest possible ROI boundary — trivial to crop or draw.
    - Always axis-aligned, which is exactly what simple array slicing
      (ROI Cropping) needs.

Limitations:
    - Wastes space around a diagonally-oriented or irregular shape —
      see Minimum Area Rectangle for a tighter, rotated fit.

Typical Applications:
    - The standard input to ROI cropping and bounding-box visualization.

Complexity:
    O(N) for a contour of N points.

OpenCV reference:
    cv2.boundingRect(contour)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    BoundingRectangleParams, BoundingRectangle
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage
from visionleaf_ai.processing.roi._common import require_largest_contour

logger = get_logger(__name__)


@dataclass(frozen=True)
class BoundingRectangleParams:
    """Parameters for `BoundingRectangle`.

    This algorithm has no tunable parameters — it's fully determined
    by the largest contour — but a (currently empty) params dataclass
    is kept for interface consistency with every other algorithm.
    """


@register_algorithm("bounding_rectangle")
class BoundingRectangle(PipelineStage):
    """Compute the smallest axis-aligned rectangle around the largest contour."""

    stage_key = "roi"

    def __init__(self, params: BoundingRectangleParams | None = None, **kwargs) -> None:
        self.params = params or BoundingRectangleParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_largest_contour(session)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        contour = require_largest_contour(session)

        x, y, w, h = cv2.boundingRect(contour)

        session.clear_roi_results()
        session.bounding_box = {"type": "axis_aligned", "x": x, "y": y, "w": w, "h": h}
        duration = time.perf_counter() - started
        session.record_event(self.stage_key, self.get_name(), details=f"x={x}, y={y}, w={w}, h={h}")
        logger.info(
            "%s | x=%d y=%d w=%d h=%d | duration=%.4fs | success",
            self.get_name(),
            x,
            y,
            w,
            h,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Bounding Rectangle"

    def get_description(self) -> str:
        return "Computes the smallest axis-aligned rectangle that fully contains the largest contour."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Compute the smallest axis-aligned rectangle that fully "
                "contains the largest contour — the simplest possible ROI "
                "boundary."
            ),
            theory=(
                "Finds the minimum and maximum x/y coordinates among the "
                "contour's points and uses them as the rectangle's "
                "corners. Always aligned to the image's axes."
            ),
            working_principle=(
                "1) Require a largest contour. 2) Find its min/max x and "
                "y. 3) Store the resulting rectangle."
            ),
            math_intuition=(
                "x = min(contour_x), y = min(contour_y); "
                "w = max(contour_x) - x, h = max(contour_y) - y"
            ),
            advantages=(
                "Simplest possible ROI boundary — trivial to crop or draw.",
                "Always axis-aligned, exactly what array-slice cropping needs.",
            ),
            limitations=(
                "Wastes space around a diagonally-oriented shape — see "
                "Minimum Area Rectangle for a tighter fit.",
            ),
            typical_applications=(
                "The standard input to ROI cropping and bounding-box visualization.",
            ),
            complexity="O(N) for a contour of N points",
            opencv_reference="cv2.boundingRect(contour)",
        )
