"""Minimum Area Rectangle.

Purpose:
    Compute the smallest possible rectangle that contains the largest
    contour, allowing rotation — tighter than Bounding Rectangle for
    shapes that aren't aligned with the image's axes.

Theory:
    Bounding Rectangle is always axis-aligned, which wastes space
    around a diagonally-oriented object. This algorithm instead finds
    the minimum-area rectangle at *any* angle that still fully
    contains the shape, via rotating calipers.

Working Principle:
    1. Require a largest contour.
    2. Compute the minimum-area enclosing rectangle at any angle.
    3. Store its center, size, angle, and the four corner points
       (`box_points`) in `session.bounding_box`.

Math intuition:
    Not a single formula — the rotating calipers technique considers
    every edge of the contour's convex hull as a candidate rectangle
    orientation and keeps the one with the smallest area.

Advantages:
    - Tighter fit than an axis-aligned box for rotated/diagonal shapes.
    - Directly gives an orientation angle, useful for later shape
      features (Milestone 7).

Limitations:
    - Not directly usable by simple NumPy array-slice cropping (which
      only supports axis-aligned rectangles) — `ROI Cropping` falls
      back to the rotated box's own axis-aligned bounds when given one
      of these.
    - Slightly more expensive to compute than the axis-aligned version.

Typical Applications:
    - Measuring a leaf's true elongation/orientation when it's
      photographed at an angle.

Complexity:
    O(N log N) for N contour points (convex hull plus rotating calipers).

OpenCV reference:
    cv2.minAreaRect(contour), cv2.boxPoints(rect)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    MinAreaRectangleParams, MinAreaRectangle
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
class MinAreaRectangleParams:
    """Parameters for `MinAreaRectangle`.

    This algorithm has no tunable parameters — it's fully determined
    by the largest contour — but a (currently empty) params dataclass
    is kept for interface consistency with every other algorithm.
    """


@register_algorithm("min_area_rectangle")
class MinAreaRectangle(PipelineStage):
    """Compute the smallest possible (possibly rotated) rectangle around the largest contour."""

    stage_key = "roi"

    def __init__(self, params: MinAreaRectangleParams | None = None, **kwargs) -> None:
        self.params = params or MinAreaRectangleParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_largest_contour(session)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        contour = require_largest_contour(session)

        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), angle = rect
        box_points = cv2.boxPoints(rect).tolist()

        session.clear_roi_results()
        session.bounding_box = {
            "type": "min_area_rect",
            "center": (center_x, center_y),
            "size": (width, height),
            "angle": angle,
            "box_points": box_points,
        }
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"size=({width:.1f}, {height:.1f}), angle={angle:.1f}",
        )
        logger.info(
            "%s | size=(%.1f, %.1f) | angle=%.1f | duration=%.4fs | success",
            self.get_name(),
            width,
            height,
            angle,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Minimum Area Rectangle"

    def get_description(self) -> str:
        return "Computes the smallest possible rectangle, at any angle, that contains the largest contour."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Compute the smallest possible rectangle that contains "
                "the largest contour, allowing rotation — tighter than "
                "Bounding Rectangle for diagonally-oriented shapes."
            ),
            theory=(
                "Bounding Rectangle is always axis-aligned, which wastes "
                "space around a diagonal object. This finds the minimum-"
                "area rectangle at any angle that still fully contains "
                "the shape, via rotating calipers."
            ),
            working_principle=(
                "1) Require a largest contour. 2) Compute the minimum-"
                "area enclosing rectangle at any angle. 3) Store its "
                "center, size, angle, and corner points."
            ),
            math_intuition=(
                "Rotating calipers: considers every edge of the contour's "
                "convex hull as a candidate rectangle orientation and "
                "keeps the smallest-area one."
            ),
            advantages=(
                "Tighter fit than an axis-aligned box for rotated shapes.",
                "Directly gives an orientation angle, useful for shape features.",
            ),
            limitations=(
                "Not directly usable by simple axis-aligned array-slice "
                "cropping — ROI Cropping falls back to its axis-aligned bounds.",
                "Slightly more expensive than the axis-aligned version.",
            ),
            typical_applications=(
                "Measuring a leaf's true elongation/orientation when "
                "photographed at an angle.",
            ),
            complexity="O(N log N) for N contour points",
            opencv_reference="cv2.minAreaRect(contour), cv2.boxPoints(rect)",
        )
