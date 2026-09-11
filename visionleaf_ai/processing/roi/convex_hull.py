"""Convex Hull.

Purpose:
    Simplify the largest contour into its convex envelope — the
    smallest convex shape containing every point of the original
    contour — smoothing over concavities for more robust downstream
    shape measurement.

Theory:
    A shape is convex if a straight line between any two of its points
    never leaves the shape. Real contours (e.g. an irregular leaf
    outline with lobes) are rarely convex; their convex hull "shrink-
    wraps" the contour, removing inward dents while keeping the
    outermost points.

Working Principle:
    1. Require a largest contour.
    2. Compute its convex hull.
    3. Overwrite `session.largest_contour` with the hull's points —
       the hull is a refinement of that same contour, not a separate
       shape, so it replaces rather than adding a new field.

Math intuition:
    hull = the minimal set of contour points such that every other
    point lies on or inside the polygon they form (Graham scan /
    Sklansky's algorithm, as implemented by OpenCV).

Advantages:
    - Produces a simpler, always-convex shape — useful for stable area/
      perimeter measurements less sensitive to small boundary noise.
    - Cheap relative to the accuracy gain for downstream measurements.

Limitations:
    - Discards genuine concave detail (e.g. a leaf's lobed edges) —
      not appropriate when the actual concave shape matters.
    - Irreversible: once applied, the original (non-convex) contour
      detail is gone unless Find Contours is re-run.

Typical Applications:
    - Producing a stable shape for area/perimeter-based feature
      extraction when minor boundary irregularities shouldn't affect
      the measurement.

Complexity:
    O(N log N) for a contour of N points.

OpenCV reference:
    cv2.convexHull(contour)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    ConvexHullParams, ConvexHull
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
class ConvexHullParams:
    """Parameters for `ConvexHull`.

    This algorithm has no tunable parameters — it's fully determined
    by the largest contour — but a (currently empty) params dataclass
    is kept for interface consistency with every other algorithm.
    """


@register_algorithm("convex_hull")
class ConvexHull(PipelineStage):
    """Replace the largest contour with its convex envelope."""

    stage_key = "roi"

    def __init__(self, params: ConvexHullParams | None = None, **kwargs) -> None:
        self.params = params or ConvexHullParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_largest_contour(session)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        contour = require_largest_contour(session)

        hull = cv2.convexHull(contour)

        original_point_count = len(contour)
        session.largest_contour = hull
        session.clear_roi_results()
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"points {original_point_count} -> {len(hull)}",
        )
        logger.info(
            "%s | points %d -> %d | duration=%.4fs | success",
            self.get_name(),
            original_point_count,
            len(hull),
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Convex Hull"

    def get_description(self) -> str:
        return "Replaces the largest contour with its convex envelope, smoothing over concavities."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Simplify the largest contour into its convex envelope — "
                "the smallest convex shape containing every original point."
            ),
            theory=(
                "A shape is convex if a straight line between any two of "
                "its points never leaves the shape. Real contours are "
                "rarely convex; the convex hull shrink-wraps the contour, "
                "removing inward dents."
            ),
            working_principle=(
                "1) Require a largest contour. 2) Compute its convex "
                "hull. 3) Overwrite the largest contour with the hull's "
                "points — a refinement, not a new shape."
            ),
            math_intuition=(
                "The minimal set of points such that every other point "
                "lies on or inside the polygon they form (Sklansky's "
                "algorithm)."
            ),
            advantages=(
                "Produces a simpler, always-convex shape for stable measurements.",
                "Cheap relative to the accuracy gain for downstream measurements.",
            ),
            limitations=(
                "Discards genuine concave detail (e.g. lobed leaf edges).",
                "Irreversible — re-run Find Contours to get the original back.",
            ),
            typical_applications=(
                "Producing a stable shape for area/perimeter-based feature extraction.",
            ),
            complexity="O(N log N) for N contour points",
            opencv_reference="cv2.convexHull(contour)",
        )
