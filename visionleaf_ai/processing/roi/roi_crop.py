"""ROI Cropping.

Purpose:
    Crop the image down to just its bounding box — the last step that
    turns "the whole photo" into "just the region Feature Extraction
    should actually look at."

Theory:
    A pure array-slice operation: extracts the rectangular sub-region
    of the active image described by `session.bounding_box`. If the
    stored box is a rotated Minimum Area Rectangle, its axis-aligned
    bounds are used instead (a true rotated crop would require an
    affine warp, which is more than a "crop" — this keeps the
    operation simple and predictable).

Working Principle:
    1. Require an existing bounding box.
    2. Derive its axis-aligned (x, y, w, h), clamped to the image's
       actual bounds.
    3. Slice the active image to that rectangle and store it in
       `session.roi_image`.

Math intuition:
    roi = image[y : y + h, x : x + w]

Advantages:
    - Trivial, lossless (no resampling), and fast.
    - Directly produces the input Feature Extraction (Milestone 7)
      will read.

Limitations:
    - Always axis-aligned, even when the source was a rotated Minimum
      Area Rectangle — some background may remain in the corners for
      a rotated shape.

Typical Applications:
    - Producing the final cropped region of interest for feature
      extraction and classification.

Complexity:
    O(h x w) to copy the cropped region.

OpenCV reference:
    NumPy array slicing (no OpenCV call needed for an axis-aligned crop).

Dependencies:
    numpy; visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    ROICropParams, ROICrop
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage
from visionleaf_ai.processing.roi._common import (
    axis_aligned_rect_from_bounding_box,
    require_bounding_box,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class ROICropParams:
    """Parameters for `ROICrop`.

    This algorithm has no tunable parameters — it's fully determined
    by the stored bounding box — but a (currently empty) params
    dataclass is kept for interface consistency with every other
    algorithm.
    """


@register_algorithm("roi_crop")
class ROICrop(PipelineStage):
    """Crop the active image down to its stored bounding box."""

    stage_key = "roi"

    def __init__(self, params: ROICropParams | None = None, **kwargs) -> None:
        self.params = params or ROICropParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_bounding_box(session)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        bounding_box = require_bounding_box(session)
        image = session.require_active_image()

        x, y, w, h = axis_aligned_rect_from_bounding_box(bounding_box)
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        session.roi_image = image[y : y + h, x : x + w].copy()
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
        return "ROI Cropping"

    def get_description(self) -> str:
        return "Crops the active image down to its stored bounding box."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Crop the image down to just its bounding box — turning "
                "the whole photo into the region Feature Extraction "
                "should look at."
            ),
            theory=(
                "A pure array-slice operation: extracts the rectangular "
                "sub-region described by the stored bounding box. A "
                "rotated Minimum Area Rectangle falls back to its "
                "axis-aligned bounds."
            ),
            working_principle=(
                "1) Require an existing bounding box. 2) Derive its "
                "axis-aligned (x, y, w, h), clamped to the image. 3) "
                "Slice the image and store the result."
            ),
            math_intuition="roi = image[y:y+h, x:x+w]",
            advantages=(
                "Trivial, lossless (no resampling), and fast.",
                "Directly produces Feature Extraction's input.",
            ),
            limitations=(
                "Always axis-aligned — some background may remain in "
                "the corners for a rotated shape.",
            ),
            typical_applications=(
                "Producing the final cropped region of interest for "
                "feature extraction and classification.",
            ),
            complexity="O(h x w) to copy the cropped region",
            opencv_reference="NumPy array slicing (no OpenCV call needed)",
        )
