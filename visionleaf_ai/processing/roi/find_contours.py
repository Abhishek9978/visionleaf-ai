"""Find Contours.

Purpose:
    Trace the boundaries of every connected foreground region in a
    binary mask, turning a pixel mask into a list of actual traceable
    coordinate boundaries — the bridge between "a mask" and "a
    specific shape" that every later contour/ROI/feature-extraction
    stage depends on.

Theory:
    Contour finding walks the boundary between foreground and
    background pixels, producing an ordered list of (x, y) points for
    each separate connected foreground region. Multiple disconnected
    regions in the mask (e.g. several disease spots) each produce
    their own contour, all stored together.

Working Principle:
    1. Require an existing mask (`session.active_mask`).
    2. Find every external contour in the mask.
    3. Store all of them in `session.contours`.
    4. Identify the one with the largest enclosed area and store it
       separately as `session.largest_contour`, since most downstream
       work (bounding box, ROI crop) cares about the single dominant
       region, not every contour individually.

Math intuition:
    Not a single formula — a boundary-tracing algorithm (Suzuki &
    Abe's border-following algorithm, as implemented by OpenCV) that
    walks pixel-by-pixel around each connected region's edge.

Advantages:
    - Converts a pixel mask into structured, traceable shape data.
    - Naturally supports multiple separate regions in one pass.

Limitations:
    - Only as good as the input mask — noisy or fragmented masks
      produce noisy or fragmented contours (run morphological cleanup
      first if needed).
    - `RETR_EXTERNAL` (used here) only finds outermost boundaries, not
      nested holes within a region.

Typical Applications:
    - The mandatory first step before Bounding Rectangle, Minimum Area
      Rectangle, Convex Hull, or any ROI cropping/masking.

Complexity:
    O(H x W) for the boundary trace, plus O(N) to compare N contours'
    areas when picking the largest.

OpenCV reference:
    cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    FindContoursParams, FindContours
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage
from visionleaf_ai.processing.roi._common import require_mask

logger = get_logger(__name__)


@dataclass(frozen=True)
class FindContoursParams:
    """Parameters for `FindContours`.

    This algorithm has no tunable parameters — behavior is fully
    determined by the input mask — but a (currently empty) params
    dataclass is kept for interface consistency with every other
    algorithm.
    """


@register_algorithm("find_contours")
class FindContours(PipelineStage):
    """Trace every connected foreground region's boundary in a mask."""

    stage_key = "roi"

    def __init__(self, params: FindContoursParams | None = None, **kwargs) -> None:
        self.params = params or FindContoursParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_mask(session)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        mask = require_mask(session)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        largest = max(contours, key=cv2.contourArea) if contours else None

        session.clear_roi_results()
        session.contours = contours
        session.largest_contour = largest
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key, self.get_name(), details=f"contours_found={len(contours)}"
        )
        logger.info(
            "%s | contours_found=%d | mask=%s | duration=%.4fs | success",
            self.get_name(),
            len(contours),
            mask.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Find Contours"

    def get_description(self) -> str:
        return "Traces the boundary of every connected foreground region in a mask."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Trace the boundaries of every connected foreground "
                "region in a mask, turning pixels into traceable shape "
                "coordinates."
            ),
            theory=(
                "Walks the boundary between foreground and background "
                "pixels, producing an ordered list of points for each "
                "separate connected region. Multiple disconnected "
                "regions each produce their own contour."
            ),
            working_principle=(
                "1) Require an existing mask. 2) Find every external "
                "contour. 3) Store all of them. 4) Identify the largest "
                "by area and store it separately."
            ),
            math_intuition=(
                "Not a single formula — a boundary-tracing algorithm "
                "(Suzuki & Abe border-following) that walks pixel-by-"
                "pixel around each region's edge."
            ),
            advantages=(
                "Converts a pixel mask into structured, traceable shape data.",
                "Naturally supports multiple separate regions in one pass.",
            ),
            limitations=(
                "Only as good as the input mask — clean it with "
                "morphology first if it's noisy.",
                "RETR_EXTERNAL only finds outermost boundaries, not "
                "nested holes.",
            ),
            typical_applications=(
                "The mandatory first step before any bounding box, hull, "
                "or ROI operation.",
            ),
            complexity="O(H x W) trace + O(N) to compare N contour areas",
            opencv_reference="cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
        )
