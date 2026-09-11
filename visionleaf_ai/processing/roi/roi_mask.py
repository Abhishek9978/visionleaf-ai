"""ROI Masking.

Purpose:
    Produce a full-size visualization of the segmented region with its
    background zeroed out — unlike ROI Cropping, this keeps the
    original image dimensions, useful for displaying "what was
    detected" in context rather than as an isolated crop.

Theory:
    Applies the active mask directly to the active image: pixels where
    the mask is foreground keep their original value; everywhere else
    is set to black. If contours have been found, the mask used is
    rebuilt from the largest contour specifically (so masking reflects
    exactly the region Find Contours/Convex Hull selected, not
    whatever noise might remain elsewhere in the raw mask).

Working Principle:
    1. Determine the mask to apply: rebuild a filled mask from
       `session.largest_contour` if one exists, otherwise fall back to
       `session.active_mask`.
    2. Apply it to the active image via `cv2.bitwise_and`.
    3. Store the result in `session.segmented_image`.

Math intuition:
    dst(x, y) = src(x, y) if mask(x, y) > 0 else 0

Advantages:
    - Preserves spatial context (background pixels are visibly zeroed,
      not removed from the frame).
    - Reflects exactly the selected contour, not incidental mask noise.

Limitations:
    - Does not reduce the array's size — later stages that assume a
      tightly-cropped ROI should use `roi_image` (from ROI Cropping)
      instead.

Typical Applications:
    - Visualizing exactly which pixels the segmentation identified as
      foreground, in their original position.

Complexity:
    O(H x W) (mask fill plus bitwise AND).

OpenCV reference:
    cv2.bitwise_and(src, src, mask=mask); cv2.drawContours(..., -1, 255, cv2.FILLED)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    ROIMaskParams, ROIMask
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage
from visionleaf_ai.processing.roi._common import require_mask

logger = get_logger(__name__)


@dataclass(frozen=True)
class ROIMaskParams:
    """Parameters for `ROIMask`.

    This algorithm has no tunable parameters — behavior is fully
    determined by the existing mask/contour — but a (currently empty)
    params dataclass is kept for interface consistency with every
    other algorithm.
    """


@register_algorithm("roi_mask")
class ROIMask(PipelineStage):
    """Zero out everything outside the segmented region, in place."""

    stage_key = "roi"

    def __init__(self, params: ROIMaskParams | None = None, **kwargs) -> None:
        self.params = params or ROIMaskParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        if session.largest_contour is None:
            require_mask(session)  # raises if neither is available

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        image = session.require_active_image()

        if session.largest_contour is not None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [session.largest_contour], -1, 255, cv2.FILLED)
            source = "largest_contour"
        else:
            mask = require_mask(session)
            source = "active_mask"

        masked = cv2.bitwise_and(image, image, mask=mask)

        session.segmented_image = masked
        duration = time.perf_counter() - started
        session.record_event(self.stage_key, self.get_name(), details=f"mask_source={source}")
        logger.info(
            "%s | mask_source=%s | image=%s | duration=%.4fs | success",
            self.get_name(),
            source,
            image.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "ROI Masking"

    def get_description(self) -> str:
        return "Zeroes out everything outside the segmented region, preserving the image's original size."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Produce a full-size visualization of the segmented "
                "region with its background zeroed out, keeping the "
                "original image dimensions."
            ),
            theory=(
                "Applies the active mask (or a mask rebuilt from the "
                "largest contour, if available) directly to the active "
                "image: foreground pixels keep their value, everything "
                "else becomes black."
            ),
            working_principle=(
                "1) Determine the mask: rebuild from largest_contour if "
                "available, else use active_mask. 2) Apply via bitwise "
                "AND. 3) Store the result."
            ),
            math_intuition="dst(x, y) = src(x, y) if mask(x, y) > 0 else 0",
            advantages=(
                "Preserves spatial context, unlike a cropped ROI.",
                "Reflects exactly the selected contour, not incidental mask noise.",
            ),
            limitations=(
                "Does not reduce array size — use ROI Cropping for a "
                "tightly-cropped region.",
            ),
            typical_applications=(
                "Visualizing exactly which pixels were identified as "
                "foreground, in their original position.",
            ),
            complexity="O(H x W)",
            opencv_reference="cv2.bitwise_and(src, src, mask=mask)",
        )
