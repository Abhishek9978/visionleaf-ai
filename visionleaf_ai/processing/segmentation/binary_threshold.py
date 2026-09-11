"""Binary Threshold.

Purpose:
    The simplest segmentation algorithm: split an image into exactly
    two intensity levels using a single fixed cutoff, producing the
    foreground/background mask later stages (morphology, contour
    detection, ROI extraction) build on.

Theory:
    Every pixel above the threshold becomes the maximum value (white);
    every pixel at or below it becomes 0 (black). This assumes the
    object of interest is reliably brighter (or darker, with the
    Inverse variant) than the background across the whole image — a
    single global cutoff works well when lighting is even.

Working Principle:
    1. Convert the active image to grayscale if it isn't already.
    2. Compare every pixel to `threshold`.
    3. Pixels greater than `threshold` become `max_value`; all others
       become 0.

Math intuition:
    dst(x, y) = max_value if src(x, y) > threshold else 0

Advantages:
    - Extremely fast and simple to reason about.
    - Produces a clean binary mask in one pass.

Limitations:
    - A single global threshold fails on unevenly-lit images (see Otsu
      or Adaptive Thresholding for that case).
    - Requires the user (or a prior stage) to pick a reasonable cutoff.

Typical Applications:
    - Separating a clearly-lit subject from a plain, contrasting background.
    - A quick first pass before adaptive refinement.

Complexity:
    O(H x W).

OpenCV reference:
    cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    BinaryThresholdParams, BinaryThreshold
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage
from visionleaf_ai.processing.segmentation._common import ensure_grayscale
from visionleaf_ai.utils.validators import ensure_threshold_in_range

logger = get_logger(__name__)


@dataclass(frozen=True)
class BinaryThresholdParams:
    """Parameters for `BinaryThreshold`.

    Attributes:
        threshold: Cutoff intensity, [0, 255]. Pixels above this
            become `max_value`; all others become 0.
        max_value: Output value assigned to pixels above `threshold`.
    """

    threshold: int = 127
    max_value: int = 255


@register_algorithm("binary_threshold")
class BinaryThreshold(PipelineStage):
    """Split an image into two levels at a fixed global cutoff."""

    stage_key = "segmentation"

    def __init__(self, params: BinaryThresholdParams | None = None, **kwargs) -> None:
        self.params = params or BinaryThresholdParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        ensure_threshold_in_range(self.params.threshold)
        ensure_threshold_in_range(self.params.max_value, "max_value")

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        grayscale = ensure_grayscale(session)

        _, binary = cv2.threshold(
            grayscale, self.params.threshold, self.params.max_value, cv2.THRESH_BINARY
        )

        session.clear_mask_results()
        session.binary_image = binary
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"threshold={self.params.threshold}, max_value={self.params.max_value}",
        )
        logger.info(
            "%s | threshold=%d | max_value=%d | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.threshold,
            self.params.max_value,
            grayscale.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Binary Threshold"

    def get_description(self) -> str:
        return "Splits pixels into two levels at a fixed cutoff: above becomes white, at-or-below becomes black."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "The simplest segmentation algorithm: split an image into "
                "two intensity levels using one fixed cutoff."
            ),
            theory=(
                "Every pixel above the threshold becomes the maximum "
                "value; every pixel at or below it becomes 0. Assumes the "
                "object of interest is reliably brighter than the "
                "background across the whole image."
            ),
            working_principle=(
                "1) Convert to grayscale if needed. 2) Compare every pixel "
                "to the threshold. 3) Assign max_value or 0 accordingly."
            ),
            math_intuition="dst(x, y) = max_value if src(x, y) > threshold else 0",
            advantages=(
                "Extremely fast and simple to reason about.",
                "Produces a clean binary mask in one pass.",
            ),
            limitations=(
                "Fails on unevenly-lit images — see Otsu or Adaptive Thresholding.",
                "Requires a reasonable cutoff to be chosen.",
            ),
            typical_applications=(
                "Separating a clearly-lit subject from a plain background.",
                "A quick first pass before adaptive refinement.",
            ),
            complexity="O(H x W)",
            opencv_reference="cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)",
        )
