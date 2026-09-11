"""To Zero Threshold.

Purpose:
    Zero out everything below a cutoff while leaving brighter pixels
    untouched — the complement of Truncate Threshold, useful for
    suppressing dim background noise without altering the foreground's
    actual intensity values.

Theory:
    Pixels at or below the threshold are zeroed; pixels above it are
    left completely unchanged (not clamped, not remapped). `max_value`
    has no effect for this mode.

Working Principle:
    1. Convert the active image to grayscale if it isn't already.
    2. Compare every pixel to `threshold`.
    3. Pixels at or below `threshold` become 0; pixels above are left
       unchanged.

Math intuition:
    dst(x, y) = src(x, y) if src(x, y) > threshold else 0

Advantages:
    - Preserves exact foreground intensities, unlike Binary Threshold.
    - Cheap way to suppress low-intensity background noise.

Limitations:
    - Like Truncate, doesn't produce a binary mask on its own —
      typically feeds into a further thresholding stage.

Typical Applications:
    - Removing dim background texture before a stricter threshold.

Complexity:
    O(H x W).

OpenCV reference:
    cv2.threshold(src, thresh, maxval, cv2.THRESH_TOZERO)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    ToZeroThresholdParams, ToZeroThreshold
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
class ToZeroThresholdParams:
    """Parameters for `ToZeroThreshold`.

    Attributes:
        threshold: Floor intensity, [0, 255]. Pixels at or below this
            are zeroed.
    """

    threshold: int = 127


@register_algorithm("to_zero_threshold")
class ToZeroThreshold(PipelineStage):
    """Zero out pixel values at or below a floor, leaving the rest unchanged."""

    stage_key = "segmentation"

    def __init__(self, params: ToZeroThresholdParams | None = None, **kwargs) -> None:
        self.params = params or ToZeroThresholdParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        ensure_threshold_in_range(self.params.threshold)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        grayscale = ensure_grayscale(session)

        _, binary = cv2.threshold(grayscale, self.params.threshold, 255, cv2.THRESH_TOZERO)

        session.clear_mask_results()
        session.binary_image = binary
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key, self.get_name(), details=f"threshold={self.params.threshold}"
        )
        logger.info(
            "%s | threshold=%d | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.threshold,
            grayscale.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "To Zero Threshold"

    def get_description(self) -> str:
        return "Zeroes out pixel values at or below a floor, leaving brighter pixels untouched."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Zero out everything below a cutoff while leaving "
                "brighter pixels untouched — the complement of Truncate "
                "Threshold."
            ),
            theory=(
                "Pixels at or below the threshold are zeroed; pixels "
                "above it are left completely unchanged, not clamped or "
                "remapped."
            ),
            working_principle=(
                "1) Convert to grayscale if needed. 2) Compare every pixel "
                "to the threshold. 3) Zero values at or below it; leave "
                "the rest unchanged."
            ),
            math_intuition="dst(x, y) = src(x, y) if src(x, y) > threshold else 0",
            advantages=(
                "Preserves exact foreground intensities, unlike Binary Threshold.",
                "Cheap way to suppress low-intensity background noise.",
            ),
            limitations=(
                "Doesn't produce a binary mask on its own — typically "
                "feeds into a further thresholding stage.",
            ),
            typical_applications=("Removing dim background texture before a stricter threshold.",),
            complexity="O(H x W)",
            opencv_reference="cv2.threshold(src, thresh, maxval, cv2.THRESH_TOZERO)",
        )
