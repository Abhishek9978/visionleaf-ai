"""Otsu Threshold.

Purpose:
    Automatically choose a global threshold instead of requiring the
    user to guess one — the standard first choice for binary
    segmentation whenever the image has a clear bimodal histogram
    (a distinct foreground peak and background peak).

Theory:
    Otsu's method searches every possible threshold value and picks
    the one that minimizes the combined variance *within* the two
    resulting classes (equivalently, maximizes the variance *between*
    them). It assumes the histogram is roughly bimodal — one peak for
    background, one for foreground.

Working Principle:
    1. Convert the active image to grayscale if it isn't already.
    2. Compute the image's histogram.
    3. For every candidate threshold, compute the within-class
       variance of pixels below vs. above it.
    4. Pick the threshold that minimizes that variance, then apply
       Binary Threshold at that value.

Math intuition:
    threshold* = argmin_t [ w_bg(t) * var_bg(t) + w_fg(t) * var_fg(t) ]
    where w and var are each class's pixel-count weight and intensity
    variance at split point t.

Advantages:
    - No manual threshold to tune — genuinely automatic.
    - Well-understood, fast, and a strong default for bimodal images.

Limitations:
    - Assumes a bimodal histogram; performs poorly on images with
      uneven lighting or more than two dominant intensity groups (see
      Adaptive Thresholding for that case).

Typical Applications:
    - Segmenting a leaf from a fairly uniform background under even lighting.

Complexity:
    O(H x W) for the histogram plus O(256) for the search over
    candidate thresholds — effectively O(H x W).

OpenCV reference:
    cv2.threshold(src, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    OtsuThresholdParams, OtsuThreshold
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
class OtsuThresholdParams:
    """Parameters for `OtsuThreshold`.

    Attributes:
        max_value: Output value assigned to pixels above the
            automatically-chosen threshold.
    """

    max_value: int = 255


@register_algorithm("otsu_threshold")
class OtsuThreshold(PipelineStage):
    """Automatically choose a global threshold via Otsu's method."""

    stage_key = "segmentation"

    def __init__(self, params: OtsuThresholdParams | None = None, **kwargs) -> None:
        self.params = params or OtsuThresholdParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        ensure_threshold_in_range(self.params.max_value, "max_value")

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        grayscale = ensure_grayscale(session)

        chosen_threshold, binary = cv2.threshold(
            grayscale, 0, self.params.max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        session.clear_mask_results()
        session.binary_image = binary
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"auto_threshold={chosen_threshold:.0f}, max_value={self.params.max_value}",
        )
        logger.info(
            "%s | auto_threshold=%.1f | max_value=%d | image=%s | duration=%.4fs | success",
            self.get_name(),
            chosen_threshold,
            self.params.max_value,
            grayscale.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Otsu Threshold"

    def get_description(self) -> str:
        return "Automatically chooses a global threshold that best separates a bimodal histogram."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Automatically choose a global threshold instead of "
                "requiring the user to guess one."
            ),
            theory=(
                "Searches every possible threshold and picks the one that "
                "minimizes the combined variance within the two resulting "
                "classes. Assumes the histogram is roughly bimodal."
            ),
            working_principle=(
                "1) Convert to grayscale. 2) Compute the histogram. "
                "3) For every candidate threshold, compute the within-"
                "class variance above/below it. 4) Pick the threshold "
                "minimizing that variance and apply Binary Threshold there."
            ),
            math_intuition=(
                "threshold* = argmin_t [ w_bg(t) * var_bg(t) + "
                "w_fg(t) * var_fg(t) ]"
            ),
            advantages=(
                "No manual threshold to tune — genuinely automatic.",
                "Well-understood, fast, and a strong default for bimodal images.",
            ),
            limitations=(
                "Assumes a bimodal histogram; performs poorly with uneven "
                "lighting — see Adaptive Thresholding for that case.",
            ),
            typical_applications=(
                "Segmenting a leaf from a fairly uniform background under even lighting.",
            ),
            complexity="O(H x W) histogram + O(256) search",
            opencv_reference="cv2.threshold(src, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)",
        )
