"""Adaptive Gaussian Threshold.

Purpose:
    Like Adaptive Mean Threshold, but weights the local neighborhood
    by a Gaussian instead of a flat average — typically smoother and
    less sensitive to a few extreme pixels within the neighborhood.

Theory:
    Each pixel's local threshold is a Gaussian-weighted sum of its
    neighborhood (nearby pixels count more than distant ones) minus a
    constant `C`, rather than a flat unweighted mean.

Working Principle:
    1. Convert the active image to grayscale if it isn't already.
    2. For each pixel, compute a Gaussian-weighted average of its
       local `block_size` window.
    3. Threshold that pixel against (weighted average - C).

Math intuition:
    T(x, y) = gaussian_weighted_mean(neighborhood(x, y)) - C
    dst(x, y) = max_value if src(x, y) > T(x, y) else 0

Advantages:
    - Generally smoother, more stable results than Adaptive Mean,
      since distant/outlier pixels in the window contribute less.
    - Same robustness to uneven lighting as Adaptive Mean.

Limitations:
    - Slightly more computation than Adaptive Mean for the same
      neighborhood size.
    - Still has block_size/C to tune.

Typical Applications:
    - The same uneven-lighting scenarios as Adaptive Mean, when a
      smoother mask is preferred.

Complexity:
    O(H x W).

OpenCV reference:
    cv2.adaptiveThreshold(src, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, blockSize, C)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    AdaptiveGaussianThresholdParams, AdaptiveGaussianThreshold
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage
from visionleaf_ai.processing.segmentation._common import (
    ensure_grayscale,
    validate_adaptive_threshold_params,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class AdaptiveGaussianThresholdParams:
    """Parameters for `AdaptiveGaussianThreshold`.

    Attributes:
        max_value: Output value assigned to foreground pixels.
        block_size: Size of the local neighborhood window; must be a
            positive odd integer greater than 1.
        C: Constant subtracted from the weighted mean before comparison.
            May be negative.
    """

    max_value: int = 255
    block_size: int = 11
    C: int = 2


@register_algorithm("adaptive_gaussian_threshold")
class AdaptiveGaussianThreshold(PipelineStage):
    """Threshold each pixel against a Gaussian-weighted local average."""

    stage_key = "segmentation"

    def __init__(self, params: AdaptiveGaussianThresholdParams | None = None, **kwargs) -> None:
        self.params = params or AdaptiveGaussianThresholdParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        validate_adaptive_threshold_params(self.params.max_value, self.params.block_size)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        grayscale = ensure_grayscale(session)

        binary = cv2.adaptiveThreshold(
            grayscale,
            self.params.max_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.params.block_size,
            self.params.C,
        )

        session.clear_mask_results()
        session.binary_image = binary
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"block_size={self.params.block_size}, C={self.params.C}",
        )
        logger.info(
            "%s | block_size=%d | C=%d | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.block_size,
            self.params.C,
            grayscale.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Adaptive Gaussian Threshold"

    def get_description(self) -> str:
        return "Thresholds each pixel against a Gaussian-weighted average of its local neighborhood."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Like Adaptive Mean Threshold, but weights the local "
                "neighborhood by a Gaussian for smoother, less outlier-"
                "sensitive results."
            ),
            theory=(
                "Each pixel's local threshold is a Gaussian-weighted sum "
                "of its neighborhood (nearby pixels count more than "
                "distant ones) minus a constant C, rather than a flat mean."
            ),
            working_principle=(
                "1) Convert to grayscale. 2) Compute a Gaussian-weighted "
                "average over a block_size window per pixel. 3) Threshold "
                "each pixel against (weighted average - C)."
            ),
            math_intuition=(
                "T(x, y) = gaussian_weighted_mean(neighborhood(x, y)) - C; "
                "dst(x, y) = max_value if src(x, y) > T(x, y) else 0"
            ),
            advantages=(
                "Smoother, more stable results than Adaptive Mean.",
                "Same robustness to uneven lighting as Adaptive Mean.",
            ),
            limitations=(
                "Slightly more computation than Adaptive Mean.",
                "Still has block_size/C to tune.",
            ),
            typical_applications=(
                "The same uneven-lighting scenarios as Adaptive Mean, "
                "when a smoother mask is preferred.",
            ),
            complexity="O(H x W)",
            opencv_reference=(
                "cv2.adaptiveThreshold(src, maxValue, "
                "cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)"
            ),
        )
