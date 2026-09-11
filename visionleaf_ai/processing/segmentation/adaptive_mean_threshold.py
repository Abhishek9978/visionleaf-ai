"""Adaptive Mean Threshold.

Purpose:
    Handle uneven lighting that defeats a single global threshold (like
    Binary Threshold or Otsu) by computing a *different* threshold for
    every pixel, based on its local neighborhood.

Theory:
    Instead of one cutoff for the whole image, each pixel's threshold
    is the mean intensity of its own neighborhood (a `block_size` x
    `block_size` window), minus a constant `C`. A pixel brighter than
    its *local* neighborhood average becomes foreground, even if that
    neighborhood itself is dim (e.g. one side of the image in shadow).

Working Principle:
    1. Convert the active image to grayscale if it isn't already.
    2. For each pixel, compute the mean of its local `block_size`
       window.
    3. Threshold that pixel against (local mean - C).

Math intuition:
    T(x, y) = mean(neighborhood(x, y)) - C
    dst(x, y) = max_value if src(x, y) > T(x, y) else 0

Advantages:
    - Robust to uneven illumination across the image.
    - Still fully automatic per pixel, just locally rather than globally.

Limitations:
    - Two parameters to tune (block_size, C) instead of Otsu's zero.
    - Can be noisier than a global threshold in genuinely flat regions,
      since local means fluctuate with sensor/compression noise.

Typical Applications:
    - Segmenting a leaf photographed under a natural light gradient
      (e.g. one side shaded).

Complexity:
    O(H x W) — computing local means is done with an efficient box
    filter, not a naive per-pixel neighborhood scan.

OpenCV reference:
    cv2.adaptiveThreshold(src, maxValue, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY, blockSize, C)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    AdaptiveMeanThresholdParams, AdaptiveMeanThreshold
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
class AdaptiveMeanThresholdParams:
    """Parameters for `AdaptiveMeanThreshold`.

    Attributes:
        max_value: Output value assigned to foreground pixels.
        block_size: Size of the local neighborhood window; must be a
            positive odd integer greater than 1.
        C: Constant subtracted from the local mean before comparison.
            May be negative.
    """

    max_value: int = 255
    block_size: int = 11
    C: int = 2


@register_algorithm("adaptive_mean_threshold")
class AdaptiveMeanThreshold(PipelineStage):
    """Threshold each pixel against its own local neighborhood's mean."""

    stage_key = "segmentation"

    def __init__(self, params: AdaptiveMeanThresholdParams | None = None, **kwargs) -> None:
        self.params = params or AdaptiveMeanThresholdParams(**kwargs)

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
            cv2.ADAPTIVE_THRESH_MEAN_C,
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
        return "Adaptive Mean Threshold"

    def get_description(self) -> str:
        return "Thresholds each pixel against the mean intensity of its own local neighborhood."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Handle uneven lighting that defeats a single global "
                "threshold by computing a different threshold per pixel."
            ),
            theory=(
                "Each pixel's threshold is the mean intensity of its own "
                "local neighborhood, minus a constant C. A pixel brighter "
                "than its local neighborhood average becomes foreground, "
                "even if that neighborhood itself is dim."
            ),
            working_principle=(
                "1) Convert to grayscale. 2) Compute the local mean over "
                "a block_size window per pixel. 3) Threshold each pixel "
                "against (local mean - C)."
            ),
            math_intuition=(
                "T(x, y) = mean(neighborhood(x, y)) - C; "
                "dst(x, y) = max_value if src(x, y) > T(x, y) else 0"
            ),
            advantages=(
                "Robust to uneven illumination across the image.",
                "Fully automatic per pixel, just locally rather than globally.",
            ),
            limitations=(
                "Two parameters to tune instead of Otsu's zero.",
                "Can be noisier than a global threshold in flat regions.",
            ),
            typical_applications=(
                "Segmenting a leaf photographed under a natural light gradient.",
            ),
            complexity="O(H x W) via box-filtered local means",
            opencv_reference=(
                "cv2.adaptiveThreshold(src, maxValue, "
                "cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)"
            ),
        )
