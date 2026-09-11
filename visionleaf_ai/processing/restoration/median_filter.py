"""Median Filter.

Purpose:
    Remove impulse ("salt-and-pepper") noise while preserving edges
    better than a Gaussian blur would — the standard restoration
    algorithm when noise appears as isolated extreme-value pixels
    rather than smooth grain.

Theory:
    For each pixel, replaces its value with the median (not the mean)
    of the values in its neighborhood. Because a single extreme
    outlier pixel can't dominate a median the way it dominates a mean,
    isolated noisy pixels get replaced by a value from their
    surroundings, while edges — where a genuine, consistent step in
    intensity exists across many neighboring pixels — are largely
    preserved.

Math intuition:
    output(x, y) = median{ input(i, j) : (i, j) in neighborhood(x, y) }
    Unlike Gaussian blur's weighted average, the median is robust to
    outliers: one pixel that's wildly different from its neighbors is
    simply outvoted, not blended in.

Advantages:
    - Excellent at removing salt-and-pepper noise specifically.
    - Better edge preservation than linear (mean-based) filters.

Limitations:
    - Less effective against Gaussian (smooth, continuous) noise than
      Gaussian blur or bilateral filtering.
    - Can erode fine details smaller than the kernel (thin lines, small
      spots) since they can be outvoted by surrounding pixels.

Complexity:
    O(H x W x k log k) per naive neighborhood sort for kernel size k
    (OpenCV uses a faster histogram-based algorithm for 8-bit images in
    practice).

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    MedianFilterParams, MedianFilter
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage
from visionleaf_ai.utils.validators import ensure_kernel_fits_image, ensure_odd_positive_int

logger = get_logger(__name__)


@dataclass(frozen=True)
class MedianFilterParams:
    """Parameters for `MedianFilter`.

    Attributes:
        kernel_size: Side length of the square neighborhood; must be a
            positive odd integer greater than 1.
    """

    kernel_size: int = 5


@register_algorithm("median_filter")
class MedianFilter(PipelineStage):
    """Replace each pixel with the median of its neighborhood."""

    stage_key = "restoration"

    def __init__(self, params: MedianFilterParams | None = None, **kwargs) -> None:
        self.params = params or MedianFilterParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        ensure_odd_positive_int(self.params.kernel_size, "kernel_size")
        if self.params.kernel_size < 3:
            raise ValidationError(
                "kernel_size must be at least 3 for a median filter to be meaningful",
                details=f"got {self.params.kernel_size!r}",
            )
        ensure_kernel_fits_image(self.params.kernel_size, session.require_active_image().shape)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        image = session.require_active_image()

        result = cv2.medianBlur(image, self.params.kernel_size)

        session.current_image = result
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"kernel_size={self.params.kernel_size}",
        )
        logger.info(
            "%s | kernel_size=%d | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.kernel_size,
            image.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Median Filter"

    def get_description(self) -> str:
        return "Replaces each pixel with the median value of its neighborhood, removing impulse noise while preserving edges."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Remove impulse (salt-and-pepper) noise while preserving "
                "edges better than a Gaussian blur."
            ),
            theory=(
                "Replaces each pixel with the median of its neighborhood. "
                "A single extreme outlier can't dominate a median, so "
                "isolated noisy pixels are replaced while genuine edges "
                "are largely preserved."
            ),
            math_intuition=(
                "output(x, y) = median{ input(i, j) : (i, j) in "
                "neighborhood(x, y) }"
            ),
            advantages=(
                "Excellent at removing salt-and-pepper noise specifically.",
                "Better edge preservation than linear (mean-based) filters.",
            ),
            limitations=(
                "Less effective against smooth Gaussian noise.",
                "Can erode fine details smaller than the kernel.",
            ),
            complexity="O(H x W x k log k) naive; faster in practice for 8-bit images",
        )
