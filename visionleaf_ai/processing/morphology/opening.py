"""Opening.

Purpose:
    Remove small noise specks without shrinking the overall shape of
    the main foreground region the way plain Erosion would — the
    standard "denoise a mask" operation.

Theory:
    Erosion followed by Dilation. The initial erosion eliminates
    anything smaller than the kernel; the following dilation grows the
    surviving regions back to roughly their original size — but
    anything erased entirely by the erosion never comes back.

Working Principle:
    1. Require an existing mask.
    2. Build a `kernel_size` x `kernel_size` structuring element.
    3. Apply `cv2.MORPH_OPEN` (erode then dilate) for `iterations` passes.

Math intuition:
    opening(src) = dilate(erode(src, kernel), kernel)

Advantages:
    - Removes small noise specks while roughly preserving the main
      region's size — unlike Erosion alone.
    - A very common, safe default cleanup step after thresholding.

Limitations:
    - Doesn't help with small holes *inside* the foreground region
      (see Closing for that).
    - Still erases anything smaller than the kernel entirely.

Typical Applications:
    - Cleaning speckle noise out of a thresholded leaf mask before
      contour detection.

Complexity:
    O(H x W x k^2) per iteration (erosion + dilation combined).

OpenCV reference:
    cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=iterations)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    OpeningParams, Opening
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.morphology._common import (
    build_kernel,
    require_mask,
    validate_morphology_params,
)
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage

logger = get_logger(__name__)


@dataclass(frozen=True)
class OpeningParams:
    """Parameters for `Opening`.

    Attributes:
        kernel_size: Side length of the structuring element; must be
            a positive odd integer.
        iterations: Number of times to apply the operation; must be a
            positive integer.
    """

    kernel_size: int = 5
    iterations: int = 1


@register_algorithm("opening")
class Opening(PipelineStage):
    """Erosion followed by Dilation — removes noise, preserves overall size."""

    stage_key = "morphology"

    def __init__(self, params: OpeningParams | None = None, **kwargs) -> None:
        self.params = params or OpeningParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        validate_morphology_params(session, self.params.kernel_size, self.params.iterations)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        mask = require_mask(session)
        kernel = build_kernel(self.params.kernel_size)

        result = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=self.params.iterations
        )

        session.clear_contour_results()
        session.segmentation_mask = result
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"kernel_size={self.params.kernel_size}, iterations={self.params.iterations}",
        )
        logger.info(
            "%s | kernel_size=%d | iterations=%d | mask=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.kernel_size,
            self.params.iterations,
            mask.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Opening"

    def get_description(self) -> str:
        return "Erosion followed by Dilation — removes small noise while preserving the region's overall size."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Remove small noise specks without shrinking the overall "
                "shape of the main foreground region."
            ),
            theory=(
                "Erosion followed by Dilation. The initial erosion "
                "eliminates anything smaller than the kernel; the "
                "following dilation grows survivors back to roughly "
                "their original size."
            ),
            working_principle=(
                "1) Require an existing mask. 2) Build a structuring "
                "element. 3) Apply MORPH_OPEN (erode then dilate) for "
                "the configured iterations."
            ),
            math_intuition="opening(src) = dilate(erode(src, kernel), kernel)",
            advantages=(
                "Removes noise while roughly preserving the main region's size.",
                "A very common, safe default cleanup step after thresholding.",
            ),
            limitations=(
                "Doesn't help with small holes inside the foreground — see Closing.",
                "Still erases anything smaller than the kernel entirely.",
            ),
            typical_applications=(
                "Cleaning speckle noise out of a thresholded leaf mask.",
            ),
            complexity="O(H x W x k^2) per iteration",
            opencv_reference="cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=iterations)",
        )
