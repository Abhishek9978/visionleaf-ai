"""Erosion.

Purpose:
    Shrink the foreground region of a binary mask — the fundamental
    morphological operation for removing small noise specks and
    thinning object boundaries before contour detection.

Theory:
    Slides a structuring element (kernel) over the mask; a foreground
    pixel survives only if the *entire* kernel fits within the
    foreground at that position. Any foreground pixel near a
    background pixel gets eroded away, shrinking the region and
    eliminating anything smaller than the kernel entirely.

Working Principle:
    1. Require an existing mask (`session.active_mask`).
    2. Build a `kernel_size` x `kernel_size` structuring element.
    3. Apply erosion for `iterations` passes.

Math intuition:
    dst(x, y) = min over (i, j) in kernel of src(x + i, y + j)
    (For a binary mask, "min" means the output is foreground only if
    every pixel under the kernel is foreground.)

Advantages:
    - Removes small noise specks smaller than the kernel entirely.
    - Simple, fast, and the basis for Opening/Closing/Gradient below.

Limitations:
    - Also shrinks genuine foreground regions, not just noise — can
      erase small legitimate features if the kernel is too large.
    - Repeated erosion (`iterations`) compounds this shrinkage.

Typical Applications:
    - Removing small speckle noise from a thresholded mask before
      finding contours.

Complexity:
    O(H x W x k^2) for kernel size k, per iteration.

OpenCV reference:
    cv2.erode(src, kernel, iterations=iterations)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    ErosionParams, Erosion
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
class ErosionParams:
    """Parameters for `Erosion`.

    Attributes:
        kernel_size: Side length of the structuring element; must be
            a positive odd integer.
        iterations: Number of times to apply erosion; must be a
            positive integer.
    """

    kernel_size: int = 5
    iterations: int = 1


@register_algorithm("erosion")
class Erosion(PipelineStage):
    """Shrink the foreground region of a binary mask."""

    stage_key = "morphology"

    def __init__(self, params: ErosionParams | None = None, **kwargs) -> None:
        self.params = params or ErosionParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        validate_morphology_params(session, self.params.kernel_size, self.params.iterations)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        mask = require_mask(session)
        kernel = build_kernel(self.params.kernel_size)

        result = cv2.erode(mask, kernel, iterations=self.params.iterations)

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
        return "Erosion"

    def get_description(self) -> str:
        return "Shrinks the foreground region of a mask, removing small noise specks."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Shrink the foreground region of a binary mask — the "
                "fundamental operation for removing small noise specks."
            ),
            theory=(
                "Slides a structuring element over the mask; a "
                "foreground pixel survives only if the entire kernel "
                "fits within the foreground at that position."
            ),
            working_principle=(
                "1) Require an existing mask. 2) Build a square "
                "structuring element. 3) Apply erosion for the "
                "configured number of iterations."
            ),
            math_intuition=(
                "dst(x, y) = min over (i, j) in kernel of src(x+i, y+j)"
            ),
            advantages=(
                "Removes small noise specks smaller than the kernel entirely.",
                "Simple, fast, and the basis for Opening/Closing/Gradient.",
            ),
            limitations=(
                "Also shrinks genuine foreground regions, not just noise.",
                "Repeated erosion compounds the shrinkage.",
            ),
            typical_applications=(
                "Removing small speckle noise before finding contours.",
            ),
            complexity="O(H x W x k^2) per iteration",
            opencv_reference="cv2.erode(src, kernel, iterations=iterations)",
        )
