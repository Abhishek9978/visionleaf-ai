"""Dilation.

Purpose:
    Grow the foreground region of a binary mask — the complement of
    Erosion, used to close small gaps/holes and reconnect fragmented
    regions before contour detection.

Theory:
    Slides a structuring element over the mask; a pixel becomes
    foreground if *any* part of the kernel overlaps foreground at that
    position. This expands regions outward and fills small background
    gaps inside or between foreground areas.

Working Principle:
    1. Require an existing mask.
    2. Build a `kernel_size` x `kernel_size` structuring element.
    3. Apply dilation for `iterations` passes.

Math intuition:
    dst(x, y) = max over (i, j) in kernel of src(x + i, y + j)
    (For a binary mask: the output is foreground if any pixel under
    the kernel is foreground.)

Advantages:
    - Reconnects fragmented regions and fills small holes.
    - Simple, fast, and the basis for Opening/Closing below.

Limitations:
    - Also grows genuine noise specks, not just the real foreground.
    - Repeated dilation can merge distinct nearby objects together.

Typical Applications:
    - Closing small gaps in a leaf's outline before finding its contour.

Complexity:
    O(H x W x k^2) for kernel size k, per iteration.

OpenCV reference:
    cv2.dilate(src, kernel, iterations=iterations)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    DilationParams, Dilation
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
class DilationParams:
    """Parameters for `Dilation`.

    Attributes:
        kernel_size: Side length of the structuring element; must be
            a positive odd integer.
        iterations: Number of times to apply dilation; must be a
            positive integer.
    """

    kernel_size: int = 5
    iterations: int = 1


@register_algorithm("dilation")
class Dilation(PipelineStage):
    """Grow the foreground region of a binary mask."""

    stage_key = "morphology"

    def __init__(self, params: DilationParams | None = None, **kwargs) -> None:
        self.params = params or DilationParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        validate_morphology_params(session, self.params.kernel_size, self.params.iterations)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        mask = require_mask(session)
        kernel = build_kernel(self.params.kernel_size)

        result = cv2.dilate(mask, kernel, iterations=self.params.iterations)

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
        return "Dilation"

    def get_description(self) -> str:
        return "Grows the foreground region of a mask, closing small gaps and holes."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Grow the foreground region of a binary mask — the "
                "complement of Erosion, closing gaps and reconnecting "
                "fragmented regions."
            ),
            theory=(
                "Slides a structuring element over the mask; a pixel "
                "becomes foreground if any part of the kernel overlaps "
                "foreground at that position."
            ),
            working_principle=(
                "1) Require an existing mask. 2) Build a square "
                "structuring element. 3) Apply dilation for the "
                "configured number of iterations."
            ),
            math_intuition=(
                "dst(x, y) = max over (i, j) in kernel of src(x+i, y+j)"
            ),
            advantages=(
                "Reconnects fragmented regions and fills small holes.",
                "Simple, fast, and the basis for Opening/Closing.",
            ),
            limitations=(
                "Also grows genuine noise specks, not just real foreground.",
                "Repeated dilation can merge distinct nearby objects together.",
            ),
            typical_applications=(
                "Closing small gaps in a leaf's outline before finding its contour.",
            ),
            complexity="O(H x W x k^2) per iteration",
            opencv_reference="cv2.dilate(src, kernel, iterations=iterations)",
        )
