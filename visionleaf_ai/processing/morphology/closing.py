"""Closing.

Purpose:
    Fill small holes and gaps *inside* the foreground region without
    shrinking it — the complement of Opening, used when the mask has
    small dark speckles trapped inside an otherwise-solid object.

Theory:
    Dilation followed by Erosion. The initial dilation fills small
    background gaps inside the foreground and bridges narrow breaks;
    the following erosion shrinks the result back down to roughly the
    original outer boundary.

Working Principle:
    1. Require an existing mask.
    2. Build a `kernel_size` x `kernel_size` structuring element.
    3. Apply `cv2.MORPH_CLOSE` (dilate then erode) for `iterations`
       passes.

Math intuition:
    closing(src) = erode(dilate(src, kernel), kernel)

Advantages:
    - Fills small holes and bridges narrow gaps in the foreground.
    - Preserves the region's overall outer size, unlike plain Dilation.

Limitations:
    - Doesn't remove small isolated noise specks *outside* the
      foreground (see Opening for that).
    - Large holes may not be fully filled if they exceed the kernel size.

Typical Applications:
    - Filling small gaps in a leaf's vein pattern that fragment its
      otherwise-solid silhouette in a thresholded mask.

Complexity:
    O(H x W x k^2) per iteration (dilation + erosion combined).

OpenCV reference:
    cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations=iterations)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    ClosingParams, Closing
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
class ClosingParams:
    """Parameters for `Closing`.

    Attributes:
        kernel_size: Side length of the structuring element; must be
            a positive odd integer.
        iterations: Number of times to apply the operation; must be a
            positive integer.
    """

    kernel_size: int = 5
    iterations: int = 1


@register_algorithm("closing")
class Closing(PipelineStage):
    """Dilation followed by Erosion — fills holes, preserves overall size."""

    stage_key = "morphology"

    def __init__(self, params: ClosingParams | None = None, **kwargs) -> None:
        self.params = params or ClosingParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        validate_morphology_params(session, self.params.kernel_size, self.params.iterations)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        mask = require_mask(session)
        kernel = build_kernel(self.params.kernel_size)

        result = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=self.params.iterations
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
        return "Closing"

    def get_description(self) -> str:
        return "Dilation followed by Erosion — fills small holes/gaps without shrinking the region."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Fill small holes and gaps inside the foreground region "
                "without shrinking it — the complement of Opening."
            ),
            theory=(
                "Dilation followed by Erosion. The initial dilation fills "
                "small background gaps inside the foreground and bridges "
                "narrow breaks; the erosion shrinks the result back down "
                "to roughly the original outer boundary."
            ),
            working_principle=(
                "1) Require an existing mask. 2) Build a structuring "
                "element. 3) Apply MORPH_CLOSE (dilate then erode) for "
                "the configured iterations."
            ),
            math_intuition="closing(src) = erode(dilate(src, kernel), kernel)",
            advantages=(
                "Fills small holes and bridges narrow gaps in the foreground.",
                "Preserves the region's overall outer size, unlike plain Dilation.",
            ),
            limitations=(
                "Doesn't remove isolated noise specks outside the foreground — see Opening.",
                "Large holes may not be fully filled if they exceed the kernel size.",
            ),
            typical_applications=(
                "Filling small gaps that fragment an otherwise-solid leaf silhouette.",
            ),
            complexity="O(H x W x k^2) per iteration",
            opencv_reference="cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations=iterations)",
        )
