"""Morphological Gradient.

Purpose:
    Extract the outline (boundary) of the foreground region — the
    difference between a dilated and eroded version of the mask
    isolates just the edge pixels.

Theory:
    Dilation grows the region outward; Erosion shrinks it inward.
    Subtracting the eroded version from the dilated version leaves
    only the ring of pixels that changed between the two — i.e. the
    boundary of the shape.

Working Principle:
    1. Require an existing mask.
    2. Build a `kernel_size` x `kernel_size` structuring element.
    3. Compute `cv2.MORPH_GRADIENT` = dilate(src) - erode(src).

Math intuition:
    gradient(src) = dilate(src, kernel) - erode(src, kernel)

Advantages:
    - Directly isolates object boundaries without a separate edge
      detector.
    - Thickness of the outline is directly controlled by `kernel_size`.

Limitations:
    - Only as accurate as the input mask — noise in the mask produces
      noise in the outline.
    - Not a substitute for `processing.roi.find_contours`, which
      produces actual traceable boundary coordinates rather than a
      pixel mask.

Typical Applications:
    - Visualizing or highlighting the exact boundary of a segmented
      region.

Complexity:
    O(H x W x k^2) (one erosion plus one dilation).

OpenCV reference:
    cv2.morphologyEx(src, cv2.MORPH_GRADIENT, kernel, iterations=iterations)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    MorphologicalGradientParams, MorphologicalGradient
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
class MorphologicalGradientParams:
    """Parameters for `MorphologicalGradient`.

    Attributes:
        kernel_size: Side length of the structuring element; must be
            a positive odd integer.
        iterations: Number of times to apply the operation; must be a
            positive integer.
    """

    kernel_size: int = 5
    iterations: int = 1


@register_algorithm("morphological_gradient")
class MorphologicalGradient(PipelineStage):
    """Extract the boundary outline of a mask's foreground region."""

    stage_key = "morphology"

    def __init__(self, params: MorphologicalGradientParams | None = None, **kwargs) -> None:
        self.params = params or MorphologicalGradientParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        validate_morphology_params(session, self.params.kernel_size, self.params.iterations)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        mask = require_mask(session)
        kernel = build_kernel(self.params.kernel_size)

        result = cv2.morphologyEx(
            mask, cv2.MORPH_GRADIENT, kernel, iterations=self.params.iterations
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
        return "Morphological Gradient"

    def get_description(self) -> str:
        return "Extracts the boundary outline of the foreground region: dilation minus erosion."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Extract the outline (boundary) of the foreground region "
                "by isolating the pixels that differ between a dilated "
                "and an eroded version of the mask."
            ),
            theory=(
                "Dilation grows the region outward; Erosion shrinks it "
                "inward. Subtracting the eroded version from the dilated "
                "version leaves only the ring of pixels that changed."
            ),
            working_principle=(
                "1) Require an existing mask. 2) Build a structuring "
                "element. 3) Compute dilate(src) - erode(src)."
            ),
            math_intuition="gradient(src) = dilate(src, kernel) - erode(src, kernel)",
            advantages=(
                "Directly isolates object boundaries without a separate edge detector.",
                "Outline thickness is directly controlled by kernel_size.",
            ),
            limitations=(
                "Only as accurate as the input mask — noise in the mask "
                "produces noise in the outline.",
                "Produces a pixel mask, not traceable coordinates — see "
                "Find Contours for that.",
            ),
            typical_applications=(
                "Visualizing or highlighting the exact boundary of a segmented region.",
            ),
            complexity="O(H x W x k^2)",
            opencv_reference="cv2.morphologyEx(src, cv2.MORPH_GRADIENT, kernel, iterations=iterations)",
        )
