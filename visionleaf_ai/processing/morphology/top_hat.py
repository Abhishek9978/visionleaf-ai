"""Top Hat.

Purpose:
    Isolate small bright details/specks that are smaller than the
    structuring element and brighter than their surroundings — the
    difference between the original mask and its Opening.

Theory:
    Opening removes small bright features (anything smaller than the
    kernel gets erased). Subtracting the opened result from the
    original leaves exactly those small bright features that Opening
    removed — nothing else.

Working Principle:
    1. Require an existing mask.
    2. Build a `kernel_size` x `kernel_size` structuring element.
    3. Compute `cv2.MORPH_TOPHAT` = src - opening(src).

Math intuition:
    top_hat(src) = src - opening(src, kernel)

Advantages:
    - Isolates small bright features regardless of a varying background.
    - Directly complementary to Black Hat (below), which does the same
      for dark features.

Limitations:
    - Only meaningful relative to the chosen kernel size — features
      larger than the kernel are considered "background," not detail.
    - Like other morphological ops, sensitive to noise in the input mask.

Typical Applications:
    - Highlighting small bright lesions/spots against a mask's main body.

Complexity:
    O(H x W x k^2) (one Opening plus one subtraction).

OpenCV reference:
    cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel, iterations=iterations)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    TopHatParams, TopHat
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
class TopHatParams:
    """Parameters for `TopHat`.

    Attributes:
        kernel_size: Side length of the structuring element; must be
            a positive odd integer.
        iterations: Number of times to apply the underlying operation;
            must be a positive integer.
    """

    kernel_size: int = 9
    iterations: int = 1


@register_algorithm("top_hat")
class TopHat(PipelineStage):
    """Isolate small bright features smaller than the structuring element."""

    stage_key = "morphology"

    def __init__(self, params: TopHatParams | None = None, **kwargs) -> None:
        self.params = params or TopHatParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        validate_morphology_params(session, self.params.kernel_size, self.params.iterations)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        mask = require_mask(session)
        kernel = build_kernel(self.params.kernel_size)

        result = cv2.morphologyEx(
            mask, cv2.MORPH_TOPHAT, kernel, iterations=self.params.iterations
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
        return "Top Hat"

    def get_description(self) -> str:
        return "Isolates small bright details smaller than the kernel: original minus its Opening."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Isolate small bright details/specks that are smaller "
                "than the structuring element and brighter than their "
                "surroundings."
            ),
            theory=(
                "Opening removes small bright features entirely. "
                "Subtracting the opened result from the original leaves "
                "exactly those small bright features that Opening removed."
            ),
            working_principle=(
                "1) Require an existing mask. 2) Build a structuring "
                "element. 3) Compute src - opening(src)."
            ),
            math_intuition="top_hat(src) = src - opening(src, kernel)",
            advantages=(
                "Isolates small bright features regardless of a varying background.",
                "Directly complementary to Black Hat.",
            ),
            limitations=(
                "Only meaningful relative to the chosen kernel size.",
                "Sensitive to noise in the input mask, like other morphological ops.",
            ),
            typical_applications=(
                "Highlighting small bright lesions/spots against a mask's main body.",
            ),
            complexity="O(H x W x k^2)",
            opencv_reference="cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel, iterations=iterations)",
        )
