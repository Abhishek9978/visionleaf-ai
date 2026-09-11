"""Black Hat.

Purpose:
    Isolate small dark details/gaps that are smaller than the
    structuring element and darker than their surroundings — the
    complement of Top Hat.

Theory:
    Closing fills small dark gaps entirely. Subtracting the original
    from the closed result leaves exactly those small dark features
    that Closing filled in — nothing else.

Working Principle:
    1. Require an existing mask.
    2. Build a `kernel_size` x `kernel_size` structuring element.
    3. Compute `cv2.MORPH_BLACKHAT` = closing(src) - src.

Math intuition:
    black_hat(src) = closing(src, kernel) - src

Advantages:
    - Isolates small dark features/gaps regardless of a varying background.
    - Directly complementary to Top Hat.

Limitations:
    - Only meaningful relative to the chosen kernel size.
    - Sensitive to noise in the input mask, like other morphological ops.

Typical Applications:
    - Highlighting small dark gaps or veins within an otherwise-solid
      foreground region.

Complexity:
    O(H x W x k^2) (one Closing plus one subtraction).

OpenCV reference:
    cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    BlackHatParams, BlackHat
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
class BlackHatParams:
    """Parameters for `BlackHat`.

    Attributes:
        kernel_size: Side length of the structuring element; must be
            a positive odd integer.
        iterations: Number of times to apply the underlying operation;
            must be a positive integer.
    """

    kernel_size: int = 9
    iterations: int = 1


@register_algorithm("black_hat")
class BlackHat(PipelineStage):
    """Isolate small dark features smaller than the structuring element."""

    stage_key = "morphology"

    def __init__(self, params: BlackHatParams | None = None, **kwargs) -> None:
        self.params = params or BlackHatParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        validate_morphology_params(session, self.params.kernel_size, self.params.iterations)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        mask = require_mask(session)
        kernel = build_kernel(self.params.kernel_size)

        result = cv2.morphologyEx(
            mask, cv2.MORPH_BLACKHAT, kernel, iterations=self.params.iterations
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
        return "Black Hat"

    def get_description(self) -> str:
        return "Isolates small dark details smaller than the kernel: its Closing minus the original."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Isolate small dark details/gaps that are smaller than "
                "the structuring element and darker than their "
                "surroundings — the complement of Top Hat."
            ),
            theory=(
                "Closing fills small dark gaps entirely. Subtracting the "
                "original from the closed result leaves exactly those "
                "small dark features that Closing filled in."
            ),
            working_principle=(
                "1) Require an existing mask. 2) Build a structuring "
                "element. 3) Compute closing(src) - src."
            ),
            math_intuition="black_hat(src) = closing(src, kernel) - src",
            advantages=(
                "Isolates small dark features/gaps regardless of a varying background.",
                "Directly complementary to Top Hat.",
            ),
            limitations=(
                "Only meaningful relative to the chosen kernel size.",
                "Sensitive to noise in the input mask, like other morphological ops.",
            ),
            typical_applications=(
                "Highlighting small dark gaps or veins within a foreground region.",
            ),
            complexity="O(H x W x k^2)",
            opencv_reference="cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)",
        )
