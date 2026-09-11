"""Binary Inverse Threshold.

Purpose:
    The mirror image of Binary Threshold: useful when the object of
    interest is *darker* than the background rather than brighter
    (e.g. a dark leaf spot on a light background).

Theory:
    Identical to Binary Threshold except the assignment is flipped:
    pixels above the cutoff become 0, pixels at or below become
    `max_value`.

Working Principle:
    1. Convert the active image to grayscale if it isn't already.
    2. Compare every pixel to `threshold`.
    3. Pixels greater than `threshold` become 0; all others become
       `max_value`.

Math intuition:
    dst(x, y) = 0 if src(x, y) > threshold else max_value

Advantages:
    - Same speed/simplicity as Binary Threshold.
    - Correct choice whenever the foreground is the darker region.

Limitations:
    - Same global-cutoff limitation as Binary Threshold.

Typical Applications:
    - Segmenting dark lesions or spots against a lighter leaf surface.

Complexity:
    O(H x W).

OpenCV reference:
    cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY_INV)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    BinaryInverseThresholdParams, BinaryInverseThreshold
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage
from visionleaf_ai.processing.segmentation._common import ensure_grayscale
from visionleaf_ai.utils.validators import ensure_threshold_in_range

logger = get_logger(__name__)


@dataclass(frozen=True)
class BinaryInverseThresholdParams:
    """Parameters for `BinaryInverseThreshold`.

    Attributes:
        threshold: Cutoff intensity, [0, 255].
        max_value: Output value assigned to pixels at or below `threshold`.
    """

    threshold: int = 127
    max_value: int = 255


@register_algorithm("binary_inverse_threshold")
class BinaryInverseThreshold(PipelineStage):
    """Split an image into two levels, inverted relative to Binary Threshold."""

    stage_key = "segmentation"

    def __init__(self, params: BinaryInverseThresholdParams | None = None, **kwargs) -> None:
        self.params = params or BinaryInverseThresholdParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        ensure_threshold_in_range(self.params.threshold)
        ensure_threshold_in_range(self.params.max_value, "max_value")

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        grayscale = ensure_grayscale(session)

        _, binary = cv2.threshold(
            grayscale, self.params.threshold, self.params.max_value, cv2.THRESH_BINARY_INV
        )

        session.clear_mask_results()
        session.binary_image = binary
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"threshold={self.params.threshold}, max_value={self.params.max_value}",
        )
        logger.info(
            "%s | threshold=%d | max_value=%d | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.threshold,
            self.params.max_value,
            grayscale.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Binary Inverse Threshold"

    def get_description(self) -> str:
        return "Like Binary Threshold, but inverted: above the cutoff becomes black, at-or-below becomes white."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "The mirror image of Binary Threshold, for objects darker "
                "than their background."
            ),
            theory=(
                "Identical to Binary Threshold except the assignment is "
                "flipped: pixels above the cutoff become 0, pixels at or "
                "below become max_value."
            ),
            working_principle=(
                "1) Convert to grayscale if needed. 2) Compare every pixel "
                "to the threshold. 3) Assign 0 or max_value, inverted "
                "relative to Binary Threshold."
            ),
            math_intuition="dst(x, y) = 0 if src(x, y) > threshold else max_value",
            advantages=(
                "Same speed/simplicity as Binary Threshold.",
                "Correct choice whenever the foreground is the darker region.",
            ),
            limitations=("Same global-cutoff limitation as Binary Threshold.",),
            typical_applications=(
                "Segmenting dark lesions or spots against a lighter leaf surface.",
            ),
            complexity="O(H x W)",
            opencv_reference="cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY_INV)",
        )
