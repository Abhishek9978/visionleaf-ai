"""Truncate Threshold.

Purpose:
    Cap intensities at a ceiling without zeroing anything out —
    useful for reducing the influence of very bright highlights (e.g.
    glare) while keeping the rest of the image's detail intact, rather
    than producing a strict binary mask.

Theory:
    Unlike Binary Threshold, this doesn't produce a two-level image:
    values above the cutoff are clamped down to the cutoff itself,
    values at or below are left unchanged. `max_value` has no effect
    for this mode (OpenCV still requires the argument, but ignores it).

Working Principle:
    1. Convert the active image to grayscale if it isn't already.
    2. Compare every pixel to `threshold`.
    3. Pixels greater than `threshold` are set to `threshold`; all
       others are left unchanged.

Math intuition:
    dst(x, y) = threshold if src(x, y) > threshold else src(x, y)

Advantages:
    - Preserves gradations below the cutoff, unlike a strict binary mask.
    - Simple way to suppress glare/overexposure before further processing.

Limitations:
    - Doesn't itself produce a segmentation mask usable by contour
      finding — typically a preprocessing step before Binary Threshold
      or Otsu, not the final segmentation stage.

Typical Applications:
    - Suppressing specular highlights before segmenting.

Complexity:
    O(H x W).

OpenCV reference:
    cv2.threshold(src, thresh, maxval, cv2.THRESH_TRUNC)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    TruncateThresholdParams, TruncateThreshold
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
class TruncateThresholdParams:
    """Parameters for `TruncateThreshold`.

    Attributes:
        threshold: Ceiling intensity, [0, 255]. Pixels above this are
            clamped down to it.
    """

    threshold: int = 127


@register_algorithm("truncate_threshold")
class TruncateThreshold(PipelineStage):
    """Clamp pixel values above a ceiling, leaving the rest unchanged."""

    stage_key = "segmentation"

    def __init__(self, params: TruncateThresholdParams | None = None, **kwargs) -> None:
        self.params = params or TruncateThresholdParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        ensure_threshold_in_range(self.params.threshold)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        grayscale = ensure_grayscale(session)

        _, binary = cv2.threshold(grayscale, self.params.threshold, 255, cv2.THRESH_TRUNC)

        session.clear_mask_results()
        session.binary_image = binary
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key, self.get_name(), details=f"threshold={self.params.threshold}"
        )
        logger.info(
            "%s | threshold=%d | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.threshold,
            grayscale.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Truncate Threshold"

    def get_description(self) -> str:
        return "Clamps pixel values above a ceiling to that ceiling, leaving darker pixels unchanged."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Cap intensities at a ceiling without zeroing anything "
                "out — useful for suppressing highlights while keeping "
                "detail intact."
            ),
            theory=(
                "Unlike Binary Threshold, this doesn't produce a two-level "
                "image: values above the cutoff are clamped to the cutoff "
                "itself; values at or below are left unchanged."
            ),
            working_principle=(
                "1) Convert to grayscale if needed. 2) Compare every pixel "
                "to the threshold. 3) Clamp values above it; leave the rest."
            ),
            math_intuition="dst(x, y) = threshold if src(x, y) > threshold else src(x, y)",
            advantages=(
                "Preserves gradations below the cutoff, unlike a strict binary mask.",
                "Simple way to suppress glare/overexposure before further processing.",
            ),
            limitations=(
                "Doesn't itself produce a mask usable by contour finding — "
                "typically a preprocessing step, not the final segmentation stage.",
            ),
            typical_applications=("Suppressing specular highlights before segmenting.",),
            complexity="O(H x W)",
            opencv_reference="cv2.threshold(src, thresh, maxval, cv2.THRESH_TRUNC)",
        )
