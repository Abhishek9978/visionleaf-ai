"""Brightness Adjustment.

Purpose:
    Uniformly raise or lower every pixel's intensity — the simplest
    possible enhancement, and a good baseline for comparing more
    sophisticated techniques against.

Theory:
    Brightness adjustment adds a constant offset to every pixel value.
    A positive offset makes the image lighter, a negative offset makes
    it darker. Because the operation is identical everywhere in the
    image, it changes overall exposure without changing contrast or
    relative differences between regions.

Math intuition:
    output(x, y) = clip(input(x, y) + delta, 0, 255)
    Clipping (saturation) is essential: without it, values would wrap
    around (e.g. 250 + 20 = 270, which isn't a valid pixel value).

Advantages:
    - Extremely fast (a single elementwise addition).
    - Fully reversible if you record `delta` (subtract it back).
    - Simple enough to reason about pixel-by-pixel.

Limitations:
    - Uniform: doesn't adapt to regions that are already well-exposed
      vs. regions that are too dark/bright.
    - Large deltas clip highlights or shadows, permanently losing detail.

Complexity:
    O(H x W) — one pass over every pixel.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    BrightnessParams, BrightnessAdjustment
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

logger = get_logger(__name__)


@dataclass(frozen=True)
class BrightnessParams:
    """Parameters for `BrightnessAdjustment`.

    Attributes:
        delta: Amount to add to every pixel, in the range [-255, 255].
            Positive brightens, negative darkens.
    """

    delta: int = 30


@register_algorithm("brightness_adjustment")
class BrightnessAdjustment(PipelineStage):
    """Add a constant offset to every pixel value."""

    stage_key = "enhancement"

    def __init__(self, params: BrightnessParams | None = None, **kwargs) -> None:
        self.params = params or BrightnessParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        if not isinstance(self.params.delta, int) or not (-255 <= self.params.delta <= 255):
            raise ValidationError(
                "delta must be an integer in [-255, 255]",
                details=f"got {self.params.delta!r}",
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        image = session.require_active_image()

        result = cv2.convertScaleAbs(image, alpha=1.0, beta=self.params.delta)

        session.current_image = result
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"delta={self.params.delta}",
        )
        logger.info(
            "%s | delta=%d | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.delta,
            image.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Brightness Adjustment"

    def get_description(self) -> str:
        return "Adds a constant offset to every pixel to lighten or darken the image."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Uniformly raise or lower every pixel's intensity — the "
                "simplest possible enhancement."
            ),
            theory=(
                "Adds a constant offset to every pixel value. A positive "
                "offset lightens the image, a negative offset darkens it, "
                "without changing contrast or relative differences between "
                "regions."
            ),
            math_intuition=(
                "output(x, y) = clip(input(x, y) + delta, 0, 255). Clipping "
                "prevents values from wrapping past 0 or 255."
            ),
            advantages=(
                "Extremely fast — a single elementwise addition.",
                "Fully reversible if delta is recorded.",
            ),
            limitations=(
                "Uniform: doesn't adapt to already well-exposed regions.",
                "Large deltas clip highlights or shadows, losing detail.",
            ),
            complexity="O(H x W)",
        )
