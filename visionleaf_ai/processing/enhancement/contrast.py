"""Contrast Adjustment.

Purpose:
    Stretch or compress the range of pixel intensities around their
    midpoint, making dark/light regions more (or less) distinct from
    each other.

Theory:
    Contrast adjustment scales pixel values by a multiplicative factor.
    A factor greater than 1 spreads values further apart (more
    contrast); a factor between 0 and 1 pulls them closer together
    (less contrast); a factor of exactly 1 leaves the image unchanged.

Math intuition:
    output(x, y) = clip(input(x, y) * alpha, 0, 255)
    Unlike brightness (an additive shift), contrast is multiplicative,
    so it affects bright and dark pixels differently — a pixel at 200
    changes more in absolute terms than a pixel at 50 for the same alpha.

Advantages:
    - Fast, single-pass elementwise operation.
    - Effective for images that look "flat" or washed out.

Limitations:
    - A single global alpha can't fix an image where some regions are
      under- and others over-exposed (that needs adaptive techniques
      like CLAHE).
    - Large alpha values clip both highlights and shadows simultaneously.

Complexity:
    O(H x W).

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    ContrastParams, ContrastAdjustment
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
class ContrastParams:
    """Parameters for `ContrastAdjustment`.

    Attributes:
        alpha: Multiplicative contrast factor. 1.0 = unchanged,
            > 1.0 = more contrast, between 0 and 1 = less contrast.
            Must be strictly positive.
    """

    alpha: float = 1.5


@register_algorithm("contrast_adjustment")
class ContrastAdjustment(PipelineStage):
    """Scale every pixel value by a multiplicative contrast factor."""

    stage_key = "enhancement"

    def __init__(self, params: ContrastParams | None = None, **kwargs) -> None:
        self.params = params or ContrastParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        if self.params.alpha <= 0:
            raise ValidationError(
                "alpha must be strictly positive", details=f"got {self.params.alpha!r}"
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        image = session.require_active_image()

        result = cv2.convertScaleAbs(image, alpha=self.params.alpha, beta=0)

        session.current_image = result
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"alpha={self.params.alpha}",
        )
        logger.info(
            "%s | alpha=%.3f | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.alpha,
            image.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Contrast Adjustment"

    def get_description(self) -> str:
        return "Scales pixel values by a multiplicative factor to increase or decrease contrast."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Stretch or compress the range of pixel intensities to make "
                "regions more or less distinct from each other."
            ),
            theory=(
                "Scales every pixel value by a multiplicative factor. "
                "Values above 1.0 spread intensities further apart; values "
                "between 0 and 1 pull them closer together."
            ),
            math_intuition="output(x, y) = clip(input(x, y) * alpha, 0, 255)",
            advantages=(
                "Fast, single-pass elementwise operation.",
                "Effective for images that look flat or washed out.",
            ),
            limitations=(
                "A single global alpha can't fix unevenly-exposed images.",
                "Large alpha clips both highlights and shadows at once.",
            ),
            complexity="O(H x W)",
        )
