"""Gamma Correction.

Purpose:
    Apply a non-linear intensity transform that brightens or darkens
    midtones without clipping highlights or shadows the way a simple
    additive brightness shift would.

Theory:
    Gamma correction applies a power-law transform. It's non-linear:
    unlike brightness (additive) or contrast (linear multiplicative),
    it affects dark and light pixels by different relative amounts,
    which better matches how human vision perceives brightness (our
    eyes are more sensitive to changes in dark regions than bright
    ones).

Math intuition:
    output = 255 * (input / 255) ** (1 / gamma)
    gamma > 1 brightens the image (expands dark tones); gamma < 1
    darkens it (expands bright tones); gamma == 1 is a no-op. Applied
    via a precomputed 256-entry lookup table for speed, rather than
    computing the power for every pixel individually.

Advantages:
    - Perceptually more natural than linear brightness/contrast changes.
    - Never fully clips highlights the way adding a large delta can.

Limitations:
    - The right gamma value is scene-dependent; there's no universal
      default that works for every image.
    - Still a single global transform — doesn't adapt regionally.

Complexity:
    O(256) to build the lookup table, then O(H x W) to apply it —
    effectively O(H x W) for any real image.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    GammaParams, GammaCorrection
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage

logger = get_logger(__name__)


@dataclass(frozen=True)
class GammaParams:
    """Parameters for `GammaCorrection`.

    Attributes:
        gamma: Gamma value. Must be strictly positive. > 1 brightens,
            < 1 darkens, == 1 leaves the image unchanged.
    """

    gamma: float = 1.5


@register_algorithm("gamma_correction")
class GammaCorrection(PipelineStage):
    """Apply a power-law (gamma) intensity transform via a lookup table."""

    stage_key = "enhancement"

    def __init__(self, params: GammaParams | None = None, **kwargs) -> None:
        self.params = params or GammaParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        if self.params.gamma <= 0:
            raise ValidationError(
                "gamma must be strictly positive", details=f"got {self.params.gamma!r}"
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        image = session.require_active_image()

        inverse_gamma = 1.0 / self.params.gamma
        lookup_table = np.array(
            [((i / 255.0) ** inverse_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        result = cv2.LUT(image, lookup_table)

        session.current_image = result
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"gamma={self.params.gamma}",
        )
        logger.info(
            "%s | gamma=%.3f | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.gamma,
            image.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Gamma Correction"

    def get_description(self) -> str:
        return "Applies a non-linear power-law transform to brighten or darken midtones."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Brighten or darken midtones without clipping highlights or "
                "shadows the way additive brightness can."
            ),
            theory=(
                "Applies a power-law transform, which is non-linear: it "
                "affects dark and light pixels by different relative "
                "amounts, matching how human vision perceives brightness."
            ),
            math_intuition=(
                "output = 255 * (input / 255) ** (1 / gamma). gamma > 1 "
                "brightens, gamma < 1 darkens, gamma == 1 is a no-op."
            ),
            advantages=(
                "Perceptually more natural than linear brightness/contrast.",
                "Doesn't fully clip highlights the way a large delta can.",
            ),
            limitations=(
                "The right gamma is scene-dependent; no universal default.",
                "Still a single global transform, not regionally adaptive.",
            ),
            complexity="O(H x W) (lookup table built once in O(256))",
        )
