"""Bilateral Filter.

Purpose:
    Smooth noise while explicitly preserving edges — useful as a final
    restoration step before segmentation, where sharp boundaries (e.g.
    a diseased spot's edge) must survive noise reduction.

Theory:
    Like Gaussian blur, each output pixel is a weighted average of its
    neighbors — but the bilateral filter uses *two* weights per
    neighbor: one based on spatial distance (as in Gaussian blur) and
    one based on intensity difference. A neighbor that's spatially
    close but very different in intensity (i.e., likely across an
    edge) contributes little, so edges are preserved even as flat
    regions get smoothed.

Math intuition:
    weight(i, j) = spatial_weight(distance) * range_weight(intensity_difference)
    `sigma_space` controls how far spatially a pixel's influence
    reaches (like Gaussian blur's sigma); `sigma_color` controls how
    much an intensity difference reduces a neighbor's influence —
    small `sigma_color` preserves edges more aggressively.

Advantages:
    - Genuinely edge-preserving, unlike Gaussian blur.
    - Effective at reducing noise without softening region boundaries.

Limitations:
    - Substantially slower than Gaussian or median filtering (each
      output pixel needs its own locally-computed range weights).
    - Two interacting parameters (sigma_color, sigma_space) can be
      harder to tune than a single kernel size.

Complexity:
    O(H x W x d^2) for diameter d — quadratic in the neighborhood size,
    notably more expensive than the separable Gaussian blur.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    BilateralFilterParams, BilateralFilter
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
class BilateralFilterParams:
    """Parameters for `BilateralFilter`.

    Attributes:
        diameter: Diameter of each pixel neighborhood. Must be a
            positive integer.
        sigma_color: Filter sigma in color space — larger values mean
            more distant intensities are still blended together.
        sigma_space: Filter sigma in coordinate space — larger values
            mean more distant pixels can influence each other,
            provided their intensities are also similar.
    """

    diameter: int = 9
    sigma_color: float = 75.0
    sigma_space: float = 75.0


@register_algorithm("bilateral_filter")
class BilateralFilter(PipelineStage):
    """Smooth noise while preserving edges via joint spatial/intensity weighting."""

    stage_key = "restoration"

    def __init__(self, params: BilateralFilterParams | None = None, **kwargs) -> None:
        self.params = params or BilateralFilterParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        if not isinstance(self.params.diameter, int) or self.params.diameter <= 0:
            raise ValidationError(
                "diameter must be a positive integer",
                details=f"got {self.params.diameter!r}",
            )
        if self.params.sigma_color <= 0:
            raise ValidationError(
                "sigma_color must be strictly positive",
                details=f"got {self.params.sigma_color!r}",
            )
        if self.params.sigma_space <= 0:
            raise ValidationError(
                "sigma_space must be strictly positive",
                details=f"got {self.params.sigma_space!r}",
            )
        height, width = session.require_active_image().shape[:2]
        if self.params.diameter > min(height, width):
            raise ValidationError(
                f"diameter {self.params.diameter} exceeds image dimensions "
                f"({width}x{height})"
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        image = session.require_active_image()

        result = cv2.bilateralFilter(
            image,
            self.params.diameter,
            self.params.sigma_color,
            self.params.sigma_space,
        )

        session.current_image = result
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=(
                f"diameter={self.params.diameter}, "
                f"sigma_color={self.params.sigma_color}, "
                f"sigma_space={self.params.sigma_space}"
            ),
        )
        logger.info(
            "%s | diameter=%d | sigma_color=%.1f | sigma_space=%.1f | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.diameter,
            self.params.sigma_color,
            self.params.sigma_space,
            image.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Bilateral Filter"

    def get_description(self) -> str:
        return "Smooths noise while preserving edges by weighting neighbors on both spatial distance and intensity similarity."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Smooth noise while explicitly preserving edges — useful "
                "before segmentation, where sharp boundaries must survive "
                "noise reduction."
            ),
            theory=(
                "Each output pixel is a weighted average of its neighbors "
                "using two weights: spatial distance (as in Gaussian blur) "
                "and intensity difference. Neighbors across an edge "
                "contribute little, so edges are preserved."
            ),
            math_intuition=(
                "weight(i, j) = spatial_weight(distance) * "
                "range_weight(intensity_difference). sigma_space controls "
                "spatial reach; sigma_color controls how much an intensity "
                "difference reduces influence."
            ),
            advantages=(
                "Genuinely edge-preserving, unlike Gaussian blur.",
                "Reduces noise without softening region boundaries.",
            ),
            limitations=(
                "Substantially slower than Gaussian or median filtering.",
                "Two interacting parameters can be harder to tune.",
            ),
            complexity="O(H x W x d^2) for diameter d",
        )
