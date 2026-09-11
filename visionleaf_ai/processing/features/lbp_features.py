"""Local Binary Pattern (LBP) Texture Features.

Purpose:
    Describe local micro-texture — how each pixel's intensity compares
    to its immediate neighbors — a texture descriptor that is
    famously robust to monotonic lighting changes (a brighter or
    dimmer photo of the same leaf produces nearly the same LBP
    pattern), complementing GLCM's more global co-occurrence view.

Theory:
    For each pixel, LBP compares it to `n_points` neighbors sampled on
    a circle of radius `radius`, producing a binary pattern (1 where
    the neighbor is brighter, 0 where darker) that gets encoded as a
    single number. "Uniform" LBP (used here) further groups patterns
    by how many 0-to-1 transitions occur around the circle — patterns
    with at most 2 transitions ("uniform" patterns, corresponding to
    corners, edges, and flat regions) each get their own bin, and every
    other ("non-uniform," noise-like) pattern shares one bin — this is
    both a texture-theoretically motivated grouping (uniform patterns
    are the ones that correspond to recognizable micro-structures) and
    a practical dimensionality reduction.

Working Principle:
    1. Require the segmented/ROI image as grayscale.
    2. Compute the uniform LBP code for every pixel.
    3. Build a histogram of LBP codes across the image.
    4. Normalize the histogram to sum to 1.

Math intuition:
    LBP(x, y) = sum_{p=0}^{P-1} s(neighbor_p - center) * 2^p,
    where s(z) = 1 if z >= 0 else 0. Uniform patterns are then grouped
    by their transition count into P+2 bins total (P "uniform" bins
    plus one catch-all "non-uniform" bin).

Advantages:
    - Robust to monotonic illumination changes (unlike raw intensity
      comparisons).
    - Computationally cheap and well-studied for texture classification.

Limitations:
    - Purely local — doesn't capture larger-scale texture patterns the
      way GLCM's configurable distance can.
    - `radius`/`n_points` are real hyperparameters that change what
      scale of micro-texture gets captured.

Typical Applications:
    - Texture classification robust to photographs taken in different
      lighting conditions — a common real-world constraint for
      field-collected leaf images.

Complexity:
    O(H x W x n_points) to compute the LBP codes, O(H x W) for the
    histogram.

References:
    Ojala, T., Pietikainen, M., Maenpaa, T. "Multiresolution Gray-Scale
    and Rotation Invariant Texture Classification with Local Binary
    Patterns." IEEE TPAMI, 2002.

Dependencies:
    numpy, scikit-image (skimage.feature); visionleaf_ai.core;
    visionleaf_ai.processing.pipeline.

Public classes:
    LBPFeaturesParams, LBPFeatures
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from skimage.feature import local_binary_pattern

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.features._common import require_grayscale_input
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage

logger = get_logger(__name__)

_EPSILON = 1e-12


@dataclass(frozen=True)
class LBPFeaturesParams:
    """Parameters for `LBPFeatures`.

    Attributes:
        radius: Radius of the circular neighborhood, in pixels; must
            be a positive integer.
        n_points: Number of neighbor points sampled on that circle;
            must be a positive integer. The resulting histogram has
            `n_points + 2` bins (uniform LBP convention).
    """

    radius: int = 1
    n_points: int = 8


@register_algorithm("lbp_features")
class LBPFeatures(PipelineStage):
    """Compute the uniform Local Binary Pattern histogram."""

    stage_key = "features"

    def __init__(self, params: LBPFeaturesParams | None = None, **kwargs) -> None:
        self.params = params or LBPFeaturesParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_grayscale_input(session)
        if self.params.radius < 1:
            raise ValidationError(
                "radius must be a positive integer", details=f"got {self.params.radius!r}"
            )
        if self.params.n_points < 1:
            raise ValidationError(
                "n_points must be a positive integer", details=f"got {self.params.n_points!r}"
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        gray = require_grayscale_input(session)

        lbp = local_binary_pattern(
            gray, P=self.params.n_points, R=self.params.radius, method="uniform"
        )
        n_bins = self.params.n_points + 2
        histogram, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        histogram_normalized = histogram / (histogram.sum() + _EPSILON)

        existing_texture_features = session.texture_features or {}
        existing_texture_features["lbp"] = {
            "histogram": histogram.tolist(),
            "histogram_normalized": histogram_normalized.tolist(),
            "radius": self.params.radius,
            "n_points": self.params.n_points,
        }
        session.texture_features = existing_texture_features

        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"radius={self.params.radius}, n_points={self.params.n_points}",
        )
        logger.info(
            "%s | radius=%d | n_points=%d | duration=%.4fs | success",
            self.get_name(),
            self.params.radius,
            self.params.n_points,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Local Binary Pattern"

    def get_description(self) -> str:
        return "Computes the uniform LBP histogram, a lighting-robust local micro-texture descriptor."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Describe local micro-texture in a way that is robust "
                "to monotonic lighting changes, complementing GLCM's "
                "more global co-occurrence view."
            ),
            theory=(
                "Compares each pixel to neighbors sampled on a circle, "
                "encoding brighter/darker as a binary pattern. 'Uniform' "
                "LBP groups patterns by their transition count: patterns "
                "with at most 2 transitions get their own bin; every "
                "other pattern shares one catch-all bin."
            ),
            working_principle=(
                "1) Require the segmented/ROI image as grayscale. "
                "2) Compute the uniform LBP code per pixel. 3) Build a "
                "histogram of codes. 4) Normalize it to sum to 1."
            ),
            math_intuition=(
                "LBP(x,y) = sum_p s(neighbor_p - center) * 2^p, where "
                "s(z) = 1 if z >= 0 else 0; uniform patterns are grouped "
                "into P+2 bins by transition count."
            ),
            advantages=(
                "Robust to monotonic illumination changes.",
                "Computationally cheap and well-studied for texture classification.",
            ),
            limitations=(
                "Purely local — doesn't capture larger-scale texture patterns.",
                "radius/n_points change what scale of micro-texture is captured.",
            ),
            typical_applications=(
                "Texture classification robust to varying field-photo lighting.",
            ),
            complexity="O(H x W x n_points)",
            opencv_reference="skimage.feature.local_binary_pattern",
        )
