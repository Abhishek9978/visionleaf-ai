"""Gray Level Co-occurrence Matrix (GLCM) Texture Features.

Purpose:
    Quantify surface texture — how often pairs of pixel intensities
    occur next to each other — the classic Haralick approach to
    distinguishing smooth healthy tissue from rough, irregular
    diseased tissue.

Theory:
    A GLCM tabulates, for a given spatial offset (distance + angle),
    how often each pair of gray levels (i, j) occurs at that offset
    across the image. A smooth, uniform region produces a GLCM
    concentrated near the diagonal (neighboring pixels have similar
    values); a rough, high-contrast region spreads the GLCM out.
    Several scalar statistics ("Haralick features") summarize that
    matrix's shape.

Working Principle:
    1. Require the segmented/ROI image, converted to grayscale.
    2. Quantize it to a small number of gray levels (fewer levels
       means a smaller, more statistically robust co-occurrence
       matrix — 256 raw gray levels would make most entries near-zero
       and noisy for a typical ROI-sized image).
    3. Build the GLCM for the configured distance/angle.
    4. Extract Contrast, Dissimilarity, Homogeneity, Energy,
       Correlation, and ASM from it.

Math intuition:
    Given a normalized GLCM P(i, j):
    - Contrast    = sum_{i,j} (i - j)^2 * P(i, j)      (weights far-apart pairs)
    - Dissimilarity = sum_{i,j} |i - j| * P(i, j)       (like Contrast, linear not squared)
    - Homogeneity = sum_{i,j} P(i, j) / (1 + (i - j)^2) (rewards similar pairs)
    - Energy      = sqrt(sum_{i,j} P(i, j)^2)           (uniformity of the matrix)
    - ASM         = sum_{i,j} P(i, j)^2                 (Energy squared)
    - Correlation = how linearly dependent a pixel's value is on its neighbor's

Advantages:
    - Well-established, interpretable texture descriptors used across
      decades of image analysis literature.
    - Captures spatial pixel relationships that a plain intensity
      histogram (Color Features) cannot.

Limitations:
    - Sensitive to the chosen quantization level, distance, and angle —
      different choices measure texture at different scales/directions.
    - Computed at a single distance/angle here (not averaged across
      multiple, which would add rotation invariance at the cost of
      more computation).

Typical Applications:
    - Distinguishing a smooth healthy leaf surface from the rough,
      irregular texture of fungal or bacterial lesions.

Complexity:
    O(H x W) to build the co-occurrence matrix, O(levels^2) to compute
    each property from it.

References:
    Haralick, R.M., Shanmugam, K., Dinstein, I. "Textural Features for
    Image Classification." IEEE Transactions on Systems, Man, and
    Cybernetics, 1973.

Dependencies:
    numpy, opencv-python (cv2), scikit-image (skimage.feature);
    visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    GLCMFeaturesParams, GLCMFeatures
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.features._common import require_grayscale_input
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage

logger = get_logger(__name__)

_GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


@dataclass(frozen=True)
class GLCMFeaturesParams:
    """Parameters for `GLCMFeatures`.

    Attributes:
        distance: Pixel-pair offset distance; must be a positive integer.
        angle_degrees: Pixel-pair offset angle in degrees (0, 45, 90,
            or 135 are the conventional choices).
        levels: Number of gray levels to quantize to before building
            the co-occurrence matrix; must be a positive integer, and
            should be small (e.g. 8-32) relative to 256 for a
            statistically meaningful matrix on ROI-sized images.
    """

    distance: int = 1
    angle_degrees: float = 0.0
    levels: int = 8


@register_algorithm("glcm_features")
class GLCMFeatures(PipelineStage):
    """Compute GLCM (Haralick) texture properties."""

    stage_key = "features"

    def __init__(self, params: GLCMFeaturesParams | None = None, **kwargs) -> None:
        self.params = params or GLCMFeaturesParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_grayscale_input(session)
        if self.params.distance < 1:
            raise ValidationError(
                "distance must be a positive integer", details=f"got {self.params.distance!r}"
            )
        if self.params.levels < 2:
            raise ValidationError(
                "levels must be at least 2", details=f"got {self.params.levels!r}"
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        gray = require_grayscale_input(session)

        levels = self.params.levels
        quantized = np.floor(gray.astype(np.float64) / 256.0 * levels).astype(np.uint8)
        quantized = np.clip(quantized, 0, levels - 1)

        angle_radians = np.deg2rad(self.params.angle_degrees)
        glcm = graycomatrix(
            quantized,
            distances=[self.params.distance],
            angles=[angle_radians],
            levels=levels,
            symmetric=True,
            normed=True,
        )

        glcm_props = {
            prop.lower(): float(graycoprops(glcm, prop)[0, 0]) for prop in _GLCM_PROPERTIES
        }

        existing_texture_features = session.texture_features or {}
        existing_texture_features["glcm"] = {
            **glcm_props,
            "distance": self.params.distance,
            "angle_degrees": self.params.angle_degrees,
            "levels": levels,
        }
        session.texture_features = existing_texture_features

        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"distance={self.params.distance}, angle={self.params.angle_degrees}, levels={levels}",
        )
        logger.info(
            "%s | distance=%d | angle=%.1f | levels=%d | duration=%.4fs | success",
            self.get_name(),
            self.params.distance,
            self.params.angle_degrees,
            levels,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "GLCM Texture Features"

    def get_description(self) -> str:
        return "Computes Haralick texture properties (contrast, correlation, energy, homogeneity, dissimilarity, ASM) from the gray-level co-occurrence matrix."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Quantify surface texture via how often pairs of pixel "
                "intensities occur next to each other — distinguishing "
                "smooth healthy tissue from rough diseased tissue."
            ),
            theory=(
                "A GLCM tabulates how often each pair of gray levels "
                "occurs at a given spatial offset. A smooth region "
                "concentrates the GLCM near the diagonal; a rough, "
                "high-contrast region spreads it out. Several scalar "
                "Haralick statistics summarize that shape."
            ),
            working_principle=(
                "1) Require the segmented/ROI image as grayscale. "
                "2) Quantize to a small number of gray levels. 3) Build "
                "the GLCM for the configured distance/angle. 4) Extract "
                "Contrast, Dissimilarity, Homogeneity, Energy, "
                "Correlation, and ASM."
            ),
            math_intuition=(
                "Contrast = sum (i-j)^2 * P(i,j); Homogeneity = "
                "sum P(i,j) / (1 + (i-j)^2); ASM = sum P(i,j)^2; "
                "Energy = sqrt(ASM); Correlation measures linear "
                "dependence between paired pixel values."
            ),
            advantages=(
                "Well-established, interpretable texture descriptors.",
                "Captures spatial pixel relationships a plain histogram cannot.",
            ),
            limitations=(
                "Sensitive to the chosen quantization level, distance, and angle.",
                "Computed at a single distance/angle, not rotation-invariant.",
            ),
            typical_applications=(
                "Distinguishing smooth healthy leaf surface from rough "
                "lesion texture.",
            ),
            complexity="O(H x W) to build the matrix, O(levels^2) per property",
            opencv_reference="skimage.feature.graycomatrix, skimage.feature.graycoprops",
        )
