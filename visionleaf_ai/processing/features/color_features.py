"""Color Features.

Purpose:
    Summarize the leaf region's color distribution — RGB and HSV
    mean/standard deviation plus per-channel histograms — the primary
    signal for color-based symptoms like chlorosis (yellowing) or
    necrosis (browning/blackening).

Theory:
    Mean and standard deviation per channel capture the overall color
    and how much it varies (a uniformly green leaf has low std; a
    heavily spotted one has high std). HSV is included alongside RGB
    because Hue isolates "what color" from Value ("how bright"),
    making HSV-based features more robust to lighting changes than RGB
    alone — two photos of the same leaf under different brightness
    have very different RGB means but similar Hue. Histograms capture
    the full distribution shape, not just its first two moments — two
    regions with the same mean/std can still have very different
    histograms (e.g. bimodal green+brown vs. uniform olive).

Working Principle:
    1. Require the segmented/ROI image (foreground-only if a mask is
       available — see `_common.require_feature_input_image`).
    2. Compute per-channel mean/std in RGB, over foreground pixels only.
    3. Convert to HSV and repeat.
    4. Compute a per-channel histogram (RGB) with the configured
       number of bins, restricted to foreground pixels, and normalize
       each channel's histogram to sum to 1 (so it describes a
       *distribution*, comparable across images of different sizes,
       not a raw pixel count).

Math intuition:
    mean_c = (1/N) * sum(pixel_c for pixel in foreground)
    std_c = sqrt((1/N) * sum((pixel_c - mean_c)^2 for pixel in foreground))
    histogram_normalized_c = histogram_c / sum(histogram_c)

Advantages:
    - Fast to compute, easy to interpret.
    - HSV mean/std are meaningfully more lighting-robust than RGB.

Limitations:
    - A histogram alone doesn't capture spatial arrangement (e.g.
      whether brown pixels are clustered in one spot or scattered) —
      that's what Shape and Texture features are for.
    - Bin count is a real hyperparameter: too few bins loses
      distinction, too many makes the histogram sparse and noisy.

Typical Applications:
    - Detecting chlorosis/necrosis via a color-distribution shift
      relative to a healthy baseline.

Complexity:
    O(H x W) for the statistics and histogram computation.

References:
    Gonzalez & Woods, "Digital Image Processing" — color image
    processing and histogram fundamentals.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    ColorFeaturesParams, ColorFeatures
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.features._common import require_feature_input_image
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage

logger = get_logger(__name__)

_EPSILON = 1e-12


@dataclass(frozen=True)
class ColorFeaturesParams:
    """Parameters for `ColorFeatures`.

    Attributes:
        bins: Number of histogram bins per channel; must be a positive
            integer.
    """

    bins: int = 32


@register_algorithm("color_features")
class ColorFeatures(PipelineStage):
    """Compute RGB/HSV mean, standard deviation, and normalized histograms."""

    stage_key = "features"

    def __init__(self, params: ColorFeaturesParams | None = None, **kwargs) -> None:
        self.params = params or ColorFeaturesParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_feature_input_image(session)
        if self.params.bins < 1:
            raise ValidationError(
                "bins must be a positive integer", details=f"got {self.params.bins!r}"
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        image, foreground_mask = require_feature_input_image(session)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pixels = image.reshape(-1, 3).astype(np.float64)
        if foreground_mask is not None:
            pixels = pixels[foreground_mask.reshape(-1)]

        rgb_mean = pixels.mean(axis=0)
        rgb_std = pixels.std(axis=0)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_pixels = hsv_image.reshape(-1, 3).astype(np.float64)
        if foreground_mask is not None:
            hsv_pixels = hsv_pixels[foreground_mask.reshape(-1)]
        hsv_mean = hsv_pixels.mean(axis=0)
        hsv_std = hsv_pixels.std(axis=0)

        histogram_normalized = []
        for channel_index in range(3):
            channel_values = pixels[:, channel_index]
            counts, _ = np.histogram(channel_values, bins=self.params.bins, range=(0, 255))
            histogram_normalized.append((counts / (counts.sum() + _EPSILON)).tolist())

        session.color_features = {
            "rgb_mean": rgb_mean.tolist(),
            "rgb_std": rgb_std.tolist(),
            "hsv_mean": hsv_mean.tolist(),
            "hsv_std": hsv_std.tolist(),
            "histogram_normalized": histogram_normalized,
            "bins": self.params.bins,
        }
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key, self.get_name(), details=f"bins={self.params.bins}"
        )
        logger.info(
            "%s | bins=%d | pixels=%d | duration=%.4fs | success",
            self.get_name(),
            self.params.bins,
            len(pixels),
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Color Features"

    def get_description(self) -> str:
        return "Computes RGB/HSV mean, standard deviation, and normalized per-channel histograms."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Summarize the leaf region's color distribution — the "
                "primary signal for color-based symptoms like chlorosis "
                "or necrosis."
            ),
            theory=(
                "Mean/standard deviation per channel capture overall "
                "color and its variation. HSV is included alongside RGB "
                "since Hue isolates color from brightness, making it "
                "more lighting-robust. Histograms capture the full "
                "distribution shape, not just its first two moments."
            ),
            working_principle=(
                "1) Require the segmented/ROI image, foreground pixels "
                "only if a mask is available. 2) Compute RGB mean/std. "
                "3) Convert to HSV and repeat. 4) Compute and normalize "
                "a per-channel histogram."
            ),
            math_intuition=(
                "mean_c = (1/N) * sum(pixel_c); std_c = sqrt(variance_c); "
                "histogram_normalized_c = histogram_c / sum(histogram_c)"
            ),
            advantages=(
                "Fast to compute, easy to interpret.",
                "HSV mean/std are meaningfully more lighting-robust than RGB.",
            ),
            limitations=(
                "Doesn't capture spatial arrangement of color regions.",
                "Bin count is a real hyperparameter affecting sensitivity.",
            ),
            typical_applications=(
                "Detecting chlorosis/necrosis via color-distribution shift.",
            ),
            complexity="O(H x W)",
            opencv_reference="cv2.cvtColor(..., cv2.COLOR_RGB2HSV), numpy histogram",
        )
