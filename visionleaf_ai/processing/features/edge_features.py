"""Edge Features.

Purpose:
    Summarize how much and how sharply intensity changes across the
    region — lesion boundaries and rough diseased surfaces both
    produce more/stronger edges than smooth healthy tissue, making
    edge statistics a cheap complement to Texture Features.

Theory:
    Three classic edge operators, each capturing a different aspect:
    Canny finds a clean binary edge map (after non-maximum suppression
    and hysteresis thresholding) — its edge-pixel density measures how
    much of the region is "edge." Sobel computes a directional
    gradient — its magnitude measures local intensity change strength
    everywhere, not just at detected edges. Laplacian is a
    second-derivative operator — its response variance is a classic
    proxy for how much fine detail/sharpness an image contains (the
    same statistic commonly used to detect blur).

Working Principle:
    1. Require the segmented/ROI image as grayscale.
    2. Run Canny edge detection; edge density = fraction of pixels
       marked as edges.
    3. Compute Sobel gradients in x and y, combine into a magnitude
       image, and take its mean.
    4. Compute the Laplacian and take its variance.

Math intuition:
    canny_edge_density = count(edge_pixels) / total_pixels
    sobel_magnitude(x, y) = sqrt(Gx(x,y)^2 + Gy(x,y)^2)
    laplacian_response_variance = Var(Laplacian(image))

Advantages:
    - Cheap to compute, three complementary views of "how much edge."
    - Laplacian variance in particular is a well-known sharpness/blur
      proxy independent of any thresholding choice.

Limitations:
    - Canny's edge density depends on its two threshold parameters —
      different thresholds meaningfully change the count.
    - All three are single scalar summaries — they say "how much edge"
      but nothing about *where* (Shape Features covers boundary
      geometry; these do not).

Typical Applications:
    - Distinguishing a smooth, healthy leaf surface from a rough,
      lesion-covered one via overall edge intensity, without needing
      to first segment individual lesions.

Complexity:
    O(H x W) for each operator.

References:
    Canny, J. "A Computational Approach to Edge Detection." IEEE
    TPAMI, 1986.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    EdgeFeaturesParams, EdgeFeatures
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.features._common import require_grayscale_input
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage

logger = get_logger(__name__)


@dataclass(frozen=True)
class EdgeFeaturesParams:
    """Parameters for `EdgeFeatures`.

    Attributes:
        canny_low_threshold: Lower hysteresis threshold for Canny;
            must be in [0, 255].
        canny_high_threshold: Upper hysteresis threshold for Canny;
            must be in [0, 255] and greater than `canny_low_threshold`.
        sobel_kernel_size: Kernel size for the Sobel operator; must be
            a positive odd integer.
    """

    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    sobel_kernel_size: int = 3


@register_algorithm("edge_features")
class EdgeFeatures(PipelineStage):
    """Compute Canny edge density, Sobel gradient magnitude, and Laplacian variance."""

    stage_key = "features"

    def __init__(self, params: EdgeFeaturesParams | None = None, **kwargs) -> None:
        self.params = params or EdgeFeaturesParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_grayscale_input(session)
        for name, value in (
            ("canny_low_threshold", self.params.canny_low_threshold),
            ("canny_high_threshold", self.params.canny_high_threshold),
        ):
            if not (0 <= value <= 255):
                raise ValidationError(f"{name} must be in [0, 255]", details=f"got {value!r}")
        if self.params.canny_high_threshold <= self.params.canny_low_threshold:
            raise ValidationError(
                "canny_high_threshold must be greater than canny_low_threshold",
                details=(
                    f"low={self.params.canny_low_threshold}, "
                    f"high={self.params.canny_high_threshold}"
                ),
            )
        if self.params.sobel_kernel_size < 1 or self.params.sobel_kernel_size % 2 == 0:
            raise ValidationError(
                "sobel_kernel_size must be a positive odd integer",
                details=f"got {self.params.sobel_kernel_size!r}",
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        gray = require_grayscale_input(session)

        edges = cv2.Canny(gray, self.params.canny_low_threshold, self.params.canny_high_threshold)
        canny_edge_density = float(np.count_nonzero(edges) / edges.size)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.params.sobel_kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.params.sobel_kernel_size)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_gradient_magnitude_mean = float(sobel_magnitude.mean())

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_response_variance = float(laplacian.var())

        session.edge_features = {
            "canny_edge_density": canny_edge_density,
            "sobel_gradient_magnitude_mean": sobel_gradient_magnitude_mean,
            "laplacian_response_variance": laplacian_response_variance,
        }
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"canny_edge_density={canny_edge_density:.4f}",
        )
        logger.info(
            "%s | canny_edge_density=%.4f | sobel_mean=%.2f | laplacian_var=%.2f | duration=%.4fs | success",
            self.get_name(),
            canny_edge_density,
            sobel_gradient_magnitude_mean,
            laplacian_response_variance,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Edge Features"

    def get_description(self) -> str:
        return "Computes Canny edge density, mean Sobel gradient magnitude, and Laplacian response variance."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Summarize how much and how sharply intensity changes "
                "across the region — a cheap complement to Texture "
                "Features for capturing lesion boundaries and rough "
                "diseased surfaces."
            ),
            theory=(
                "Canny finds a clean binary edge map; its density "
                "measures how much of the region is 'edge.' Sobel "
                "computes a directional gradient measuring local "
                "intensity change everywhere. Laplacian is a second-"
                "derivative operator whose response variance is a "
                "classic sharpness/blur proxy."
            ),
            working_principle=(
                "1) Require the segmented/ROI image as grayscale. "
                "2) Run Canny, measure edge density. 3) Compute Sobel "
                "gradients, take magnitude mean. 4) Compute Laplacian, "
                "take variance."
            ),
            math_intuition=(
                "canny_edge_density = count(edges) / total_pixels; "
                "sobel_magnitude = sqrt(Gx^2 + Gy^2); "
                "laplacian_response_variance = Var(Laplacian(image))"
            ),
            advantages=(
                "Cheap, three complementary views of 'how much edge.'",
                "Laplacian variance is a well-known sharpness/blur proxy.",
            ),
            limitations=(
                "Canny's density depends on its two threshold parameters.",
                "All three are scalar summaries — say nothing about "
                "edge location.",
            ),
            typical_applications=(
                "Distinguishing smooth healthy leaf surface from rough "
                "lesion-covered surface via overall edge intensity.",
            ),
            complexity="O(H x W) per operator",
            opencv_reference="cv2.Canny, cv2.Sobel, cv2.Laplacian",
        )
