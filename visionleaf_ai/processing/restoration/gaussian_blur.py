"""Gaussian Blur.

Purpose:
    Smooth an image by averaging each pixel with its neighbors,
    weighted by a Gaussian (bell-curve) function — the standard way to
    simulate or reduce mild blur/noise and a common pre-processing step
    before edge-sensitive later stages (e.g. segmentation).

Theory:
    Convolves the image with a 2D Gaussian kernel. Nearby pixels
    contribute more to the output than distant ones, with the falloff
    controlled by `sigma`. Because a Gaussian kernel is separable, it
    can be applied as two 1D passes (horizontal then vertical), which
    is why Gaussian blur is fast even for larger kernels.

Math intuition:
    G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))
    The output at each pixel is the weighted sum of its neighborhood
    under this kernel. Larger `sigma` (or a larger kernel) means more
    smoothing.

Advantages:
    - Removes high-frequency noise effectively.
    - Separable kernel makes it computationally cheap.
    - Simple, well-understood, and reversible in simulation contexts
      (used in Experiment Mode to *create* controlled blur too).

Limitations:
    - Blurs edges along with noise — it cannot distinguish "noise" from
      "fine detail."
    - Not edge-preserving (compare to the Bilateral Filter).

Complexity:
    O(H x W x k) for a naive 2D kernel of size k, or O(H x W x k) split
    into two O(H x W) passes thanks to separability — effectively
    linear in image size for a fixed kernel size.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    GaussianBlurParams, GaussianBlur
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
from visionleaf_ai.utils.validators import ensure_kernel_fits_image, ensure_odd_positive_int

logger = get_logger(__name__)


@dataclass(frozen=True)
class GaussianBlurParams:
    """Parameters for `GaussianBlur`.

    Attributes:
        kernel_size: Side length of the square kernel; must be a
            positive odd integer.
        sigma: Standard deviation of the Gaussian. 0 means "derive it
            automatically from kernel_size" (OpenCV's convention).
    """

    kernel_size: int = 5
    sigma: float = 0.0


@register_algorithm("gaussian_blur")
class GaussianBlur(PipelineStage):
    """Smooth an image by convolving it with a Gaussian kernel."""

    stage_key = "restoration"

    def __init__(self, params: GaussianBlurParams | None = None, **kwargs) -> None:
        self.params = params or GaussianBlurParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        ensure_odd_positive_int(self.params.kernel_size, "kernel_size")
        if self.params.sigma < 0:
            raise ValidationError(
                "sigma must be non-negative", details=f"got {self.params.sigma!r}"
            )
        ensure_kernel_fits_image(self.params.kernel_size, session.require_active_image().shape)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        image = session.require_active_image()

        result = cv2.GaussianBlur(
            image,
            (self.params.kernel_size, self.params.kernel_size),
            self.params.sigma,
        )

        session.current_image = result
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"kernel_size={self.params.kernel_size}, sigma={self.params.sigma}",
        )
        logger.info(
            "%s | kernel_size=%d | sigma=%.3f | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.kernel_size,
            self.params.sigma,
            image.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Gaussian Blur"

    def get_description(self) -> str:
        return "Smooths the image by convolving with a Gaussian-weighted kernel."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Smooth an image by averaging each pixel with its "
                "neighbors, weighted by a Gaussian function."
            ),
            theory=(
                "Convolves the image with a 2D Gaussian kernel; nearby "
                "pixels contribute more than distant ones, controlled by "
                "sigma. The kernel is separable, so it's applied as two "
                "fast 1D passes."
            ),
            math_intuition=(
                "G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2)); "
                "the output is the weighted sum of each pixel's neighborhood."
            ),
            advantages=(
                "Removes high-frequency noise effectively.",
                "Separable kernel makes it computationally cheap.",
            ),
            limitations=(
                "Blurs edges along with noise — not edge-preserving.",
                "Cannot distinguish noise from genuine fine detail.",
            ),
            complexity="O(H x W) for a fixed kernel size (separable convolution)",
        )
