"""Histogram Equalization.

Purpose:
    Automatically improve global contrast by spreading out the most
    frequent intensity values across the full available range —
    useful when an image's histogram is bunched into a narrow band
    (e.g. a hazy or low-contrast photo).

Theory:
    Computes the image's cumulative distribution function (CDF) of
    pixel intensities and remaps each intensity so the output
    histogram is as close to uniform as possible. For color images,
    equalizing each RGB channel independently would distort color
    balance, so this implementation converts to YCrCb, equalizes only
    the luminance (Y) channel, and converts back — preserving color
    while still improving contrast.

Math intuition:
    output(x, y) = round(255 * CDF(input(x, y)) / (H x W))
    Intensities that occur very often get spread across a wider output
    range; intensities that barely occur get compressed. Unlike gamma
    or contrast (which apply a fixed formula), the transform here is
    derived directly from each image's own histogram.

Advantages:
    - Fully automatic — no parameter to tune.
    - Very effective on genuinely low-contrast images.

Limitations:
    - Can over-amplify noise in flat regions.
    - Purely global — CLAHE (a separate algorithm) addresses this by
      equalizing local tiles instead of the whole image at once.

Complexity:
    O(H x W) to build the histogram/CDF, O(H x W) to remap — O(H x W)
    overall.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    HistogramEqualizationParams, HistogramEqualization
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
class HistogramEqualizationParams:
    """Parameters for `HistogramEqualization`.

    This algorithm has no tunable parameters — it's fully derived from
    each image's own histogram — but a (currently empty) params
    dataclass is kept for interface consistency with every other
    algorithm, and so a future parameter (e.g. an optional channel
    selection) can be added without changing the constructor's shape.
    """


@register_algorithm("histogram_equalization")
class HistogramEqualization(PipelineStage):
    """Equalize the luminance channel's histogram to spread out contrast."""

    stage_key = "enhancement"

    def __init__(self, params: HistogramEqualizationParams | None = None, **kwargs) -> None:
        self.params = params or HistogramEqualizationParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        image = session.require_active_image()
        if image.dtype != np.uint8:
            raise ValidationError(
                "Histogram equalization requires an 8-bit image",
                details=f"got dtype {image.dtype}",
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        image = session.require_active_image()

        if image.ndim == 2:
            result = cv2.equalizeHist(image)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_equalized = cv2.equalizeHist(y)
            result = cv2.cvtColor(cv2.merge([y_equalized, cr, cb]), cv2.COLOR_YCrCb2RGB)

        session.current_image = result
        duration = time.perf_counter() - started
        session.record_event(self.stage_key, self.get_name())
        logger.info(
            "%s | image=%s | duration=%.4fs | success",
            self.get_name(),
            image.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Histogram Equalization"

    def get_description(self) -> str:
        return "Spreads out pixel intensities using the image's cumulative histogram to improve global contrast."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Automatically improve global contrast for images whose "
                "histogram is bunched into a narrow band."
            ),
            theory=(
                "Computes the cumulative distribution function (CDF) of "
                "pixel intensities and remaps each intensity so the output "
                "histogram approaches uniform. Color images are equalized "
                "on the luminance (Y) channel only, in YCrCb space, to "
                "avoid distorting color balance."
            ),
            math_intuition=(
                "output(x, y) = round(255 * CDF(input(x, y)) / (H x W)). "
                "Frequently-occurring intensities spread across a wider "
                "output range; rare ones compress."
            ),
            advantages=(
                "Fully automatic — no parameter to tune.",
                "Very effective on genuinely low-contrast images.",
            ),
            limitations=(
                "Can over-amplify noise in flat regions.",
                "Purely global — see CLAHE for a local alternative.",
            ),
            complexity="O(H x W)",
        )
