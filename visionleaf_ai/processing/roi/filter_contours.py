"""Contour Filtering by Area.

Purpose:
    Discard contours that are too small (noise specks) or too large
    (e.g. an accidental full-frame border) to be genuine regions of
    interest, keeping only contours within a plausible area range.

Theory:
    `Find Contours` often returns many small, spurious contours from
    noise alongside the genuine region(s) of interest. Filtering by
    enclosed area is the simplest, most direct way to discard them
    without needing shape analysis.

Working Principle:
    1. Require existing contours (`session.contours`).
    2. Compute each contour's enclosed area.
    3. Keep only contours whose area falls within `[min_area, max_area]`.
    4. Recompute `largest_contour` from the surviving set (it may have
       been filtered out, or a smaller one may now be the largest of
       what remains).

Math intuition:
    keep(contour) = min_area <= area(contour) <= max_area
    (area computed via the shoelace formula over the contour's points.)

Advantages:
    - Cheap, simple, and effective for removing speckle noise.
    - `max_area` guards against an accidental full-frame contour
      (e.g. from a mask that's mostly foreground).

Limitations:
    - Area alone can't distinguish a genuine small region from a
      similarly-small noise speck — shape-based filtering would be
      needed for that distinction.
    - Requires the user to know a plausible size range in advance.

Typical Applications:
    - Removing speckle-noise contours left over after thresholding,
      before selecting the genuine region of interest.

Complexity:
    O(N) for N contours (one area computation each).

OpenCV reference:
    cv2.contourArea(contour)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    ContourAreaFilterParams, ContourAreaFilter
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
from visionleaf_ai.processing.roi._common import require_contours

logger = get_logger(__name__)


@dataclass(frozen=True)
class ContourAreaFilterParams:
    """Parameters for `ContourAreaFilter`.

    Attributes:
        min_area: Minimum enclosed area (in pixels) a contour must
            have to survive filtering.
        max_area: Maximum enclosed area a contour may have to survive
            filtering.
    """

    min_area: float = 50.0
    max_area: float = 1_000_000.0


@register_algorithm("contour_area_filter")
class ContourAreaFilter(PipelineStage):
    """Discard contours outside a plausible area range."""

    stage_key = "roi"

    def __init__(self, params: ContourAreaFilterParams | None = None, **kwargs) -> None:
        self.params = params or ContourAreaFilterParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_contours(session)
        if self.params.min_area < 0:
            raise ValidationError(
                "min_area must be non-negative", details=f"got {self.params.min_area!r}"
            )
        if self.params.max_area <= self.params.min_area:
            raise ValidationError(
                "max_area must be greater than min_area",
                details=f"min_area={self.params.min_area!r}, max_area={self.params.max_area!r}",
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        contours = require_contours(session)

        filtered = [
            contour
            for contour in contours
            if self.params.min_area <= cv2.contourArea(contour) <= self.params.max_area
        ]
        largest = max(filtered, key=cv2.contourArea) if filtered else None

        session.contours = filtered
        session.largest_contour = largest
        session.clear_roi_results()
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=(
                f"min_area={self.params.min_area}, max_area={self.params.max_area}, "
                f"kept={len(filtered)}/{len(contours)}"
            ),
        )
        logger.info(
            "%s | min_area=%.1f | max_area=%.1f | kept=%d/%d | duration=%.4fs | success",
            self.get_name(),
            self.params.min_area,
            self.params.max_area,
            len(filtered),
            len(contours),
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Contour Filtering by Area"

    def get_description(self) -> str:
        return "Keeps only contours whose enclosed area falls within a configured range."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Discard contours that are too small (noise) or too "
                "large (e.g. a full-frame border) to be genuine regions "
                "of interest."
            ),
            theory=(
                "Find Contours often returns many small, spurious "
                "contours alongside the genuine region(s). Filtering by "
                "enclosed area is the simplest way to discard them."
            ),
            working_principle=(
                "1) Require existing contours. 2) Compute each contour's "
                "area. 3) Keep only those within [min_area, max_area]. "
                "4) Recompute the largest contour from survivors."
            ),
            math_intuition="keep(contour) = min_area <= area(contour) <= max_area",
            advantages=(
                "Cheap, simple, and effective for removing speckle noise.",
                "max_area guards against an accidental full-frame contour.",
            ),
            limitations=(
                "Area alone can't distinguish a genuine small region from "
                "a similarly-small noise speck.",
                "Requires knowing a plausible size range in advance.",
            ),
            typical_applications=(
                "Removing speckle-noise contours before selecting the "
                "genuine region of interest.",
            ),
            complexity="O(N) for N contours",
            opencv_reference="cv2.contourArea(contour)",
        )
