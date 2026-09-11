"""Shape Features.

Purpose:
    Quantify the geometry of the segmented region — different diseases
    tend to produce differently-shaped lesions or affected regions
    (compact circular fungal spots vs. sprawling irregular blight), so
    shape is an independent signal from color and texture.

Theory:
    Area, perimeter, and the ratios/derived quantities built from them
    (extent, solidity, circularity) describe how compact vs. irregular
    a shape is. Hu moments go further: they're combinations of
    normalized central image moments specifically constructed to stay
    invariant under translation, scale, and rotation — two photos of
    the same lesion at different zoom levels or angles should still
    produce very similar Hu moments.

Working Principle:
    1. Require the largest contour, rejecting a zero-area one (a
       degenerate contour makes every ratio below divide by zero).
    2. Compute area and perimeter directly from the contour.
    3. Compute the axis-aligned bounding rectangle for aspect ratio and
       extent.
    4. Compute the convex hull for solidity.
    5. Derive circularity and equivalent diameter from area/perimeter.
    6. Compute the 7 Hu moments from the contour's image moments,
       applying a log transform (`-sign(h) * log10(|h|)`) since raw Hu
       moments span many orders of magnitude — the log-transformed
       values are what's actually usable as ML features.

Math intuition:
    aspect_ratio = bounding_box_width / bounding_box_height
    extent = contour_area / bounding_box_area
    solidity = contour_area / convex_hull_area
    circularity = 4 * pi * area / perimeter^2   (1.0 for a perfect circle)
    equivalent_diameter = sqrt(4 * area / pi)   (diameter of a circle with the same area)

Advantages:
    - Fully invariant to translation (all of these); circularity,
      solidity, aspect ratio, and Hu moments are also scale-invariant.
    - Directly interpretable — "circularity" and "solidity" have
      obvious real-world meaning, unlike an opaque learned feature.

Limitations:
    - Entirely dependent on segmentation quality — a poorly segmented
      contour produces meaningless shape statistics regardless of how
      well-designed these formulas are.
    - Describes the region's outer boundary only; says nothing about
      internal texture or color (that's Texture/Color Features' job).

Typical Applications:
    - Distinguishing round, compact fungal leaf-spot lesions from the
      irregular, spreading patterns typical of blight.

Complexity:
    O(N) for a contour of N points (area/perimeter/hull), O(1) for the
    moment-based Hu moments.

References:
    Hu, M.K. "Visual Pattern Recognition by Moment Invariants." IRE
    Transactions on Information Theory, 1962.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core;
    visionleaf_ai.processing.pipeline; visionleaf_ai.processing.features._common.

Public classes:
    ShapeFeaturesParams, ShapeFeatures
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.features._common import require_nonzero_area_contour
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage

logger = get_logger(__name__)


@dataclass(frozen=True)
class ShapeFeaturesParams:
    """Parameters for `ShapeFeatures`.

    This algorithm has no tunable parameters — it's fully determined
    by the largest contour — but a (currently empty) params dataclass
    is kept for interface consistency with every other algorithm.
    """


def _log_scale_hu_moments(raw_hu_moments: np.ndarray) -> list[float]:
    """Apply the conventional log transform to raw Hu moments.

    Raw Hu moments can span many orders of magnitude, which makes them
    poorly scaled as ML features. The standard transform
    `-sign(h) * log10(|h|)` compresses that range while preserving
    sign and relative ordering.
    """
    with np.errstate(divide="ignore"):
        scaled = -np.sign(raw_hu_moments) * np.log10(np.abs(raw_hu_moments) + 1e-30)
    return scaled.flatten().tolist()


@register_algorithm("shape_features")
class ShapeFeatures(PipelineStage):
    """Compute contour-derived geometric shape descriptors."""

    stage_key = "features"

    def __init__(self, params: ShapeFeaturesParams | None = None, **kwargs) -> None:
        self.params = params or ShapeFeaturesParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_nonzero_area_contour(session)

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        contour = require_nonzero_area_contour(session)

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        _, _, bbox_width, bbox_height = cv2.boundingRect(contour)
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0.0
        bbox_area = bbox_width * bbox_height
        extent = area / bbox_area if bbox_area > 0 else 0.0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        circularity = (4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
        equivalent_diameter = float(np.sqrt(4 * area / np.pi))

        moments = cv2.moments(contour)
        raw_hu_moments = cv2.HuMoments(moments)
        hu_moments = _log_scale_hu_moments(raw_hu_moments)

        session.shape_features = {
            "area": float(area),
            "perimeter": float(perimeter),
            "aspect_ratio": float(aspect_ratio),
            "extent": float(extent),
            "solidity": float(solidity),
            "circularity": float(circularity),
            "equivalent_diameter": equivalent_diameter,
            "hu_moments": hu_moments,
        }
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"area={area:.1f}, circularity={circularity:.3f}",
        )
        logger.info(
            "%s | area=%.1f | perimeter=%.1f | circularity=%.3f | duration=%.4fs | success",
            self.get_name(),
            area,
            perimeter,
            circularity,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Shape Features"

    def get_description(self) -> str:
        return "Computes contour-derived geometric descriptors: area, perimeter, circularity, solidity, and Hu moments."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Quantify the geometry of the segmented region — "
                "different diseases tend to produce differently-shaped "
                "lesions, an independent signal from color and texture."
            ),
            theory=(
                "Area/perimeter-derived ratios (extent, solidity, "
                "circularity) describe how compact vs. irregular a "
                "shape is. Hu moments are normalized central image "
                "moments constructed to stay invariant under "
                "translation, scale, and rotation."
            ),
            working_principle=(
                "1) Require a non-degenerate largest contour. "
                "2) Compute area/perimeter. 3) Compute bounding rect for "
                "aspect ratio/extent. 4) Compute convex hull for "
                "solidity. 5) Derive circularity/equivalent diameter. "
                "6) Compute log-scaled Hu moments."
            ),
            math_intuition=(
                "circularity = 4*pi*area / perimeter^2 (1.0 for a "
                "perfect circle); solidity = area / hull_area; "
                "equivalent_diameter = sqrt(4*area/pi)"
            ),
            advantages=(
                "Fully translation-invariant; most measures are also "
                "scale-invariant.",
                "Directly interpretable, unlike an opaque learned feature.",
            ),
            limitations=(
                "Entirely dependent on segmentation quality.",
                "Describes only the outer boundary, not internal texture/color.",
            ),
            typical_applications=(
                "Distinguishing round fungal leaf-spot lesions from "
                "irregular blight patterns.",
            ),
            complexity="O(N) for a contour of N points",
            opencv_reference="cv2.contourArea, cv2.arcLength, cv2.HuMoments",
        )
