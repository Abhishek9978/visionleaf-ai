"""Bounding Box Visualization.

Purpose:
    Draw the detected bounding box directly onto a copy of the image
    — a visual sanity check that segmentation found the right region,
    without needing to inspect raw coordinates.

Theory:
    A pure annotation/drawing operation: overlays a rectangle outline
    (axis-aligned or rotated, matching whichever bounding box type is
    stored) onto the image. Does not alter pixel values anywhere except
    the drawn outline itself.

Working Principle:
    1. Require an existing bounding box.
    2. Draw its outline onto a copy of the active image — a plain
       rectangle for an axis-aligned box, or the four connected corner
       points for a rotated Minimum Area Rectangle.
    3. Store the annotated copy in `session.current_image`, since this
       is a visualization of "the current state," following the same
       convention Enhancement/Restoration use for their output.

Math intuition:
    Not a pixel-transform formula — a direct line-drawing operation
    along the rectangle's edges.

Advantages:
    - Immediate visual confirmation of what the pipeline detected.
    - Works identically for axis-aligned or rotated boxes.

Limitations:
    - Purely cosmetic — provides no information a later stage could
      use computationally (that's what `bounding_box` itself is for).

Typical Applications:
    - Visual QA of segmentation results before proceeding to cropping
      or feature extraction.

Complexity:
    O(perimeter) — drawing four line segments.

OpenCV reference:
    cv2.rectangle(...) or cv2.drawContours(..., [box_points], ...)

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    BoundingBoxVisualizationParams, BoundingBoxVisualization
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
from visionleaf_ai.processing.roi._common import require_bounding_box

logger = get_logger(__name__)


@dataclass(frozen=True)
class BoundingBoxVisualizationParams:
    """Parameters for `BoundingBoxVisualization`.

    Attributes:
        color: RGB outline color, e.g. (255, 0, 0) for red.
        thickness: Outline thickness in pixels; must be a positive integer.
    """

    color: tuple[int, int, int] = (255, 0, 0)
    thickness: int = 2


@register_algorithm("bounding_box_visualization")
class BoundingBoxVisualization(PipelineStage):
    """Draw the detected bounding box onto a copy of the image."""

    stage_key = "roi"

    def __init__(self, params: BoundingBoxVisualizationParams | None = None, **kwargs) -> None:
        self.params = params or BoundingBoxVisualizationParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        require_bounding_box(session)
        if self.params.thickness < 1:
            raise ValidationError(
                "thickness must be a positive integer", details=f"got {self.params.thickness!r}"
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        bounding_box = require_bounding_box(session)
        image = session.require_active_image().copy()
        color = self.params.color
        thickness = self.params.thickness

        if bounding_box["type"] == "axis_aligned":
            x, y, w, h = bounding_box["x"], bounding_box["y"], bounding_box["w"], bounding_box["h"]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        else:
            box_points = np.array(bounding_box["box_points"], dtype=np.int32)
            cv2.drawContours(image, [box_points], 0, color, thickness)

        session.current_image = image
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key, self.get_name(), details=f"type={bounding_box['type']}"
        )
        logger.info(
            "%s | type=%s | duration=%.4fs | success",
            self.get_name(),
            bounding_box["type"],
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Bounding Box Visualization"

    def get_description(self) -> str:
        return "Draws the detected bounding box outline onto a copy of the image."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Draw the detected bounding box directly onto a copy of "
                "the image — a visual sanity check that segmentation "
                "found the right region."
            ),
            theory=(
                "A pure annotation/drawing operation: overlays a "
                "rectangle outline (axis-aligned or rotated) onto the "
                "image, without altering pixel values elsewhere."
            ),
            working_principle=(
                "1) Require an existing bounding box. 2) Draw its "
                "outline onto a copy of the active image. 3) Store the "
                "annotated copy as the current image."
            ),
            math_intuition=(
                "Not a pixel-transform formula — a direct line-drawing "
                "operation along the rectangle's edges."
            ),
            advantages=(
                "Immediate visual confirmation of what the pipeline detected.",
                "Works identically for axis-aligned or rotated boxes.",
            ),
            limitations=(
                "Purely cosmetic — provides no information a later "
                "stage could use computationally.",
            ),
            typical_applications=(
                "Visual QA of segmentation results before cropping or "
                "feature extraction.",
            ),
            complexity="O(perimeter)",
            opencv_reference="cv2.rectangle(...) or cv2.drawContours(...)",
        )
