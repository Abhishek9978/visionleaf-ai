"""CLAHE (Contrast Limited Adaptive Histogram Equalization).

Purpose:
    Improve local contrast without over-amplifying noise or blowing
    out already-bright regions — the adaptive counterpart to global
    Histogram Equalization.

Theory:
    Standard histogram equalization computes one histogram for the
    entire image. CLAHE instead divides the image into a grid of small
    tiles, equalizes each tile's histogram independently, and clips the
    histogram at a configurable limit before equalizing (preventing
    noise in near-uniform tiles from being amplified into visible
    artifacts). Tile borders are then smoothed with bilinear
    interpolation so the result doesn't show a visible grid pattern.
    As with Histogram Equalization, color images are processed in the
    L channel of LAB color space to avoid distorting color balance.

Math intuition:
    Within each tile, CLAHE equalizes as usual, but any histogram bin
    taller than `clip_limit` has its excess redistributed evenly across
    all bins first — "clipping" the contrast amplification so a tile
    that's almost entirely one intensity doesn't get stretched into
    visible noise.

Advantages:
    - Handles images with uneven lighting far better than global
      equalization (e.g. one side of a leaf in shadow, the other in sun).
    - The clip limit directly controls how much noise amplification is
      tolerated.

Limitations:
    - Two parameters to tune (clip limit, tile grid size) instead of
      zero for global equalization.
    - More computationally expensive than a single global histogram.

Complexity:
    O(H x W) — each tile is processed independently and interpolation
    is a constant-factor overhead.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core; visionleaf_ai.processing.pipeline.

Public classes:
    CLAHEParams, CLAHEEnhancement
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
class CLAHEParams:
    """Parameters for `CLAHEEnhancement`.

    Attributes:
        clip_limit: Threshold for contrast limiting. Must be strictly
            positive; higher values allow more contrast (and more
            noise amplification) per tile.
        tile_grid_size: (columns, rows) the image is divided into.
            Both must be positive integers.
    """

    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)


@register_algorithm("clahe")
class CLAHEEnhancement(PipelineStage):
    """Apply Contrast Limited Adaptive Histogram Equalization."""

    stage_key = "enhancement"

    def __init__(self, params: CLAHEParams | None = None, **kwargs) -> None:
        self.params = params or CLAHEParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        if self.params.clip_limit <= 0:
            raise ValidationError(
                "clip_limit must be strictly positive",
                details=f"got {self.params.clip_limit!r}",
            )
        cols, rows = self.params.tile_grid_size
        if cols <= 0 or rows <= 0:
            raise ValidationError(
                "tile_grid_size dimensions must be positive integers",
                details=f"got {self.params.tile_grid_size!r}",
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()
        image = session.require_active_image()

        clahe = cv2.createCLAHE(
            clipLimit=self.params.clip_limit, tileGridSize=self.params.tile_grid_size
        )

        if image.ndim == 2:
            result = clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_equalized = clahe.apply(l_channel)
            result = cv2.cvtColor(
                cv2.merge([l_equalized, a_channel, b_channel]), cv2.COLOR_LAB2RGB
            )

        session.current_image = result
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=(
                f"clip_limit={self.params.clip_limit}, "
                f"tile_grid_size={self.params.tile_grid_size}"
            ),
        )
        logger.info(
            "%s | clip_limit=%.2f | tile_grid_size=%s | image=%s | duration=%.4fs | success",
            self.get_name(),
            self.params.clip_limit,
            self.params.tile_grid_size,
            image.shape,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "CLAHE"

    def get_description(self) -> str:
        return "Applies local, contrast-limited adaptive histogram equalization to improve contrast in unevenly-lit images."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Improve local contrast without over-amplifying noise — "
                "the adaptive counterpart to global Histogram Equalization."
            ),
            theory=(
                "Divides the image into a grid of tiles, equalizes each "
                "tile's histogram independently after clipping it at a "
                "configurable limit, then blends tile borders with "
                "bilinear interpolation. Color images are processed in "
                "the L channel of LAB space."
            ),
            math_intuition=(
                "Within each tile, any histogram bin taller than "
                "clip_limit has its excess redistributed evenly across "
                "all bins before equalizing, capping noise amplification."
            ),
            advantages=(
                "Handles unevenly-lit images far better than global "
                "equalization.",
                "clip_limit directly controls noise-amplification tolerance.",
            ),
            limitations=(
                "Two parameters to tune instead of zero.",
                "More computationally expensive than global equalization.",
            ),
            complexity="O(H x W)",
        )
