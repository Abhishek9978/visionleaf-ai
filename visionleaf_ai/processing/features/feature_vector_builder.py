"""Feature Vector Builder.

Purpose:
    Combine whichever feature groups (Color, Texture, Shape, Edge)
    have already been computed into one flat, normalized
    `np.ndarray` — the actual input PCA and SVM (Milestone 8) will
    consume. This is the seam between "images" and "numbers": every
    stage before this one in the pipeline produces images or
    intermediate dicts; every stage after Milestone 8 only ever
    touches `session.feature_vector`.

Theory:
    PCA and SVM both require a fixed-length numeric vector per sample.
    Concatenating each feature group in a fixed, documented order (see
    `_common.flatten_*`) guarantees that vector length and meaning are
    consistent across every image processed, which is the hard
    requirement those downstream algorithms impose. Min-max
    normalization to `[0, 1]` is applied per-vector (not per-dataset,
    since no fitted, dataset-level scaler exists at this stage in the
    pipeline) so that features on very different natural scales — a
    Hu moment vs. a raw pixel-count area vs. a normalized histogram
    bin — don't let one group dominate purely due to units. Milestone
    8 may still apply its own dataset-level `StandardScaler` on top of
    this; that's a separate, later concern from "make one image's
    features internally comparable," which is this stage's job.

Working Principle:
    1. Check that at least one feature group has been computed.
    2. Flatten each present group into a list of floats, in the fixed
       order: Color, Texture (GLCM then LBP), Shape, Edge.
    3. Concatenate into one raw vector.
    4. Min-max normalize that vector to `[0, 1]`.
    5. Store the result in `session.feature_vector`.

Math intuition:
    normalized = (raw - min(raw)) / (max(raw) - min(raw) + epsilon)

Advantages:
    - Guarantees a consistent, documented feature order and length.
    - Per-vector min-max normalization prevents any single
      large-magnitude feature (e.g. raw pixel area) from dominating a
      distance-based or margin-based classifier purely due to scale.

Limitations:
    - The final vector's length depends on which feature groups were
      actually run before this stage — running Color + Shape produces
      a shorter vector than Color + Texture + Shape + Edge. Callers
      building a dataset for Milestone 8 must run the same set of
      extractors, with the same parameters, on every image.
    - Per-vector (not per-dataset) normalization means this stage
      alone cannot correct for systematic differences between images
      (e.g. one batch photographed brighter than another) — that's a
      dataset-level concern for Milestone 8, not this stage.

Typical Applications:
    - The final step of every feature-extraction run before handing a
      sample to PCA/SVM training or inference.

Complexity:
    O(F) where F is the total number of extracted feature values —
    linear in the size of the already-computed feature dicts.

References:
    Standard practice in classical (pre-deep-learning) computer vision
    pipelines — see e.g. Duda, Hart & Stork, "Pattern Classification,"
    on feature vector construction and normalization for classifiers.

Dependencies:
    numpy; visionleaf_ai.core; visionleaf_ai.processing.pipeline;
    visionleaf_ai.processing.features._common.

Public classes:
    FeatureVectorBuilderParams, FeatureVectorBuilder
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.features._common import (
    flatten_color_features,
    flatten_edge_features,
    flatten_shape_features,
    flatten_texture_features,
)
from visionleaf_ai.processing.pipeline.registry import register_algorithm
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage

logger = get_logger(__name__)

_EPSILON = 1e-12


@dataclass(frozen=True)
class FeatureVectorBuilderParams:
    """Parameters for `FeatureVectorBuilder`.

    Attributes:
        normalize: Whether to min-max normalize the concatenated
            vector to [0, 1]. Disabling it returns the raw
            concatenation, useful for inspecting each feature's
            natural scale during development.
    """

    normalize: bool = True


@register_algorithm("feature_vector_builder")
class FeatureVectorBuilder(PipelineStage):
    """Combine every computed feature group into one flat, normalized vector."""

    stage_key = "features"

    def __init__(self, params: FeatureVectorBuilderParams | None = None, **kwargs) -> None:
        self.params = params or FeatureVectorBuilderParams(**kwargs)

    def validate(self, session: ImageSession) -> None:
        super().validate(session)
        if not any(
            (
                session.color_features,
                session.texture_features,
                session.shape_features,
                session.edge_features,
            )
        ):
            raise ValidationError(
                "No features have been extracted yet",
                details="run at least one of Color/Texture/Shape/Edge Features first",
            )

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        started = time.perf_counter()

        values: list[float] = []
        groups_included: list[str] = []
        if session.color_features:
            values.extend(flatten_color_features(session.color_features))
            groups_included.append("color")
        if session.texture_features:
            values.extend(flatten_texture_features(session.texture_features))
            groups_included.append("texture")
        if session.shape_features:
            values.extend(flatten_shape_features(session.shape_features))
            groups_included.append("shape")
        if session.edge_features:
            values.extend(flatten_edge_features(session.edge_features))
            groups_included.append("edge")

        raw_vector = np.array(values, dtype=np.float64)
        if self.params.normalize:
            value_range = raw_vector.max() - raw_vector.min()
            feature_vector = (raw_vector - raw_vector.min()) / (value_range + _EPSILON)
        else:
            feature_vector = raw_vector

        session.feature_vector = feature_vector
        duration = time.perf_counter() - started
        session.record_event(
            self.stage_key,
            self.get_name(),
            details=f"length={len(feature_vector)}, groups={'+'.join(groups_included)}",
        )
        logger.info(
            "%s | length=%d | groups=%s | duration=%.4fs | success",
            self.get_name(),
            len(feature_vector),
            groups_included,
            duration,
        )
        return session

    def get_name(self) -> str:
        return "Feature Vector Builder"

    def get_description(self) -> str:
        return "Combines every computed feature group into one flat, normalized numeric vector, ready for PCA/SVM."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose=(
                "Combine whichever feature groups have been computed "
                "into one flat, normalized vector — the actual input "
                "PCA and SVM will consume."
            ),
            theory=(
                "PCA and SVM require a fixed-length numeric vector per "
                "sample. Concatenating each group in a fixed, documented "
                "order guarantees consistent vector length and meaning "
                "across every image. Per-vector min-max normalization "
                "keeps features on very different natural scales from "
                "dominating purely due to units."
            ),
            working_principle=(
                "1) Check at least one feature group exists. 2) Flatten "
                "each present group in fixed order (Color, Texture, "
                "Shape, Edge). 3) Concatenate. 4) Min-max normalize. "
                "5) Store the result."
            ),
            math_intuition="normalized = (raw - min(raw)) / (max(raw) - min(raw) + epsilon)",
            advantages=(
                "Guarantees a consistent, documented feature order and length.",
                "Prevents any single large-magnitude feature from "
                "dominating purely due to scale.",
            ),
            limitations=(
                "Vector length depends on which extractors actually ran "
                "— callers must run the same set on every image in a dataset.",
                "Per-vector normalization can't correct for systematic "
                "differences between images — that's a dataset-level "
                "concern for Milestone 8.",
            ),
            typical_applications=(
                "The final step before handing a sample to PCA/SVM "
                "training or inference.",
            ),
            complexity="O(F) in the total number of extracted feature values",
        )
