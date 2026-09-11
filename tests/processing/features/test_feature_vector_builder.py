"""Tests for feature_vector_builder.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.features.{color_features,edge_features,
    feature_vector_builder}.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.features.color_features import ColorFeatures
from visionleaf_ai.processing.features.edge_features import EdgeFeatures
from visionleaf_ai.processing.features.feature_vector_builder import (
    FeatureVectorBuilder,
    FeatureVectorBuilderParams,
)
from visionleaf_ai.processing.features.glcm_features import GLCMFeatures
from visionleaf_ai.processing.features.shape_features import ShapeFeatures


def test_builder_rejects_when_no_features_extracted(fully_prepared_session):
    with pytest.raises(ValidationError):
        FeatureVectorBuilder().process(fully_prepared_session)


def test_builder_succeeds_with_only_one_group(fully_prepared_session):
    session = ColorFeatures().process(fully_prepared_session)
    result = FeatureVectorBuilder().process(session)
    assert result.feature_vector is not None
    assert len(result.feature_vector) > 0


def test_builder_vector_length_grows_with_more_groups(fully_prepared_session):
    session_a = ColorFeatures().process(fully_prepared_session)
    length_one_group = len(FeatureVectorBuilder().process(session_a).feature_vector)

    session_b = ColorFeatures().process(fully_prepared_session)
    session_b = EdgeFeatures().process(session_b)
    length_two_groups = len(FeatureVectorBuilder().process(session_b).feature_vector)

    assert length_two_groups > length_one_group


def test_builder_normalizes_to_unit_range_by_default(fully_prepared_session):
    session = ColorFeatures().process(fully_prepared_session)
    session = EdgeFeatures().process(session)
    result = FeatureVectorBuilder().process(session)
    assert result.feature_vector.min() >= 0.0
    assert result.feature_vector.max() <= 1.0 + 1e-9


def test_builder_can_skip_normalization(fully_prepared_session):
    session = ColorFeatures().process(fully_prepared_session)
    result = FeatureVectorBuilder(FeatureVectorBuilderParams(normalize=False)).process(session)
    # Raw RGB means are in [0, 255], not [0, 1] — confirms normalization was skipped.
    assert result.feature_vector.max() > 1.0


def test_builder_includes_all_four_groups_when_all_present(fully_prepared_session):
    session = ColorFeatures().process(fully_prepared_session)
    session = GLCMFeatures().process(session)
    session = ShapeFeatures().process(session)
    session = EdgeFeatures().process(session)
    result = FeatureVectorBuilder().process(session)
    # 12 (color stats) + 96 (3x32 hist) + 6 (glcm) + 14 (shape) + 3 (edge) = 131
    assert len(result.feature_vector) == 12 + 96 + 6 + 14 + 3


def test_builder_records_history_with_length_and_groups(fully_prepared_session):
    session = ColorFeatures().process(fully_prepared_session)
    result = FeatureVectorBuilder().process(session)
    details = result.processing_history[-1].details
    assert "length=" in details
    assert "color" in details


def test_builder_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        FeatureVectorBuilder().process(empty_session)


def test_builder_get_info_structure():
    info = FeatureVectorBuilder().get_info()
    assert info.name == "Feature Vector Builder"
    assert info.category == "features"
