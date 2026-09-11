"""Tests for lbp_features.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.features.lbp_features.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.features.lbp_features import LBPFeatures, LBPFeaturesParams


def test_lbp_features_histogram_length_matches_n_points(fully_prepared_session):
    result = LBPFeatures(LBPFeaturesParams(n_points=8)).process(fully_prepared_session)
    assert len(result.texture_features["lbp"]["histogram"]) == 10  # n_points + 2


def test_lbp_features_normalized_histogram_sums_to_one(fully_prepared_session):
    result = LBPFeatures().process(fully_prepared_session)
    total = sum(result.texture_features["lbp"]["histogram_normalized"])
    assert abs(total - 1.0) < 1e-6


def test_lbp_features_merges_with_existing_texture_features(fully_prepared_session):
    fully_prepared_session.texture_features = {"glcm": {"contrast": 1.0}}
    result = LBPFeatures().process(fully_prepared_session)
    assert "glcm" in result.texture_features
    assert "lbp" in result.texture_features


def test_lbp_features_records_history(fully_prepared_session):
    result = LBPFeatures().process(fully_prepared_session)
    assert result.processing_history[-1].label == "Local Binary Pattern"


def test_lbp_features_rejects_missing_segmentation(session_without_segmentation):
    with pytest.raises(ValidationError):
        LBPFeatures().process(session_without_segmentation)


def test_lbp_features_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        LBPFeatures().process(empty_session)


def test_lbp_features_rejects_invalid_radius(fully_prepared_session):
    with pytest.raises(ValidationError):
        LBPFeatures(LBPFeaturesParams(radius=0)).process(fully_prepared_session)


def test_lbp_features_rejects_invalid_n_points(fully_prepared_session):
    with pytest.raises(ValidationError):
        LBPFeatures(LBPFeaturesParams(n_points=0)).process(fully_prepared_session)


def test_lbp_features_get_info_structure():
    info = LBPFeatures().get_info()
    assert info.name == "Local Binary Pattern"
    assert info.category == "features"
