"""Tests for glcm_features.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.features.glcm_features.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.features.glcm_features import GLCMFeatures, GLCMFeaturesParams


def test_glcm_features_produces_expected_properties(fully_prepared_session):
    result = GLCMFeatures().process(fully_prepared_session)
    glcm = result.texture_features["glcm"]
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm"):
        assert prop in glcm


def test_glcm_features_uniform_image_has_zero_contrast():
    # A perfectly flat image has no intensity variation between
    # neighboring pixels, so GLCM contrast must be exactly 0.
    flat_mask = np.full((30, 30), 200, dtype=np.uint8)
    image = np.stack([flat_mask] * 3, axis=-1)
    session = ImageSession(
        original_image=image.copy(), current_image=image.copy(), roi_image=image.copy()
    )
    result = GLCMFeatures().process(session)
    assert result.texture_features["glcm"]["contrast"] == pytest.approx(0.0, abs=1e-6)


def test_glcm_features_merges_with_existing_texture_features(fully_prepared_session):
    fully_prepared_session.texture_features = {"lbp": {"histogram_normalized": [0.1]}}
    result = GLCMFeatures().process(fully_prepared_session)
    assert "lbp" in result.texture_features
    assert "glcm" in result.texture_features


def test_glcm_features_records_history(fully_prepared_session):
    result = GLCMFeatures().process(fully_prepared_session)
    assert result.processing_history[-1].label == "GLCM Texture Features"


def test_glcm_features_rejects_missing_segmentation(session_without_segmentation):
    with pytest.raises(ValidationError):
        GLCMFeatures().process(session_without_segmentation)


def test_glcm_features_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        GLCMFeatures().process(empty_session)


def test_glcm_features_rejects_invalid_distance(fully_prepared_session):
    with pytest.raises(ValidationError):
        GLCMFeatures(GLCMFeaturesParams(distance=0)).process(fully_prepared_session)


def test_glcm_features_rejects_invalid_levels(fully_prepared_session):
    with pytest.raises(ValidationError):
        GLCMFeatures(GLCMFeaturesParams(levels=1)).process(fully_prepared_session)


def test_glcm_features_get_info_structure():
    info = GLCMFeatures().get_info()
    assert info.name == "GLCM Texture Features"
    assert "Haralick" in info.purpose or "Haralick" in info.theory
