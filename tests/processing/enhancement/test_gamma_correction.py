"""Tests for visionleaf_ai.processing.enhancement.gamma_correction.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.enhancement.gamma_correction.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.enhancement.gamma_correction import (
    GammaCorrection,
    GammaParams,
)


def test_gamma_above_one_brightens_midtones(loaded_session):
    original = loaded_session.active_image.copy()
    stage = GammaCorrection(GammaParams(gamma=2.0))
    result = stage.process(loaded_session)
    assert result.current_image.astype(float).mean() > original.astype(float).mean()


def test_gamma_below_one_darkens_midtones(loaded_session):
    original = loaded_session.active_image.copy()
    stage = GammaCorrection(GammaParams(gamma=0.5))
    result = stage.process(loaded_session)
    assert result.current_image.astype(float).mean() < original.astype(float).mean()


def test_gamma_equal_one_is_a_near_no_op(loaded_session):
    original = loaded_session.active_image.copy()
    stage = GammaCorrection(GammaParams(gamma=1.0))
    result = stage.process(loaded_session)
    np.testing.assert_allclose(result.current_image, original, atol=1)


def test_gamma_output_stays_in_valid_range(loaded_session):
    stage = GammaCorrection(GammaParams(gamma=5.0))
    result = stage.process(loaded_session)
    assert result.current_image.min() >= 0
    assert result.current_image.max() <= 255


def test_gamma_records_history_event(loaded_session):
    stage = GammaCorrection(GammaParams(gamma=1.5))
    result = stage.process(loaded_session)
    assert result.processing_history[0].label == "Gamma Correction"
    assert "gamma=1.5" in result.processing_history[0].details


def test_gamma_rejects_zero(loaded_session):
    stage = GammaCorrection(GammaParams(gamma=0.0))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_gamma_rejects_negative(loaded_session):
    stage = GammaCorrection(GammaParams(gamma=-1.0))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_gamma_rejects_missing_image(empty_session):
    stage = GammaCorrection(GammaParams(gamma=1.5))
    with pytest.raises(ValidationError):
        stage.process(empty_session)


def test_gamma_works_on_grayscale(loaded_grayscale_session):
    stage = GammaCorrection(GammaParams(gamma=1.5))
    result = stage.process(loaded_grayscale_session)
    assert result.current_image.ndim == 2


def test_gamma_get_info_structure():
    info = GammaCorrection().get_info()
    assert info.name == "Gamma Correction"
    assert info.complexity is not None
