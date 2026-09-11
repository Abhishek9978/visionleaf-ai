"""Tests for visionleaf_ai.processing.enhancement.contrast.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.enhancement.contrast.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.enhancement.contrast import (
    ContrastAdjustment,
    ContrastParams,
)


def test_contrast_above_one_increases_spread(loaded_session):
    original = loaded_session.active_image.copy()
    stage = ContrastAdjustment(ContrastParams(alpha=2.0))
    result = stage.process(loaded_session)
    assert result.current_image.astype(float).std() > original.astype(float).std()


def test_contrast_below_one_decreases_spread(loaded_session):
    original = loaded_session.active_image.copy()
    stage = ContrastAdjustment(ContrastParams(alpha=0.5))
    result = stage.process(loaded_session)
    assert result.current_image.astype(float).std() < original.astype(float).std()


def test_contrast_alpha_one_is_a_near_no_op(loaded_session):
    original = loaded_session.active_image.copy()
    stage = ContrastAdjustment(ContrastParams(alpha=1.0))
    result = stage.process(loaded_session)
    np.testing.assert_array_equal(result.current_image, original)


def test_contrast_clips_at_255(loaded_session):
    stage = ContrastAdjustment(ContrastParams(alpha=10.0))
    result = stage.process(loaded_session)
    assert result.current_image.max() <= 255


def test_contrast_records_history_event(loaded_session):
    stage = ContrastAdjustment(ContrastParams(alpha=1.5))
    result = stage.process(loaded_session)
    assert result.processing_history[0].label == "Contrast Adjustment"


def test_contrast_rejects_zero_alpha(loaded_session):
    stage = ContrastAdjustment(ContrastParams(alpha=0.0))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_contrast_rejects_negative_alpha(loaded_session):
    stage = ContrastAdjustment(ContrastParams(alpha=-1.0))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_contrast_rejects_missing_image(empty_session):
    stage = ContrastAdjustment(ContrastParams(alpha=1.5))
    with pytest.raises(ValidationError):
        stage.process(empty_session)


def test_contrast_works_on_grayscale(loaded_grayscale_session):
    stage = ContrastAdjustment(ContrastParams(alpha=1.5))
    result = stage.process(loaded_grayscale_session)
    assert result.current_image.ndim == 2


def test_contrast_get_info_structure():
    info = ContrastAdjustment().get_info()
    assert info.name == "Contrast Adjustment"
    assert info.category == "enhancement"
