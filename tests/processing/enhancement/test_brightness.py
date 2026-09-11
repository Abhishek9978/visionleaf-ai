"""Tests for visionleaf_ai.processing.enhancement.brightness.

Purpose:
    Verify normal operation, invalid parameters, missing image, and
    clipping edge cases for Brightness Adjustment.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.enhancement.brightness.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.enhancement.brightness import (
    BrightnessAdjustment,
    BrightnessParams,
)


def test_brightness_increases_pixel_values(loaded_session):
    original = loaded_session.active_image.copy()
    stage = BrightnessAdjustment(BrightnessParams(delta=20))

    result = stage.process(loaded_session)

    assert result.current_image.mean() > original.astype(float).mean()


def test_brightness_negative_delta_decreases_pixel_values(loaded_session):
    original = loaded_session.active_image.copy()
    stage = BrightnessAdjustment(BrightnessParams(delta=-20))

    result = stage.process(loaded_session)

    assert result.current_image.mean() < original.astype(float).mean()


def test_brightness_clips_at_255(loaded_session):
    stage = BrightnessAdjustment(BrightnessParams(delta=255))
    result = stage.process(loaded_session)
    assert result.current_image.max() <= 255


def test_brightness_clips_at_0(loaded_session):
    stage = BrightnessAdjustment(BrightnessParams(delta=-255))
    result = stage.process(loaded_session)
    assert result.current_image.min() >= 0


def test_brightness_records_history_event(loaded_session):
    stage = BrightnessAdjustment(BrightnessParams(delta=10))
    result = stage.process(loaded_session)
    assert len(result.processing_history) == 1
    assert result.processing_history[0].label == "Brightness Adjustment"
    assert "delta=10" in result.processing_history[0].details


def test_brightness_default_params_via_kwargs():
    stage = BrightnessAdjustment(delta=5)
    assert stage.params.delta == 5


def test_brightness_rejects_out_of_range_delta(loaded_session):
    stage = BrightnessAdjustment(BrightnessParams(delta=300))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_brightness_rejects_missing_image(empty_session):
    stage = BrightnessAdjustment(BrightnessParams(delta=10))
    with pytest.raises(ValidationError):
        stage.process(empty_session)


def test_brightness_works_on_grayscale(loaded_grayscale_session):
    stage = BrightnessAdjustment(BrightnessParams(delta=15))
    result = stage.process(loaded_grayscale_session)
    assert result.current_image is not None
    assert result.current_image.ndim == 2


def test_brightness_zero_delta_is_a_near_no_op(loaded_session):
    original = loaded_session.active_image.copy()
    stage = BrightnessAdjustment(BrightnessParams(delta=0))
    result = stage.process(loaded_session)
    np.testing.assert_array_equal(result.current_image, original)


def test_brightness_get_name_and_description_and_info():
    stage = BrightnessAdjustment()
    assert stage.get_name() == "Brightness Adjustment"
    assert isinstance(stage.get_description(), str) and stage.get_description()
    info = stage.get_info()
    assert info.category == "enhancement"
    assert info.purpose and info.theory and info.math_intuition
    assert len(info.advantages) > 0
    assert len(info.limitations) > 0
