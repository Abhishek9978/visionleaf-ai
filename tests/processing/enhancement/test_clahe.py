"""Tests for visionleaf_ai.processing.enhancement.clahe.

Dependencies:
    pytest; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.enhancement.clahe.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.enhancement.clahe import CLAHEEnhancement, CLAHEParams


def test_clahe_runs_successfully_on_rgb(loaded_session):
    result = CLAHEEnhancement(CLAHEParams()).process(loaded_session)
    assert result.current_image.shape == loaded_session.original_image.shape


def test_clahe_runs_successfully_on_grayscale(loaded_grayscale_session):
    result = CLAHEEnhancement(CLAHEParams()).process(loaded_grayscale_session)
    assert result.current_image.ndim == 2


def test_clahe_output_stays_in_valid_range(loaded_session):
    result = CLAHEEnhancement(CLAHEParams(clip_limit=4.0)).process(loaded_session)
    assert result.current_image.min() >= 0
    assert result.current_image.max() <= 255


def test_clahe_records_history_event(loaded_session):
    result = CLAHEEnhancement(CLAHEParams(clip_limit=2.0)).process(loaded_session)
    assert result.processing_history[0].label == "CLAHE"
    assert "clip_limit=2.0" in result.processing_history[0].details


def test_clahe_rejects_zero_clip_limit(loaded_session):
    stage = CLAHEEnhancement(CLAHEParams(clip_limit=0.0))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_clahe_rejects_negative_clip_limit(loaded_session):
    stage = CLAHEEnhancement(CLAHEParams(clip_limit=-1.0))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_clahe_rejects_non_positive_tile_grid_size(loaded_session):
    stage = CLAHEEnhancement(CLAHEParams(tile_grid_size=(0, 8)))
    with pytest.raises(ValidationError):
        stage.process(loaded_session)


def test_clahe_rejects_missing_image(empty_session):
    stage = CLAHEEnhancement(CLAHEParams())
    with pytest.raises(ValidationError):
        stage.process(empty_session)


def test_clahe_default_params_via_kwargs():
    stage = CLAHEEnhancement(clip_limit=3.0)
    assert stage.params.clip_limit == 3.0
    assert stage.params.tile_grid_size == (8, 8)


def test_clahe_get_info_structure():
    info = CLAHEEnhancement().get_info()
    assert info.name == "CLAHE"
    assert info.category == "enhancement"
