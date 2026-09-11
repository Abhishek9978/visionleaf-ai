"""Tests for otsu_threshold.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.segmentation.otsu_threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.segmentation.otsu_threshold import (
    OtsuThreshold,
    OtsuThresholdParams,
)


def test_otsu_finds_a_reasonable_split_on_bimodal_image(loaded_session):
    result = OtsuThreshold().process(loaded_session)
    assert result.binary_image[20, 30] == 255  # bright square
    assert result.binary_image[2, 2] == 0  # dark field


def test_otsu_records_the_chosen_threshold_in_history(loaded_session):
    result = OtsuThreshold().process(loaded_session)
    assert "auto_threshold=" in result.processing_history[-1].details


def test_otsu_produces_two_level_output(loaded_session):
    result = OtsuThreshold().process(loaded_session)
    assert set(np.unique(result.binary_image).tolist()) <= {0, 255}


def test_otsu_rejects_invalid_max_value(loaded_session):
    with pytest.raises(ValidationError):
        OtsuThreshold(OtsuThresholdParams(max_value=300)).process(loaded_session)


def test_otsu_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        OtsuThreshold().process(empty_session)


def test_otsu_works_on_grayscale(loaded_grayscale_session):
    result = OtsuThreshold().process(loaded_grayscale_session)
    assert result.binary_image.ndim == 2


def test_otsu_get_info_mentions_bimodal():
    info = OtsuThreshold().get_info()
    assert "bimodal" in info.theory.lower()
