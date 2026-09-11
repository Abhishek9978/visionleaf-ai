"""Tests for adaptive_mean_threshold.py and adaptive_gaussian_threshold.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.segmentation.{adaptive_mean_threshold,adaptive_gaussian_threshold}.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.segmentation.adaptive_gaussian_threshold import (
    AdaptiveGaussianThreshold,
    AdaptiveGaussianThresholdParams,
)
from visionleaf_ai.processing.segmentation.adaptive_mean_threshold import (
    AdaptiveMeanThreshold,
    AdaptiveMeanThresholdParams,
)


def test_adaptive_mean_produces_binary_output(loaded_session):
    result = AdaptiveMeanThreshold().process(loaded_session)
    assert set(np.unique(result.binary_image).tolist()) <= {0, 255}


def test_adaptive_mean_records_history(loaded_session):
    result = AdaptiveMeanThreshold(
        AdaptiveMeanThresholdParams(block_size=15, C=3)
    ).process(loaded_session)
    assert "block_size=15" in result.processing_history[-1].details


def test_adaptive_mean_rejects_even_block_size(loaded_session):
    with pytest.raises(ValidationError):
        AdaptiveMeanThreshold(AdaptiveMeanThresholdParams(block_size=10)).process(loaded_session)


def test_adaptive_mean_rejects_block_size_too_small(loaded_session):
    with pytest.raises(ValidationError):
        AdaptiveMeanThreshold(AdaptiveMeanThresholdParams(block_size=1)).process(loaded_session)


def test_adaptive_mean_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        AdaptiveMeanThreshold().process(empty_session)


def test_adaptive_mean_works_on_grayscale(loaded_grayscale_session):
    result = AdaptiveMeanThreshold().process(loaded_grayscale_session)
    assert result.binary_image.ndim == 2


def test_adaptive_gaussian_produces_binary_output(loaded_session):
    result = AdaptiveGaussianThreshold().process(loaded_session)
    assert set(np.unique(result.binary_image).tolist()) <= {0, 255}


def test_adaptive_gaussian_rejects_even_block_size(loaded_session):
    with pytest.raises(ValidationError):
        AdaptiveGaussianThreshold(
            AdaptiveGaussianThresholdParams(block_size=8)
        ).process(loaded_session)


def test_adaptive_gaussian_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        AdaptiveGaussianThreshold().process(empty_session)


def test_adaptive_gaussian_get_info_structure():
    info = AdaptiveGaussianThreshold().get_info()
    assert info.name == "Adaptive Gaussian Threshold"
    assert info.opencv_reference is not None
