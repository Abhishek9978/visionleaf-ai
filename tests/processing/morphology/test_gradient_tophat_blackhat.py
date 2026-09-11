"""Tests for gradient.py, top_hat.py, and black_hat.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.morphology.{gradient,top_hat,black_hat}.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.morphology.black_hat import BlackHat, BlackHatParams
from visionleaf_ai.processing.morphology.gradient import (
    MorphologicalGradient,
    MorphologicalGradientParams,
)
from visionleaf_ai.processing.morphology.top_hat import TopHat, TopHatParams


def test_gradient_isolates_only_the_boundary(loaded_session_with_binary):
    result = MorphologicalGradient(MorphologicalGradientParams(kernel_size=3)).process(
        loaded_session_with_binary
    )
    # Deep interior of the solid square should be 0 (no boundary there)...
    assert result.segmentation_mask[32, 32] == 0
    # ...but right at the square's edge, the gradient should be nonzero.
    assert result.segmentation_mask[16, 32] == 255


def test_gradient_rejects_missing_mask(loaded_session_without_mask):
    with pytest.raises(ValidationError):
        MorphologicalGradient().process(loaded_session_without_mask)


def test_gradient_get_info_structure():
    info = MorphologicalGradient().get_info()
    assert info.name == "Morphological Gradient"


def test_top_hat_isolates_small_bright_speck(loaded_session_with_noisy_binary):
    result = TopHat(TopHatParams(kernel_size=9)).process(loaded_session_with_noisy_binary)
    # The isolated 3x3 speck should appear in the top-hat result...
    assert result.segmentation_mask[5, 5] == 255
    # ...but the large solid square's deep interior should not.
    assert result.segmentation_mask[30, 30] == 0


def test_top_hat_rejects_missing_mask(loaded_session_without_mask):
    with pytest.raises(ValidationError):
        TopHat().process(loaded_session_without_mask)


def test_top_hat_rejects_even_kernel_size(loaded_session_with_binary):
    with pytest.raises(ValidationError):
        TopHat(TopHatParams(kernel_size=10)).process(loaded_session_with_binary)


def test_black_hat_isolates_small_dark_hole(loaded_session_with_noisy_binary):
    result = BlackHat(BlackHatParams(kernel_size=9)).process(loaded_session_with_noisy_binary)
    # The small internal hole should appear in the black-hat result.
    assert result.segmentation_mask[30, 30] == 255


def test_black_hat_rejects_missing_mask(loaded_session_without_mask):
    with pytest.raises(ValidationError):
        BlackHat().process(loaded_session_without_mask)


def test_black_hat_rejects_non_positive_iterations(loaded_session_with_binary):
    with pytest.raises(ValidationError):
        BlackHat(BlackHatParams(iterations=0)).process(loaded_session_with_binary)


def test_black_hat_get_info_structure():
    info = BlackHat().get_info()
    assert info.name == "Black Hat"
    assert info.category == "morphology"
