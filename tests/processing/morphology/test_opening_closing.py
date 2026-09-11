"""Tests for opening.py and closing.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.morphology.{opening,closing}.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.morphology.closing import Closing, ClosingParams
from visionleaf_ai.processing.morphology.opening import Opening, OpeningParams


def test_opening_removes_small_isolated_speck(loaded_session_with_noisy_binary):
    result = Opening(OpeningParams(kernel_size=5)).process(loaded_session_with_noisy_binary)
    # The small 3x3 speck at [4:7, 4:7] should be removed by a 5x5 opening.
    assert result.segmentation_mask[5, 5] == 0


def test_opening_roughly_preserves_main_region_size(
    loaded_session_with_noisy_binary, noisy_mask
):
    result = Opening(OpeningParams(kernel_size=5)).process(loaded_session_with_noisy_binary)
    # A point in the main square away from the internal hole should
    # still be foreground (the hole itself, at [28:32, 28:32], is
    # correctly left unfilled by Opening — that's Closing's job).
    assert result.segmentation_mask[20, 20] == 255


def test_opening_rejects_even_kernel_size(loaded_session_with_binary):
    with pytest.raises(ValidationError):
        Opening(OpeningParams(kernel_size=4)).process(loaded_session_with_binary)


def test_opening_rejects_missing_mask(loaded_session_without_mask):
    with pytest.raises(ValidationError):
        Opening().process(loaded_session_without_mask)


def test_closing_fills_small_internal_hole(loaded_session_with_noisy_binary):
    result = Closing(ClosingParams(kernel_size=7)).process(loaded_session_with_noisy_binary)
    # The 4x4 hole at [28:32, 28:32] should be filled by a 7x7 closing.
    assert result.segmentation_mask[30, 30] == 255


def test_closing_records_history(loaded_session_with_binary):
    result = Closing(ClosingParams(kernel_size=3, iterations=2)).process(
        loaded_session_with_binary
    )
    assert result.processing_history[-1].label == "Closing"


def test_closing_rejects_non_positive_iterations(loaded_session_with_binary):
    with pytest.raises(ValidationError):
        Closing(ClosingParams(iterations=-1)).process(loaded_session_with_binary)


def test_closing_rejects_missing_mask(loaded_session_without_mask):
    with pytest.raises(ValidationError):
        Closing().process(loaded_session_without_mask)


def test_closing_get_info_structure():
    info = Closing().get_info()
    assert info.name == "Closing"
    assert info.math_intuition
