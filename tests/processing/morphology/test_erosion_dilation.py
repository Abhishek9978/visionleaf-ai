"""Tests for erosion.py and dilation.py.

Dependencies:
    pytest; numpy; visionleaf_ai.core.exceptions;
    visionleaf_ai.processing.morphology.{erosion,dilation}.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.morphology.dilation import Dilation, DilationParams
from visionleaf_ai.processing.morphology.erosion import Erosion, ErosionParams


def test_erosion_shrinks_the_foreground_region(loaded_session_with_binary, solid_mask):
    result = Erosion(ErosionParams(kernel_size=5)).process(loaded_session_with_binary)
    assert result.segmentation_mask.sum() < solid_mask.sum()


def test_erosion_records_history_and_clears_contour_results(loaded_session_with_binary):
    loaded_session_with_binary.contours = ["placeholder"]
    result = Erosion(ErosionParams(kernel_size=3, iterations=2)).process(
        loaded_session_with_binary
    )
    assert result.processing_history[-1].label == "Erosion"
    assert "iterations=2" in result.processing_history[-1].details
    assert result.contours is None


def test_erosion_rejects_even_kernel_size(loaded_session_with_binary):
    with pytest.raises(ValidationError):
        Erosion(ErosionParams(kernel_size=4)).process(loaded_session_with_binary)


def test_erosion_rejects_non_positive_iterations(loaded_session_with_binary):
    with pytest.raises(ValidationError):
        Erosion(ErosionParams(iterations=0)).process(loaded_session_with_binary)


def test_erosion_rejects_kernel_larger_than_mask(loaded_session_with_binary):
    with pytest.raises(ValidationError):
        Erosion(ErosionParams(kernel_size=999)).process(loaded_session_with_binary)


def test_erosion_rejects_missing_mask(loaded_session_without_mask):
    with pytest.raises(ValidationError):
        Erosion().process(loaded_session_without_mask)


def test_erosion_rejects_missing_image(empty_session):
    with pytest.raises(ValidationError):
        Erosion().process(empty_session)


def test_dilation_grows_the_foreground_region(loaded_session_with_binary, solid_mask):
    result = Dilation(DilationParams(kernel_size=5)).process(loaded_session_with_binary)
    assert result.segmentation_mask.sum() > solid_mask.sum()


def test_dilation_rejects_even_kernel_size(loaded_session_with_binary):
    with pytest.raises(ValidationError):
        Dilation(DilationParams(kernel_size=6)).process(loaded_session_with_binary)


def test_dilation_rejects_missing_mask(loaded_session_without_mask):
    with pytest.raises(ValidationError):
        Dilation().process(loaded_session_without_mask)


def test_dilation_get_info_structure():
    info = Dilation().get_info()
    assert info.name == "Dilation"
    assert info.category == "morphology"
