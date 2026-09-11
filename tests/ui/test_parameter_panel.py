"""Tests for visionleaf_ai.ui.components.parameter_panel.

Purpose:
    Verify ALGORITHM_PARAM_SPECS is internally consistent and matches
    every algorithm actually registered — this is the guard against
    the UI parameter schema silently drifting out of sync with the
    real Pipeline Engine algorithms (e.g. a renamed constructor
    argument that the UI would still send under the old name).

Dependencies:
    pytest; visionleaf_ai.processing.pipeline; visionleaf_ai.ui.components.parameter_panel.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.processing.pipeline import AlgorithmRegistry, get_pipeline_engine
from visionleaf_ai.ui.components.parameter_panel import ALGORITHM_PARAM_SPECS


@pytest.fixture(autouse=True)
def _ensure_registry_populated():
    get_pipeline_engine()  # triggers bootstrap.ensure_registered()


def test_every_enhancement_algorithm_has_a_param_spec_entry():
    for name in AlgorithmRegistry.list_algorithms("enhancement"):
        assert name in ALGORITHM_PARAM_SPECS


def test_every_restoration_algorithm_has_a_param_spec_entry():
    for name in AlgorithmRegistry.list_algorithms("restoration"):
        assert name in ALGORITHM_PARAM_SPECS


def test_every_spec_key_is_a_valid_constructor_kwarg():
    """Each ParamSpec's key must actually be accepted by its algorithm."""
    for algorithm_name, specs in ALGORITHM_PARAM_SPECS.items():
        kwargs = {spec.key: spec.default for spec in specs}
        # tile_grid_size is special-cased to a tuple by the panel, not
        # the raw slider default — substitute a valid tuple here.
        if "tile_grid_size" in kwargs:
            kwargs["tile_grid_size"] = (8, 8)
        instance = AlgorithmRegistry.create(algorithm_name, **kwargs)
        assert instance is not None


def test_odd_slider_defaults_are_actually_odd():
    for specs in ALGORITHM_PARAM_SPECS.values():
        for spec in specs:
            if spec.widget == "odd_slider":
                assert int(spec.default) % 2 == 1


def test_every_spec_has_non_empty_help_text():
    for specs in ALGORITHM_PARAM_SPECS.values():
        for spec in specs:
            assert spec.help.strip()


def test_min_is_less_than_or_equal_to_max_for_every_spec():
    for specs in ALGORITHM_PARAM_SPECS.values():
        for spec in specs:
            assert spec.min_value <= spec.max_value


def test_default_is_within_min_max_range_for_every_spec():
    for specs in ALGORITHM_PARAM_SPECS.values():
        for spec in specs:
            assert spec.min_value <= spec.default <= spec.max_value
