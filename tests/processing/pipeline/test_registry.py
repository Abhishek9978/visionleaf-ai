"""Tests for visionleaf_ai.processing.pipeline.registry.

Purpose:
    Verify registration, lookup, instantiation, category filtering, and
    duplicate-name protection — using an isolated in-test registry
    state so these tests never depend on (or interfere with) the real
    application's registered algorithms.

Dependencies:
    pytest; visionleaf_ai.processing.pipeline.{stage,registry}.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.pipeline.registry import (
    AlgorithmRegistry,
    register_algorithm,
)
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Snapshot and restore the module-level registry around each test.

    The registry is process-global by design (like `get_settings`'s
    cache), so tests that register throwaway test-only stages must not
    leak them into other tests or into the real application's registry.
    """
    from visionleaf_ai.processing.pipeline import registry as registry_module

    snapshot = dict(registry_module._REGISTRY)
    yield
    registry_module._REGISTRY.clear()
    registry_module._REGISTRY.update(snapshot)


def _make_stage_class(name: str = "Test Stage"):
    class _TestStage(PipelineStage):
        stage_key = "enhancement"

        def process(self, session):
            self.validate(session)
            session.current_image = session.active_image
            return session

        def get_name(self):
            return name

        def get_description(self):
            return "A test-only stage."

        def get_info(self):
            return AlgorithmInfo(
                name=name,
                category=self.stage_key,
                purpose="test",
                theory="test",
                math_intuition="test",
                advantages=("test",),
                limitations=("test",),
            )

    return _TestStage


def test_register_and_get_returns_the_same_class():
    stage_cls = _make_stage_class()
    register_algorithm("test_stage_a")(stage_cls)
    assert AlgorithmRegistry.get("test_stage_a") is stage_cls


def test_get_unknown_algorithm_raises_validation_error():
    with pytest.raises(ValidationError):
        AlgorithmRegistry.get("not_a_real_algorithm")


def test_create_instantiates_with_kwargs():
    class _ParamStage(PipelineStage):
        stage_key = "restoration"

        def __init__(self, value: int = 1):
            self.value = value

        def process(self, session):
            return session

        def get_name(self):
            return "Param Stage"

        def get_description(self):
            return "test"

        def get_info(self):
            return AlgorithmInfo(
                name="Param Stage",
                category=self.stage_key,
                purpose="test",
                theory="test",
                math_intuition="test",
                advantages=("test",),
                limitations=("test",),
            )

    register_algorithm("test_param_stage")(_ParamStage)
    instance = AlgorithmRegistry.create("test_param_stage", value=42)
    assert isinstance(instance, _ParamStage)
    assert instance.value == 42


def test_list_algorithms_filters_by_category():
    enhancement_cls = _make_stage_class("Enh")
    enhancement_cls.stage_key = "enhancement"
    register_algorithm("test_enh_stage")(enhancement_cls)

    restoration_cls = _make_stage_class("Rest")
    restoration_cls.stage_key = "restoration"
    register_algorithm("test_rest_stage")(restoration_cls)

    enhancement_only = AlgorithmRegistry.list_algorithms("enhancement")
    assert "test_enh_stage" in enhancement_only
    assert "test_rest_stage" not in enhancement_only


def test_list_algorithms_with_no_category_returns_everything_registered():
    stage_cls = _make_stage_class()
    register_algorithm("test_stage_b")(stage_cls)
    all_names = AlgorithmRegistry.list_algorithms()
    assert "test_stage_b" in all_names


def test_registering_same_name_twice_with_different_class_raises():
    stage_cls_1 = _make_stage_class("One")
    stage_cls_2 = _make_stage_class("Two")
    register_algorithm("test_duplicate_name")(stage_cls_1)
    with pytest.raises(ValueError):
        register_algorithm("test_duplicate_name")(stage_cls_2)


def test_registering_same_name_with_same_class_is_a_no_op():
    stage_cls = _make_stage_class()
    register_algorithm("test_idempotent_name")(stage_cls)
    register_algorithm("test_idempotent_name")(stage_cls)  # should not raise
    assert AlgorithmRegistry.get("test_idempotent_name") is stage_cls


def test_get_info_returns_algorithm_info_without_needing_manual_instantiation():
    stage_cls = _make_stage_class("Info Stage")
    register_algorithm("test_info_stage")(stage_cls)
    info = AlgorithmRegistry.get_info("test_info_stage")
    assert isinstance(info, AlgorithmInfo)
    assert info.name == "Info Stage"
