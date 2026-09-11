"""Tests for visionleaf_ai.processing.pipeline.engine.

Purpose:
    Verify single-stage execution, multi-stage chains, cancellation
    on failure, timing, and that unexpected (non-VisionLeafError)
    exceptions are also caught and surfaced as a failed StageResult
    rather than crashing the caller.

Dependencies:
    pytest; numpy; visionleaf_ai.core; visionleaf_ai.processing.pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.pipeline.engine import PipelineEngine
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage


def _make_session() -> ImageSession:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    return ImageSession(original_image=image.copy(), current_image=image.copy())


class _AlwaysSucceeds(PipelineStage):
    stage_key = "enhancement"

    def process(self, session):
        self.validate(session)
        session.current_image = session.active_image
        session.record_event(self.stage_key, self.get_name())
        return session

    def get_name(self):
        return "Always Succeeds"

    def get_description(self):
        return "test"

    def get_info(self):
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose="test",
            theory="test",
            math_intuition="test",
            advantages=("test",),
            limitations=("test",),
        )


class _AlwaysFailsValidation(PipelineStage):
    stage_key = "enhancement"

    def validate(self, session):
        raise ValidationError("deliberately invalid for testing")

    def process(self, session):
        self.validate(session)
        return session

    def get_name(self):
        return "Always Fails"

    def get_description(self):
        return "test"

    def get_info(self):
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose="test",
            theory="test",
            math_intuition="test",
            advantages=("test",),
            limitations=("test",),
        )


class _RaisesUnexpectedError(PipelineStage):
    stage_key = "enhancement"

    def process(self, session):
        raise RuntimeError("boom — a genuine bug, not a validation failure")

    def get_name(self):
        return "Raises Unexpected"

    def get_description(self):
        return "test"

    def get_info(self):
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose="test",
            theory="test",
            math_intuition="test",
            advantages=("test",),
            limitations=("test",),
        )


def test_run_stage_success_returns_updated_session_and_result():
    engine = PipelineEngine()
    session = _make_session()

    updated, result = engine.run_stage(_AlwaysSucceeds(), session)

    assert result.success is True
    assert result.algorithm_name == "Always Succeeds"
    assert result.stage_key == "enhancement"
    assert result.error_message is None
    assert result.duration_seconds >= 0
    assert len(updated.processing_history) == 1


def test_run_stage_validation_failure_returns_failed_result_not_exception():
    engine = PipelineEngine()
    session = _make_session()

    updated, result = engine.run_stage(_AlwaysFailsValidation(), session)

    assert result.success is False
    assert "deliberately invalid" in result.error_message
    assert updated is session
    assert updated.processing_history == []  # never got to record an event


def test_run_stage_unexpected_exception_is_caught_not_raised():
    engine = PipelineEngine()
    session = _make_session()

    updated, result = engine.run_stage(_RaisesUnexpectedError(), session)

    assert result.success is False
    assert "Unexpected error" in result.error_message
    assert "boom" in result.error_message


def test_run_pipeline_runs_all_stages_when_all_succeed():
    engine = PipelineEngine()
    session = _make_session()
    stages = [_AlwaysSucceeds(), _AlwaysSucceeds(), _AlwaysSucceeds()]

    updated, results = engine.run_pipeline(stages, session)

    assert len(results) == 3
    assert all(r.success for r in results)
    assert len(updated.processing_history) == 3


def test_run_pipeline_cancels_remaining_stages_after_failure():
    engine = PipelineEngine()
    session = _make_session()
    stages = [_AlwaysSucceeds(), _AlwaysFailsValidation(), _AlwaysSucceeds()]

    updated, results = engine.run_pipeline(stages, session, stop_on_error=True)

    # Only the first two stages were attempted; the third was cancelled.
    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is False
    assert len(updated.processing_history) == 1  # only the first stage recorded


def test_run_pipeline_continues_past_failure_when_stop_on_error_is_false():
    engine = PipelineEngine()
    session = _make_session()
    stages = [_AlwaysSucceeds(), _AlwaysFailsValidation(), _AlwaysSucceeds()]

    updated, results = engine.run_pipeline(stages, session, stop_on_error=False)

    assert len(results) == 3
    assert [r.success for r in results] == [True, False, True]
    assert len(updated.processing_history) == 2  # the two successes recorded


def test_run_by_name_looks_up_and_runs_a_registered_algorithm():
    engine = PipelineEngine()
    session = _make_session()

    updated, result = engine.run_by_name("brightness_adjustment", session, delta=10)

    assert result.success is True
    assert result.algorithm_name == "Brightness Adjustment"
    assert updated.current_image is not None


def test_run_by_name_unknown_algorithm_raises_immediately():
    engine = PipelineEngine()
    session = _make_session()

    with pytest.raises(ValidationError):
        engine.run_by_name("not_a_real_algorithm", session)
