"""Tests for visionleaf_ai.processing.pipeline.stage.

Purpose:
    Verify PipelineStage's default validate() behavior and that it
    genuinely cannot be instantiated without implementing every
    abstract method.

Dependencies:
    pytest; numpy; visionleaf_ai.core.image_session;
    visionleaf_ai.processing.pipeline.stage.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage


class _MinimalStage(PipelineStage):
    """Smallest possible concrete stage, for exercising the base class."""

    stage_key = "enhancement"

    def process(self, session: ImageSession) -> ImageSession:
        self.validate(session)
        session.current_image = session.active_image
        session.record_event(self.stage_key, self.get_name())
        return session

    def get_name(self) -> str:
        return "Minimal Stage"

    def get_description(self) -> str:
        return "A no-op stage used only for testing the base class."

    def get_info(self) -> AlgorithmInfo:
        return AlgorithmInfo(
            name=self.get_name(),
            category=self.stage_key,
            purpose="test",
            theory="test",
            math_intuition="test",
            advantages=("test",),
            limitations=("test",),
        )


def test_pipeline_stage_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        PipelineStage()  # type: ignore[abstract]


def test_subclass_missing_abstract_method_cannot_be_instantiated():
    class _Incomplete(PipelineStage):
        stage_key = "enhancement"

        def process(self, session):
            return session

        def get_name(self):
            return "Incomplete"

        # get_description and get_info deliberately not implemented

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


def test_default_validate_raises_when_no_image_loaded():
    stage = _MinimalStage()
    empty_session = ImageSession.empty()
    with pytest.raises(ValidationError):
        stage.validate(empty_session)


def test_default_validate_passes_when_image_loaded():
    stage = _MinimalStage()
    session = ImageSession(original_image=np.zeros((4, 4, 3), dtype=np.uint8))
    stage.validate(session)  # should not raise


def test_process_updates_session_and_records_history():
    stage = _MinimalStage()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    session = ImageSession(original_image=image, current_image=image)

    result = stage.process(session)

    assert result is session
    assert len(session.processing_history) == 1
    assert session.processing_history[0].label == "Minimal Stage"
    assert session.processing_history[0].stage_key == "enhancement"


def test_get_info_returns_algorithm_info():
    stage = _MinimalStage()
    info = stage.get_info()
    assert isinstance(info, AlgorithmInfo)
    assert info.name == "Minimal Stage"
    assert info.category == "enhancement"
