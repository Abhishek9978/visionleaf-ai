"""Pipeline package — the Core Digital Image Processing Engine.

Purpose:
    Re-export the pieces that make every DIP algorithm interchangeable
    and centrally executable: the `PipelineStage` interface, the
    `AlgorithmRegistry`, and the `PipelineEngine` that the UI calls
    instead of any algorithm directly.

Description:
    Built in Milestone 4. No Streamlit code lives here or anywhere
    under `processing/` — this package is UI-agnostic by design, so
    every algorithm is unit-testable with plain pytest.

Dependencies:
    visionleaf_ai.processing.pipeline.{stage,registry,engine,bootstrap}

Public functions/classes:
    PipelineStage, AlgorithmInfo, register_algorithm, AlgorithmRegistry,
    PipelineEngine, StageResult
"""

from visionleaf_ai.processing.pipeline.engine import (
    PipelineEngine,
    StageResult,
    get_pipeline_engine,
)
from visionleaf_ai.processing.pipeline.registry import (
    AlgorithmRegistry,
    register_algorithm,
)
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage

__all__ = [
    "PipelineStage",
    "AlgorithmInfo",
    "register_algorithm",
    "AlgorithmRegistry",
    "PipelineEngine",
    "StageResult",
    "get_pipeline_engine",
]
