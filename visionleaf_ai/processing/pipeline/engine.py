"""PipelineEngine — the only way the UI is allowed to run an algorithm.

Purpose:
    Provide one execution surface for running DIP algorithms against an
    `ImageSession`: single-stage execution, multi-stage chains with
    cancellation, timing, logging, and uniform error handling. The UI
    (in later milestones) calls only this — never a `PipelineStage`
    subclass directly.

Description:
    `run_stage()` executes one stage and always returns a
    `StageResult`, whether the stage succeeded or failed — callers
    check `result.success` rather than wrapping every call in
    try/except. `run_pipeline()` runs a sequence of stages in order;
    by default (`stop_on_error=True`) a failed stage cancels every
    stage after it, leaving `ImageSession` exactly as it was after the
    last successful stage (never partially corrupted by a stage that
    started but shouldn't have run). `run_by_name()` is the primary
    UI-facing entry point: look an algorithm up in the
    `AlgorithmRegistry` by string name and run it, so calling code
    never imports an algorithm class.

    Stages already validate, log, and record their own history inside
    `process()` (see `pipeline.stage.PipelineStage`) — this engine adds
    an *outer* layer: wall-clock timing independent of the stage's own
    logging, catching `VisionLeafError` and unexpected exceptions alike
    so a bad algorithm can never crash the whole app, and sequencing/
    cancellation semantics for multi-stage runs.

Dependencies:
    time; visionleaf_ai.core; visionleaf_ai.processing.pipeline.{stage,
    registry,bootstrap}.

Public classes/functions:
    StageResult
    PipelineEngine
        run_stage(stage, session) -> tuple[ImageSession, StageResult]
        run_pipeline(stages, session, stop_on_error=True) -> tuple[ImageSession, list[StageResult]]
        run_by_name(name, session, **params) -> tuple[ImageSession, StageResult]
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from visionleaf_ai.core.exceptions import VisionLeafError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.processing.pipeline.bootstrap import ensure_registered
from visionleaf_ai.processing.pipeline.registry import AlgorithmRegistry
from visionleaf_ai.processing.pipeline.stage import PipelineStage

logger = get_logger(__name__)


@dataclass(frozen=True)
class StageResult:
    """The outcome of running one `PipelineStage` through the engine.

    Attributes:
        algorithm_name: The stage's `get_name()` value.
        stage_key: The stage's pipeline stage key (e.g. "enhancement").
        success: Whether the stage completed without error.
        duration_seconds: Wall-clock time the engine measured around
            the call, independent of any timing the stage logs itself.
        error_message: Human-readable error if `success` is False,
            otherwise `None`.
    """

    algorithm_name: str
    stage_key: str
    success: bool
    duration_seconds: float
    error_message: str | None = None


class PipelineEngine:
    """Central execution engine for running DIP algorithms.

    Constructing an engine ensures every algorithm module has been
    imported (and therefore registered) via
    `pipeline.bootstrap.ensure_registered()` — so `run_by_name()` works
    immediately without callers needing to import algorithm modules.
    """

    def __init__(self) -> None:
        ensure_registered()

    def run_stage(
        self, stage: PipelineStage, session: ImageSession
    ) -> tuple[ImageSession, StageResult]:
        """Run exactly one stage against `session`.

        Args:
            stage: The `PipelineStage` instance to run.
            session: The `ImageSession` to process.

        Returns:
            `(session, result)`. `session` is the same object passed
            in — updated in place if the stage succeeded, unchanged if
            it failed. `result.success` tells the caller which happened;
            this method never raises for an algorithm-level failure.
        """
        name = stage.get_name()
        started = time.perf_counter()
        try:
            updated_session = stage.process(session)
            duration = time.perf_counter() - started
            logger.info(
                "PipelineEngine: '%s' (%s) succeeded in %.4fs",
                name,
                stage.stage_key,
                duration,
            )
            return updated_session, StageResult(
                algorithm_name=name,
                stage_key=stage.stage_key,
                success=True,
                duration_seconds=duration,
            )
        except VisionLeafError as exc:
            duration = time.perf_counter() - started
            logger.warning(
                "PipelineEngine: '%s' (%s) failed after %.4fs: %s",
                name,
                stage.stage_key,
                duration,
                exc,
            )
            return session, StageResult(
                algorithm_name=name,
                stage_key=stage.stage_key,
                success=False,
                duration_seconds=duration,
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 — deliberate: never let an
            # algorithm bug crash the whole app; surface it as a failed result.
            duration = time.perf_counter() - started
            logger.error(
                "PipelineEngine: '%s' (%s) raised an unexpected error after %.4fs: %s",
                name,
                stage.stage_key,
                duration,
                exc,
                exc_info=True,
            )
            return session, StageResult(
                algorithm_name=name,
                stage_key=stage.stage_key,
                success=False,
                duration_seconds=duration,
                error_message=f"Unexpected error: {exc}",
            )

    def run_pipeline(
        self,
        stages: Sequence[PipelineStage],
        session: ImageSession,
        stop_on_error: bool = True,
    ) -> tuple[ImageSession, list[StageResult]]:
        """Run a sequence of stages against `session`, in order.

        Args:
            stages: Stages to run, in the order they should be applied.
            session: The `ImageSession` to process.
            stop_on_error: If True (default), a failed stage cancels
                every stage after it — the pipeline "cancellation"
                behavior required by the architecture. If False, every
                stage runs regardless of earlier failures (useful for,
                e.g., Experiment Mode's side-by-side comparisons where
                a failed variant shouldn't block the others).

        Returns:
            `(session, results)` — `session` reflects every stage that
            actually ran; `results` has one `StageResult` per stage
            that was attempted (stages skipped due to cancellation are
            not included).
        """
        results: list[StageResult] = []
        for stage in stages:
            session, result = self.run_stage(stage, session)
            results.append(result)
            if not result.success and stop_on_error:
                logger.warning(
                    "PipelineEngine: cancelling remaining %d stage(s) after '%s' failed",
                    len(stages) - len(results),
                    result.algorithm_name,
                )
                break
        return session, results

    def run_by_name(
        self, name: str, session: ImageSession, **params
    ) -> tuple[ImageSession, StageResult]:
        """Look up an algorithm by name, instantiate it, and run it.

        This is the primary entry point later UI code will use — it
        never needs to import an algorithm class, only know its
        registered name and parameters.

        Args:
            name: Registered algorithm key, e.g. "brightness_adjustment".
            session: The `ImageSession` to process.
            **params: Forwarded to the algorithm's constructor, e.g.
                `delta=40`.

        Returns:
            Same as `run_stage`.

        Raises:
            ValidationError: If `name` is not a registered algorithm
                (raised immediately — this is a programming error, not
                a runtime processing failure, so it is not wrapped into
                a failed `StageResult`).
        """
        stage = AlgorithmRegistry.create(name, **params)
        return self.run_stage(stage, session)


def get_pipeline_engine() -> PipelineEngine:
    """Return a `PipelineEngine`, ready to use.

    Convenience accessor for pages — equivalent to `PipelineEngine()`
    but named for readability at call sites, matching
    `processing.acquisition.get_image_manager()`.
    """
    return PipelineEngine()
