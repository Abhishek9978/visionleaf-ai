"""Automatic ROI Extraction.

Purpose:
    Provide the "just extract the region of interest for me" one-call
    convenience the Blueprint asks for, without inventing a new
    monolithic algorithm that duplicates what Otsu Threshold, Closing,
    Find Contours, Bounding Rectangle, and ROI Cropping already do
    individually.

Description:
    This is NOT a `PipelineStage` and is NOT registered with
    `AlgorithmRegistry` — it's a thin orchestration function that
    reuses `PipelineEngine.run_pipeline()` to run a sensible default
    sequence of *existing* stages in order. Each individual algorithm
    remains independently registered, tested, and usable on its own;
    this function only saves the caller from having to name all five
    steps by hand for the common case.

    Sequence: Otsu Threshold (automatic global segmentation) -> Closing
    (fill small gaps) -> Find Contours -> Bounding Rectangle -> ROI
    Cropping. `stop_on_error=True` (the engine's default) means a
    failure at any step correctly cancels the rest, exactly as
    `PipelineEngine.run_pipeline` already guarantees for any caller.

Dependencies:
    visionleaf_ai.core.image_session.ImageSession;
    visionleaf_ai.processing.pipeline.{PipelineEngine, StageResult, AlgorithmRegistry}.

Public functions:
    run_automatic_roi_extraction(engine, session) -> tuple[ImageSession, list[StageResult]]
"""

from __future__ import annotations

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.pipeline.engine import PipelineEngine, StageResult
from visionleaf_ai.processing.pipeline.registry import AlgorithmRegistry

#: The default stage sequence for automatic extraction, in order.
#: Each name must be a real registered algorithm — see
#: test_auto_extract.py for a test enforcing that.
DEFAULT_SEQUENCE: tuple[str, ...] = (
    "otsu_threshold",
    "closing",
    "find_contours",
    "bounding_rectangle",
    "roi_crop",
)


def run_automatic_roi_extraction(
    engine: PipelineEngine, session: ImageSession
) -> tuple[ImageSession, list[StageResult]]:
    """Run the default Otsu -> Closing -> Contours -> BBox -> Crop sequence.

    Args:
        engine: The `PipelineEngine` to run the sequence through.
        session: The `ImageSession` to process.

    Returns:
        `(session, results)` exactly as `PipelineEngine.run_pipeline`
        returns — one `StageResult` per stage that was attempted,
        stopping early if any stage fails.
    """
    stages = [AlgorithmRegistry.create(name) for name in DEFAULT_SEQUENCE]
    return engine.run_pipeline(stages, session, stop_on_error=True)
