"""Contour Processing and ROI Extraction algorithms.

Purpose:
    Concrete `PipelineStage` implementations that turn a cleaned mask
    into structured shape data (Find Contours, Contour Filtering by
    Area, Bounding Rectangle, Minimum Area Rectangle, Convex Hull) and
    then into the actual region-of-interest images later stages
    consume (ROI Cropping, ROI Masking, Bounding Box Visualization).

Description:
    Every module here registers its algorithm via `@register_algorithm`
    at import time — `pipeline.bootstrap.ensure_registered()` imports
    all of them. `_common.py` holds private validation/derivation
    helpers shared across this package. `auto_extract.py` is a plain
    orchestration function (not a registered algorithm) that composes
    several of these stages via `PipelineEngine.run_pipeline()` for
    the common "just extract the ROI" case — see its own docstring for
    why it isn't a `PipelineStage` itself.

Dependencies:
    visionleaf_ai.processing.roi.{find_contours,filter_contours,
    bounding_rectangle,min_area_rectangle,convex_hull,roi_crop,
    roi_mask,bounding_box_visualization}

Public functions:
    run_automatic_roi_extraction (see auto_extract.py)
"""

from visionleaf_ai.processing.roi.auto_extract import run_automatic_roi_extraction

__all__ = ["run_automatic_roi_extraction"]
