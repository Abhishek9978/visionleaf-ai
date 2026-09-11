"""Tests for auto_extract.py (Automatic ROI Extraction).

Purpose:
    Verify the orchestration function composes real, registered
    algorithms correctly (not a duplicate algorithm implementation),
    succeeds end to end on a normal image, and correctly cancels the
    remaining sequence when an early stage fails.

Dependencies:
    pytest; numpy; visionleaf_ai.core.image_session;
    visionleaf_ai.processing.pipeline; visionleaf_ai.processing.roi.auto_extract.
"""

from __future__ import annotations

import numpy as np

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.pipeline import AlgorithmRegistry, get_pipeline_engine
from visionleaf_ai.processing.roi.auto_extract import (
    DEFAULT_SEQUENCE,
    run_automatic_roi_extraction,
)


def test_every_stage_in_default_sequence_is_a_real_registered_algorithm():
    get_pipeline_engine()  # ensure the registry is populated
    for name in DEFAULT_SEQUENCE:
        AlgorithmRegistry.get(name)  # raises ValidationError if not registered


def test_automatic_extraction_succeeds_end_to_end_on_a_normal_image():
    engine = get_pipeline_engine()
    image = np.full((100, 100, 3), 30, dtype=np.uint8)
    image[20:70, 30:80] = 220
    session = ImageSession(original_image=image.copy(), current_image=image.copy())

    session, results = run_automatic_roi_extraction(engine, session)

    assert len(results) == len(DEFAULT_SEQUENCE)
    assert all(r.success for r in results)
    assert session.roi_image is not None
    assert session.bounding_box is not None


def test_automatic_extraction_cancels_on_failure_with_empty_mask():
    engine = get_pipeline_engine()
    # A fully black image: Otsu's threshold degenerates to 0, and since
    # THRESH_BINARY requires strictly greater-than, every pixel (also 0)
    # becomes background — Find Contours then finds zero contours, so
    # Bounding Rectangle should fail and cancel ROI Crop after it.
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    session = ImageSession(original_image=image.copy(), current_image=image.copy())

    session, results = run_automatic_roi_extraction(engine, session)

    assert not all(r.success for r in results)
    failure_index = next(i for i, r in enumerate(results) if not r.success)
    assert failure_index < len(DEFAULT_SEQUENCE) - 1  # something after it was cancelled
