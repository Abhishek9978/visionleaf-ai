"""Cross-module integration tests.

Purpose:
    Verify the two places that independently define "what pipeline
    stages exist" — `processing.pipeline.AlgorithmRegistry` (each
    algorithm declares its own `stage_key`) and
    `ui.pipeline_stages.STAGES` (the UI's canonical stage list) — never
    drift apart. Added during the Milestone 6 stabilization review,
    which found this integration point had no test coverage: nothing
    previously caught a typo'd `stage_key` on a new algorithm before
    it silently broke the pipeline rail / sidebar status rollup for
    that algorithm's page.

Description:
    This module deliberately lives at the top level of `tests/`, not
    under `tests/processing/` or `tests/ui/`, because it tests the
    seam between those two layers rather than either one in isolation.

Dependencies:
    pytest; visionleaf_ai.processing.pipeline; visionleaf_ai.ui.pipeline_stages.
"""

from __future__ import annotations

from visionleaf_ai.processing.pipeline import AlgorithmRegistry, get_pipeline_engine
from visionleaf_ai.ui.pipeline_stages import STAGES


def test_every_registered_algorithms_stage_key_is_a_valid_ui_stage():
    """Every algorithm's stage_key must match a real ui.pipeline_stages.STAGES key.

    Guards against exactly the class of bug a typo introduces: an
    algorithm registers successfully (the registry doesn't validate
    stage_key against anything), but its result would silently never
    show up in the pipeline rail or sidebar status rollup for any
    page, because no page's `stage_keys` tuple would ever match it.
    """
    get_pipeline_engine()  # ensure the registry is fully populated
    valid_stage_keys = {stage["key"] for stage in STAGES}

    mismatched = []
    for algorithm_name in AlgorithmRegistry.list_algorithms():
        stage_cls = AlgorithmRegistry.get(algorithm_name)
        if stage_cls.stage_key not in valid_stage_keys:
            mismatched.append((algorithm_name, stage_cls.stage_key))

    assert mismatched == [], f"algorithms with invalid stage_key: {mismatched}"


def test_at_least_one_algorithm_registered_per_implemented_stage():
    """Every stage that has algorithms implemented (M4/M6) has at least one.

    A coarser sanity check than the per-algorithm test above: confirms
    the categories users actually expect to find something in
    (enhancement, restoration, segmentation, morphology, roi) are
    non-empty, catching a whole category accidentally failing to
    register (e.g. a bootstrap import silently raising and being
    swallowed) rather than just one algorithm within it.
    """
    get_pipeline_engine()
    implemented_categories = ("enhancement", "restoration", "segmentation", "morphology", "roi")
    for category in implemented_categories:
        assert AlgorithmRegistry.list_algorithms(category), (
            f"expected at least one registered algorithm for '{category}'"
        )
