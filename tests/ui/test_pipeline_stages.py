"""Tests for visionleaf_ai.ui.pipeline_stages.

Purpose:
    Verify the stage/page metadata is internally consistent — every
    stage a page claims to cover must actually exist in STAGES, every
    stage must be covered by exactly one page, and lookups work.

Dependencies:
    pytest; visionleaf_ai.ui.pipeline_stages.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.ui.pipeline_stages import PAGES, STAGES, get_page, stage_label


def test_eleven_stages_defined():
    assert len(STAGES) == 11


def test_stage_keys_are_unique():
    keys = [stage["key"] for stage in STAGES]
    assert len(keys) == len(set(keys))


def test_page_keys_are_unique():
    keys = [page["key"] for page in PAGES]
    assert len(keys) == len(set(keys))


def test_exactly_one_default_page():
    defaults = [page for page in PAGES if page["default"]]
    assert len(defaults) == 1
    assert defaults[0]["key"] == "dashboard"


def test_every_stage_covered_by_exactly_one_page():
    stage_keys = {stage["key"] for stage in STAGES}
    covered = []
    for page in PAGES:
        covered.extend(page["stage_keys"])
    assert set(covered) == stage_keys
    assert len(covered) == len(set(covered))  # no stage covered twice


def test_dashboard_covers_no_stages():
    dashboard = get_page("dashboard")
    assert dashboard["stage_keys"] == ()


def test_get_page_raises_for_unknown_key():
    with pytest.raises(KeyError):
        get_page("not-a-real-page")


def test_stage_label_lookup():
    assert stage_label("morphology") == "Morphology"


def test_stage_label_raises_for_unknown_key():
    with pytest.raises(KeyError):
        stage_label("not-a-real-stage")
