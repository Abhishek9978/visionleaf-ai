"""Tests for visionleaf_ai.ui.session (non-image-session behavior).

Purpose:
    Verify session-state initialization is idempotent and that
    stage-status get/set works correctly. ImageSession-specific
    behavior (get_image_session/set_image_session) is covered
    separately in test_session_image.py.

Dependencies:
    pytest; streamlit; visionleaf_ai.ui.session; visionleaf_ai.ui.pipeline_stages.
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.ui.pipeline_stages import STAGES
from visionleaf_ai.ui.session import (
    get_stage_status,
    init_session_state,
    set_stage_status,
)


def test_init_session_state_sets_all_defaults():
    st.session_state.clear()
    init_session_state()
    assert st.session_state["learning_mode"] is False
    assert st.session_state["image_session"].is_loaded() is False
    assert set(st.session_state["pipeline_status"].keys()) == {
        stage["key"] for stage in STAGES
    }


def test_init_session_state_does_not_overwrite_existing_values():
    st.session_state.clear()
    init_session_state()
    st.session_state["learning_mode"] = True
    init_session_state()  # simulate a rerun
    assert st.session_state["learning_mode"] is True


def test_get_stage_status_defaults_to_pending():
    st.session_state.clear()
    init_session_state()
    assert get_stage_status("acquisition") == "pending"


def test_get_stage_status_reflects_updates():
    st.session_state.clear()
    init_session_state()
    st.session_state["pipeline_status"]["acquisition"] = "complete"
    assert get_stage_status("acquisition") == "complete"


def test_get_stage_status_unknown_key_defaults_to_pending():
    st.session_state.clear()
    init_session_state()
    assert get_stage_status("not-a-real-stage") == "pending"


def test_set_stage_status_updates_a_single_stage():
    st.session_state.clear()
    init_session_state()
    set_stage_status("enhancement", "active")
    assert get_stage_status("enhancement") == "active"
    # Other stages remain untouched.
    assert get_stage_status("segmentation") == "pending"
