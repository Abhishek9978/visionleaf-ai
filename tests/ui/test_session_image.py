"""Tests for ImageSession integration in visionleaf_ai.ui.session.

Purpose:
    Verify get_image_session/set_image_session provide the single
    source of truth contract: session state always holds exactly one
    ImageSession, and replacing it correctly resets pipeline_status.

Dependencies:
    pytest; streamlit; visionleaf_ai.ui.session; visionleaf_ai.core.image_session.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.ui.session import (
    get_image_session,
    init_session_state,
    set_image_session,
    set_stage_status,
)


def test_get_image_session_returns_empty_session_by_default():
    st.session_state.clear()
    init_session_state()
    session = get_image_session()
    assert session.is_loaded() is False


def test_get_image_session_returns_same_object_across_calls():
    st.session_state.clear()
    init_session_state()
    first = get_image_session()
    second = get_image_session()
    assert first is second


def test_set_image_session_replaces_the_authoritative_session():
    st.session_state.clear()
    init_session_state()

    new_session = ImageSession(original_image=np.zeros((4, 4, 3), dtype=np.uint8))
    set_image_session(new_session)

    assert get_image_session() is new_session
    assert get_image_session().is_loaded() is True


def test_set_image_session_marks_acquisition_complete_when_loaded():
    st.session_state.clear()
    init_session_state()

    loaded_session = ImageSession(original_image=np.zeros((4, 4, 3), dtype=np.uint8))
    set_image_session(loaded_session)

    assert st.session_state["pipeline_status"]["acquisition"] == "complete"


def test_set_image_session_resets_downstream_stage_statuses():
    st.session_state.clear()
    init_session_state()
    set_stage_status("segmentation", "complete")
    assert st.session_state["pipeline_status"]["segmentation"] == "complete"

    new_session = ImageSession(original_image=np.zeros((4, 4, 3), dtype=np.uint8))
    set_image_session(new_session)

    # Loading a new image invalidates every downstream result.
    assert st.session_state["pipeline_status"]["segmentation"] == "pending"


def test_set_image_session_with_empty_session_marks_acquisition_pending():
    st.session_state.clear()
    init_session_state()
    set_image_session(ImageSession.empty())
    assert st.session_state["pipeline_status"]["acquisition"] == "pending"
