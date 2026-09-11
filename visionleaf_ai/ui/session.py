"""Session state initialization and management.

Purpose:
    Define every `st.session_state` key the application uses, in one
    place, with its default value — so no page has to guess whether a
    key exists yet, and so later milestones extend this dict instead
    of scattering new session keys through page code.

Description:
    Streamlit reruns the entire script on every user interaction, so
    initialization must be idempotent: `init_session_state()` only
    sets a key if it is not already present, meaning it's safe to call
    at the top of every page render without resetting user state.

    `image_session` holds the single authoritative `ImageSession` (see
    `core.image_session`) — the one image object every pipeline stage
    reads from and writes to. Pages must fetch it via
    `get_image_session()` / replace it via `set_image_session()` rather
    than storing images under any other session-state key, so there is
    never more than one source of truth for "the current image".

    `pipeline_status` is a dict of stage_key -> "pending" | "active" |
    "complete", read by the sidebar and pipeline rail to show progress;
    stages flip their own status as they run. `learning_mode` is read
    by `components.cards.learning_note`.

Dependencies:
    streamlit; visionleaf_ai.core.image_session; visionleaf_ai.ui.pipeline_stages.

Public functions:
    init_session_state() -> None
        Populate any missing session-state keys with their defaults.
    get_stage_status(stage_key: str) -> str
        Read a stage's current status, defaulting to "pending".
    set_stage_status(stage_key: str, status: str) -> None
        Update a stage's status.
    get_image_session() -> ImageSession
        Return the single authoritative ImageSession.
    set_image_session(session: ImageSession) -> None
        Replace the authoritative ImageSession (e.g. after a new image
        is loaded, or on reset) and reset every stage's status to
        "pending" except Acquisition, since a new image invalidates
        every downstream result.
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.ui.pipeline_stages import STAGES


def _default_pipeline_status() -> dict[str, str]:
    return {stage["key"]: "pending" for stage in STAGES}


def _default_session_state() -> dict[str, object]:
    return {
        "learning_mode": False,
        "image_session": ImageSession.empty(),
        "pipeline_status": _default_pipeline_status(),
    }


def init_session_state() -> None:
    """Populate any missing `st.session_state` keys with their defaults.

    Idempotent: keys that already exist (e.g. because the user toggled
    Learning Mode or loaded an image earlier in the session) are left
    untouched.
    """
    for key, default in _default_session_state().items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_stage_status(stage_key: str) -> str:
    """Return a stage's current status.

    Args:
        stage_key: One of the keys in `pipeline_stages.STAGES`.

    Returns:
        One of "pending", "active", "complete". Defaults to "pending"
        if the stage has no recorded status yet.
    """
    return st.session_state.get("pipeline_status", {}).get(stage_key, "pending")


def set_stage_status(stage_key: str, status: str) -> None:
    """Update a single stage's status.

    Args:
        stage_key: One of the keys in `pipeline_stages.STAGES`.
        status: One of "pending", "active", "complete".
    """
    st.session_state.setdefault("pipeline_status", _default_pipeline_status())
    st.session_state["pipeline_status"][stage_key] = status


def get_image_session() -> ImageSession:
    """Return the single authoritative `ImageSession` for this session.

    Returns:
        The `ImageSession` stored in `st.session_state["image_session"]`.
        Callers should treat this as the one source of truth — read
        `session.active_image` for input, write results to the
        session's own fields, and call `session.record_event(...)`.
    """
    session = st.session_state.get("image_session")
    if session is None:
        session = ImageSession.empty()
        st.session_state["image_session"] = session
    return session


def set_image_session(session: ImageSession) -> None:
    """Replace the authoritative `ImageSession` and reset stage statuses.

    Args:
        session: The new `ImageSession` (e.g. freshly loaded, or reset).

    Notes:
        Loading a new image invalidates every downstream result, so
        every stage's status is reset to "pending" here except
        Acquisition, which is set to "complete" if the new session has
        an image loaded (and "pending" otherwise, e.g. after a reset).
    """
    st.session_state["image_session"] = session
    status = _default_pipeline_status()
    status["acquisition"] = "complete" if session.is_loaded() else "pending"
    st.session_state["pipeline_status"] = status
