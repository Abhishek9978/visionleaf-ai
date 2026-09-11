"""Integration tests for the running application (app.py).

Purpose:
    Exercise the real app end to end using Streamlit's AppTest
    framework: it starts without error, the sidebar has a link for
    every page, and Learning Mode actually toggles the learning notes.

Description:
    This is the test that would catch a broken import, a bad page-key
    mismatch between `app.py` and `pipeline_stages.py`, or a component
    that raises when actually rendered inside a full app context —
    none of which the per-module unit tests can catch on their own.

Dependencies:
    pytest; streamlit.testing.v1.AppTest.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from visionleaf_ai.ui.pipeline_stages import PAGES

APP_PATH = str(Path(__file__).resolve().parents[2] / "app.py")


def test_app_starts_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception


def test_sidebar_has_a_link_for_every_page():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    links = at.sidebar.get("page_link")
    labels = {link.label for link in links}
    expected_labels = {page["label"] for page in PAGES}
    assert labels == expected_labels


def test_learning_mode_toggle_shows_learning_note():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert len(at.get("info")) == 0  # off by default on the Dashboard

    toggle = at.sidebar.get("toggle")[0]
    toggle.set_value(True).run()
    assert not at.exception
    assert len(at.get("info")) >= 1  # learning_note now visible


def test_learning_mode_off_by_default():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert at.session_state["learning_mode"] is False
