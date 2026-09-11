"""Integration tests for the Image Acquisition page (both UI states).

Purpose:
    Exercise the real Acquisition page end to end via Streamlit's
    AppTest — both before any image is loaded (upload panel + sample
    gallery visible) and after loading a sample image (preview,
    metadata, history visible) — catching wiring bugs unit tests on
    ImageManager/ImageSession alone can't.

Description:
    `AppTest.switch_page()` only supports file-based multipage apps;
    since VisionLeaf AI's pages are callables wrapped in `st.Page`
    (see app.py's docstring for why), navigating to a specific page
    through the full running app isn't directly testable. Instead,
    each state is exercised through a small standalone harness script
    in `tests/ui/fixtures/` that imports and calls
    `acquisition.render()` directly — the same page code the real app
    runs, just invoked without going through `st.navigation`.

Dependencies:
    pytest; streamlit.testing.v1.AppTest.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EMPTY_HARNESS = str(_FIXTURES_DIR / "acquisition_empty_harness.py")
LOADED_HARNESS = str(_FIXTURES_DIR / "acquisition_loaded_harness.py")


def test_acquisition_page_empty_state_shows_upload_and_gallery():
    at = AppTest.from_file(EMPTY_HARNESS, default_timeout=20)
    at.run()
    assert not at.exception

    assert len(at.get("file_uploader")) == 1
    sample_buttons = [b for b in at.get("button") if b.label == "Use this image"]
    assert len(sample_buttons) >= 1  # synthetic sample gallery is populated


def test_acquisition_page_loaded_state_shows_preview_and_metadata():
    at = AppTest.from_file(LOADED_HARNESS, default_timeout=20)
    at.run()
    assert not at.exception

    # Loaded state: metadata metrics and the reset button are present,
    # and the upload panel is gone.
    assert len(at.get("metric")) >= 6
    assert any(b.label == "🔄 Load a different image" for b in at.get("button"))
    assert len(at.get("file_uploader")) == 0


def test_acquisition_reset_returns_to_empty_state():
    at = AppTest.from_file(LOADED_HARNESS, default_timeout=20)
    at.run()

    reset_button = next(
        b for b in at.get("button") if b.label == "🔄 Load a different image"
    )
    reset_button.click().run()
    at.run()  # the reset button triggers st.rerun(); let the pending rerun execute
    assert not at.exception
    assert len(at.get("file_uploader")) == 1


def test_loading_sample_marks_acquisition_complete_in_session_state():
    at = AppTest.from_file(LOADED_HARNESS, default_timeout=20)
    at.run()

    assert at.session_state["pipeline_status"]["acquisition"] == "complete"
    assert at.session_state["image_session"].is_loaded()


def test_selecting_sample_in_empty_state_loads_it():
    at = AppTest.from_file(EMPTY_HARNESS, default_timeout=20)
    at.run()

    sample_button = next(b for b in at.get("button") if b.label == "Use this image")
    sample_button.click().run()
    at.run()  # loading triggers st.rerun(); let the pending rerun execute
    assert not at.exception

    assert at.session_state["image_session"].is_loaded()
    assert at.session_state["pipeline_status"]["acquisition"] == "complete"
    assert len(at.get("file_uploader")) == 0
