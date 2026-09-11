"""Integration tests for the Image Processing Lab page.

Purpose:
    Exercise the real page end to end via Streamlit's AppTest: empty
    state, applying enhancement/restoration algorithms in sequence
    (verifying they chain correctly onto ImageSession), Learning Mode,
    Reset, and graceful error handling for invalid parameters — all
    through the actual page code and actual PipelineEngine, not mocks.

Dependencies:
    pytest; streamlit.testing.v1.AppTest.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EMPTY_HARNESS = str(_FIXTURES_DIR / "processing_lab_empty_harness.py")
LOADED_HARNESS = str(_FIXTURES_DIR / "processing_lab_loaded_harness.py")
TINY_HARNESS = str(_FIXTURES_DIR / "processing_lab_tiny_harness.py")


def test_empty_state_shows_prompt_to_load_an_image():
    at = AppTest.from_file(EMPTY_HARNESS, default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.get("button")) == 0  # no algorithm controls without a loaded image


def test_loaded_state_renders_all_sections_without_exception():
    at = AppTest.from_file(LOADED_HARNESS, default_timeout=20)
    at.run()
    assert not at.exception

    apply_buttons = [b for b in at.get("button") if b.label.startswith("▶ Apply")]
    assert len(apply_buttons) == 2  # one for enhancement, one for restoration
    assert len(at.get("download_button")) == 1
    assert any("Reset to Original" in b.label for b in at.get("button"))


def test_applying_enhancement_records_history_and_marks_stage_complete():
    at = AppTest.from_file(LOADED_HARNESS, default_timeout=20)
    at.run()

    enhancement_apply = next(
        b for b in at.get("button") if b.label.startswith("▶ Apply") and b.key == "apply__enhancement"
    )
    enhancement_apply.click().run()
    assert not at.exception

    history = at.session_state["image_session"].processing_history
    assert len(history) == 2  # Image Loaded, then the applied enhancement
    assert at.session_state["pipeline_status"]["enhancement"] == "complete"


def test_applying_enhancement_then_restoration_chains_onto_same_session():
    at = AppTest.from_file(LOADED_HARNESS, default_timeout=20)
    at.run()

    enhancement_apply = next(
        b for b in at.get("button") if b.key == "apply__enhancement"
    )
    enhancement_apply.click().run()
    restoration_apply = next(
        b for b in at.get("button") if b.key == "apply__restoration"
    )
    restoration_apply.click().run()
    assert not at.exception

    history = at.session_state["image_session"].processing_history
    assert len(history) == 3
    assert history[0].stage_key == "acquisition"
    assert history[1].stage_key == "enhancement"
    assert history[2].stage_key == "restoration"
    assert at.session_state["pipeline_status"]["enhancement"] == "complete"
    assert at.session_state["pipeline_status"]["restoration"] == "complete"


def test_learning_mode_shows_algorithm_documentation():
    at = AppTest.from_file(LOADED_HARNESS, default_timeout=20)
    at.run()
    assert len(at.get("code")) == 0  # math_intuition renders via st.code — absent when off

    at.session_state["learning_mode"] = True
    at.run()
    assert not at.exception
    assert len(at.get("code")) >= 1  # at least one algorithm's math intuition is shown


def test_reset_restores_original_and_clears_history():
    at = AppTest.from_file(LOADED_HARNESS, default_timeout=20)
    at.run()

    enhancement_apply = next(b for b in at.get("button") if b.key == "apply__enhancement")
    enhancement_apply.click().run()
    assert len(at.session_state["image_session"].processing_history) == 2

    reset_button = next(b for b in at.get("button") if "Reset to Original" in b.label)
    reset_button.click().run()
    assert not at.exception

    session = at.session_state["image_session"]
    assert len(session.processing_history) == 1
    assert session.processing_history[0].label == "Image Loaded"
    assert at.session_state["pipeline_status"]["enhancement"] == "pending"
    assert at.session_state["pipeline_status"]["restoration"] == "pending"


def test_invalid_parameter_shows_error_without_crashing():
    at = AppTest.from_file(TINY_HARNESS, default_timeout=20)
    at.run()

    # Switch restoration to gaussian_blur, then set a kernel size (15)
    # that genuinely exceeds the tiny 10x10 image's dimensions.
    restoration_select = next(
        s for s in at.get("selectbox") if "algorithm_select__restoration" in s.key
    )
    restoration_select.set_value("gaussian_blur").run()

    kernel_slider = next(s for s in at.get("select_slider") if "kernel_size" in s.key)
    kernel_slider.set_value(15).run()

    apply_button = next(b for b in at.get("button") if b.key == "apply__restoration")
    apply_button.click().run()

    assert not at.exception  # the failure is caught and shown as st.error, never raised
    errors = at.get("error")
    assert len(errors) == 1
    assert "exceeds image dimensions" in errors[0].value
    # The session must be untouched by the failed stage.
    assert len(at.session_state["image_session"].processing_history) == 1
