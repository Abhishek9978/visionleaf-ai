"""Test harness: renders the Processing Lab page with a sample image loaded.

Not a real application entry point — used only by
tests/ui/test_processing_lab_page.py via Streamlit's AppTest.from_file.
Seeds exactly once (via a dedicated flag, not `is_loaded()`) so a
Reset button click during a test doesn't get immediately undone by
this harness re-seeding on the next rerun — see
tests/ui/fixtures/acquisition_loaded_harness.py for the same pattern.
"""

import streamlit as st

from visionleaf_ai.processing.acquisition import get_image_manager
from visionleaf_ai.ui.pages import processing_lab
from visionleaf_ai.ui.session import init_session_state, set_image_session
from visionleaf_ai.ui.theme import inject_theme

init_session_state()
inject_theme()

if "_test_seeded" not in st.session_state:
    st.session_state["_test_seeded"] = True
    manager = get_image_manager()
    samples = manager.list_sample_images()
    if samples:
        set_image_session(manager.load_sample(samples[0]))

processing_lab.render()
