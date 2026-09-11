"""Test harness: renders the Acquisition page with a sample image loaded.

Not a real application entry point — used only by
tests/ui/test_acquisition_page.py via Streamlit's AppTest.from_file.
Pre-loads the first available sample image before rendering, so the
test can assert on the "image loaded" branch of the page.

The pre-load is guarded by a one-time seed flag rather than by
`session.is_loaded()`: this script reruns from the top on every
interaction (e.g. the reset button), just like the real app would, and
checking `is_loaded()` would re-seed a sample image immediately after
the user explicitly reset it — exactly the behavior the reset test
needs to rule out.
"""

import streamlit as st

from visionleaf_ai.processing.acquisition import get_image_manager
from visionleaf_ai.ui.pages import acquisition
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

acquisition.render()
