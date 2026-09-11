"""Test harness: renders the Processing Lab page in its empty state.

Not a real application entry point — used only by
tests/ui/test_processing_lab_page.py via Streamlit's AppTest.from_file.
"""

from visionleaf_ai.ui.pages import processing_lab
from visionleaf_ai.ui.session import init_session_state
from visionleaf_ai.ui.theme import inject_theme

init_session_state()
inject_theme()
processing_lab.render()
