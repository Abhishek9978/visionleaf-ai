"""Test harness: renders the Acquisition page in its empty state.

Not a real application entry point — used only by
tests/ui/test_acquisition_page.py via Streamlit's AppTest.from_file,
which requires an actual script path (a callable-based st.Page can't
be navigated to directly in tests; see that test module's docstring).
"""

from visionleaf_ai.ui.pages import acquisition
from visionleaf_ai.ui.session import init_session_state
from visionleaf_ai.ui.theme import inject_theme

init_session_state()
inject_theme()
acquisition.render()
