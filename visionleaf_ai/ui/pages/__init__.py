"""Streamlit pages package.

Purpose:
    Each module here defines one navigable page via a `render()`
    function. `app.py` imports these and wraps each in an `st.Page`.

Description:
    No re-exports here deliberately — `app.py` imports each page
    module by name so the mapping from page key to render function is
    explicit and easy to audit in one place (see `app.py`).

Dependencies:
    Each page module depends on visionleaf_ai.ui.components.

Public functions:
    None at the package level.
"""
