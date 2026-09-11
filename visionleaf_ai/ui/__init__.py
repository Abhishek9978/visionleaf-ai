"""UI package — Streamlit application shell.

Purpose:
    Everything needed to render VisionLeaf AI's front end: theming,
    session state, pipeline/page metadata, reusable components, and
    the pages themselves. Contains no DIP/ML logic (Milestone 2 scope).

Description:
    The entry point is the top-level `app.py`, not this package
    directly — `app.py` wires theme + session + navigation together.

Dependencies:
    streamlit.

Public functions:
    None at the package level — see theme.py, session.py,
    pipeline_stages.py, components/, and pages/.
"""
