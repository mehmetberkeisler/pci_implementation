"""TMS-EEG PCIst Workbench - UI modules.

The Streamlit entry point is ``app.py`` at the repository root. Each
submodule below owns one visual concern:

- ``theme``      - CSS + matplotlib defaults (single source of look & feel)
- ``state``      - session-state keys and their default values
- ``sidebar``    - file upload, live preview, analysis parameters
- ``results``    - session cards, summary table, export buttons
- ``plots``      - matplotlib figures (timeline, bar, per-session grid, GFP)
- ``about``      - methodology tab content
"""
