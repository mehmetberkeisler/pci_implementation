"""TMS-EEG PCIst Workbench - Streamlit entry point.

Thin orchestrator: page config, theme, tabs. All substantial UI lives in
``ui/`` submodules (sidebar, analyze_tab, results, plots, about, theme,
state). PCIst itself is the vendored renzocom/PCIst reference
implementation, called via ``pcist.calc_PCIst``.

Run:
    streamlit run app.py --server.port 8501 --server.address localhost
"""

from __future__ import annotations

import logging
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import streamlit as st

# Quiet MNE + matplotlib cache dirs (we don't use MNE here, but keep
# environment stable for any callers that do).
for var in ("MNE_DATA", "MNE_CACHE_DIR"):
    p = os.path.join(tempfile.gettempdir(), "mne_cache")
    os.makedirs(p, exist_ok=True)
    os.environ[var] = p
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config MUST come before any other st.* call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TMS-EEG PCIst Workbench",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import about as about_tab
from ui import analyze_tab
from ui import sidebar as sidebar_mod
from ui import state as state_mod
from ui import theme

theme.inject_css()
theme.apply_matplotlib_defaults()
state_mod.init()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">TMS-EEG · Perturbational Complexity</div>
      <p class="hero-title">PCIst Workbench</p>
      <p class="hero-sub">
        Sensor-space PCIst computed with the <code>renzocom/PCIst</code>
        reference implementation (Comolatti et&nbsp;al., Brain Stimulation, 2019).
        Upload a BrainVision triple - sessions are auto-detected and analysed
        with QC indicators visible alongside every result.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar (file upload + parameters + live preview build)
# ---------------------------------------------------------------------------
sidebar_mod.render()

# ---------------------------------------------------------------------------
# Main: two tabs - Analyse, About
# ---------------------------------------------------------------------------
tab_analyse, tab_about = st.tabs(["Analyse", "About PCIst"])

with tab_analyse:
    analyze_tab.render()

with tab_about:
    about_tab.render()
