"""Visual theme for the PCIst Workbench.

A single place to tune colour, typography, spacing, and matplotlib defaults.
The CSS is intentionally minimal — enough to make the Streamlit defaults
feel modern and consistent, nothing more.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st


# ── Palette ────────────────────────────────────────────────────────────────
INK_900 = "#0f1c26"
INK_700 = "#2a3e4d"
INK_500 = "#566878"
INK_300 = "#9fb0bd"
BG_SURFACE = "#ffffff"
BG_MUTED = "#f6f8fa"
BG_CARD = "#ffffff"
ACCENT = "#1e6091"         # deep sea blue
ACCENT_WEAK = "#e4eff6"
OK = "#2a7f3b"
WARN = "#b36b00"
FAIL = "#b02a37"
SESSION_COLORS = ["#1e6091", "#b36b00", "#2a7f3b", "#6a3b8d", "#a02457"]


_CSS = f"""
<style>
  :root {{
    --ink-900: {INK_900};
    --ink-700: {INK_700};
    --ink-500: {INK_500};
    --ink-300: {INK_300};
    --bg-surface: {BG_SURFACE};
    --bg-muted: {BG_MUTED};
    --bg-card: {BG_CARD};
    --accent: {ACCENT};
    --accent-weak: {ACCENT_WEAK};
    --ok: {OK};
    --warn: {WARN};
    --fail: {FAIL};
  }}

  html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                 Roboto, Helvetica, Arial, sans-serif;
    color: var(--ink-900);
  }}

  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}

  h1, h2, h3, h4 {{ color: var(--ink-900); letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.65rem; font-weight: 650; margin-bottom: 0.2rem; }}
  h2 {{ font-size: 1.20rem; font-weight: 600; margin-top: 1.6rem; }}
  h3 {{ font-size: 1.02rem; font-weight: 600; margin-top: 1.4rem; }}

  /* Hero */
  .hero {{
    border-bottom: 1px solid #e3e8ed;
    padding-bottom: 1rem; margin-bottom: 1.6rem;
  }}
  .hero-eyebrow {{
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--accent); font-weight: 600; margin-bottom: 0.35rem;
  }}
  .hero-title {{ font-size: 1.7rem; font-weight: 650; margin: 0; }}
  .hero-sub   {{ color: var(--ink-500); margin: 0.3rem 0 0 0; font-size: 0.95rem; }}

  /* Card primitives */
  .card {{
    background: var(--bg-card);
    border: 1px solid #e3e8ed;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
  }}
  .card.muted {{ background: var(--bg-muted); }}

  /* Session grid */
  .session-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 0.9rem; margin: 0.5rem 0 1rem 0;
  }}
  .session-card {{
    background: var(--bg-card); border: 1px solid #e3e8ed;
    border-left: 4px solid var(--ink-300); border-radius: 10px;
    padding: 0.9rem 1rem;
  }}
  .session-card.ok   {{ border-left-color: var(--ok); }}
  .session-card.warn {{ border-left-color: var(--warn); }}
  .session-card.fail {{ border-left-color: var(--fail); }}

  .session-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.3rem; }}
  .session-name {{ font-weight: 600; font-size: 0.95rem; color: var(--ink-900); }}
  .badge {{
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    padding: 2px 8px; border-radius: 999px; letter-spacing: 0.06em;
  }}
  .badge.ok   {{ background: #e6f4ea; color: var(--ok); }}
  .badge.warn {{ background: #fdf0e2; color: var(--warn); }}
  .badge.fail {{ background: #fbe6e8; color: var(--fail); }}

  .score {{ font-size: 2.0rem; font-weight: 700; color: var(--ink-900); line-height: 1.05; }}
  .score-sub {{ font-size: 0.72rem; color: var(--ink-500); text-transform: uppercase;
                letter-spacing: 0.08em; margin-top: 0.1rem; }}

  .mini-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.4rem;
               margin-top: 0.7rem; }}
  .mini {{ text-align: left; }}
  .mini-label {{ font-size: 0.66rem; color: var(--ink-500); text-transform: uppercase;
                 letter-spacing: 0.08em; }}
  .mini-value {{ font-size: 0.95rem; font-weight: 600; color: var(--ink-900); }}

  .note {{ background: var(--bg-muted); border-left: 3px solid var(--accent);
           padding: 0.55rem 0.8rem; border-radius: 6px; font-size: 0.82rem;
           color: var(--ink-700); margin: 0.4rem 0; }}

  /* De-emphasize Streamlit stock chrome we do not need */
  header[data-testid="stHeader"] {{ background: transparent; }}
  div[data-testid="stSidebarUserContent"] {{ padding-top: 1.4rem; }}

  /* Metrics */
  div[data-testid="stMetricLabel"] {{ color: var(--ink-500); font-size: 0.72rem;
                                      text-transform: uppercase; letter-spacing: 0.08em; }}
  div[data-testid="stMetricValue"] {{ color: var(--ink-900); font-size: 1.25rem; font-weight: 600; }}

  /* Buttons */
  .stButton > button {{ border-radius: 8px; font-weight: 600; }}
  .stDownloadButton > button {{ border-radius: 8px; }}

  /* Caption under figures */
  .fig-caption {{ font-size: 0.78rem; color: var(--ink-500); margin: 0.2rem 0 1rem 0; }}
</style>
"""


def inject_css() -> None:
    """Call once at app start (after ``st.set_page_config``)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def apply_matplotlib_defaults() -> None:
    """Matplotlib defaults that match the web theme."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Inter", "Helvetica Neue", "Arial", "DejaVu Sans",
        ],
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "600",
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#d6dde3",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": INK_500,
        "ytick.color": INK_500,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.color": "#edf0f3",
        "grid.linewidth": 0.5,
        "figure.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    })
