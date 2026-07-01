"""Session-state keys and defaults for the PCIst Workbench.

Centralising these means the sidebar / results modules can rely on
``st.session_state[k]`` being present without defensive lookups, and
there is exactly one place to look when reasoning about what the app
remembers across reruns.
"""

from __future__ import annotations

import streamlit as st


DEFAULTS = {
    # Upload state
    "files": None,             # dict[str, UploadedFile]  ({"vhdr":..,"vmrk":..,"eeg":..})
    "vhdr_path": None,          # str; path under a persistent tmp dir
    # Preview (parsed from header/markers, no MNE)
    "preview": None,            # dict | None
    # Analysis parameters (user-overridable in sidebar)
    "reject_uv": 150,
    "decimate_to": 1000,
    "gap_seconds": 60.0,
    "artifact_start_ms": -2,
    "artifact_end_ms": 10,
    "pcist_k": 1.2,
    "pcist_min_snr": 1.1,
    "pcist_max_var": 99.0,
    "pcist_n_steps": 100,
    "min_snr_gate": 1.4,
    # Marker selection
    "tms_marker": "",          # e.g. "R256"; empty = auto-detect
    "tms_marker_type": "",     # "Stimulus" or "Response"; empty = auto-detect
    "dedup_gap_ms": 10.0,      # strip duplicate markers within this window
    # Epoch balancing
    "epoch_balance_enabled": False,   # toggle for the epoch cap section
    "max_epochs": 0,                  # 0 = disabled; e.g. 60 to cap all subjects at 60
    # Result
    "result": None,             # dict from analyze_pci.analyze_file | None
}


def init() -> None:
    """Populate any missing session-state keys with their defaults."""
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_result() -> None:
    st.session_state["result"] = None
