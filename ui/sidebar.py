"""Sidebar: file upload, live preview, analysis parameters.

The sidebar is intentionally small. It does three things:

1. Takes a BrainVision triple (.vhdr/.vmrk/.eeg) and stashes them on disk.
2. Parses header + markers *without MNE* and builds a preview dict
   (``n_channels``, ``sfreq``, ``duration``, ``n_markers``, ``sessions``)
   so the main tab can show a recording summary immediately.
3. Exposes the analysis knobs behind a single collapsed expander — most
   users never need them.

The actual expensive work (epoching, PCIst per session) runs from the
main tab's "Run analysis" button, not here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_pci import (  # noqa: E402
    parse_vhdr,
    parse_vmrk,
    detect_sessions,
    _label_sessions_from_comments,
)

from . import state as state_mod


_PERSIST_DIR = os.path.join(tempfile.gettempdir(), "pcist_workbench")
os.makedirs(_PERSIST_DIR, exist_ok=True)


def _save_triple(uploaded: Dict[str, Any]) -> str:
    """Persist the uploaded .vhdr/.vmrk/.eeg to a stable tmp dir.

    The .vhdr references the other files by name; keeping all three in the
    same directory means we can feed the vhdr path to analyze_pci.
    """
    base = uploaded["vhdr"].name.rsplit(".", 1)[0]
    target = Path(_PERSIST_DIR) / base
    target.mkdir(exist_ok=True)
    for ext in ("vhdr", "vmrk", "eeg"):
        if ext in uploaded:
            f = uploaded[ext]
            f.seek(0)
            (target / f"{base}.{ext}").write_bytes(f.read())
            f.seek(0)
    return str(target / f"{base}.vhdr")


def _build_preview(vhdr_path: str, gap_seconds: float) -> Dict[str, Any]:
    """Header + marker summary for the preview card."""
    info = parse_vhdr(vhdr_path)
    base = os.path.dirname(vhdr_path)
    markers = []
    if info.get("marker_file"):
        markers = parse_vmrk(os.path.join(base, info["marker_file"]))

    stim_markers = [m for m in markers if m["type"] == "Stimulus"]
    resp_markers = [m for m in markers if m["type"] == "Response"]
    comment_markers = [m for m in markers if m["type"] == "Comment"]

    sfreq = info.get("sfreq", 0.0) or 0.0
    # Duration from .eeg file size
    fmt = info.get("binary_format", "IEEE_FLOAT_32")
    bps = 4 if fmt == "IEEE_FLOAT_32" else 2
    eeg_path = os.path.join(base, info["data_file"]) if info.get("data_file") else None
    n_samples = 0
    if eeg_path and os.path.exists(eeg_path):
        n_samples = os.path.getsize(eeg_path) // (max(info["n_channels"], 1) * bps)
    duration = n_samples / sfreq if sfreq else 0.0

    # Decide marker source and detect sessions (detect_sessions wants sample
    # positions + sfreq, not seconds).
    stim_positions = [int(m["position"]) for m in stim_markers]
    if not stim_positions and resp_markers:
        stim_positions = [int(m["position"]) for m in resp_markers]
        marker_source = "Response proxy"
    else:
        marker_source = "Stimulus"

    sessions = []
    if stim_positions and sfreq:
        sessions = detect_sessions(
            stim_positions, sfreq=sfreq, gap_seconds=gap_seconds
        ) or []
        # Mutates sessions in place; returns None — don't reassign.
        _label_sessions_from_comments(sessions, comment_markers, sfreq)

    return {
        "n_channels": info.get("n_channels", 0),
        "sfreq": sfreq,
        "duration": duration,
        "n_markers": len(markers),
        "n_stim": len(stim_markers),
        "n_resp": len(resp_markers),
        "n_comments": len(comment_markers),
        "marker_source": marker_source,
        "sessions": sessions,
        "comment_markers": [
            {"description": m["description"], "time_s": m["position"] / sfreq if sfreq else 0.0}
            for m in comment_markers
        ],
    }


def _file_uploader() -> Optional[Dict[str, Any]]:
    """Grab the .vhdr/.vmrk/.eeg triple from the user."""
    st.markdown("#### Upload recording")
    files = st.file_uploader(
        "Drop a BrainVision triple (.vhdr + .vmrk + .eeg)",
        type=["vhdr", "vmrk", "eeg"],
        accept_multiple_files=True,
        key="uploader",
        help="All three files are required. They should share the same filename stem.",
    )
    if not files:
        return None
    triple = {f.name.rsplit(".", 1)[-1].lower(): f for f in files}
    missing = [ext for ext in ("vhdr", "vmrk", "eeg") if ext not in triple]
    if missing:
        st.warning("Still waiting for: " + ", ".join(f".{m}" for m in missing))
        return None
    return triple


def render() -> None:
    """Render the whole sidebar and update session state in place."""
    with st.sidebar:
        st.markdown("### PCIst Workbench")
        st.caption("TMS-EEG perturbational complexity (Comolatti 2019).")

        triple = _file_uploader()
        if triple is not None:
            same_as_before = (
                st.session_state["files"] is not None
                and {ext: f.name for ext, f in triple.items()}
                == {ext: f.name for ext, f in st.session_state["files"].items()}
            )
            if not same_as_before:
                st.session_state["files"] = triple
                st.session_state["vhdr_path"] = _save_triple(triple)
                state_mod.reset_result()

        # Build / refresh preview whenever we have a vhdr path
        if st.session_state["vhdr_path"]:
            try:
                st.session_state["preview"] = _build_preview(
                    st.session_state["vhdr_path"],
                    gap_seconds=float(st.session_state["gap_seconds"]),
                )
            except Exception as e:  # pragma: no cover - UX fallback
                st.error(f"Couldn't read header/markers: {e}")
                st.session_state["preview"] = None

        st.markdown("---")
        st.markdown("#### Analysis parameters")
        with st.expander("Advanced (Comolatti defaults)", expanded=False):
            st.caption("Defaults follow Comolatti et al. 2019 TMS/EEG settings.")
            c1, c2 = st.columns(2)
            with c1:
                st.session_state["reject_uv"] = st.number_input(
                    "Reject ±µV", 50, 2000,
                    int(st.session_state["reject_uv"]), 25,
                    help="Peak-to-peak amplitude threshold for epoch rejection.",
                )
                st.session_state["gap_seconds"] = st.number_input(
                    "Session gap (s)", 10.0, 600.0,
                    float(st.session_state["gap_seconds"]), 10.0,
                    help="Minimum gap between stimulus trains to split sessions.",
                )
                st.session_state["artifact_start_ms"] = st.number_input(
                    "Artifact start (ms)", -20, 0,
                    int(st.session_state["artifact_start_ms"]), 1,
                )
                st.session_state["artifact_end_ms"] = st.number_input(
                    "Artifact end (ms)", 0, 50,
                    int(st.session_state["artifact_end_ms"]), 1,
                )
            with c2:
                st.session_state["decimate_to"] = st.selectbox(
                    "Target fs (Hz)", [500, 725, 1000, 1500, 2000],
                    index=[500, 725, 1000, 1500, 2000].index(
                        int(st.session_state["decimate_to"])
                    ),
                )
                st.session_state["pcist_k"] = st.number_input(
                    "k (noise control)", 1.0, 2.0,
                    float(st.session_state["pcist_k"]), 0.1, format="%.1f",
                )
                st.session_state["pcist_min_snr"] = st.number_input(
                    "min component SNR", 0.5, 3.0,
                    float(st.session_state["pcist_min_snr"]), 0.1, format="%.1f",
                )
                st.session_state["pcist_max_var"] = st.number_input(
                    "max variance (%)", 80.0, 100.0,
                    float(st.session_state["pcist_max_var"]), 1.0, format="%.0f",
                )
                st.session_state["pcist_n_steps"] = st.number_input(
                    "threshold steps", 10, 200,
                    int(st.session_state["pcist_n_steps"]), 10,
                )
                st.session_state["min_snr_gate"] = st.number_input(
                    "QC SNR gate", 0.5, 5.0,
                    float(st.session_state["min_snr_gate"]), 0.1, format="%.1f",
                    help="Sessions below this SNR are flagged UNRELIABLE.",
                )

        st.caption(
            "Pipeline: artifact interpolation → decimate → CAR → "
            "bandpass 0.1–45 Hz → epoch → PCIst (renzocom/PCIst reference)."
        )
