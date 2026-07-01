"""The main "Analyze" tab.

Orchestrates the flow between preview (from the sidebar) and results. Owns
the "Run analysis" button and the call to ``analyze_pci.analyze_file``.
"""

from __future__ import annotations

import traceback
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_pci import analyze_file  # noqa: E402

from . import plots as plots_mod
from . import results as results_mod
from . import state as state_mod


def _empty_state() -> None:
    st.markdown(
        '<div class="card muted">'
        "Upload a <strong>BrainVision triple</strong> (.vhdr + .vmrk + .eeg) "
        "in the sidebar. The recording is analysed with the reference PCIst "
        "implementation from <code>renzocom/PCIst</code> (Comolatti et&nbsp;al. 2019). "
        "Stimulation sessions are auto-detected from marker timing."
        "</div>",
        unsafe_allow_html=True,
    )


def _preview_block(preview: dict) -> None:
    st.markdown("### Recording preview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Channels", f'{preview["n_channels"]}')
    c2.metric("Sampling rate", f'{preview["sfreq"]:.0f} Hz')
    c3.metric(
        "Duration",
        f'{preview["duration"]:.0f} s ({preview["duration"]/60:.1f} min)',
    )
    c4.metric(
        "Stim markers",
        f'{preview["n_stim"] or preview["n_resp"]}',
        delta=preview["marker_source"],
        delta_color="off",
    )

    sessions = preview.get("sessions") or []
    if sessions:
        st.markdown(
            f"Detected **{len(sessions)} session(s)**: "
            + " · ".join(
                f'**{s["label"]}** ({s["n_events"]} pulses)' for s in sessions
            )
        )
        fig = plots_mod.session_timeline(preview)
        st.pyplot(fig); plt.close(fig)
    else:
        st.info(
            "No distinct sessions detected — all stimulus events may belong to "
            "a single session, or the gap threshold is too large."
        )

    if preview["marker_source"] == "Response proxy":
        st.warning(
            "No explicit Stimulus markers were detected. The pipeline is "
            "using Response markers as TMS proxies because they form a "
            "periodic train. Confirm from the acquisition log that these are "
            "TMS TTL markers, not button presses."
        )


def _run_analysis() -> None:
    """Invoke analyze_file with current sidebar parameters, write to session state."""
    ss = st.session_state
    with st.spinner("Running PCIst pipeline — this takes ~30 s per session…"):
        try:
            _tms_marker = (ss.get("tms_marker") or "").strip() or None
            _tms_marker_type = (ss.get("tms_marker_type") or "").strip() or None
            _max_epochs = int(ss.get("max_epochs", 0)) or None
            _epoch_mode = ss.get("epoch_mode", "off")  # "off" | "cap" | "exact"
            result = analyze_file(
                ss["vhdr_path"],
                gap_seconds=float(ss["gap_seconds"]),
                reject_uv=float(ss["reject_uv"]),
                artifact_window_ms=(int(ss["artifact_start_ms"]),
                                    int(ss["artifact_end_ms"])),
                decimate_to=float(ss["decimate_to"]),
                min_snr=float(ss["min_snr_gate"]),
                tms_marker=_tms_marker,
                tms_marker_type=_tms_marker_type,
                max_epochs=_max_epochs,
                exact_epochs=(_epoch_mode == "exact"),
                dedup_gap_ms=float(ss.get("dedup_gap_ms", 10.0)),
                pcist_baseline_window=(-0.400, -0.050),
                pcist_response_window=(0.0, 0.300),
                pcist_k=float(ss["pcist_k"]),
                pcist_min_snr=float(ss["pcist_min_snr"]),
                pcist_max_var=float(ss["pcist_max_var"]),
                pcist_n_steps=int(ss["pcist_n_steps"]),
            )
            ss["result"] = result
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


def render() -> None:
    ss = st.session_state

    if not ss.get("vhdr_path") or ss.get("preview") is None:
        _empty_state()
        return

    _preview_block(ss["preview"])

    # Run button
    has_result = ss.get("result") is not None
    label = "Re-run analysis (updated parameters)" if has_result else "Run analysis"
    if st.button(label, key="run_analysis", type="primary", use_container_width=True):
        state_mod.reset_result()
        _run_analysis()
        st.rerun()

    # Results
    if ss.get("result") is not None:
        st.markdown("---")
        results_mod.render(ss["result"])
