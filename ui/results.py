"""Result rendering: recording summary, session cards, exports, per-session detail.

Reads the dict returned by ``analyze_pci.analyze_file`` and produces the
main-area UI. All plot code lives in ``ui.plots``; this module is layout,
text, tables, and exports only.
"""

from __future__ import annotations

import csv
import html
import io
import json as _json
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import streamlit as st

from . import plots as plots_mod


def _qc_status(sess: Dict[str, Any]) -> str:
    if sess.get("error") or sess.get("pcist") is None or sess.get("n_components", 0) == 0:
        return "fail"
    if sess.get("warnings") or not sess.get("snr_pass", True):
        return "warn"
    return "ok"


def _session_card(sess: Dict[str, Any]) -> str:
    status = _qc_status(sess)
    pv = sess.get("pcist")
    score = f"{pv:.1f}" if pv is not None else "—"
    n_acc = int(sess.get("n_accepted", 0))
    n_evt = int(sess.get("n_events", 0))
    n_bad = len(sess.get("bad_channels", []) or [])
    snr = float(sess.get("snr", 0.0))
    label = html.escape(str(sess.get("label", "session")))
    badge = {"ok": "ok", "warn": "review", "fail": "failed"}[status]
    # NOTE: no leading whitespace on any line — Streamlit's markdown parser
    # treats 4+ space indented lines as code blocks and would print the raw
    # HTML verbatim instead of rendering it.
    return (
        f'<div class="session-card {status}">'
        f'<div class="session-head">'
        f'<div class="session-name">{label}</div>'
        f'<span class="badge {status}">{badge}</span>'
        f'</div>'
        f'<div class="score">{score}</div>'
        f'<div class="score-sub">PCIst · state transitions</div>'
        f'<div class="mini-row">'
        f'<div class="mini"><div class="mini-label">SNR</div>'
        f'<div class="mini-value">{snr:.2f}</div></div>'
        f'<div class="mini"><div class="mini-label">Epochs</div>'
        f'<div class="mini-value">{n_acc}/{n_evt}</div></div>'
        f'<div class="mini"><div class="mini-label">Bad ch.</div>'
        f'<div class="mini-value">{n_bad}</div></div>'
        f'</div>'
        f'</div>'
    )


def _recording_summary(res: Dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Channels", f'{res["n_channels"]}')
    c2.metric(
        "Sampling rate",
        f'{res["sfreq"]:.0f} → {res.get("sfreq_proc", res["sfreq"]):.0f} Hz',
    )
    c3.metric(
        "Duration",
        f'{res["duration"]:.0f} s ({res["duration"]/60:.1f} min)',
    )
    c4.metric("Total stimuli", f'{res.get("n_stim_total", 0)}')
    c5.metric("Sessions", f'{len(res.get("sessions", []))}')


def _interpretation_note() -> None:
    st.markdown(
        '<div class="note">'
        '<strong>Interpretation guardrail.</strong> '
        "PCIst here is sensor-space (Comolatti 2019 reference implementation). "
        "Use it for within-study comparisons. Do not apply the source-space "
        "Casali 2013 LZ-PCI thresholds (0.31, 0.44) to these values without a "
        "pipeline-specific validation."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_exports(res: Dict[str, Any]) -> None:
    sessions = res.get("sessions", [])
    valid = [s for s in sessions if s.get("pcist") is not None]
    if not valid:
        return

    # CSV
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Session", "PCIst", "n_components", "SNR", "SNR_pass",
                "Epochs_accepted", "Epochs_total", "Bad_channels", "Warnings"])
    for s in valid:
        w.writerow([
            s["label"],
            f'{s["pcist"]:.6f}',
            s.get("n_components", ""),
            f'{s.get("snr", 0):.4f}',
            "Yes" if s.get("snr_pass", True) else "No",
            s.get("n_accepted", 0),
            s.get("n_events", 0),
            "|".join(s.get("bad_channels", []) or []),
            "|".join(s.get("warnings", []) or []),
        ])

    # JSON
    json_export = {
        "recording": {
            "n_channels": res.get("n_channels"),
            "sfreq": res.get("sfreq"),
            "sfreq_proc": res.get("sfreq_proc"),
            "duration_s": res.get("duration"),
            "excluded_channels": res.get("excluded_channels", []),
        },
        "parameters": {
            "reject_uv": res.get("reject_uv"),
            "artifact_window_ms": list(res.get("artifact_window_ms", (-2, 10))),
            "min_snr": res.get("min_snr"),
            "pcist": {
                "baseline_window": list(res.get("pcist_baseline_window", (-0.400, -0.050))),
                "response_window": list(res.get("pcist_response_window", (0.0, 0.300))),
                "k": res.get("pcist_k"),
                "min_snr": res.get("pcist_min_snr"),
                "max_var": res.get("pcist_max_var"),
                "n_steps": res.get("pcist_n_steps"),
            },
        },
        "sessions": [
            {
                k: s.get(k) for k in (
                    "label", "pcist", "n_components", "dNST",
                    "var_explained", "component_snrs", "cumvar",
                    "components_kept", "snr", "snr_pass",
                    "trigger_timing", "n_accepted", "n_events",
                    "bad_channels", "warnings",
                    "start_time", "end_time",
                )
            }
            for s in valid
        ],
    }

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download CSV summary", buf.getvalue(),
            file_name="pcist_results.csv", mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download JSON report",
            _json.dumps(json_export, indent=2, default=float),
            file_name="pcist_report.json", mime="application/json",
            use_container_width=True,
        )


def _render_summary_table(sessions: List[Dict[str, Any]]) -> None:
    rows = []
    for s in sessions:
        pv = s.get("pcist")
        n_bad = len(s.get("bad_channels", []) or [])
        n_used = s.get("n_channels_used", "—")
        rows.append({
            "Session": s["label"],
            "PCIst": f'{pv:.2f}' if pv is not None else "—",
            "n_components": s.get("n_components", "—"),
            "SNR": f'{s.get("snr", 0):.2f}' + (" ✓" if s.get("snr_pass", True) else " ✗"),
            "Epochs": f'{s.get("n_accepted", 0)}/{s.get("n_events", 0)}',
            "Channels": f'{n_used}' + (f' (-{n_bad})' if n_bad else ''),
            "Warnings": "·" if not s.get("warnings") else str(len(s["warnings"])),
        })
    st.table(rows)


def render(result: Dict[str, Any]) -> None:
    """Render the full result block."""
    sessions = result.get("sessions", [])
    art_win = tuple(result.get("artifact_window_ms", (-2, 10)))

    # 1. Recording summary
    st.markdown("### Recording")
    _recording_summary(result)

    # 2. Session cards (big PCIst numbers, always visible)
    st.markdown("### Session results")
    cards = "".join(_session_card(s) for s in sessions)
    st.markdown(f'<div class="session-grid">{cards}</div>', unsafe_allow_html=True)
    _interpretation_note()

    valid = [s for s in sessions if s.get("pcist") is not None]
    failed = [s for s in sessions if s.get("pcist") is None]

    # 3. Cross-session comparison
    if valid:
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = plots_mod.pcist_bar(valid)
            st.pyplot(fig); plt.close(fig)
        with c2:
            fig = plots_mod.gfp_overlay(valid, art_win=art_win)
            st.pyplot(fig); plt.close(fig)

        # 4. Summary table
        with st.expander("Summary table", expanded=True):
            _render_summary_table(valid)

        # 5. Exports
        with st.expander("Export", expanded=False):
            _render_exports(result)

        # 6. Per-session detail
        st.markdown("### Per-session detail")
        for i, s in enumerate(valid):
            title = (
                f'{s["label"]} — PCIst = {s["pcist"]:.1f}'
                f'  · {s.get("n_components", "?")} components'
                f'  · {s.get("n_accepted", 0)}/{s.get("n_events", 0)} epochs'
            )
            with st.expander(title, expanded=(i == 0)):
                for w in s.get("warnings", []) or []:
                    st.warning(w)
                fig = plots_mod.session_detail(s, art_win=art_win)
                st.pyplot(fig); plt.close(fig)

    # 7. Failed sessions
    if failed:
        st.markdown("### Failed sessions")
        for f in failed:
            st.warning(
                f'**{f["label"]}** — {f.get("error", "Unknown error")} '
                f'(events: {f.get("n_events", 0)}, accepted: {f.get("n_accepted", 0)})'
            )
