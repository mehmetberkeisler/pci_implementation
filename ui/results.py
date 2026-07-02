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


_INTERP_OPTIONS = {
    "drop": "Drop (exclude)",
    "interpolate_spline": "Interpolate: spherical spline (MNE)",
    "interpolate_neighbors": "Interpolate: average neighbors",
    "keep": "Keep as-is (include despite flag)",
}


def _rejection_drivers(sess: Dict[str, Any], flagged: Dict[str, Dict]) -> list:
    """Channels that drive epoch rejection but the variance detector did NOT
    flag (e.g. intermittent drifts/pops). Returns [(name, count, max_pp), ...]
    sorted by rejection count, excluding channels already flagged."""
    rs = sess.get("reject_stats") or {}
    names = rs.get("ch_names", []) or []
    counts = rs.get("ch_reject_count", []) or []
    pps = rs.get("ch_max_pp_uv", []) or []
    total = (rs.get("n_accepted", 0) + rs.get("n_rejected", 0)) or 1
    thresh = max(3, int(0.05 * total))  # >=5% of epochs, at least 3
    drivers = [
        (n, c, p)
        for n, c, p in zip(names, counts, pps)
        if c >= thresh and n not in flagged
    ]
    drivers.sort(key=lambda x: x[1], reverse=True)
    return drivers


def _render_bad_channel_manager(sess: Dict[str, Any], sess_key: str) -> None:
    """Per-channel action dropdowns for both variance-flagged channels and
    channels that drive epoch rejection."""
    bad_stats: Dict[str, Dict] = sess.get("bad_ch_stats") or {}
    flagged = {ch: s for ch, s in bad_stats.items() if s.get("flagged")}
    overrides_applied = sess.get("ch_overrides_applied") or {}
    drivers = _rejection_drivers(sess, flagged)

    if not flagged and not drivers:
        return

    ch_overrides = dict(st.session_state.get("ch_overrides") or {})
    changed = False

    def _row(ch: str, caption: str, default_action: str, key_suffix: str) -> None:
        nonlocal changed
        current = ch_overrides.get(ch, default_action)
        col1, col2, col3 = st.columns([2, 2, 4])
        col1.markdown(f"**{ch}**")
        col2.caption(caption)
        chosen = col3.selectbox(
            f"Action for {ch}",
            options=list(_INTERP_OPTIONS.keys()),
            index=list(_INTERP_OPTIONS.keys()).index(current),
            format_func=lambda k: _INTERP_OPTIONS[k],
            key=f"ch_override_{sess_key}_{key_suffix}_{ch}",
            label_visibility="collapsed",
        )
        if chosen != current:
            ch_overrides[ch] = chosen
            changed = True
        elif default_action != "keep" and ch not in ch_overrides:
            # record the effective default for flagged channels so it persists
            ch_overrides[ch] = default_action

    # ── Section 1: variance-flagged channels (default: drop) ────────────────
    if flagged:
        st.markdown(
            f"**Channel quality: {len(flagged)} flagged channel(s)** "
            f"(set an action per channel, then Re-run)"
        )
        st.caption("High-variance channels. Default action is Drop.")
        for ch, info in sorted(flagged.items()):
            reason = info.get("reason", "")
            rms = info.get("rms_uv", 0.0)
            ratio = info.get("var_ratio", 0.0)
            _row(ch, f"{reason}  \n{rms:.1f} µV RMS · {ratio:.1f}× median",
                 default_action="drop", key_suffix="flag")

    # ── Section 2: channels driving epoch rejection (default: keep) ─────────
    if drivers:
        total = (sess.get("reject_stats", {}).get("n_accepted", 0)
                 + sess.get("reject_stats", {}).get("n_rejected", 0))
        st.markdown(
            f"**Rejection-driving channels: {len(drivers)} channel(s)** "
            f"(not auto-flagged, but cause epoch loss)"
        )
        st.caption(
            "These passed the variance check but exceed the amplitude threshold "
            "in many epochs' baselines. Interpolating them can recover epochs. "
            "Default is Keep (no change)."
        )
        for ch, count, pp in drivers:
            _row(ch, f"caused {count}/{total} epoch rejections  \n"
                     f"max {pp:.0f} µV peak-to-peak",
                 default_action="keep", key_suffix="rej")

    if changed:
        st.session_state["ch_overrides"] = ch_overrides
        st.info("Action updated - press **Re-run analysis** to apply.")
    elif overrides_applied:
        shown = {ch: act for ch, act in overrides_applied.items()
                 if act and act != "keep"}
        if shown:
            st.caption(
                "Last run actions: "
                + ", ".join(
                    f"{ch}: *{_INTERP_OPTIONS.get(act, act)}*"
                    for ch, act in shown.items()
                )
            )


def _render_pipeline_log(result: Dict[str, Any]) -> None:
    """Collapsed expander showing captured logger output from the pipeline run."""
    lines = result.get("pipeline_log") or []
    if not lines:
        return
    with st.expander("Pipeline log", expanded=False):
        st.code("\n".join(lines), language=None)


def _render_rejection_details(sess: Dict[str, Any]) -> None:
    """Expandable artifact rejection breakdown for one session."""
    rs = sess.get("reject_stats")
    if not rs:
        return

    n_acc = rs.get("n_accepted", 0)
    n_rej = rs.get("n_rejected", 0)
    total = n_acc + n_rej
    thresh = rs.get("threshold_uv", 150)
    win = rs.get("window_ms", (-500, 0))
    ch_names = rs.get("ch_names", [])
    ch_rej = rs.get("ch_reject_count", [])
    ch_pp = rs.get("ch_max_pp_uv", [])

    pct = 100 * n_rej / total if total else 0
    st.markdown(
        f"**Artifact rejection: {n_acc}/{total} accepted** "
        f"({pct:.0f}% rejected) · threshold {thresh:.0f} µV · "
        f"window {win[0]} to {win[1]} ms (baseline)"
    )
    st.caption(
        f"**Method:** peak-to-peak amplitude in baseline window "
        f"[{win[0]} ms, {win[1]} ms]. Epoch rejected if ANY channel "
        f"exceeds **{thresh:.0f} µV**."
    )

    post_thr = rs.get("threshold_post_uv")
    if post_thr:
        pwin = rs.get("post_window_ms", (0, 300))
        n_post = rs.get("n_rejected_post", 0)
        st.caption(
            f"**Post-stimulus check active:** epochs also rejected if any "
            f"channel exceeds **{post_thr:.0f} µV** peak-to-peak in the "
            f"response window [{pwin[0]} ms, {pwin[1]} ms]. "
            f"{n_post} epoch(s) rejected on this criterion alone."
        )
    else:
        st.caption(
            "Note: rejection uses the baseline window only. Post-stimulus "
            "artifacts in the response window are not caught here; enable "
            "ICA or the optional post-stimulus rejection to control them."
        )

    if ch_names and ch_rej:
        # Build per-channel table, sorted by rejection count descending
        rows = sorted(
            zip(ch_names, ch_rej, ch_pp),
            key=lambda x: x[1],
            reverse=True,
        )
        offenders = [(n, c, p) for n, c, p in rows if c > 0]
        if offenders:
            st.markdown("**Top offending channels:**")
            table_rows = [
                {"Channel": n, "Epochs rejected": c, "Max p-p (µV)": f"{p:.1f}"}
                for n, c, p in offenders[:15]
            ]
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
        else:
            st.success("No channel exceeded the threshold - all epochs accepted.")
    else:
        st.info("Channel-level breakdown not available.")


def _qc_status(sess: Dict[str, Any]) -> str:
    if sess.get("error") or sess.get("pcist") is None or sess.get("n_components", 0) == 0:
        return "fail"
    if sess.get("warnings") or not sess.get("snr_pass", True):
        return "warn"
    return "ok"


def _session_card(sess: Dict[str, Any]) -> str:
    status = _qc_status(sess)
    pv = sess.get("pcist")
    score = f"{pv:.1f}" if pv is not None else "-"
    # n_used = epochs actually fed to PCIst (after any epoch cap); fall back to
    # n_accepted for results produced before n_used was tracked.
    n_acc = int(sess.get("n_used", sess.get("n_accepted", 0)))
    n_evt = int(sess.get("n_events", 0))
    n_bad = len(sess.get("bad_channels", []) or [])
    snr = float(sess.get("snr", 0.0))
    label = html.escape(str(sess.get("label", "session")))
    badge = {"ok": "ok", "warn": "review", "fail": "failed"}[status]
    # NOTE: no leading whitespace on any line - Streamlit's markdown parser
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
                "Epochs_used", "Epochs_accepted", "Epochs_total",
                "Bad_channels", "Warnings"])
    for s in valid:
        w.writerow([
            s["label"],
            f'{s["pcist"]:.6f}',
            s.get("n_components", ""),
            f'{s.get("snr", 0):.4f}',
            "Yes" if s.get("snr_pass", True) else "No",
            s.get("n_used", s.get("n_accepted", 0)),
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
                    "trigger_timing", "n_used", "n_accepted", "n_events",
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
        n_used = s.get("n_channels_used", "-")
        rows.append({
            "Session": s["label"],
            "PCIst": f'{pv:.2f}' if pv is not None else "-",
            "n_components": s.get("n_components", "-"),
            "SNR": f'{s.get("snr", 0):.2f}' + (" ✓" if s.get("snr_pass", True) else " ✗"),
            "Epochs": f'{s.get("n_used", s.get("n_accepted", 0))}/{s.get("n_events", 0)}',
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
    _render_pipeline_log(result)

    # 2. Session cards (big PCIst numbers, always visible)
    st.markdown("### Session results")
    cards = "".join(_session_card(s) for s in sessions)
    st.markdown(f'<div class="session-grid">{cards}</div>', unsafe_allow_html=True)
    _interpretation_note()

    valid = [s for s in sessions if s.get("pcist") is not None]
    failed = [s for s in sessions if s.get("pcist") is None]

    # 3. Cross-session comparison (or a single-session hero when there is one)
    if valid:
        if len(valid) == 1:
            fig = plots_mod.single_session_summary(valid[0], art_win=art_win)
            st.pyplot(fig); plt.close(fig)
        else:
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
                f'{s["label"]} - PCIst = {s["pcist"]:.1f}'
                f'  · {s.get("n_components", "?")} components'
                f'  · {s.get("n_used", s.get("n_accepted", 0))}/{s.get("n_events", 0)} epochs'
            )
            with st.expander(title, expanded=(i == 0)):
                for w in s.get("warnings", []) or []:
                    st.warning(w)
                _render_bad_channel_manager(s, sess_key=str(i))
                _render_rejection_details(s)
                fig = plots_mod.session_detail(s, art_win=art_win)
                st.pyplot(fig); plt.close(fig)

    # 7. Failed sessions
    if failed:
        st.markdown("### Failed sessions")
        for i, f in enumerate(failed):
            st.warning(
                f'**{f["label"]}** - {f.get("error", "Unknown error")} '
                f'(events: {f.get("n_events", 0)}, accepted: {f.get("n_accepted", 0)})'
            )
            _render_bad_channel_manager(f, sess_key=f"failed_{i}")
            _render_rejection_details(f)
