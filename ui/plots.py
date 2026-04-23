"""Matplotlib figures used by the PCIst Workbench.

All figures share the style established by ``theme.apply_matplotlib_defaults``
so callers don't need to tweak rcParams.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from .theme import SESSION_COLORS, ACCENT, OK, WARN, FAIL


# ── Preview timeline ───────────────────────────────────────────────────────
def session_timeline(preview: Dict[str, Any]):
    """Recording-level timeline: session blocks + Comment ticks."""
    total = preview.get("duration", 0.0) or 0.0
    sessions = preview.get("sessions", []) or []
    fig, ax = plt.subplots(figsize=(10, 1.6))

    ax.barh(0, total, height=0.3, color="#e9edf2", edgecolor="none")
    for i, s in enumerate(sessions):
        c = SESSION_COLORS[i % len(SESSION_COLORS)]
        ax.barh(
            0, s["duration"], left=s["start_time"],
            height=0.3, color=c, alpha=0.75, edgecolor="none",
        )
        mid = s["start_time"] + s["duration"] / 2
        ax.text(mid, 0, s["label"], ha="center", va="center",
                fontsize=9, fontweight="600", color="white")

    for cm in preview.get("comment_markers", []) or []:
        ax.axvline(cm["time_s"], color=ACCENT, ls="--", lw=0.6, alpha=0.5)

    ax.set_xlim(0, max(total, 1))
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig


# ── Cross-session PCIst bar ────────────────────────────────────────────────
def pcist_bar(sessions: List[Dict[str, Any]]):
    valid = [s for s in sessions if s.get("pcist") is not None]
    fig, ax = plt.subplots(figsize=(7, 3.0))
    if not valid:
        ax.text(0.5, 0.5, "No valid sessions",
                ha="center", va="center", transform=ax.transAxes, color="#999")
        return fig

    labels = [s["label"] for s in valid]
    values = [float(s["pcist"]) for s in valid]
    colors = [SESSION_COLORS[i % len(SESSION_COLORS)] for i in range(len(valid))]

    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="600", color="#0f1c26")
    ax.set_ylabel("PCIst (state transitions)")
    ax.set_ylim(0, max(values) * 1.18 + 1)
    ax.set_title("PCIst by session")
    fig.tight_layout()
    return fig


# ── Cross-session GFP overlay ──────────────────────────────────────────────
def gfp_overlay(sessions: List[Dict[str, Any]], art_win=(-2, 10)):
    gfp_sessions = [
        s for s in sessions
        if s.get("evoked_times") and s.get("evoked_gfp")
    ]
    fig, ax = plt.subplots(figsize=(10, 3.0))
    if len(gfp_sessions) < 2:
        ax.text(0.5, 0.5, "Need ≥2 valid sessions for overlay",
                ha="center", va="center", transform=ax.transAxes, color="#999")
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    for i, s in enumerate(gfp_sessions):
        t_ms = np.asarray(s["evoked_times"], dtype=float)
        gfp = np.asarray(s["evoked_gfp"], dtype=float)
        c = SESSION_COLORS[i % len(SESSION_COLORS)]
        pv = s.get("pcist")
        lbl = f'{s["label"]} (PCIst={pv:.1f})' if pv is not None else s["label"]
        ax.plot(t_ms, gfp, color=c, lw=1.5, label=lbl)
        ax.fill_between(t_ms, gfp, alpha=0.07, color=c)

    ax.axvline(0, color=FAIL, ls="--", lw=0.8)
    ax.axvspan(art_win[0], art_win[1], alpha=0.12, color="#fbe6e8")
    for pname, ptime in {"P30": 30, "N45": 45, "N100": 100, "P180": 180}.items():
        if -50 < ptime < 350:
            ax.axvline(ptime, color="#c8cfd6", ls=":", lw=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("GFP (µV)")
    ax.set_xlim(-50, 350)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Global Field Power — sessions overlaid")
    fig.tight_layout()
    return fig


# ── Per-session detail grid ────────────────────────────────────────────────
def session_detail(session: Dict[str, Any], art_win=(-2, 10)):
    """2×2 grid: evoked+GFP, SVD variance, per-component ΔNST, component SNRs."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))

    evoked = np.asarray(session.get("evoked_data") or [])
    t_ms = np.asarray(session.get("evoked_times") or [])
    gfp = np.asarray(session.get("evoked_gfp") or [])

    # (a) Evoked butterfly + GFP
    ax = axes[0, 0]
    if evoked.size and t_ms.size:
        n_ch = evoked.shape[0]
        alpha = max(0.08, min(0.5, 8.0 / max(n_ch, 1)))
        for i in range(n_ch):
            ax.plot(t_ms, evoked[i], color="#39475a", alpha=alpha, lw=0.4)
        if gfp.size:
            ax.plot(t_ms, gfp, color="#0f1c26", lw=1.6, label="GFP")
        ax.axvline(0, color=FAIL, ls="--", lw=1)
        ax.axvspan(art_win[0], art_win[1], alpha=0.15, color="#fbe6e8")
        ax.axvspan(0, 300, alpha=0.07, color="#e6f4ea")
        ax.set_xlim(-100, 350)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "no evoked data", ha="center", va="center",
                transform=ax.transAxes, color="#999")
    ax.set_title(f"(a) TEP butterfly + GFP")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude (µV)")

    # (b) SVD variance
    ax = axes[0, 1]
    var_exp = session.get("var_explained") or []
    if var_exp:
        x = np.arange(1, len(var_exp) + 1)
        ax.bar(x, var_exp, color=ACCENT, alpha=0.55, label="per-component")
        ax.plot(x, np.cumsum(var_exp), color="#0f1c26", marker="o",
                markersize=3.5, lw=1.2, label="cumulative")
        ax.set_xlabel("SVD component"); ax.set_ylabel("Variance (%)")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    else:
        ax.text(0.5, 0.5, "—", ha="center", va="center",
                transform=ax.transAxes, color="#999")
    ax.set_title("(b) SVD variance")

    # (c) Per-component ΔNST
    ax = axes[1, 0]
    dnst = session.get("dNST") or []
    comp_snrs = session.get("component_snrs") or []
    if dnst:
        x = np.arange(1, len(dnst) + 1)
        cs = comp_snrs if len(comp_snrs) == len(dnst) else [1.1] * len(dnst)
        colors = [ACCENT if s >= 1.1 else "#c8d6e0" for s in cs]
        ax.bar(x, dnst, color=colors, edgecolor="white")
        ax.set_xlabel("SVD component")
        ax.set_ylabel("ΔNST × n_response")
    else:
        ax.text(0.5, 0.5, "—", ha="center", va="center",
                transform=ax.transAxes, color="#999")
    ax.set_title("(c) Per-component complexity (ΔNST)")

    # (d) Component SNRs
    ax = axes[1, 1]
    if comp_snrs:
        x = np.arange(1, len(comp_snrs) + 1)
        colors = [OK if s >= 1.1 else WARN for s in comp_snrs]
        ax.bar(x, comp_snrs, color=colors, edgecolor="white")
        ax.axhline(1.1, color=OK, ls="--", lw=0.8, alpha=0.6,
                   label="min SNR = 1.1")
        ax.set_xlabel("SVD component"); ax.set_ylabel("SNR")
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "—", ha="center", va="center",
                transform=ax.transAxes, color="#999")
    ax.set_title("(d) Component SNRs")

    fig.tight_layout()
    return fig
