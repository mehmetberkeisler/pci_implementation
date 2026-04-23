"""
PCIst wrapper — Perturbational Complexity Index based on State Transitions.

This module is a thin adapter around the **vendored reference implementation**
from Renzo Comolatti's repository (`third_party/pcist/pci_st.py`,
https://github.com/renzocom/PCIst). The reference code itself is not modified;
everything below is unit conversion, input validation, and packaging the result
in the shape the rest of this pipeline (and the Streamlit UI) expect.

Reference
---------
Comolatti R, Pigorini A, Casarotto S, Fecchio M, Faria G, Sarasso S,
Rosanova M, Gosseries O, Boly M, Bodart O, Ledoux D, Brichant J-F,
Nobili L, Laureys S, Tononi G, Massimini M, Casali AG (2019).
*A fast and general method to empirically estimate the complexity of brain
responses to transcranial and intracranial stimulations.*
Brain Stimulation, 12(5), 1280-1289. https://doi.org/10.1016/j.brs.2019.05.013

Design notes
------------
- The reference `calc_PCIst` expects **milliseconds** for `times`, and for the
  `baseline_window` / `response_window` tuples. Our pipeline works in seconds,
  so the wrapper converts at the boundary.
- The reference returns `dNST` as a Python list (after a comprehension); we
  always return lists/floats so the result dict is JSON-serialisable.
- Any keys the pipeline consumes are built here explicitly — if the upstream
  schema drifts on a refresh, this file is the only thing to update.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make the vendored reference importable regardless of where this file is used
# from (CLI, Streamlit, tests).
_VENDORED = Path(__file__).resolve().parent / "third_party"
if str(_VENDORED) not in sys.path:
    sys.path.insert(0, str(_VENDORED))

from pcist_ref.pci_st import calc_PCIst as _ref_calc_PCIst  # noqa: E402

logger = logging.getLogger("pcist")

PCIST_METHOD = "recurrence_normalized_state_transitions"  # kept for back-compat


def _null_result(**extra: Any) -> Dict[str, Any]:
    """Shape-complete zero-PCIst result (used for NaN / empty / degenerate input)."""
    result: Dict[str, Any] = {
        "PCIst": 0.0,
        "n_components": 0,
        "dNST": [],
        "var_explained": [],
        "snrs": [],
        "cumvar": [],
        "components_kept": [],
        "max_thresholds": [],
        "NST_diff": [],
        "pcist_method": PCIST_METHOD,
    }
    result.update(extra)
    return result


def calc_PCIst(
    evoked: np.ndarray,
    times: np.ndarray,
    baseline_window: Tuple[float, float] = (-0.400, -0.050),
    response_window: Tuple[float, float] = (0.000, 0.300),
    k: float = 1.2,
    min_snr: float = 1.1,
    max_var: float = 99.0,
    n_steps: int = 100,
    embed: bool = False,
    L: Optional[int] = None,
    tau: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute PCIst via the vendored Comolatti 2019 reference implementation.

    Parameters
    ----------
    evoked : ndarray, shape (n_channels, n_times)
        Trial-averaged TMS-EEG evoked response (µV).
    times : ndarray, shape (n_times,)
        Time vector in **seconds**, aligned so the TMS pulse is at t = 0.
    baseline_window, response_window : (float, float)
        Intervals in **seconds**. Defaults follow Comolatti 2019 TMS/EEG.
    k : float
        Baseline penalty factor (default 1.2).
    min_snr : float
        Minimum per-component SNR = sqrt(mean(response²) / mean(baseline²)).
    max_var : float
        Cumulative variance (%) retained from SVD of the response.
    n_steps : int
        Number of distance thresholds to scan per component.
    embed, L, tau : advanced time-delay-embedding options (off by default).

    Returns
    -------
    dict
        PCIst, n_components, dNST, var_explained, snrs, cumvar, components_kept,
        max_thresholds, NST_diff, pcist_method.
    """
    evoked = np.asarray(evoked, dtype=float)
    times = np.asarray(times, dtype=float)

    if evoked.ndim != 2:
        raise ValueError(f"`evoked` must be 2D (channels × time); got {evoked.ndim}D.")
    if evoked.shape[1] != times.shape[0]:
        raise ValueError(
            f"evoked.shape[1]={evoked.shape[1]} does not match len(times)={times.shape[0]}."
        )

    if np.any(np.isnan(evoked)):
        logger.warning("[PCIst] evoked contains NaN — returning PCIst = 0.")
        return _null_result()

    # ── Convert seconds → milliseconds for the reference implementation ────
    times_ms = times * 1000.0
    bw_ms = (baseline_window[0] * 1000.0, baseline_window[1] * 1000.0)
    rw_ms = (response_window[0] * 1000.0, response_window[1] * 1000.0)

    # Sanity checks on sample counts
    n_base = int(np.sum((times >= baseline_window[0]) & (times < baseline_window[1])))
    n_resp = int(np.sum((times >= response_window[0]) & (times <= response_window[1])))
    if n_base < 5 or n_resp < 5:
        raise ValueError(
            f"Too few samples in analysis windows "
            f"(baseline={n_base}, response={n_resp}). Need ≥5 each."
        )

    par: Dict[str, Any] = dict(
        baseline_window=bw_ms,
        response_window=rw_ms,
        k=k,
        min_snr=min_snr,
        max_var=max_var,
        n_steps=n_steps,
        embed=embed,
    )
    if embed:
        par.update(L=L, tau=tau)

    ref = _ref_calc_PCIst(evoked, times_ms, full_return=True, **par)

    # The reference returns 0 (not a dict) on NaN — guard that shape.
    if not isinstance(ref, dict):
        logger.warning("[PCIst] reference returned scalar fallback — PCIst = 0.")
        return _null_result()

    pcist_value = float(ref.get("PCI", 0.0))
    dNST = [float(x) for x in np.asarray(ref.get("dNST", []), dtype=float).ravel()]
    n_components = int(ref.get("n_dims", len(dNST)))

    eigenvalues = np.asarray(ref.get("eigenvalues", []), dtype=float)
    if eigenvalues.size > 0:
        total = float(np.sum(eigenvalues ** 2))
        if total > 0:
            var_full = 100.0 * (eigenvalues ** 2) / total
            cumvar_full = np.cumsum(var_full)
        else:
            var_full = np.zeros_like(eigenvalues)
            cumvar_full = np.zeros_like(eigenvalues)
    else:
        var_full = np.array([])
        cumvar_full = np.array([])

    # `var_exp` from the reference is already truncated to the selected set;
    # we additionally expose the first n_components of the full-rank variance
    # explained so the UI's cumvar chart is informative.
    var_explained = var_full[: max(n_components, 1)].tolist() if var_full.size else []
    cumvar = cumvar_full[: max(n_components, 1)].tolist() if cumvar_full.size else []

    snrs_ref = np.asarray(ref.get("snrs", []), dtype=float).ravel().tolist()

    max_thr = np.asarray(ref.get("max_thresholds", []), dtype=float).ravel().tolist()

    nst_diff = ref.get("NST_diff", None)
    if isinstance(nst_diff, np.ndarray):
        nst_diff = nst_diff.tolist()
    else:
        nst_diff = []

    # The reference does not expose which original SVD component indices survived
    # the SNR filter; surviving components are packed densely 0..n_components-1.
    components_kept = list(range(n_components))

    logger.info(
        f"[PCIst] = {pcist_value:.4f}  ({n_components} components, "
        f"dNST={', '.join(f'{d:.2f}' for d in dNST)})"
    )

    return {
        "PCIst": pcist_value,
        "n_components": n_components,
        "dNST": dNST,
        "var_explained": var_explained,
        "snrs": snrs_ref,
        "cumvar": cumvar,
        "components_kept": components_kept,
        "max_thresholds": max_thr,
        "NST_diff": nst_diff,
        "pcist_method": PCIST_METHOD,
    }


__all__ = ["calc_PCIst", "PCIST_METHOD"]
