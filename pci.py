"""
Perturbational Complexity Index (PCI) — Core Algorithm
======================================================

Reference implementation following:

    Casali AG, Gosseries O, Rosanova M, et al. (2013)
    "A Theoretically Based Index of Consciousness Independent of
     Sensory Processing and Behavior."
    Science Translational Medicine, 5(198), 198ra105.
    DOI: 10.1126/scitranslmed.3006294

Key equation (Eq. 2 from paper):

    PCI  =  c_L  ×  log₂(L)  /  [ L × H(L) ]

Where:
    c_L  : Lempel-Ziv complexity of the binary significance matrix SS(x,t)
    L    : L₁ × L₂  — total number of spatiotemporal samples
    H(L) : −p₁ log₂(p₁) − (1−p₁) log₂(1−p₁)  — source entropy
    p₁   : fraction of '1' entries in SS(x,t)

Processing pipeline:
    1. TMS artifact interpolation
    2. Bandpass filtering (0.1–45 Hz)
    3. Epoching & artifact rejection
    4. Source estimation (CSD or sensor-level)
    5. Bootstrap significance testing → binary matrix SS(x,t)
    6. Source sorting by activation (optimal ordination)
    7. Lempel-Ziv compression of SS
    8. Normalisation by source entropy → PCI

IMPORTANT NOTES:
    - CSD (surface Laplacian) is NOT equivalent to full source reconstruction
      used in the original paper (3-sphere BERG model). The published PCI
      thresholds (0.31 / 0.44) were calibrated with ~5000 cortical dipoles
      from an inverse solution.  When using CSD or sensor-level data the
      thresholds may require recalibration (see Casarotto et al., 2016).
    - The bootstrap uses *spatially-correlated* resampling (same random
      time indices across all sources) to preserve cross-source correlation
      structure, matching Global Maximum Statistics (Nichols & Holmes, 2002).
"""

import logging
import re
import warnings
import numpy as np
from typing import Tuple, Any, List, Dict, Optional, Union

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pci")


# ═══════════════════════════════════════════════════════════════════════════
# §1  SOURCE ENTROPY
# ═══════════════════════════════════════════════════════════════════════════

def compute_entropy(ss_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Compute binary source entropy H(L) and fraction of active samples p₁.

        H(L) = −p₁ log₂ p₁ − (1 − p₁) log₂(1 − p₁)        (Eq. 1)

    Parameters
    ----------
    ss_matrix : ndarray of {0, 1}, shape (n_sources, n_times)
        Binary significance matrix.

    Returns
    -------
    H : float
        Source entropy ∈ [0, 1].
    p1 : float
        Fraction of '1' entries ∈ [0, 1].
    """
    L = ss_matrix.size
    if L == 0:
        return 0.0, 0.0

    p1 = float(np.sum(ss_matrix)) / L

    if p1 == 0.0 or p1 == 1.0:
        return 0.0, p1

    H = -p1 * np.log2(p1) - (1.0 - p1) * np.log2(1.0 - p1)
    return float(H), float(p1)


# ═══════════════════════════════════════════════════════════════════════════
# §2  LEMPEL-ZIV COMPLEXITY (LZ76)
# ═══════════════════════════════════════════════════════════════════════════

def lempel_ziv_complexity(sequence: Union[str, np.ndarray, list]) -> int:
    """
    Compute the Lempel-Ziv complexity c_L of a binary sequence using the
    LZ76 exhaustive-parsing algorithm (Lempel & Ziv 1976; Kaspar &
    Schuster, Phys. Rev. A 36, 842, 1987).

    Algorithm
    ---------
    Scan the string left-to-right.  At each position *i*, find the
    longest prefix of s[i:] that already appears in the history s[0:i].
    The new *word* consists of this longest match plus one additional
    symbol.  c_L counts the number of such distinct words.

    Complexity
    ----------
    O(n² log n) average-case via Python's optimised C string search.

    Parameters
    ----------
    sequence : str, list[int], or 1-D ndarray
        Binary input (elements must be 0/1 or '0'/'1').

    Returns
    -------
    c : int
        LZ76 complexity (number of distinct words).
    """
    if isinstance(sequence, np.ndarray):
        s = "".join(str(int(x)) for x in sequence.ravel())
    elif isinstance(sequence, list):
        s = "".join(str(int(x)) for x in sequence)
    else:
        s = str(sequence)

    n = len(s)
    if n == 0:
        return 0
    if n == 1:
        return 1

    c = 1  # first symbol is always word #1
    i = 1  # scan position

    while i < n:
        max_match = 0
        for k in range(1, n - i + 1):
            if s[i : i + k] in s[:i]:
                max_match = k
            else:
                break  # longer prefixes cannot match if shorter did not
        # new word = matched prefix (length max_match) + 1 new symbol
        i += max_match + 1
        c += 1

    return c


# Backward-compatible alias
lempel_ziv_complexity_lz76 = lempel_ziv_complexity


def lempel_ziv_2d(ss_matrix: np.ndarray) -> int:
    """
    Compute Lempel-Ziv complexity of a 2-D binary matrix via column-major
    (Fortran-order) flattening followed by 1-D LZ76 parsing.

    Convention (Casali et al. 2013, Supplementary Section 3.2):
        The matrix SS(x, t) is scanned column by column — i.e. for each
        time sample t, the full spatial vector across sources is
        concatenated — before applying the Lempel-Ziv measure.

    Column-major flattening (``order='F'``) reproduces this scanning
    convention because it traverses all rows of column 0, then all rows
    of column 1, and so on.

    Parameters
    ----------
    ss_matrix : ndarray of {0, 1}, shape (n_sources, n_times)

    Returns
    -------
    c_L : int
        Lempel-Ziv complexity of the flattened binary sequence.
    """
    if ss_matrix.size == 0:
        return 0
    flat = ss_matrix.flatten(order="F")
    return lempel_ziv_complexity(flat)


# ═══════════════════════════════════════════════════════════════════════════
# §3  PCI COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _sort_sources(ss_matrix: np.ndarray) -> np.ndarray:
    """
    Sort rows of a binary matrix by descending total activation with
    deterministic tie-breaking that is independent of the original row
    order (permutation-invariant).

    Primary key : −activity  (descending total number of 1s)
    Secondary key: row content interpreted as a binary number (descending)

    This ensures that permuting the input rows before sorting always
    produces the identical sorted matrix.
    """
    n_sources, n_times = ss_matrix.shape
    activity = np.sum(ss_matrix, axis=1)  # (n_sources,)

    # Secondary key: interpret each row as a large integer (MSB-first).
    # For efficiency with wide matrices we use the dot-product with
    # descending powers of 2 clipped to float64 range.  For matrices
    # wider than ~1020 columns the large integers would overflow, so we
    # fall back to lexicographic tuple comparison.
    if n_times <= 1020:
        powers = 2.0 ** np.arange(n_times - 1, -1, -1)
        row_value = ss_matrix.astype(np.float64) @ powers  # unique per row content
        # Sort by (-activity, -row_value) → deterministic & order-independent
        sort_idx = np.lexsort((-row_value, -activity))
    else:
        # Fallback: pure Python tuple sort for very wide matrices
        keys = [(-int(activity[i]),) + tuple(-int(x) for x in ss_matrix[i]) for i in range(n_sources)]
        sort_idx = np.array(sorted(range(n_sources), key=lambda i: keys[i]))

    return ss_matrix[sort_idx, :]



def compute_pci(
    ss_matrix: np.ndarray,
    min_entropy: float = 0.08,
    min_p1: float = 0.01,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute the Perturbational Complexity Index from a binary significance
    matrix SS(x, t).

        PCI  =  c_L  ×  log₂(L)  /  [ L × H(L) ]         (Eq. 2)

    Sources are first sorted in descending order of total significant
    activation to approximate the optimal ordination that minimises
    spatial complexity (Supplementary Section 3.3).

    Quality gates (Supplementary Section 3.3):
        • p₁ must exceed the false-positive rate (α = 0.01), i.e. H > 0.08.
        • Sessions with SNR ≤ 1.4 at the sensor level were excluded.

    Parameters
    ----------
    ss_matrix : ndarray of {0, 1}, shape (n_sources, n_times)
    min_entropy : float
        Minimum source entropy to consider the signal valid.
    min_p1 : float
        Minimum fraction of active samples.
    verbose : bool
        Emit log messages.

    Returns
    -------
    result : dict
        pci     – float  : Perturbational Complexity Index
        pci_t   – float  : PCI^T  (temporal-domain variant, Section 3.4)
        c_L     – int    : raw LZ complexity (spatial)
        c_L_t   – int    : raw LZ complexity (temporal / transposed)
        H       – float  : source entropy H(L)
        p1      – float  : fraction of active samples
    """
    n_sources, n_times = ss_matrix.shape
    L = n_sources * n_times

    null = dict(pci=0.0, pci_t=0.0, c_L=0, c_L_t=0, H=0.0, p1=0.0)
    if L == 0:
        return null

    H, p1 = compute_entropy(ss_matrix)

    if verbose:
        logger.info(
            f"SS matrix: {n_sources}×{n_times} = L={L},  "
            f"H = {H:.4f},  p₁ = {p1:.4f}"
        )

    # Quality gate: reject if entropy is below threshold
    if H < min_entropy or p1 < min_p1:
        if verbose:
            logger.warning(
                f"Insufficient signal (H={H:.4f}, p₁={p1:.4f}). "
                f"Thresholds: H ≥ {min_entropy}, p₁ ≥ {min_p1}. → PCI = 0."
            )
        null["H"] = H
        null["p1"] = p1
        return null

    # --- Optimal ordination: sort sources by descending activity ----------
    # Uses deterministic tie-breaking (permutation-invariant).
    SS_sorted = _sort_sources(ss_matrix)

    # --- Spatial PCI (standard) -------------------------------------------
    c_L = lempel_ziv_2d(SS_sorted)
    log2L = np.log2(L)
    normaliser = L * H / log2L          # theoretical asymptotic c for random
    pci = c_L / normaliser if normaliser > 0 else 0.0

    # --- Temporal PCI^T (Section 3.4) -------------------------------------
    c_L_t = lempel_ziv_2d(SS_sorted.T)
    pci_t = c_L_t / normaliser if normaliser > 0 else 0.0

    if verbose:
        logger.info(
            f"c_L = {c_L},  normaliser = {normaliser:.1f}  →  PCI = {pci:.4f}  |  "
            f"c_L^T = {c_L_t}  →  PCI^T = {pci_t:.4f}"
        )

    return dict(
        pci=float(pci),
        pci_t=float(pci_t),
        c_L=int(c_L),
        c_L_t=int(c_L_t),
        H=float(H),
        p1=float(p1),
    )


def compute_pci_temporal(
    ss_matrix: np.ndarray,
    min_entropy: float = 0.08,
    min_p1: float = 0.01,
) -> np.ndarray:
    """
    Compute the temporal evolution PCI(t)  (cf. Fig. 3C).

        PCI(t) = c_i(t) × log₂ l(t) / [ l(t) × H(L) ]

    where l(t) = L₁ × t and c_i(t) is the LZ complexity of the first t
    columns of the (sorted) SS matrix.

    Parameters
    ----------
    ss_matrix : ndarray of {0, 1}, shape (n_sources, n_times)
    min_entropy : float
        Minimum source entropy (same gate as ``compute_pci``).
    min_p1 : float
        Minimum fraction of active samples.

    Returns
    -------
    pci_curve : ndarray of shape (n_times,)
        PCI(t) for t = 1 … L₂.
    """
    n_sources, n_times = ss_matrix.shape
    L = n_sources * n_times
    H, p1 = compute_entropy(ss_matrix)

    # Use the SAME quality gates as compute_pci for consistency
    if H < min_entropy or p1 < min_p1:
        return np.zeros(n_times)

    # Sort sources (same deterministic tie-breaking as compute_pci)
    SS_sorted = _sort_sources(ss_matrix)

    # Compute PCI at each time step
    pci_curve = np.zeros(n_times)
    for t in range(1, n_times + 1):
        sub = SS_sorted[:, :t]
        l_t = n_sources * t
        if l_t < 2:
            continue
        c_t = lempel_ziv_2d(sub)
        log2_lt = np.log2(l_t)
        # Normalise using the FULL-matrix entropy (H of entire SS)
        norm = l_t * H / log2_lt
        pci_curve[t - 1] = c_t / norm if norm > 0 else 0.0

    return pci_curve


# ═══════════════════════════════════════════════════════════════════════════
# §4  BOOTSTRAP SIGNIFICANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════

def compute_significance_matrix(
    source_data: np.ndarray,
    times: np.ndarray,
    n_bootstrap: int = 500,
    alpha: float = 0.01,
    baseline_window: Tuple[float, float] = (-0.5, -0.001),
    post_stim_window: Tuple[float, float] = (0.008, 0.300),
    seed: Optional[int] = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Construct binary significance matrix SS(x, t) via non-parametric
    bootstrap with Global Maximum Statistics correction.

    Procedure (Supplementary Materials, Section 2):
        1. Compute trial-averaged evoked response.
        2. Z-score each source against its baseline mean and std.
        3. Build null distribution: for each of *n_bootstrap* iterations
           resample baseline time-points **with the same random indices
           across all sources** (preserving spatial correlation), then
           take the global maximum |z| across all sources.
        4. Set threshold τ at the (1−α) percentile of the null distribution.
        5. SS(x, t) = 1  if  |z(x, t)| > τ,   0 otherwise.

    Parameters
    ----------
    source_data : ndarray, shape (n_sources, n_times, n_trials)
    times : ndarray, shape (n_times,)
        Time axis in seconds.
    n_bootstrap : int
        Number of bootstrap iterations (paper used 500).
    alpha : float
        Significance level (paper used 0.01).
    baseline_window : (float, float)
        Baseline period in seconds (paper: −500 to −1 ms).
    post_stim_window : (float, float)
        Post-stimulus period in seconds (paper: 8 to 300 ms).
    seed : int or None
        Random seed for reproducibility.  Set to *None* for a
        non-deterministic bootstrap.
    verbose : bool

    Returns
    -------
    SS : ndarray of {0, 1}, shape (n_sources, n_post_times)
    threshold : float
        Statistical threshold τ.
    diagnostics : dict
        Intermediate values for display/debugging.
    """
    n_src, n_times_total, n_trials = source_data.shape

    baseline_mask = (times >= baseline_window[0]) & (times < baseline_window[1])
    post_mask = (times >= post_stim_window[0]) & (times <= post_stim_window[1])

    n_baseline = int(np.sum(baseline_mask))
    n_post = int(np.sum(post_mask))

    if n_baseline < 10:
        raise ValueError(
            f"Baseline contains only {n_baseline} samples. "
            f"Window [{baseline_window[0]:.3f}, {baseline_window[1]:.3f}] s "
            f"does not overlap sufficiently with data "
            f"[{times[0]:.3f}, {times[-1]:.3f}] s."
        )
    if n_post < 10:
        raise ValueError(
            f"Post-stimulus contains only {n_post} samples. "
            f"Window [{post_stim_window[0]:.3f}, {post_stim_window[1]:.3f}] s "
            f"may be outside data range."
        )

    # --- Trial-averaged evoked response -----------------------------------
    evoked = np.mean(source_data, axis=2)  # (n_src, n_times_total)
    baseline_evoked = evoked[:, baseline_mask]  # (n_src, n_baseline)
    post_evoked = evoked[:, post_mask]  # (n_src, n_post)

    # --- Per-source baseline statistics -----------------------------------
    mu = np.mean(baseline_evoked, axis=1, keepdims=True)  # (n_src, 1)
    sigma = np.std(baseline_evoked, axis=1, keepdims=True, ddof=1)

    # --- Exclude dead / flat channels (σ ≈ 0) instead of flooring ---------
    min_sigma = 1e-12
    alive_mask = sigma.ravel() > min_sigma
    n_excluded = int(np.sum(~alive_mask))
    if n_excluded > 0:
        if verbose:
            logger.warning(
                f"Excluding {n_excluded}/{n_src} flat/dead source(s) "
                f"(σ ≤ {min_sigma:.0e})."
            )
        if not np.any(alive_mask):
            raise ValueError("All sources have zero variance — no usable data.")

    # Restrict to alive sources for z-scoring
    sigma_alive = sigma[alive_mask]  # (n_alive, 1)
    mu_alive = mu[alive_mask]

    # --- Z-scores (alive sources only) ------------------------------------
    z_post_alive = (post_evoked[alive_mask] - mu_alive) / sigma_alive
    z_baseline_alive = (baseline_evoked[alive_mask] - mu_alive) / sigma_alive

    if verbose:
        logger.info(
            f"Baseline: {n_baseline} samples,  "
            f"σ ∈ [{sigma_alive.min():.2e}, {sigma_alive.max():.2e}]  "
            f"({int(np.sum(alive_mask))}/{n_src} active sources)"
        )
        logger.info(
            f"Post-stimulus: {n_post} samples,  "
            f"max|z| = {np.max(np.abs(z_post_alive)):.2f}"
        )

    # --- Bootstrap null distribution (Global Maximum Statistics) -----------
    # CRITICAL: use the *same* random time indices across all sources to
    # preserve spatial correlation structure (Nichols & Holmes 2002).
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    n_alive = int(np.sum(alive_mask))
    null_maxima = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        # Same random column indices for every source (correlated resampling)
        idx = rng.randint(0, n_baseline, size=n_post)          # (n_post,)
        sampled = z_baseline_alive[:, idx]                      # (n_alive, n_post)
        null_maxima[b] = np.max(np.abs(sampled))

    threshold = float(np.percentile(null_maxima, (1.0 - alpha) * 100))

    if verbose:
        logger.info(
            f"Bootstrap (n={n_bootstrap}, α={alpha}, seed={seed}):  "
            f"τ = {threshold:.3f},  null mean = {null_maxima.mean():.2f},  "
            f"null std = {null_maxima.std():.2f}"
        )

    # --- Binary significance matrix (ALL sources; dead ones → 0) ----------
    SS_full = np.zeros((n_src, n_post), dtype=np.int8)
    SS_full[alive_mask] = (np.abs(z_post_alive) > threshold).astype(np.int8)
    active_pct = 100.0 * np.sum(SS_full) / SS_full.size

    if verbose:
        logger.info(
            f"SS({SS_full.shape[0]}×{SS_full.shape[1]}):  "
            f"active = {active_pct:.1f}%  ({np.sum(SS_full)}/{SS_full.size} samples)"
        )

    diagnostics = dict(
        n_baseline=n_baseline,
        n_post=n_post,
        n_sources_excluded=n_excluded,
        sigma_range=(float(sigma_alive.min()), float(sigma_alive.max())),
        z_post_max=float(np.max(np.abs(z_post_alive))),
        null_mean=float(null_maxima.mean()),
        null_std=float(null_maxima.std()),
    )

    return SS_full, threshold, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# §5  SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def validate_window_overlap(
    artifact_end_ms: float,
    post_stim_start_ms: float,
) -> None:
    """
    Warn when the TMS-artifact interpolation window overlaps the
    post-stimulus analysis window.

    If the artifact interpolation extends to, say, +10 ms but analysis
    starts at +8 ms, 2 ms of *interpolated* (artificial) data would enter
    the significance matrix.
    """
    if artifact_end_ms > post_stim_start_ms:
        overlap = artifact_end_ms - post_stim_start_ms
        warnings.warn(
            f"TMS artifact interpolation ends at +{artifact_end_ms:.0f} ms but "
            f"post-stimulus analysis starts at +{post_stim_start_ms:.0f} ms "
            f"({overlap:.0f} ms overlap).  Interpolated data will enter the "
            f"significance matrix.  Consider setting post-stimulus start "
            f"> {artifact_end_ms:.0f} ms or shortening the artifact window.",
            UserWarning,
            stacklevel=2,
        )


def interpolate_tms_artifact(
    raw: Any,
    events: np.ndarray,
    tms_id: int,
    window_ms: Tuple[float, float] = (-2, 10),
    verbose: bool = False,
) -> Any:
    """
    Remove TMS pulse artifact by linear interpolation.

    For each TMS event, the EEG samples within *window_ms* around the
    pulse are replaced by a linear ramp connecting the boundary samples.

    Parameters
    ----------
    raw : mne.io.Raw
    events : ndarray, shape (n_events, 3)
    tms_id : int
        Event ID corresponding to TMS pulses.
    window_ms : (float, float)
        Interpolation window relative to pulse onset (ms).
    verbose : bool

    Returns
    -------
    raw : mne.io.Raw
        Modified in-place.
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data()

    win_s = int(window_ms[0] * sfreq / 1000)
    win_e = int(window_ms[1] * sfreq / 1000)

    tms_samples = events[events[:, 2] == tms_id, 0]
    if len(tms_samples) == 0:
        return raw

    if verbose:
        logger.info(
            f"Interpolating {len(tms_samples)} TMS artifacts "
            f"[{window_ms[0]:.0f}, {window_ms[1]:.0f}] ms"
        )

    for sample in tms_samples:
        s0 = max(0, sample + win_s)
        s1 = min(data.shape[1] - 1, sample + win_e)
        if s0 > 0 and s1 < data.shape[1] - 1 and s1 > s0:
            n = s1 - s0 + 1
            for ch in range(data.shape[0]):
                ramp = np.linspace(data[ch, s0 - 1], data[ch, s1 + 1], n + 2)
                data[ch, s0 : s1 + 1] = ramp[1:-1]

    raw._data = data
    return raw


def compute_csd(epochs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Current Source Density transformation using MNE.

    CSD is a spatial filter that estimates cortical current density from
    scalp potentials by computing the surface Laplacian.  It attenuates
    volume-conducted signals and provides reference-free data.

    .. warning::
        CSD is **not** the same as the full inverse-solution source
        reconstruction (3-sphere BERG model) used in Casali et al. (2013).
        The published PCI thresholds (PCI* = 0.31, conscious ≥ 0.44) were
        validated with ~5 000 cortical dipoles.  CSD produces one "source"
        per EEG channel and captures a different spatial scale.  PCI values
        from CSD are **not directly comparable** to the published reference
        ranges.  See Casarotto et al. (2016) for sensor-level validation.

    Parameters
    ----------
    epochs : mne.Epochs

    Returns
    -------
    source_data : ndarray, shape (n_channels, n_times, n_epochs)
    times : ndarray, shape (n_times,)
    """
    import mne

    epochs_csd = mne.preprocessing.compute_current_source_density(
        epochs, stiffness=4, lambda2=1e-5, verbose=False
    )
    data = epochs_csd.get_data()  # (n_epochs, n_ch, n_times)
    return np.transpose(data, (1, 2, 0)), epochs_csd.times


def compute_snr(epochs: Any, verbose: bool = True) -> float:
    """
    Compute sensor-level signal-to-noise ratio.

        SNR = mean|amplitude|_post  /  mean(σ_baseline)

    Post-stimulus window: 25–300 ms.
    Baseline window: −400 to −10 ms.

    Sessions with SNR ≤ 1.4 were excluded in Casali et al. (2013)
    (Supplementary Fig. S2).

    Parameters
    ----------
    epochs : mne.Epochs
    verbose : bool

    Returns
    -------
    snr : float
    """
    evoked = epochs.average()
    data, times = evoked.data, evoked.times

    post_mask = (times >= 0.025) & (times <= 0.300)
    base_mask = (times >= -0.4) & (times < -0.01)

    if not np.any(post_mask) or np.sum(base_mask) < 10:
        return 0.0

    signal = np.mean(np.abs(data[:, post_mask]))
    noise = np.mean(np.std(data[:, base_mask], axis=1))
    snr = float(signal / noise) if noise > 0 else 0.0

    if verbose:
        logger.info(f"SNR = {snr:.2f}  (signal = {signal:.2e}, noise = {noise:.2e})")

    return snr


# ═══════════════════════════════════════════════════════════════════════════
# §6  TRIGGER CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def classify_trigger(name: str, count: int) -> Tuple[str, str]:
    """
    Classify an EEG event marker as TMS, RESPONSE, or OTHER.

    Parameters
    ----------
    name : str
        Event name from the recording system.
    count : int
        Number of occurrences of this event.

    Returns
    -------
    category : str
        One of ``'tms_likely'``, ``'tms_possible'``, ``'response'``,
        ``'annotation'``, ``'unknown_likely'``, ``'other'``.
    description : str
        Human-readable explanation.
    """
    low = " ".join(name.lower().strip().split())

    response_kw = ["response", "resp", "button", "key", "target", "correct", "incorrect"]
    response_label = bool(
        re.search(r"(^|[/\s])r\s*\d+($|[/\s])", low)
        or re.search(r"(^|[/\s])response", low)
    )
    if any(k in low for k in response_kw) or response_label:
        return "response", "Button-press / response marker"

    annotation_kw = ["comment", "new segment", "boundary", "annotation"]
    if any(k in low for k in annotation_kw):
        return "annotation", "Recording annotation"

    stim_label = bool(
        re.search(r"(^|[/\s])s\s*\d+($|[/\s])", low)
        or "stimulus/" in low
        or low.startswith("stim_")
    )
    tms_kw = ["stim", "tms", "pulse", "trigger"]
    if any(k in low for k in tms_kw) or stim_label:
        if 30 <= count <= 500:
            return "tms_likely", f"TMS trigger ({count} pulses)"
        return "tms_possible", f"Possible TMS ({count} events)"

    if 50 <= count <= 300:
        return "unknown_likely", f"Likely trigger ({count} events)"

    return "other", "Unclassified event"


def is_perturbation_trigger(category: str) -> bool:
    """
    Return True when a trigger category is suitable for PCI perturbation timing.
    """
    return category in {"tms_likely", "tms_possible", "unknown_likely"}


def split_event_blocks(
    event_samples: np.ndarray,
    sfreq: float,
    gap_seconds: float = 12.0,
) -> List[Tuple[int, int]]:
    """
    Split an ordered event train into blocks using long temporal gaps.

    Parameters
    ----------
    event_samples : ndarray, shape (n_events,)
        Event sample indices.
    sfreq : float
        Sampling frequency in Hz.
    gap_seconds : float
        Gap threshold (seconds). Consecutive events farther apart than this
        start a new block.

    Returns
    -------
    blocks : list[(start_idx, end_idx)]
        Inclusive index ranges over the sorted event train.
    """
    samples = np.asarray(event_samples, dtype=float).ravel()
    if samples.size == 0:
        return []
    if samples.size == 1:
        return [(0, 0)]

    samples = np.sort(samples)
    isi = np.diff(samples) / float(sfreq)
    boundaries = np.where(isi > gap_seconds)[0]

    starts = [0] + [int(i + 1) for i in boundaries]
    ends = [int(i) for i in boundaries] + [int(samples.size - 1)]
    return list(zip(starts, ends))


def detect_stimulation_like_response_train(
    event_samples: np.ndarray,
    sfreq: float,
    min_events: int = 30,
    min_block_events: int = 10,
    min_isi: float = 0.2,
    max_isi: float = 10.0,
    max_cv: float = 0.05,
    gap_seconds: float = 12.0,
    min_periodic_block_fraction: float = 0.8,
    min_periodic_event_fraction: float = 0.8,
) -> Dict[str, Any]:
    """
    Detect whether a response-labeled event train looks like periodic stimulation.

    This is useful for exception cases where hardware TTL pulses are saved as
    "Response/R xx" markers in BrainVision files.

    Parameters
    ----------
    event_samples : ndarray
        Event sample indices.
    sfreq : float
        Sampling frequency in Hz.
    min_events : int
        Minimum total events to consider.
    min_block_events : int
        Minimum events per block.
    min_isi, max_isi : float
        Acceptable ISI range (seconds).
    max_cv : float
        Maximum coefficient of variation of ISIs within a block.
        Default 0.05 (5 %) accommodates minor jitter from manual
        TMS triggering and low sampling rates.
    gap_seconds : float
        Inter-block gap threshold (seconds).

    Returns
    -------
    info : dict
        is_stimulation_like : bool
        n_events : int
        n_blocks : int
        blocks : list[dict] with keys:
            range, n_events, median_isi, cv_isi
    """
    samples = np.asarray(event_samples, dtype=float).ravel()
    samples = np.sort(samples)
    n_events = int(samples.size)

    info: Dict[str, Any] = dict(
        is_stimulation_like=False,
        n_events=n_events,
        n_blocks=0,
        blocks=[],
    )
    if n_events < min_events:
        return info

    blocks = split_event_blocks(samples, sfreq=sfreq, gap_seconds=gap_seconds)
    block_stats: List[Dict[str, Any]] = []
    periodic_blocks = 0
    periodic_events = 0
    qualified_blocks = 0
    qualified_events = 0

    for s, e in blocks:
        n_block = e - s + 1
        if n_block < min_block_events:
            block_stats.append(
                dict(range=(int(s), int(e)), n_events=int(n_block), median_isi=None, cv_isi=None)
            )
            continue

        seg = samples[s : e + 1]
        isi = np.diff(seg) / float(sfreq)
        med_isi = float(np.median(isi)) if isi.size > 0 else None
        mean_isi = float(np.mean(isi)) if isi.size > 0 else None
        cv_isi = float(np.std(isi) / mean_isi) if isi.size > 1 and mean_isi and mean_isi > 0 else None

        block_stats.append(
            dict(
                range=(int(s), int(e)),
                n_events=int(n_block),
                median_isi=med_isi,
                cv_isi=cv_isi,
            )
        )

        qualified_blocks += 1
        qualified_events += int(n_block)

        if (
            med_isi is not None
            and cv_isi is not None
            and min_isi <= med_isi <= max_isi
            and cv_isi <= max_cv
        ):
            periodic_blocks += 1
            periodic_events += int(n_block)

    info["blocks"] = block_stats
    info["n_blocks"] = len(block_stats)
    periodic_block_fraction = (
        float(periodic_blocks) / float(qualified_blocks)
        if qualified_blocks > 0 else 0.0
    )
    periodic_event_fraction = (
        float(periodic_events) / float(qualified_events)
        if qualified_events > 0 else 0.0
    )
    info["periodic_blocks"] = periodic_blocks
    info["qualified_blocks"] = qualified_blocks
    info["periodic_events"] = periodic_events
    info["qualified_events"] = qualified_events
    info["periodic_block_fraction"] = periodic_block_fraction
    info["periodic_event_fraction"] = periodic_event_fraction
    info["is_stimulation_like"] = (
        qualified_blocks > 0
        and periodic_blocks > 0
        and periodic_block_fraction >= min_periodic_block_fraction
        and periodic_event_fraction >= min_periodic_event_fraction
    )
    return info


# ═══════════════════════════════════════════════════════════════════════════
# §7  VALIDATION & QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════

def validate_results(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run quality checks on PCI results against expected physiological
    ranges from Casali et al. (2013).

    Returns
    -------
    dict with keys:
        checks   – dict[str, tuple(status, detail)]
        errors   – list[str]
        warnings – list[str]
        valid    – bool
    """
    checks: Dict[str, Tuple[str, str]] = {}
    errors: List[str] = []
    warnings: List[str] = []

    active = result.get("active_pct", 0)
    H = result.get("H", 0)
    thresh = result.get("threshold", 0)
    snr = result.get("snr", 0)
    pci = result.get("pci", 0)

    # --- Active percentage ------------------------------------------------
    if 4 <= active <= 15:
        checks["active"] = ("PASS", f"{active:.1f}%")
    elif active > 25:
        checks["active"] = ("FAIL", f"{active:.1f}% — excessively high")
        errors.append(
            "Active > 25%: likely wrong trigger or residual artifact."
        )
    elif active < 2:
        checks["active"] = ("FAIL", f"{active:.1f}% — insufficient")
        errors.append(
            "Active < 2%: weak or absent TMS-evoked cortical response."
        )
    else:
        checks["active"] = ("WARN", f"{active:.1f}%")

    # --- Source entropy ----------------------------------------------------
    if 0.15 <= H <= 0.60:
        checks["entropy"] = ("PASS", f"H = {H:.3f}")
    elif H > 0.70:
        checks["entropy"] = ("FAIL", f"H = {H:.3f} — noise-dominated")
        errors.append("H > 0.70: significance matrix dominated by noise.")
    elif H < 0.05:
        checks["entropy"] = ("FAIL", f"H = {H:.3f} — too low")
        errors.append("H < 0.05: insufficient cortical activation.")
    else:
        checks["entropy"] = ("WARN", f"H = {H:.3f}")

    # --- Threshold --------------------------------------------------------
    if 3.0 <= thresh <= 6.0:
        checks["threshold"] = ("PASS", f"\u03c4 = {thresh:.2f}")
    elif thresh > 8.0:
        checks["threshold"] = ("FAIL", f"\u03c4 = {thresh:.2f} — anomalous")
        errors.append("Threshold > 8: bootstrap error or artifact.")
    elif thresh < 2.5:
        checks["threshold"] = ("WARN", f"\u03c4 = {thresh:.2f} — low")
        warnings.append("Low threshold may admit false positives.")
    else:
        checks["threshold"] = ("WARN", f"\u03c4 = {thresh:.2f}")

    # --- SNR (Fig. S2 threshold: 1.4) ------------------------------------
    if snr >= 1.4:
        checks["snr"] = ("PASS", f"SNR = {snr:.2f}")
    elif snr >= 1.0:
        checks["snr"] = ("WARN", f"SNR = {snr:.2f}")
        warnings.append("Marginal SNR (1.0–1.4); interpretation with caution.")
    else:
        checks["snr"] = ("FAIL", f"SNR = {snr:.2f} — below threshold")
        errors.append("SNR < 1.0: data quality insufficient for reliable PCI.")

    # --- PCI value --------------------------------------------------------
    if pci > 1.10:
        checks["pci"] = ("FAIL", f"PCI = {pci:.3f} — out of range")
        errors.append("PCI > 1.0 is theoretically impossible; computation error.")
    elif pci >= 0.44:
        checks["pci"] = ("PASS", f"PCI = {pci:.3f}")
    elif pci >= 0.31:
        checks["pci"] = ("WARN", f"PCI = {pci:.3f}")
    else:
        checks["pci"] = ("INFO", f"PCI = {pci:.3f}")

    return dict(
        checks=checks,
        errors=errors,
        warnings=warnings,
        valid=len(errors) == 0,
    )


def validate_trigger_selection(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Diagnostics for possible wrong-trigger selection.

    Returns
    -------
    issues : list[dict]
        Each dict has 'severity', 'message', 'suggestion'.
    """
    issues: List[Dict[str, str]] = []

    active = result.get("active_pct", 0)
    snr = result.get("snr", 0)
    thresh = result.get("threshold", 0)

    if thresh > 8:
        issues.append(dict(
            severity="critical",
            message=f"Anomalous threshold (\u03c4 = {thresh:.1f})",
            suggestion="Bootstrap may have failed. Try different settings.",
        ))
    if snr < 1.0 and active < 2:
        issues.append(dict(
            severity="critical",
            message=f"Low SNR ({snr:.2f}) with minimal activation ({active:.1f}%)",
            suggestion="Verify that the selected trigger aligns with TMS pulses.",
        ))
    if active > 30:
        issues.append(dict(
            severity="critical",
            message=f"Excessive activation ({active:.1f}%)",
            suggestion="Wrong trigger? Response markers are button presses, not TMS.",
        ))

    return issues
