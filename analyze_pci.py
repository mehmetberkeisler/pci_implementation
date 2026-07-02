#!/usr/bin/env python3
"""
TMS-EEG PCIst Analyzer - Single-file BrainVision loader with session detection.
Computes PCIst (Perturbational Complexity Index based on State Transitions)
per Comolatti et al. 2019 (Brain Stimulation, 12(5):1280-1289).

Standalone implementation - requires only numpy (+ standard library).
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("pcist_analyzer")


class _ListHandler(logging.Handler):
    """Capture log records into a list so we can return them from analyze_file."""
    def __init__(self):
        super().__init__()
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))

# ═══════════════════════════════════════════════════════════════════════════
# §1 BRAINVISION FILE PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_vhdr(vhdr_path: str) -> Dict[str, Any]:
    """Parse BrainVision header file (.vhdr)."""
    info = {
        "data_file": None,
        "marker_file": None,
        "data_format": "BINARY",
        "data_orientation": "MULTIPLEXED",
        "n_channels": 0,
        "sampling_interval_us": 0.0,
        "sfreq": 0.0,
        "binary_format": "IEEE_FLOAT_32",
        "channels": [],
    }

    with open(vhdr_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("["):
            section = line.strip("[]").strip()
            continue

        if section == "Common Infos":
            if line.startswith("DataFile="):
                info["data_file"] = line.split("=", 1)[1].strip()
            elif line.startswith("MarkerFile="):
                info["marker_file"] = line.split("=", 1)[1].strip()
            elif line.startswith("NumberOfChannels="):
                info["n_channels"] = int(line.split("=", 1)[1].strip())
            elif line.startswith("SamplingInterval="):
                info["sampling_interval_us"] = float(line.split("=", 1)[1].strip())
                info["sfreq"] = 1e6 / info["sampling_interval_us"]
            elif line.startswith("DataOrientation="):
                info["data_orientation"] = line.split("=", 1)[1].strip()

        elif section == "Binary Infos":
            if line.startswith("BinaryFormat="):
                info["binary_format"] = line.split("=", 1)[1].strip()

        elif section == "Channel Infos":
            m = re.match(r"Ch(\d+)=(.+)", line)
            if m:
                parts = m.group(2).split(",")
                ch_name = parts[0].strip() if parts else f"Ch{m.group(1)}"
                unit = parts[3].strip() if len(parts) > 3 else "µV"
                resolution = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 1.0
                info["channels"].append({
                    "name": ch_name,
                    "unit": unit,
                    "resolution": resolution,
                })

    return info


def parse_vmrk(vmrk_path: str) -> List[Dict[str, Any]]:
    """Parse BrainVision marker file (.vmrk)."""
    markers = []

    with open(vmrk_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("["):
            section = line.strip("[]").strip()
            continue

        if section == "Marker Infos":
            m = re.match(r"Mk(\d+)=(.+)", line)
            if m:
                parts = m.group(2).split(",")
                if len(parts) >= 3:
                    markers.append({
                        "number": int(m.group(1)),
                        "type": parts[0].strip(),
                        "description": parts[1].strip(),
                        "position": int(parts[2].strip()),
                        "size": int(parts[3].strip()) if len(parts) > 3 and parts[3].strip() else 0,
                        "channel": int(parts[4].strip()) if len(parts) > 4 and parts[4].strip() else 0,
                    })

    return markers


def load_eeg_data(eeg_path: str, info: Dict[str, Any]) -> np.ndarray:
    """Load binary EEG data from .eeg file. Uses float32 to save memory."""
    n_ch = info["n_channels"]
    fmt = info["binary_format"]

    if fmt == "IEEE_FLOAT_32":
        dtype = np.float32
    elif fmt == "INT_16":
        dtype = np.int16
    else:
        raise ValueError(f"Unsupported binary format: {fmt}")

    file_size = os.path.getsize(eeg_path)
    bytes_per_sample = np.dtype(dtype).itemsize
    n_samples_total = file_size // (n_ch * bytes_per_sample)

    data = np.fromfile(eeg_path, dtype=dtype)

    if info["data_orientation"] == "MULTIPLEXED":
        data = data.reshape(n_samples_total, n_ch).T  # (n_ch, n_samples)
    else:
        data = data.reshape(n_ch, n_samples_total)

    # Apply resolution - use float32 to save memory on large files
    data = data.astype(np.float32)
    for i, ch in enumerate(info["channels"]):
        if ch["resolution"] != 0:
            data[i] *= ch["resolution"]

    return data


def load_eeg_segment(eeg_path: str, info: Dict[str, Any],
                     start_sample: int, end_sample: int,
                     channel_mask: Optional[List[bool]] = None) -> np.ndarray:
    """Load a segment of EEG data from .eeg file (memory-efficient)."""
    n_ch = info["n_channels"]
    fmt = info["binary_format"]

    if fmt == "IEEE_FLOAT_32":
        dtype = np.float32
    elif fmt == "INT_16":
        dtype = np.int16
    else:
        raise ValueError(f"Unsupported binary format: {fmt}")

    bytes_per_sample = np.dtype(dtype).itemsize
    frame_size = n_ch * bytes_per_sample
    n_samples = end_sample - start_sample

    with open(eeg_path, "rb") as f:
        f.seek(start_sample * frame_size)
        raw_bytes = f.read(n_samples * frame_size)

    data = np.frombuffer(raw_bytes, dtype=dtype).reshape(n_samples, n_ch).T.astype(np.float32)

    # Apply resolution
    for i, ch in enumerate(info["channels"]):
        if ch["resolution"] != 0:
            data[i] *= ch["resolution"]

    # Apply channel mask
    if channel_mask is not None:
        data = data[channel_mask]

    return data


def load_brainvision(vhdr_path: str) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """Load complete BrainVision dataset (header + data + markers)."""
    vhdr_path = str(Path(vhdr_path).resolve())
    base_dir = str(Path(vhdr_path).parent)

    info = parse_vhdr(vhdr_path)

    eeg_path = os.path.join(base_dir, info["data_file"])
    data = load_eeg_data(eeg_path, info)

    markers = []
    if info["marker_file"]:
        vmrk_path = os.path.join(base_dir, info["marker_file"])
        markers = parse_vmrk(vmrk_path)

    logger.info(
        f"Loaded: {data.shape[0]} channels x {data.shape[1]} samples "
        f"@ {info['sfreq']:.1f} Hz, {len(markers)} markers"
    )

    return data, info, markers


# ═══════════════════════════════════════════════════════════════════════════
# §2 SIGNAL PROCESSING (no MNE)
# ═══════════════════════════════════════════════════════════════════════════

def bandpass_filter(data: np.ndarray, sfreq: float, low: float = 0.1, high: float = 45.0, order: int = 4) -> np.ndarray:
    """Zero-phase bandpass filter using forward-backward FFT filtering.

    Applies the frequency-domain filter twice (forward + time-reversed) to
    achieve zero phase distortion, analogous to scipy.signal.filtfilt or
    MNE's default zero-phase FIR.  The squared amplitude response ensures
    sharper rolloff without TEP latency shifts.

    Transition bands use cosine tapers to prevent ringing (Gibbs phenomenon):
        Low:  0.05-0.1 Hz  (cosine taper)
        High: 45-49.5 Hz   (cosine taper)
    """
    from numpy.fft import rfft, irfft, rfftfreq

    n_ch, n_samples = data.shape
    freqs = rfftfreq(n_samples, d=1.0/sfreq)

    # Create frequency domain filter
    filt = np.ones(len(freqs))
    if low > 0:
        filt[freqs < low] = 0
    filt[freqs > high] = 0

    # Smooth transitions (cosine taper)
    if low > 0:
        trans_width = min(low * 0.5, 0.05)
        trans_low = (freqs >= low - trans_width) & (freqs < low)
        if np.any(trans_low) and trans_width > 0:
            filt[trans_low] = 0.5 * (1 + np.cos(np.pi * (freqs[trans_low] - low) / trans_width))

    trans_high_width = min(high * 0.1, 5.0)
    trans_high = (freqs > high) & (freqs <= high + trans_high_width)
    if np.any(trans_high) and trans_high_width > 0:
        filt[trans_high] = 0.5 * (1 + np.cos(np.pi * (freqs[trans_high] - high) / trans_high_width))

    # Forward-backward filtering for zero phase distortion
    filtered = np.zeros_like(data)
    for i in range(n_ch):
        # Forward pass
        spectrum = rfft(data[i])
        forward = irfft(spectrum * filt, n=n_samples)
        # Backward pass (time-reverse, filter, time-reverse)
        spectrum_rev = rfft(forward[::-1])
        backward_rev = irfft(spectrum_rev * filt, n=n_samples)
        filtered[i] = backward_rev[::-1]

    return filtered


def interpolate_tms_artifact(
    data: np.ndarray,
    stim_positions: List[int],
    sfreq: float,
    window_ms: Tuple[float, float] = (-2, 10),
    method: str = "cubic",
) -> np.ndarray:
    """Remove TMS pulse artifact by cubic spline or linear interpolation.

    Cubic spline (default) uses 4 boundary points on each side for a smooth
    transition, following TESA / Comolatti 2019 PCIst recommendations.
    """
    n_ch, n_samples = data.shape
    win_s = int(window_ms[0] * sfreq / 1000)
    win_e = int(window_ms[1] * sfreq / 1000)

    for pos in stim_positions:
        s0 = max(0, pos + win_s)
        s1 = min(n_samples - 1, pos + win_e)
        if s0 < 2 or s1 >= n_samples - 2 or s1 <= s0:
            continue

        n = s1 - s0 + 1

        if method == "cubic" and s0 >= 4 and s1 < n_samples - 4:
            # Cubic spline using 4 boundary points on each side
            n_boundary = 4
            x_pre = np.arange(-n_boundary, 0)
            x_post = np.arange(n, n + n_boundary)
            x_fit = np.concatenate([x_pre, x_post])
            x_interp = np.arange(n)

            for ch in range(n_ch):
                y_pre = data[ch, s0 - n_boundary:s0]
                y_post = data[ch, s1 + 1:s1 + 1 + n_boundary]
                y_fit = np.concatenate([y_pre, y_post]).astype(np.float64)
                # Fit cubic polynomial
                coeffs = np.polyfit(x_fit, y_fit, 3)
                data[ch, s0:s1 + 1] = np.polyval(coeffs, x_interp).astype(np.float32)
        else:
            # Fallback: linear interpolation
            for ch in range(n_ch):
                ramp = np.linspace(data[ch, s0 - 1], data[ch, s1 + 1], n + 2)
                data[ch, s0:s1 + 1] = ramp[1:-1]

    return data


def average_rereference(data: np.ndarray) -> np.ndarray:
    """Apply common average reference (CAR).

    Per 2023 Brain Stimulation consensus guidelines for TMS-EEG,
    average re-referencing spreads forward modelling error evenly
    across channels and is standard for PCI computation.
    """
    mean_across_ch = np.mean(data, axis=0, keepdims=True)
    return data - mean_across_ch


def detect_bad_channels(
    data: np.ndarray,
    ch_names: List[str],
    sfreq: float,
    variance_threshold: float = 5.0,
) -> Tuple[List[bool], List[str], Dict[str, str], Dict[str, Dict]]:
    """Detect bad channels based on variance statistics.

    Returns
    -------
    mask       : list[bool] - True = good channel
    bad_names  : list[str]
    reasons    : dict name → short reason string
    stats      : dict name → {rms_uv, var_ratio, reason, flagged}
    """
    n_ch = data.shape[0]
    ch_var = np.var(data, axis=1)
    ch_rms = np.sqrt(ch_var)
    median_var = float(np.median(ch_var))

    mask = [True] * n_ch
    bad_names: List[str] = []
    reasons: Dict[str, str] = {}
    stats: Dict[str, Dict] = {}

    for i in range(n_ch):
        v = float(ch_var[i])
        ratio = v / median_var if median_var > 0 else 0.0
        # Data is already in µV throughout the pipeline (BrainVision resolution
        # applied at load; the reject_uv threshold operates on these same values).
        rms_uv = float(ch_rms[i])
        flagged = False
        reason = ""

        if median_var > 0 and v > variance_threshold * median_var:
            mask[i] = False
            bad_names.append(ch_names[i])
            reason = f"NOISY ({ratio:.1f}× median variance)"
            reasons[ch_names[i]] = reason
            flagged = True
        elif v < median_var * 0.01:
            mask[i] = False
            bad_names.append(ch_names[i])
            reason = "DEAD (near-zero variance)"
            reasons[ch_names[i]] = reason
            flagged = True

        stats[ch_names[i]] = {
            "rms_uv": round(rms_uv, 2),
            "var_ratio": round(ratio, 2),
            "reason": reason,
            "flagged": flagged,
        }

    return mask, bad_names, reasons, stats


def interpolate_bad_channels(
    data: np.ndarray,
    ch_names: List[str],
    channels_to_interp: List[str],
    method: str = "spline",
    sfreq: float = 1000.0,
    exclude_from_basis: Optional[List[str]] = None,
) -> np.ndarray:
    """Interpolate bad channels in-place using MNE (spline) or neighbor average.

    Parameters
    ----------
    method : "spline" | "neighbors"
        "spline"    - MNE spherical spline (standard 10-20 montage auto-matched)
        "neighbors" - unweighted average of nearest channels by name heuristic
    exclude_from_basis : list[str] | None
        Other flagged-bad channels that must NOT be used as reference when
        reconstructing ``channels_to_interp``. Critical: without this, a
        catastrophically noisy channel that is about to be dropped would still
        poison the spline/neighbour fit of the channels we want to keep.
    """
    if not channels_to_interp:
        return data

    data = data.copy()
    exclude = set(exclude_from_basis or [])

    if method == "spline":
        try:
            import mne
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
            raw_tmp = mne.io.RawArray(data.astype(np.float64), info, verbose=False)
            # Try to match standard montage for channel positions
            montage = mne.channels.make_standard_montage("standard_1020")
            try:
                raw_tmp.set_montage(montage, match_case=False,
                                    on_missing="ignore", verbose=False)
            except Exception:
                pass
            # Mark the interpolation targets AND every other flagged-bad channel
            # as bad, so MNE reconstructs from the genuinely-clean channels only.
            # The excluded channels get interpolated values too, but the caller
            # either drops them or re-interpolates them separately.
            raw_tmp.info["bads"] = list(dict.fromkeys(list(channels_to_interp) + list(exclude)))
            raw_tmp.interpolate_bads(reset_bads=True, verbose=False)
            out = raw_tmp.get_data().astype(data.dtype)
            # Only keep the reconstructed values for the requested targets;
            # leave excluded channels untouched (caller handles them).
            idx = {n: i for i, n in enumerate(ch_names)}
            for ch in channels_to_interp:
                if ch in idx:
                    data[idx[ch]] = out[idx[ch]]
            return data
        except Exception as e:
            logger.warning(f"  [INTERP] Spline failed ({e}), falling back to neighbors")
            method = "neighbors"

    if method == "neighbors":
        ch_idx = {n: i for i, n in enumerate(ch_names)}
        # Basis excludes the interpolation targets themselves AND any other
        # flagged-bad channel passed in exclude_from_basis.
        good_idx = [
            i for i, n in enumerate(ch_names)
            if n not in channels_to_interp and n not in exclude
        ]
        for ch in channels_to_interp:
            if ch not in ch_idx:
                continue
            if not good_idx:
                logger.warning(f"  [INTERP] No good channels for neighbor avg of {ch}")
                continue
            # Use up to 4 nearest channels by index distance (simple heuristic)
            ci = ch_idx[ch]
            nearest = sorted(good_idx, key=lambda x: abs(x - ci))[:4]
            data[ci] = np.mean(data[nearest], axis=0)
            logger.info(f"  [INTERP] {ch} ← avg of {[ch_names[n] for n in nearest]}")

    return data


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition, same as scipy.stats.kurtosis default)."""
    mean = np.mean(x)
    m2 = np.mean((x - mean) ** 2)
    m4 = np.mean((x - mean) ** 4)
    return float(m4 / (m2 ** 2) - 3.0) if m2 > 0 else 0.0


def _apply_ica(
    seg_data: np.ndarray,
    ch_names: List[str],
    sfreq: float,
    n_components: int = 20,
    kurtosis_thresh: float = 5.0,
) -> Tuple[np.ndarray, List[int]]:
    """Fit FastICA and remove high-kurtosis artifact components.

    Returns (cleaned_data_uv, excluded_component_indices).
    Falls back to (original_data, []) if MNE is unavailable or ICA fails.
    """
    try:
        import mne
        from mne.preprocessing import ICA as _ICA
    except ImportError:
        logger.warning("  [ICA] MNE not installed - skipping ICA step")
        return seg_data, []

    try:
        n_ch = seg_data.shape[0]
        n_comp = min(n_components, n_ch - 1, 25)

        info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(seg_data.astype(np.float64) * 1e-6, info, verbose=False)
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="ignore", verbose=False)
        except Exception:
            pass

        ica = _ICA(n_components=n_comp, method="fastica", random_state=42,
                   max_iter=500, verbose=False)
        ica.fit(raw, verbose=False)

        sources = ica.get_sources(raw).get_data()
        k_scores = [_kurtosis(s) for s in sources]
        excluded = [i for i, k in enumerate(k_scores) if k > kurtosis_thresh]

        if excluded:
            ica.apply(raw, exclude=excluded, verbose=False)
            logger.info(
                f"  [ICA] Excluded {len(excluded)} components "
                f"(kurtosis > {kurtosis_thresh}): {excluded}"
            )
        else:
            logger.info(f"  [ICA] No high-kurtosis components found (thresh={kurtosis_thresh})")

        return raw.get_data().astype(np.float32) * 1e6, excluded

    except Exception as e:
        logger.warning(f"  [ICA] Failed - skipping: {e}")
        return seg_data, []


def extract_epochs(
    data: np.ndarray,
    stim_positions: List[int],
    sfreq: float,
    tmin: float = -0.5,
    tmax: float = 0.35,
    reject_uv: float = 150.0,
    reject_tmax: float = 0.0,
    reject_post_uv: Optional[float] = None,
    reject_post_window: Tuple[float, float] = (0.0, 0.3),
    max_epochs: Optional[int] = None,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int, int, Dict]:
    """Extract epochs around stimulus positions.

    Returns
    -------
    epochs      : ndarray (n_ch, n_times, n_epochs)
    times       : ndarray (n_times,)
    n_accepted  : int  - accepted count before capping
    n_rejected  : int
    reject_stats: dict with per-channel rejection details
    """
    n_ch, n_samples = data.shape
    s_start = int(tmin * sfreq)
    s_end = int(tmax * sfreq)
    n_times = s_end - s_start + 1
    times = np.linspace(tmin, tmax, n_times)
    rej_end_idx = int((reject_tmax - tmin) * sfreq)
    rej_end_idx = max(1, min(rej_end_idx, n_times))

    # Optional post-stimulus gross-artifact window (indices into the epoch).
    post_ini_idx = post_end_idx = None
    if reject_post_uv is not None:
        post_ini_idx = max(0, int((reject_post_window[0] - tmin) * sfreq))
        post_end_idx = min(n_times, int((reject_post_window[1] - tmin) * sfreq))

    epochs_list = []
    n_rejected = 0
    n_rejected_post = 0
    # Per-channel: how many times did this channel cause a rejection
    ch_reject_count = np.zeros(n_ch, dtype=int)
    # Per-channel: max peak-to-peak seen in the rejection window (across all epochs)
    ch_max_pp = np.zeros(n_ch)

    for pos in stim_positions:
        start = pos + s_start
        end = pos + s_end + 1

        if start < 0 or end > n_samples:
            n_rejected += 1
            continue

        epoch = data[:, start:end]
        pp = np.ptp(epoch[:, :rej_end_idx], axis=1)
        ch_max_pp = np.maximum(ch_max_pp, pp)

        bad_mask = pp > reject_uv

        # Optional: reject gross post-stimulus artifacts (movement, big muscle)
        # in the response window, using a separate, higher threshold so the
        # genuine TMS-evoked response is preserved.
        if reject_post_uv is not None:
            pp_post = np.ptp(epoch[:, post_ini_idx:post_end_idx], axis=1)
            post_bad = pp_post > reject_post_uv
            if np.any(post_bad) and not np.any(bad_mask):
                n_rejected_post += 1
            bad_mask = bad_mask | post_bad

        if np.any(bad_mask):
            ch_reject_count[bad_mask] += 1
            n_rejected += 1
            continue

        epochs_list.append(epoch)

    if not epochs_list:
        reject_stats = _make_reject_stats(
            reject_uv, tmin, reject_tmax, 0, n_rejected,
            ch_reject_count, ch_max_pp, reject_post_uv, n_rejected_post,
            reject_post_window,
        )
        return np.empty((n_ch, n_times, 0)), times, 0, n_rejected, reject_stats

    n_accepted = len(epochs_list)

    if max_epochs is not None and 0 < max_epochs < n_accepted:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n_accepted, size=max_epochs, replace=False)
        idx.sort()
        epochs_list = [epochs_list[i] for i in idx]
        logger.info(f"  [EPOCHS] Subsampled {n_accepted} → {max_epochs} epochs (seed={random_seed})")

    epochs = np.stack(epochs_list, axis=2)
    reject_stats = _make_reject_stats(
        reject_uv, tmin, reject_tmax, n_accepted, n_rejected,
        ch_reject_count, ch_max_pp, reject_post_uv, n_rejected_post,
        reject_post_window,
    )
    return epochs, times, n_accepted, n_rejected, reject_stats


def _make_reject_stats(
    threshold_uv: float,
    tmin: float,
    reject_tmax: float,
    n_accepted: int,
    n_rejected: int,
    ch_reject_count: "np.ndarray",
    ch_max_pp: "np.ndarray",
    reject_post_uv: Optional[float] = None,
    n_rejected_post: int = 0,
    reject_post_window: Tuple[float, float] = (0.0, 0.3),
) -> Dict:
    return {
        "threshold_uv": threshold_uv,
        "window_ms": (int(tmin * 1000), int(reject_tmax * 1000)),
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "ch_reject_count": ch_reject_count.tolist(),
        # Data is already in µV (reject_uv threshold operates on these same
        # peak-to-peak values), so no unit conversion is needed here.
        "ch_max_pp_uv": ch_max_pp.tolist(),
        # Optional post-stimulus gross-artifact rejection (off when None)
        "threshold_post_uv": reject_post_uv,
        "n_rejected_post": n_rejected_post,
        "post_window_ms": (int(reject_post_window[0] * 1000),
                           int(reject_post_window[1] * 1000)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# §3 PCIst COMPUTATION (Comolatti et al. 2019, Brain Stimulation)
# ═══════════════════════════════════════════════════════════════════════════
#
# PCIst is computed by the vendored reference implementation from
# renzocom/PCIst (see `third_party/pcist/pci_st.py`). The wrapper
# `pcist.calc_PCIst` adapts units (seconds → ms) and packs the result in the
# dict shape the rest of this pipeline consumes. We re-export it under the
# same name so call sites in this file don't change.
#
# Reference: Comolatti R et al. (2019). A fast and general method to
# empirically estimate the complexity of brain responses to transcranial
# and intracranial stimulations. Brain Stimulation, 12(5):1280-1289.
# ═══════════════════════════════════════════════════════════════════════════

from pcist import calc_PCIst  # noqa: E402, F401  - re-exported


def compute_snr(epochs: np.ndarray, times: np.ndarray) -> float:
    """SNR from epochs (n_ch, n_times, n_trials)."""
    evoked = np.mean(epochs, axis=2)
    post_mask = (times >= 0.025) & (times <= 0.300)
    base_mask = (times >= -0.4) & (times < -0.01)

    if not np.any(post_mask) or np.sum(base_mask) < 10:
        return 0.0

    signal = np.mean(np.abs(evoked[:, post_mask]))
    noise = np.mean(np.std(evoked[:, base_mask], axis=1))
    return float(signal / noise) if noise > 0 else 0.0


def verify_trigger_timing(
    data: np.ndarray,
    stim_positions: List[int],
    sfreq: float,
    window_ms: Tuple[float, float] = (-5, 15),
) -> Dict[str, Any]:
    """Verify that TMS artifact peaks align with trigger positions.

    Examines single-trial data around each trigger to find the actual
    artifact onset (maximum absolute amplitude). Reports systematic
    offset between trigger marker and artifact peak.

    This addresses Dr. Toplutas's F3 session observation: GFP deflection
    started AFTER the marker, suggesting a trigger timestamp offset.

    Parameters
    ----------
    data : ndarray, shape (n_ch, n_samples)
        Raw EEG data (before artifact interpolation).
    stim_positions : list of int
        Trigger sample indices.
    sfreq : float
        Sampling frequency.
    window_ms : (float, float)
        Window around trigger to search for artifact peak.

    Returns
    -------
    info : dict
        offset_ms : float - median offset (positive = artifact after trigger)
        offset_samples : int - median offset in samples
        offsets_ms : list - per-trial offsets
        recommendation : str - suggested action
    """
    win_s = int(window_ms[0] * sfreq / 1000)
    win_e = int(window_ms[1] * sfreq / 1000)
    n_ch, n_samples = data.shape

    offsets = []
    for pos in stim_positions:
        s0 = pos + win_s
        s1 = pos + win_e
        if s0 < 0 or s1 >= n_samples:
            continue
        # GFP in the window around trigger
        snippet = data[:, s0:s1 + 1]
        gfp = np.std(snippet, axis=0)
        peak_idx = np.argmax(gfp)
        # Offset: peak position relative to trigger
        offset_samples = peak_idx + win_s  # relative to trigger
        offset_ms = offset_samples / sfreq * 1000
        offsets.append(offset_ms)

    if not offsets:
        return {
            "offset_ms": 0.0,
            "offset_samples": 0,
            "offsets_ms": [],
            "recommendation": "No valid triggers to analyze.",
        }

    median_offset = float(np.median(offsets))
    median_samples = int(round(median_offset * sfreq / 1000))

    recommendation = ""
    if abs(median_offset) < 1.0:
        recommendation = "Trigger alignment OK - artifact peak within ±1 ms of trigger."
    elif median_offset > 0:
        recommendation = (
            f"Systematic delay: artifact peaks {median_offset:.1f} ms AFTER trigger marker. "
            f"This is common with BrainVision TTL response ports (0.5-2 ms lag). "
            f"Consider shifting trigger positions by {-median_samples} samples before processing, "
            f"or ensure artifact interpolation window covers [-2, {10 + median_offset:.0f}] ms."
        )
    else:
        recommendation = (
            f"Artifact peaks {-median_offset:.1f} ms BEFORE trigger marker. "
            f"Check marker extraction code for off-by-one errors."
        )

    logger.info(
        f"  [TRIGGER VERIFY] Median artifact offset: {median_offset:.1f} ms "
        f"({median_samples} samples) - {recommendation}"
    )

    return {
        "offset_ms": median_offset,
        "offset_samples": median_samples,
        "offsets_ms": offsets,
        "recommendation": recommendation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# §4 SESSION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_sessions(
    stim_positions: List[int],
    sfreq: float,
    gap_seconds: float = 30.0,
    min_events_per_session: int = 10,
) -> List[Dict[str, Any]]:
    """
    Detect recording sessions based on gaps in stimulus timing.
    """
    if len(stim_positions) < min_events_per_session:
        return [{"label": "Session 1", "events": stim_positions, "start_idx": 0, "end_idx": len(stim_positions) - 1}]

    positions = np.sort(np.array(stim_positions))
    isi = np.diff(positions) / sfreq

    # Find large gaps
    gap_indices = np.where(isi > gap_seconds)[0]

    sessions = []
    starts = [0] + [int(idx + 1) for idx in gap_indices]
    ends = [int(idx) for idx in gap_indices] + [len(positions) - 1]

    for i, (s, e) in enumerate(zip(starts, ends)):
        n_events = e - s + 1
        if n_events >= min_events_per_session:
            session_events = positions[s:e + 1].tolist()
            median_isi = float(np.median(np.diff(session_events) / sfreq)) if n_events > 1 else 0
            sessions.append({
                "label": f"Session {i + 1}",
                "events": session_events,
                "start_idx": s,
                "end_idx": e,
                "n_events": n_events,
                "start_sample": int(positions[s]),
                "end_sample": int(positions[e]),
                "start_time": float(positions[s] / sfreq),
                "end_time": float(positions[e] / sfreq),
                "duration": float((positions[e] - positions[s]) / sfreq),
                "median_isi": median_isi,
            })

    if not sessions:
        sessions = [{
            "label": "Session 1",
            "events": positions.tolist(),
            "start_idx": 0,
            "end_idx": len(positions) - 1,
            "n_events": len(positions),
            "start_sample": int(positions[0]),
            "end_sample": int(positions[-1]),
            "start_time": float(positions[0] / sfreq),
            "end_time": float(positions[-1] / sfreq),
            "duration": float((positions[-1] - positions[0]) / sfreq),
            "median_isi": float(np.median(isi)),
        }]

    return sessions


# ═══════════════════════════════════════════════════════════════════════════
# §5 MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def _dedup_markers(positions: List[int], sfreq: float,
                   min_gap_ms: float = 10.0) -> List[int]:
    """Remove duplicate marker positions that are within min_gap_ms of each other.

    BrainVision TMS setups sometimes fire two TTL pulses per stimulus (e.g.
    codes 256 and 257 on the same port within a few samples). Keeping both
    doubles the effective marker count and produces a bimodal ISI distribution
    that fails the periodicity check. This function retains the *first* marker
    in each cluster and discards the rest.

    Parameters
    ----------
    positions  : list of sample indices (already 0-indexed, any order)
    sfreq      : sampling frequency in Hz
    min_gap_ms : markers separated by less than this are treated as duplicates
    """
    if len(positions) < 2:
        return list(positions)
    min_gap_samples = int(min_gap_ms * sfreq / 1000.0)
    sorted_pos = sorted(positions)
    kept = [sorted_pos[0]]
    for p in sorted_pos[1:]:
        if p - kept[-1] >= min_gap_samples:
            kept.append(p)
    n_removed = len(positions) - len(kept)
    if n_removed:
        logger.info(
            f"  [DEDUP] Removed {n_removed} duplicate marker(s) within {min_gap_ms} ms "
            f"({len(kept)} remaining)"
        )
    return kept


def _detect_periodic_response_train(positions: List[int], sfreq: float,
                                     min_events: int = 20, max_cv: float = 0.30,
                                     gap_seconds: float = 30.0) -> bool:
    """Check if event positions form periodic blocks (likely TMS TTL pulses).

    Handles multi-session recordings where large gaps exist between sessions
    but within each session the ISI is highly regular.
    """
    if len(positions) < min_events:
        return False
    pos = np.sort(np.array(positions))
    isi = np.diff(pos) / sfreq

    # First check: if all ISIs are periodic (single session)
    median_isi = float(np.median(isi))
    if 0.2 <= median_isi <= 15.0:
        cv = float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else 1.0
        if cv <= max_cv:
            return True

    # Second check: split into blocks by large gaps and test each block
    gap_mask = isi > gap_seconds
    block_starts = [0] + list(np.where(gap_mask)[0] + 1)
    block_ends = list(np.where(gap_mask)[0]) + [len(pos) - 1]

    periodic_blocks = 0
    for s, e in zip(block_starts, block_ends):
        n_block = e - s + 1
        if n_block < 10:
            continue
        block_isi = np.diff(pos[s:e + 1]) / sfreq
        med = float(np.median(block_isi))
        if not (0.2 <= med <= 15.0):
            continue
        cv = float(np.std(block_isi) / np.mean(block_isi)) if np.mean(block_isi) > 0 else 1.0
        if cv <= max_cv:
            periodic_blocks += 1

    return periodic_blocks > 0


def _label_sessions_from_comments(sessions: List[Dict], comment_markers: List[Dict], sfreq: float):
    """Label sessions using Comment markers that precede each session's first event."""
    # Sort comments by position
    comments = sorted(comment_markers, key=lambda m: m["position"])

    for sess in sessions:
        first_event_sample = sess["start_sample"]
        # Find the last comment before this session's first event
        label = None
        for cm in reversed(comments):
            if cm["position"] < first_event_sample:
                desc = cm["description"].strip()
                # Skip generic comments
                if desc.lower() not in ("", "new segment"):
                    label = desc
                break
        if label:
            sess["label"] = label
            sess["site"] = label  # Keep the stimulation site label


def _decimate_data(data: np.ndarray, sfreq: float, target_sfreq: float = 500.0) -> Tuple[np.ndarray, float]:
    """Downsample data by integer factor using anti-alias filtering."""
    if sfreq <= target_sfreq:
        return data, sfreq

    factor = int(sfreq / target_sfreq)
    actual_new_sfreq = sfreq / factor
    logger.info(f"Downsampling: {sfreq:.0f} Hz → {actual_new_sfreq:.0f} Hz (factor {factor})")

    # Anti-alias: low-pass filter at Nyquist of target
    nyquist = actual_new_sfreq / 2.0
    data_filtered = bandpass_filter(data, sfreq, low=0.0, high=nyquist * 0.9)

    # Decimate
    data_dec = data_filtered[:, ::factor]
    return data_dec, actual_new_sfreq


def analyze_file(
    vhdr_path: str,
    gap_seconds: float = 30.0,
    reject_uv: float = 150.0,
    artifact_window_ms: Tuple[float, float] = (-2, 10),
    decimate_to: Optional[float] = 1000.0,
    min_snr: float = 1.4,
    tms_marker: Optional[str] = None,
    tms_marker_type: Optional[str] = None,
    ch_overrides: Optional[Dict[str, str]] = None,
    max_epochs: Optional[int] = None,
    exact_epochs: bool = False,
    dedup_gap_ms: float = 10.0,
    apply_ica: bool = False,
    ica_kurtosis_thresh: float = 5.0,
    reject_post_uv: Optional[float] = None,
    auto_trigger_shift: bool = False,
    # PCIst parameters (Comolatti et al. 2019)
    pcist_baseline_window: Tuple[float, float] = (-0.400, -0.050),
    pcist_response_window: Tuple[float, float] = (0.000, 0.300),
    pcist_k: float = 1.2,
    pcist_min_snr: float = 1.1,
    pcist_max_var: float = 99.0,
    pcist_n_steps: int = 100,
) -> Dict[str, Any]:
    """Full PCIst analysis pipeline for a single BrainVision file.

    Pipeline (Comolatti 2019 + 2023 Brain Stimulation consensus):
      1. Parse BrainVision → 2. Load segment → 3. TMS artifact interpolation →
      4. Downsample → 5. Bad channel detection → 6. CAR re-reference →
      7. Bandpass 0.1-45 Hz → 8. Epoch extraction → 9. PCIst computation

    Parameters
    ----------
    min_snr : float
        Minimum signal-to-noise ratio for reliable PCIst.
        Sessions with SNR < min_snr are flagged as unreliable.
    pcist_baseline_window : (float, float)
        Baseline window in seconds (default -400 to -50 ms).
    pcist_response_window : (float, float)
        Response window in seconds (default 0 to 300 ms).
    pcist_k : float
        Baseline penalty factor (default 1.2).
    pcist_min_snr : float
        Minimum per-component SNR in SVD (default 1.1).
    pcist_max_var : float
        Cumulative variance to retain in SVD (default 99%).
    pcist_n_steps : int
        Number of distance thresholds to scan (default 100).
    """

    _log_handler = _ListHandler()
    _log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_log_handler)

    logger.info(f"Loading header & markers from {vhdr_path}")

    # Parse header and markers without loading data
    vhdr_path = str(Path(vhdr_path).resolve())
    base_dir = str(Path(vhdr_path).parent)
    info = parse_vhdr(vhdr_path)
    markers = []
    if info["marker_file"]:
        vmrk_path = os.path.join(base_dir, info["marker_file"])
        markers = parse_vmrk(vmrk_path)

    eeg_path = os.path.join(base_dir, info["data_file"])

    sfreq = info["sfreq"]
    ch_names = [ch["name"] for ch in info["channels"]]
    n_ch_total = info["n_channels"]

    # Calculate total samples from file size
    fmt = info["binary_format"]
    dtype = np.float32 if fmt == "IEEE_FLOAT_32" else np.int16
    bps = np.dtype(dtype).itemsize
    file_size = os.path.getsize(eeg_path)
    n_samples = file_size // (n_ch_total * bps)
    duration = n_samples / sfreq

    logger.info(f"Recording: {n_ch_total} channels, {sfreq:.0f} Hz, {duration:.1f} s ({duration/60:.1f} min), {len(markers)} markers")

    # ── Exclude non-EEG channels (HEOG, VEOG, EOG, ECG, EMG, Status) ──
    non_eeg_names = {"heog", "veog", "eog", "ecg", "emg", "status"}
    eeg_mask = [ch.lower() not in non_eeg_names for ch in ch_names]
    excluded_ch = [ch for ch, keep in zip(ch_names, eeg_mask) if not keep]
    if excluded_ch:
        logger.info(f"Excluding non-EEG channels: {', '.join(excluded_ch)}")
    ch_names_eeg = [ch for ch, keep in zip(ch_names, eeg_mask) if keep]
    n_ch = len(ch_names_eeg)

    # ── Extract markers by type ──
    stim_markers = [m for m in markers if m["type"] == "Stimulus"]
    resp_markers = [m for m in markers if m["type"] == "Response"]
    comment_markers = [m for m in markers if m["type"] == "Comment"]

    # ── Explicit TMS marker selection (user override) ──
    # tms_marker_type: "Stimulus" or "Response"
    # tms_marker:      description string (e.g. "R256", "S8192", "R 16")
    # When both are set the user has explicitly identified the TMS markers -
    # skip all auto-detection and periodicity checks entirely.
    explicit_selection = bool(tms_marker and tms_marker_type)
    if explicit_selection:
        tms_marker = str(tms_marker).strip()
        tms_marker_type = str(tms_marker_type).strip()
        pool = stim_markers if tms_marker_type == "Stimulus" else resp_markers
        all_descs = sorted({m["description"] for m in pool})
        selected = [m for m in pool if m["description"] == tms_marker]
        logger.info(
            f"User selected [{tms_marker_type}] '{tms_marker}': "
            f"{len(selected)}/{len(pool)} markers kept (all codes: {all_descs})"
        )
        # Dedup in case of dual-port double-markers
        sel_positions = _dedup_markers(
            [m["position"] - 1 for m in selected], sfreq, min_gap_ms=dedup_gap_ms
        )
        logger.info(
            f"Using {len(sel_positions)} markers as TMS triggers "
            f"(periodicity check skipped - explicit selection)."
        )
        stim_positions = sel_positions
        stim_markers = selected
        resp_positions: List[int] = []
        resp_used_as_stim = tms_marker_type == "Response"
    else:
        # ── Auto-detection ──
        stim_positions = [m["position"] - 1 for m in stim_markers]
        resp_positions = [m["position"] - 1 for m in resp_markers]
        resp_used_as_stim = False

        if len(stim_positions) == 0 and len(resp_positions) > 0:
            resp_positions = _dedup_markers(resp_positions, sfreq, min_gap_ms=dedup_gap_ms)
            if _detect_periodic_response_train(resp_positions, sfreq):
                logger.info(
                    f"Auto-detected {len(resp_positions)} periodic Response markers "
                    f"as TMS triggers."
                )
                stim_positions = resp_positions
                stim_markers = resp_markers
                resp_used_as_stim = True
            else:
                logger.warning(
                    "Response markers exist but do not look periodic - not using as TMS triggers. "
                    "Select the correct marker code in the sidebar."
                )

    # Also collect all markers for display
    all_marker_positions = [m["position"] - 1 for m in markers]
    all_marker_types = [f"{m['type']}: {m['description']}" for m in markers]

    logger.info(f"Found {len(stim_positions)} TMS trigger markers")

    if len(stim_positions) == 0:
        raise ValueError(
            "No TMS trigger markers found. File has "
            f"{len(resp_markers)} Response markers and {len(stim_markers)} Stimulus markers, "
            "but none form a periodic stimulation train."
        )

    # ── Detect sessions from inter-stimulus gaps ──
    sessions = detect_sessions(stim_positions, sfreq, gap_seconds=gap_seconds)

    # ── Label sessions from Comment markers (e.g., "f3", "fz", "p4") ──
    if comment_markers:
        _label_sessions_from_comments(sessions, comment_markers, sfreq)

    logger.info(f"Detected {len(sessions)} session(s): {[s['label'] for s in sessions]}")

    # ── Prepare downsampled EEG for full-recording display ──
    # Load display data in chunks to avoid OOM on large files
    # Target ~25 Hz for display of long recordings
    ds_factor = max(1, int(sfreq / 25))
    n_display_samples = n_samples // ds_factor
    logger.info(f"Loading display data: {n_display_samples} samples at ~{sfreq/ds_factor:.0f} Hz")

    # Load display data in segments to avoid memory issues
    chunk_size = 500000  # samples per chunk
    display_chunks = []
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        chunk = load_eeg_segment(eeg_path, info, chunk_start, chunk_end, channel_mask=eeg_mask)
        display_chunks.append(chunk[:, ::ds_factor])
        del chunk

    data_display = np.concatenate(display_chunks, axis=1)
    del display_chunks
    times_display = np.arange(data_display.shape[1]) * ds_factor / sfreq
    data_display = np.round(data_display, 1)

    # ── Process PCIst for each session individually (memory-efficient) ──
    # Per-session pipeline (step numbers match the `Step N` labels below and the
    # analyze_file docstring), following the 2023 Brain Stimulation consensus +
    # Comolatti 2019:
    #   2. Load segment  → 3. TMS artifact interpolation (cubic) →
    #   4. Downsample    → 5. Bad channel detection / interpolation →
    #   6. Average re-reference (CAR) → 7. Bandpass filter (+7b optional ICA) →
    #   8. Epoch extraction → 9. PCIst (SVD + state transitions)

    sfreq_proc = sfreq
    dec_factor = 1
    if sfreq > 1000:
        target_sfreq = decimate_to if decimate_to else 1000.0
        dec_factor = max(1, int(sfreq / target_sfreq))
        sfreq_proc = sfreq / dec_factor
        logger.info(f"Will downsample from {sfreq:.0f} Hz to {sfreq_proc:.0f} Hz (factor {dec_factor})")

    session_results = []

    for sess in sessions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {sess['label']} ({sess['n_events']} events)")
        logger.info(f"{'='*60}")

        # ── Step 2: Load segment (±3s padding) ──
        padding_samples = int(3.0 * sfreq)
        seg_start = max(0, sess["start_sample"] - padding_samples)
        seg_end = min(n_samples, sess["end_sample"] + padding_samples)

        logger.info(f"  [LOAD] Segment [{seg_start}:{seg_end}] ({(seg_end-seg_start)/sfreq:.1f} s)")
        seg_data = load_eeg_segment(eeg_path, info, seg_start, seg_end, channel_mask=eeg_mask)

        # Event positions relative to segment start
        sess_events_local = [int(p - seg_start) for p in sess["events"]]

        # ── Step 2b: Trigger timing verification (before artifact removal) ──
        trigger_timing = verify_trigger_timing(
            seg_data, sess_events_local, sfreq
        )

        # ── Step 2c: Optional automatic trigger-shift correction ──
        # The trigger marker can lag the true artifact peak (TTL/response-port
        # latency + the artifact's own rise time). If enabled, realign every
        # event to the detected artifact peak so interpolation and epoching are
        # centred correctly. Capped at 20 ms to avoid pathological shifts.
        #
        # Aligning t=0 to the peak moves the pre-peak rise to about -offset ms,
        # so the interpolation window start is widened to cover it; otherwise
        # that residual spike would land in the baseline and reject every epoch.
        trigger_timing["shift_applied_samples"] = 0
        trigger_timing["shift_applied_ms"] = 0.0
        art_win = artifact_window_ms
        if auto_trigger_shift:
            _shift = int(trigger_timing.get("offset_samples", 0))
            _max_shift = int(0.020 * sfreq)
            if 0 < abs(_shift) <= _max_shift:
                _off_ms = float(trigger_timing.get("offset_ms", 0.0))
                sess_events_local = [p + _shift for p in sess_events_local]
                art_win = (min(artifact_window_ms[0], -(abs(_off_ms) + 2.0)),
                           artifact_window_ms[1])
                trigger_timing["shift_applied_samples"] = _shift
                trigger_timing["shift_applied_ms"] = _off_ms
                logger.info(
                    f"  [TRIGGER SHIFT] Realigned events by {_shift} samples "
                    f"({_off_ms:.1f} ms) to artifact peak; interpolation window "
                    f"widened to [{art_win[0]:.1f}, {art_win[1]:.1f}] ms."
                )

        # ── Step 3: TMS artifact interpolation (cubic spline) ──
        logger.info(f"  [ARTIFACT] Cubic spline interpolation [{art_win[0]}, {art_win[1]}] ms")
        seg_data = interpolate_tms_artifact(
            seg_data, sess_events_local, sfreq, art_win, method="cubic"
        )

        # ── Step 4: Downsample ──
        if dec_factor > 1:
            from numpy.fft import rfft, irfft, rfftfreq
            nyq = sfreq_proc / 2.0
            n_seg = seg_data.shape[1]
            freqs = rfftfreq(n_seg, d=1.0/sfreq)
            filt = np.ones(len(freqs), dtype=np.float32)
            filt[freqs > nyq * 0.9] = 0
            rolloff = (freqs > nyq * 0.7) & (freqs <= nyq * 0.9)
            if np.any(rolloff):
                filt[rolloff] = (0.5 * (1 + np.cos(
                    np.pi * (freqs[rolloff] - nyq * 0.7) / (nyq * 0.2)
                ))).astype(np.float32)

            for ch in range(seg_data.shape[0]):
                spec = rfft(seg_data[ch].astype(np.float64))
                seg_data[ch] = irfft(spec * filt, n=n_seg).astype(np.float32)

            seg_data = seg_data[:, ::dec_factor]
            sess_events_local = [int(p / dec_factor) for p in sess_events_local]
            logger.info(f"  [DECIMATE] {sfreq:.0f} → {sfreq_proc:.0f} Hz (factor {dec_factor})")

        # ── Step 5: Bad channel detection + user overrides ──
        ch_mask, bad_ch_names, bad_reasons, bad_stats = detect_bad_channels(
            seg_data, ch_names_eeg, sfreq_proc
        )

        # Apply user overrides: drop / interpolate_spline / interpolate_neighbors / keep
        _overrides = ch_overrides or {}
        final_bad: List[str] = []       # channels to drop after interpolation
        to_interp_spline: List[str] = []
        to_interp_neighbors: List[str] = []

        for ch in bad_ch_names:
            action = _overrides.get(ch, "drop")
            if action == "drop":
                final_bad.append(ch)
            elif action == "interpolate_spline":
                to_interp_spline.append(ch)
            elif action == "interpolate_neighbors":
                to_interp_neighbors.append(ch)
            elif action == "keep":
                pass  # include as-is

        # Also honour explicit overrides for channels the detector did NOT flag
        # (e.g. a channel that drives epoch rejection but whose whole-segment
        # variance stays under the noisy threshold). All four actions apply.
        for ch, action in _overrides.items():
            if ch in bad_ch_names or ch not in ch_names_eeg:
                continue  # flagged channels already handled above
            if action == "drop":
                final_bad.append(ch)
                bad_stats[ch] = {**bad_stats.get(ch, {}), "reason": "User-forced drop", "flagged": True}
            elif action == "interpolate_spline":
                to_interp_spline.append(ch)
                bad_stats[ch] = {**bad_stats.get(ch, {}), "reason": "User-forced interpolate", "flagged": True}
            elif action == "interpolate_neighbors":
                to_interp_neighbors.append(ch)
                bad_stats[ch] = {**bad_stats.get(ch, {}), "reason": "User-forced interpolate", "flagged": True}
            # "keep" on an unflagged channel is a no-op (it's already included)

        # Every flagged-bad channel must be excluded from the interpolation
        # basis so a channel about to be dropped can't poison the fit of the
        # channels we're keeping.
        _all_flagged_bad = set(to_interp_spline) | set(to_interp_neighbors) | set(final_bad)
        if to_interp_spline:
            logger.info(f"  [BAD CH] Interpolating (spline): {to_interp_spline}")
            seg_data = interpolate_bad_channels(
                seg_data, ch_names_eeg, to_interp_spline, method="spline", sfreq=sfreq_proc,
                exclude_from_basis=list(_all_flagged_bad - set(to_interp_spline)),
            )
        if to_interp_neighbors:
            logger.info(f"  [BAD CH] Interpolating (neighbors): {to_interp_neighbors}")
            seg_data = interpolate_bad_channels(
                seg_data, ch_names_eeg, to_interp_neighbors, method="neighbors", sfreq=sfreq_proc,
                exclude_from_basis=list(_all_flagged_bad - set(to_interp_neighbors)),
            )

        # Build final mask: drop channels in final_bad
        ch_mask = [n not in final_bad for n in ch_names_eeg]
        kept_names = [n for n, ok in zip(ch_names_eeg, ch_mask) if ok]
        all_handled = sorted(set(to_interp_spline + to_interp_neighbors + final_bad))

        if all_handled:
            logger.info(
                f"  [BAD CH] Detected: {bad_ch_names}  |  "
                f"Dropped: {final_bad}  |  "
                f"Interpolated: {to_interp_spline + to_interp_neighbors}  |  "
                f"Kept: {[c for c in bad_ch_names if c not in all_handled]}"
            )
        else:
            logger.info("  [BAD CH] No bad channels detected")

        if final_bad:
            seg_data = seg_data[ch_mask]
            ch_names_eeg = kept_names

        # ── Step 6: Average re-reference (CAR) - now on clean channels only ──
        seg_data = average_rereference(seg_data)
        logger.info(f"  [REREF] Common average reference applied ({seg_data.shape[0]} channels)")

        # ── Step 7: Bandpass filter 0.1-45 Hz ──
        seg_data = bandpass_filter(seg_data, sfreq_proc, low=0.1, high=45.0)
        logger.info("  [FILTER] Bandpass 0.1-45 Hz applied")

        # ── Step 7b: ICA artifact removal (optional) ──
        ica_excluded: List[int] = []
        if apply_ica:
            seg_data, ica_excluded = _apply_ica(
                seg_data, ch_names_eeg, sfreq_proc,
                kurtosis_thresh=ica_kurtosis_thresh,
            )

        # ── Step 8: Epoch extraction ──
        epochs, times, n_accepted, n_rejected, reject_stats = extract_epochs(
            seg_data, sess_events_local, sfreq_proc,
            tmin=-0.5, tmax=0.35,
            reject_uv=reject_uv,
            reject_tmax=0.0,  # baseline check window (-500ms to 0ms)
            reject_post_uv=reject_post_uv,  # optional post-stimulus gross-artifact check
            reject_post_window=pcist_response_window,
            max_epochs=max_epochs,
        )
        # Attach channel names to rejection stats
        reject_stats["ch_names"] = list(ch_names_eeg)
        n_used = epochs.shape[2]  # actual count after optional cap
        _cap_msg = f", capped to {n_used}" if max_epochs and n_used < n_accepted else ""
        _exact_warn = ""
        if exact_epochs and max_epochs is not None and n_accepted < max_epochs:
            _exact_warn = (
                f" - WARNING: exact mode requested {max_epochs} epochs but only "
                f"{n_accepted} clean epochs available"
            )
            logger.warning(f"  [EPOCHS] Exact-epoch shortfall: {n_accepted} < {max_epochs}")
        logger.info(
            f"  [EPOCHS] {n_accepted} accepted, {n_rejected} rejected "
            f"(threshold: {reject_uv} µV){_cap_msg}{_exact_warn}"
        )

        del seg_data  # Free memory

        if n_accepted < 5:
            logger.warning(f"  Too few epochs ({n_accepted}), skipping PCIst")
            session_results.append({
                "label": sess["label"],
                "n_events": sess["n_events"],
                "n_accepted": n_accepted,
                "n_used": n_used,
                "n_rejected": n_rejected,
                "reject_stats": reject_stats,
                "bad_channels": bad_ch_names if bad_ch_names else [],
                "bad_ch_stats": bad_stats,
                "ch_overrides_applied": dict(_overrides),
                "ica_excluded": ica_excluded,
                "pcist": None,
                "error": "Too few epochs after rejection",
                "start_time": sess["start_time"],
                "end_time": sess["end_time"],
                "duration": sess["duration"],
                "median_isi": sess.get("median_isi", 0),
            })
            continue

        # ── Compute SNR ──
        snr = compute_snr(epochs, times)
        snr_pass = snr >= min_snr
        if not snr_pass:
            logger.warning(
                f"  [SNR GATE] SNR = {snr:.2f} < {min_snr:.1f} - "
                f"PCIst will be computed but flagged as UNRELIABLE."
            )

        # ── Evoked response & GFP ──
        evoked = np.mean(epochs, axis=2)  # (n_ch, n_times)
        gfp = np.std(evoked, axis=0)

        # ── Step 9: PCIst computation (Comolatti et al. 2019) ──
        try:
            pcist_result = calc_PCIst(
                evoked, times,
                baseline_window=pcist_baseline_window,
                response_window=pcist_response_window,
                k=pcist_k,
                min_snr=pcist_min_snr,
                max_var=pcist_max_var,
                n_steps=pcist_n_steps,
            )

            pcist_value = pcist_result["PCIst"]

            # Post-stim times for display
            post_mask = (times >= pcist_response_window[0]) & (times <= pcist_response_window[1])
            post_times = times[post_mask]

            # Build warnings list for this session
            warnings_list = []
            if not snr_pass:
                warnings_list.append(
                    f"SNR={snr:.2f} < {min_snr} - below exclusion threshold. "
                    f"PCIst value may be UNRELIABLE."
                )
            if pcist_result["n_components"] == 0:
                warnings_list.append(
                    "No SVD components survived SNR filtering. "
                    "Possible causes: weak TMS response or noisy data."
                )
            if trigger_timing.get("shift_applied_samples"):
                warnings_list.append(
                    f"Trigger timing offset: {trigger_timing['offset_ms']:.1f} ms. "
                    f"Auto-correction applied: events realigned by "
                    f"{trigger_timing['shift_applied_samples']} samples to the "
                    f"artifact peak."
                )
            elif abs(trigger_timing["offset_ms"]) > 2.0:
                warnings_list.append(
                    f"Trigger timing offset: {trigger_timing['offset_ms']:.1f} ms. "
                    f"{trigger_timing['recommendation']}"
                )
            if n_rejected > 0 and n_rejected / (n_accepted + n_rejected) > 0.20:
                reject_pct = 100.0 * n_rejected / (n_accepted + n_rejected)
                warnings_list.append(
                    f"High epoch rejection rate ({reject_pct:.0f}%). "
                    f"Suggests residual artifact or suboptimal impedances."
                )
            if exact_epochs and max_epochs is not None and n_accepted < max_epochs:
                warnings_list.append(
                    f"Exact-epoch shortfall: requested {max_epochs} epochs but only "
                    f"{n_accepted} clean epochs were available. "
                    f"Results use {n_accepted} epochs - consider lowering the epoch cap."
                )
            if n_used < n_accepted:
                warnings_list.append(
                    f"Epoch cap active: {n_accepted} clean epochs available, "
                    f"{n_used} used for PCIst (random subsample, seed=42)."
                )

            n_ch_used = int(np.sum(ch_mask)) if bad_ch_names else len(ch_names_eeg)
            session_results.append({
                "label": sess["label"],
                "n_events": sess["n_events"],
                "n_accepted": n_accepted,
                "n_used": n_used,
                "n_rejected": n_rejected,
                "reject_stats": reject_stats,
                "n_channels_used": n_ch_used,
                "bad_channels": bad_ch_names if bad_ch_names else [],
                "bad_ch_stats": bad_stats,
                "ch_overrides_applied": dict(_overrides),
                "ica_excluded": ica_excluded,
                # PCIst results (primary metric)
                "pcist": pcist_value,
                "n_components": pcist_result["n_components"],
                "dNST": pcist_result["dNST"],
                "var_explained": pcist_result["var_explained"],
                "component_snrs": pcist_result["snrs"],
                "cumvar": pcist_result["cumvar"],
                "components_kept": pcist_result["components_kept"],
                # Signal quality
                "snr": snr,
                "snr_pass": snr_pass,
                "trigger_timing": trigger_timing,
                "warnings": warnings_list,
                # Display data
                "evoked_times": (times * 1000).tolist(),
                "evoked_gfp": gfp.tolist(),
                "evoked_data": evoked.tolist(),
                "post_times_ms": (post_times * 1000).tolist(),
                "start_time": sess["start_time"],
                "end_time": sess["end_time"],
                "duration": sess["duration"],
                "median_isi": sess.get("median_isi", 0),
                "error": None,
            })

            logger.info(f"  PCIst = {pcist_value:.4f}, SNR = {snr:.2f}")

        except Exception as e:
            logger.error(f"  PCIst computation failed: {e}")
            import traceback
            traceback.print_exc()
            session_results.append({
                "label": sess["label"],
                "n_events": sess["n_events"],
                "n_accepted": n_accepted,
                "n_rejected": n_rejected,
                "reject_stats": reject_stats,
                "bad_channels": bad_ch_names if bad_ch_names else [],
                "bad_ch_stats": bad_stats,
                "ch_overrides_applied": dict(_overrides),
                "ica_excluded": ica_excluded,
                "pcist": None,
                "error": str(e),
                "start_time": sess["start_time"],
                "end_time": sess["end_time"],
                "duration": sess["duration"],
                "median_isi": sess.get("median_isi", 0),
            })

    logger.removeHandler(_log_handler)

    return {
        "file": os.path.basename(vhdr_path),
        "pipeline_log": _log_handler.lines,
        "n_channels": n_ch,
        "ch_names": ch_names_eeg,
        "display_ch_names": ch_names_eeg[:20],
        "sfreq": sfreq,
        "sfreq_proc": sfreq_proc,
        "duration": duration,
        "n_samples": n_samples,
        "n_stim_total": len(stim_positions),
        "stim_positions": stim_positions,
        "stim_times": [p / sfreq for p in stim_positions],
        "sessions": session_results,
        "eeg_display_times": times_display.tolist(),
        "eeg_display_data": data_display.tolist(),
        "ds_factor": ds_factor,
        "all_marker_times": [p / sfreq for p in all_marker_positions],
        "all_marker_types": all_marker_types,
        "excluded_channels": excluded_ch,
        "resp_used_as_stim": resp_used_as_stim,
        "comment_labels": [m["description"] for m in comment_markers],
        "artifact_window_ms": artifact_window_ms,
        "decimate_to": decimate_to,
        "reject_uv": reject_uv,
        "min_snr": min_snr,
        # PCIst parameters (Comolatti 2019)
        "pcist_baseline_window": pcist_baseline_window,
        "pcist_response_window": pcist_response_window,
        "pcist_k": pcist_k,
        "pcist_min_snr": pcist_min_snr,
        "pcist_max_var": pcist_max_var,
        "pcist_n_steps": pcist_n_steps,
    }


def analyze_multiple_files(vhdr_paths: List[str], **kwargs) -> Dict[str, Any]:
    """Analyze multiple files (split sessions) as separate sessions."""
    all_results = []
    for i, path in enumerate(vhdr_paths):
        result = analyze_file(path, **kwargs)
        # Override session labels with file names
        for sess in result["sessions"]:
            sess["label"] = Path(path).stem
        all_results.append(result)

    # Combine into a unified result
    combined = {
        "files": [r["file"] for r in all_results],
        "results": all_results,
        "total_sessions": sum(len(r["sessions"]) for r in all_results),
    }
    return combined

# The static HTML report generator and CLI now live in report_html.py.
if __name__ == "__main__":
    from report_html import main
    main()
