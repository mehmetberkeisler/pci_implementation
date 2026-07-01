#!/usr/bin/env python3
"""
TMS-EEG PCIst Analyzer — Single-file BrainVision loader with session detection.
Computes PCIst (Perturbational Complexity Index based on State Transitions)
per Comolatti et al. 2019 (Brain Stimulation, 12(5):1280-1289).

Standalone implementation — requires only numpy (+ standard library).
"""

import os
import sys
import re
import struct
import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("pcist_analyzer")

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

    # Apply resolution — use float32 to save memory on large files
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
        Low:  0.05–0.1 Hz  (cosine taper)
        High: 45–49.5 Hz   (cosine taper)
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
) -> Tuple[List[bool], List[str], Dict[str, str]]:
    """Detect bad channels based on variance statistics.

    Flags channels whose variance exceeds variance_threshold × median
    or is near zero (dead channels).

    Returns:
        mask: boolean list (True = good channel)
        bad_names: list of bad channel names
        reasons: dict mapping channel name → reason string
    """
    n_ch = data.shape[0]
    ch_var = np.var(data, axis=1)
    median_var = float(np.median(ch_var))

    mask = [True] * n_ch
    bad_names = []
    reasons = {}

    for i in range(n_ch):
        v = float(ch_var[i])
        if median_var > 0 and v > variance_threshold * median_var:
            mask[i] = False
            bad_names.append(ch_names[i])
            reasons[ch_names[i]] = f"NOISY (var={v:.1f}, {v/median_var:.1f}× median)"
        elif v < median_var * 0.01:
            mask[i] = False
            bad_names.append(ch_names[i])
            reasons[ch_names[i]] = f"DEAD (var={v:.4f})"

    return mask, bad_names, reasons


def extract_epochs(
    data: np.ndarray,
    stim_positions: List[int],
    sfreq: float,
    tmin: float = -0.5,
    tmax: float = 0.35,
    reject_uv: float = 150.0,
    reject_tmax: float = 0.0,
    max_epochs: Optional[int] = None,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Extract epochs around stimulus positions.

    Parameters
    ----------
    max_epochs : int or None
        If set, randomly subsample accepted epochs to this number after
        artifact rejection. Use the same value across all subjects to
        keep epoch counts balanced (recommended: minimum clean count
        across the cohort, or a fixed target such as 60).
    random_seed : int
        Seed for the random subsampler so results are reproducible.

    Returns
    -------
    epochs : ndarray, shape (n_ch, n_times, n_epochs)
    times  : ndarray, shape (n_times,)
    n_accepted : int   (before capping)
    n_rejected : int
    """
    n_ch, n_samples = data.shape
    s_start = int(tmin * sfreq)
    s_end = int(tmax * sfreq)
    n_times = s_end - s_start + 1
    times = np.linspace(tmin, tmax, n_times)

    epochs_list = []
    n_rejected = 0

    for pos in stim_positions:
        start = pos + s_start
        end = pos + s_end + 1

        if start < 0 or end > n_samples:
            n_rejected += 1
            continue

        epoch = data[:, start:end]  # (n_ch, n_times)

        # Artifact rejection: peak-to-peak in the rejection window only.
        # By default reject_tmax=0.0 so we check the PRE-stimulus baseline
        # (-500ms to 0ms). This avoids the post-stimulus TMS artifact window
        # contaminating the rejection decision — a standard TMS-EEG practice.
        rej_end_idx = int((reject_tmax - tmin) * sfreq)
        rej_end_idx = max(1, min(rej_end_idx, epoch.shape[1]))
        pp = np.ptp(epoch[:, :rej_end_idx], axis=1)
        if np.any(pp > reject_uv):
            n_rejected += 1
            continue

        epochs_list.append(epoch)

    if not epochs_list:
        return np.empty((n_ch, n_times, 0)), times, 0, n_rejected

    n_accepted = len(epochs_list)

    # Optional epoch cap: subsample to max_epochs for cross-subject balance
    if max_epochs is not None and 0 < max_epochs < n_accepted:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n_accepted, size=max_epochs, replace=False)
        idx.sort()
        epochs_list = [epochs_list[i] for i in idx]
        logger.info(f"  [EPOCHS] Subsampled {n_accepted} → {max_epochs} epochs (seed={random_seed})")

    epochs = np.stack(epochs_list, axis=2)  # (n_ch, n_times, n_epochs)
    return epochs, times, n_accepted, n_rejected


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

from pcist import calc_PCIst  # noqa: E402, F401  — re-exported


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
        offset_ms : float — median offset (positive = artifact after trigger)
        offset_samples : int — median offset in samples
        offsets_ms : list — per-trial offsets
        recommendation : str — suggested action
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
        recommendation = "Trigger alignment OK — artifact peak within ±1 ms of trigger."
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
        f"({median_samples} samples) — {recommendation}"
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
    max_epochs: Optional[int] = None,
    exact_epochs: bool = False,
    dedup_gap_ms: float = 10.0,
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
      7. Bandpass 0.1–45 Hz → 8. Epoch extraction → 9. PCIst computation

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
    segment_markers = [m for m in markers if m["type"] == "New Segment"]

    # ── Optional: filter Response markers to a specific description ──
    # BrainVision TMS setups with dual-port triggering often record both a
    # primary code (e.g. "256") and an auxiliary code (e.g. "257") within a
    # few samples of the same pulse.  Supply tms_marker="256" to keep only
    # those markers and avoid the bimodal ISI that fails the periodicity check.
    if tms_marker:
        tms_marker = str(tms_marker).strip()
        all_resp_descs = sorted({m["description"] for m in resp_markers})
        filtered = [m for m in resp_markers if m["description"] == tms_marker]
        logger.info(
            f"tms_marker='{tms_marker}': kept {len(filtered)}/{len(resp_markers)} "
            f"Response markers (all codes seen: {all_resp_descs})"
        )
        resp_markers = filtered

    stim_positions = [m["position"] - 1 for m in stim_markers]  # 1-indexed → 0-indexed

    # ── Handle Response markers as TMS triggers (common in BrainVision TMS setups) ──
    # In many TMS systems, TTL pulses arrive on the Response port.
    # Detect if Response markers form a periodic stimulation train.
    resp_positions = [m["position"] - 1 for m in resp_markers]
    resp_used_as_stim = False

    # ── Deduplication + periodicity check ──
    # When tms_marker is explicitly set, the user has identified these markers
    # as TMS triggers — skip the periodicity check and use them directly.
    # This supports jittered-ISI protocols (CV > 0.10) which fail the check.
    # Auto-detection (no tms_marker) still requires periodicity to avoid false
    # positives from Response markers that are not TMS triggers.
    if len(stim_positions) == 0 and len(resp_positions) > 0:
        if tms_marker:
            # Explicit selection: trust the user, only dedup if needed
            resp_positions = _dedup_markers(resp_positions, sfreq, min_gap_ms=dedup_gap_ms)
            logger.info(
                f"tms_marker explicitly set — skipping periodicity check. "
                f"Using {len(resp_positions)} Response markers as TMS triggers."
            )
            stim_positions = resp_positions
            stim_markers = resp_markers
            resp_used_as_stim = True
        else:
            # Auto-detection: dedup first, then require periodicity
            resp_positions = _dedup_markers(resp_positions, sfreq, min_gap_ms=dedup_gap_ms)
            if _detect_periodic_response_train(resp_positions, sfreq):
                logger.info(
                    f"No Stimulus markers found, but {len(resp_positions)} Response markers "
                    f"form a periodic train (likely TMS TTL). Using Response markers as stimuli."
                )
                stim_positions = resp_positions
                stim_markers = resp_markers
                resp_used_as_stim = True
            else:
                logger.warning(
                    "Response markers exist but do not look periodic — not using as TMS triggers. "
                    "Tip: set tms_marker='256' (or the correct code) in the sidebar to select "
                    "only the TMS pulse markers and ignore auxiliary codes."
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
    # Pipeline per 2023 Brain Stimulation consensus + Comolatti 2019:
    #   1. Load segment → 2. TMS artifact interpolation (cubic) →
    #   3. Downsample → 4. Bad channel detection → 5. Average re-reference (CAR) →
    #   6. Bandpass filter → 7. Epoch extraction → 8. PCIst (SVD + state transitions)

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

        # ── Step 3: TMS artifact interpolation (cubic spline, -2 to +10 ms) ──
        logger.info(f"  [ARTIFACT] Cubic spline interpolation [{artifact_window_ms[0]}, {artifact_window_ms[1]}] ms")
        seg_data = interpolate_tms_artifact(
            seg_data, sess_events_local, sfreq, artifact_window_ms, method="cubic"
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

        # ── Step 5: Bad channel detection (BEFORE CAR to avoid contamination) ──
        # Per 2023 Brain Stimulation TMS-EEG consensus: detect bad channels
        # first so noisy channels don't contaminate the average reference.
        ch_mask, bad_ch_names, bad_reasons = detect_bad_channels(
            seg_data, ch_names_eeg, sfreq_proc
        )
        if bad_ch_names:
            logger.info(f"  [BAD CH] Excluding {len(bad_ch_names)}: {', '.join(bad_ch_names)}")
            for name, reason in bad_reasons.items():
                logger.info(f"    {name}: {reason}")
            seg_data = seg_data[ch_mask]
        else:
            logger.info(f"  [BAD CH] No bad channels detected")

        # ── Step 6: Average re-reference (CAR) — now on clean channels only ──
        seg_data = average_rereference(seg_data)
        logger.info(f"  [REREF] Common average reference applied ({seg_data.shape[0]} channels)")

        # ── Step 7: Bandpass filter 0.1–45 Hz ──
        seg_data = bandpass_filter(seg_data, sfreq_proc, low=0.1, high=45.0)
        logger.info(f"  [FILTER] Bandpass 0.1–45 Hz applied")

        # ── Step 8: Epoch extraction ──
        epochs, times, n_accepted, n_rejected = extract_epochs(
            seg_data, sess_events_local, sfreq_proc,
            tmin=-0.5, tmax=0.35,
            reject_uv=reject_uv,
            reject_tmax=0.0,  # check baseline only (-500ms to 0ms)
            max_epochs=max_epochs,
        )
        n_used = epochs.shape[2]  # actual count after optional cap
        _cap_msg = f", capped to {n_used}" if max_epochs and n_used < n_accepted else ""
        _exact_warn = ""
        if exact_epochs and max_epochs is not None and n_accepted < max_epochs:
            _exact_warn = (
                f" — WARNING: exact mode requested {max_epochs} epochs but only "
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
                "n_rejected": n_rejected,
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
                f"  [SNR GATE] SNR = {snr:.2f} < {min_snr:.1f} — "
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
                    f"SNR={snr:.2f} < {min_snr} — below exclusion threshold. "
                    f"PCIst value may be UNRELIABLE."
                )
            if pcist_result["n_components"] == 0:
                warnings_list.append(
                    "No SVD components survived SNR filtering. "
                    "Possible causes: weak TMS response or noisy data."
                )
            if abs(trigger_timing["offset_ms"]) > 2.0:
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
                    f"Results use {n_accepted} epochs — consider lowering the epoch cap."
                )

            n_ch_used = int(np.sum(ch_mask)) if bad_ch_names else len(ch_names_eeg)
            session_results.append({
                "label": sess["label"],
                "n_events": sess["n_events"],
                "n_accepted": n_accepted,
                "n_rejected": n_rejected,
                "n_channels_used": n_ch_used,
                "bad_channels": bad_ch_names if bad_ch_names else [],
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
                "pcist": None,
                "error": str(e),
                "start_time": sess["start_time"],
                "end_time": sess["end_time"],
                "duration": sess["duration"],
                "median_isi": sess.get("median_isi", 0),
            })

    return {
        "file": os.path.basename(vhdr_path),
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


# ═══════════════════════════════════════════════════════════════════════════
# §6 HTML REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_html_report(analysis: Dict[str, Any], output_path: str):
    """Generate interactive HTML report with Plotly."""

    results = analysis.get("results", [analysis])

    # Collect all session data for the report
    all_sessions_json = []
    all_eeg_data = []

    for result in results:
        for sess in result["sessions"]:
            all_sessions_json.append(sess)

        # Limit EEG display channels to 20 evenly spaced for file size
        ch_names = result["ch_names"]
        eeg_data = result["eeg_display_data"]
        n_total_ch = len(ch_names)
        if n_total_ch > 20:
            step = n_total_ch // 20
            display_indices = list(range(0, n_total_ch, step))[:20]
        else:
            display_indices = list(range(n_total_ch))

        all_eeg_data.append({
            "file": result["file"],
            "ch_names": [ch_names[i] for i in display_indices],
            "all_ch_names": ch_names,
            "sfreq": result["sfreq"],
            "duration": result["duration"],
            "n_stim": result["n_stim_total"],
            "stim_times": [round(t, 3) for t in result["stim_times"]],
            "eeg_times": [round(t, 3) for t in result["eeg_display_times"]],
            "eeg_data": [eeg_data[i] for i in display_indices],
            "ds_factor": result["ds_factor"],
            "all_marker_times": [round(t, 3) for t in result["all_marker_times"]],
            "all_marker_types": result["all_marker_types"],
            "display_indices": display_indices,
        })

    # Serialize session data (without heavy EEG data)
    sessions_for_json = []
    for s in all_sessions_json:
        entry = {k: v for k, v in s.items() if k not in ("evoked_data",)}
        # evoked_data can be large; we keep it for per-session evoked plots
        if "evoked_data" in s and s["evoked_data"] is not None:
            # Keep all channels but round values
            entry["evoked_data"] = [[round(v, 4) for v in ch] for ch in s["evoked_data"]]
        sessions_for_json.append(entry)

    html = _build_html(sessions_for_json, all_eeg_data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Report saved to {output_path}")


def _build_html(sessions: List[Dict], eeg_data_list: List[Dict]) -> str:
    """Build the complete HTML string."""

    sessions_json = json.dumps(sessions, default=_json_default)
    eeg_json = json.dumps(eeg_data_list, default=_json_default)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TMS-EEG PCI Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
    --ink-900: #102231;
    --ink-700: #264255;
    --accent: #0b6e8a;
    --accent-light: #1483a5;
    --warm: #c86b33;
    --bg: #f5f8fb;
    --panel: #ffffff;
    --line: #d6e2e8;
    --green: #2E7D32;
    --orange: #E65100;
    --grey: #37474F;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--ink-900);
    line-height: 1.5;
}}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
.header {{
    background: linear-gradient(135deg, #0b6e8a 0%, #1a3a4a 100%);
    color: white;
    padding: 30px 40px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(11, 110, 138, 0.3);
}}
.header h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }}
.header .subtitle {{ font-size: 0.95rem; opacity: 0.85; }}
.card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}
.card h2 {{
    font-size: 1.15rem;
    color: var(--ink-900);
    border-bottom: 1px solid var(--line);
    padding-bottom: 8px;
    margin-bottom: 16px;
}}
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}}
.metric {{
    background: linear-gradient(180deg, #f8fbfd 0%, #f0f5f8 100%);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}}
.metric-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--ink-900);
}}
.metric-label {{
    font-size: 0.75rem;
    color: #5b7282;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.pci-badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}}
.pci-conscious {{ background: #E8F5E9; color: #2E7D32; border: 1px solid #2E7D32; }}
.pci-intermediate {{ background: #FFF3E0; color: #E65100; border: 1px solid #E65100; }}
.pci-unconscious {{ background: #ECEFF1; color: #37474F; border: 1px solid #37474F; }}
.sessions-tabs {{
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}}
.session-tab {{
    padding: 8px 20px;
    border: 2px solid var(--line);
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.2s;
    background: white;
}}
.session-tab:hover {{ border-color: var(--accent); }}
.session-tab.active {{
    background: var(--accent);
    color: white;
    border-color: var(--accent);
}}
.session-content {{ display: none; }}
.session-content.active {{ display: block; }}
.plot-container {{ width: 100%; min-height: 300px; }}
.summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}}
.summary-table th {{
    background: #eaf2f6;
    color: #183448;
    font-weight: 650;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 2px solid #c7d9e2;
}}
.summary-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid #e5eef3;
}}
.summary-table tr:hover {{ background: #f8fbfd; }}
.eeg-controls {{
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 8px;
    flex-wrap: wrap;
}}
.eeg-controls label {{ font-size: 0.85rem; font-weight: 600; }}
.eeg-controls select, .eeg-controls input {{
    padding: 4px 8px;
    border: 1px solid var(--line);
    border-radius: 6px;
    font-size: 0.85rem;
}}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>Perturbational Complexity Index — TMS-EEG Analysis</h1>
    <div class="subtitle">Casali et al. (2013) framework | Bootstrap significance + Lempel-Ziv complexity</div>
</div>

<!-- Summary Card -->
<div class="card" id="summary-card"></div>

<!-- Full EEG Display -->
<div class="card">
    <h2>Full EEG Recording with Stimulus Markers</h2>
    <div class="eeg-controls">
        <label>Channels:</label>
        <select id="ch-select" onchange="updateEEGPlot()">
            <option value="all">All channels</option>
            <option value="subset">Key channels (Fz, Cz, Pz, C3, C4)</option>
        </select>
        <label>Scale (µV):</label>
        <input type="range" id="scale-slider" min="10" max="500" value="100" oninput="updateEEGPlot()">
        <span id="scale-value">100</span>
    </div>
    <div id="eeg-full-plot" class="plot-container" style="min-height:500px;"></div>
</div>

<!-- Session Tabs -->
<div class="card">
    <h2>Session-Based Analysis</h2>
    <div class="sessions-tabs" id="session-tabs"></div>
    <div id="session-panels"></div>
</div>

<!-- Comparison Table -->
<div class="card">
    <h2>Session Comparison</h2>
    <div id="comparison-table"></div>
</div>

<!-- Interpretation -->
<div class="card">
    <h2>Interpretation Notes</h2>
    <table class="summary-table">
        <tr><th>Item</th><th>Meaning</th><th>Use</th></tr>
        <tr><td>PCIst</td><td>Sensor-level normalized state-transition sum</td><td>Compare sessions processed with identical settings and QC gates</td></tr>
        <tr><td>Scale</td><td>Not bounded to 0-1</td><td>Do not apply Casali LZ-PCI thresholds directly</td></tr>
        <tr><td>QC</td><td>SNR, rejected epochs, bad channels, and trigger timing</td><td>Interpret before ranking sessions</td></tr>
    </table>
</div>

</div>

<script>
const sessions = {sessions_json};
const eegDataList = {eeg_json};

// ── INITIALIZATION ──
function init() {{
    buildSummary();
    buildEEGPlot();
    buildSessionTabs();
    buildComparisonTable();
}}

// ── SUMMARY ──
function buildSummary() {{
    const card = document.getElementById('summary-card');
    let totalStim = 0;
    eegDataList.forEach(d => totalStim += d.n_stim);

    const validSessions = sessions.filter(s => s.pcist !== null && s.pcist !== undefined);
    const avgPCIst = validSessions.length > 0 ? (validSessions.reduce((a, s) => a + s.pcist, 0) / validSessions.length) : 0;

    let html = '<h2>Recording Overview</h2><div class="metrics-grid">';
    html += `<div class="metric"><div class="metric-value">${{eegDataList.length}}</div><div class="metric-label">File(s)</div></div>`;
    html += `<div class="metric"><div class="metric-value">${{eegDataList[0]?.ch_names?.length || 0}}</div><div class="metric-label">Channels</div></div>`;
    html += `<div class="metric"><div class="metric-value">${{(eegDataList[0]?.sfreq || 0).toFixed(0)}} Hz</div><div class="metric-label">Sampling Rate</div></div>`;
    html += `<div class="metric"><div class="metric-value">${{totalStim}}</div><div class="metric-label">Total Stimuli</div></div>`;
    html += `<div class="metric"><div class="metric-value">${{sessions.length}}</div><div class="metric-label">Sessions</div></div>`;
    if (validSessions.length > 0) {{
        html += `<div class="metric"><div class="metric-value">${{avgPCIst.toFixed(1)}}</div><div class="metric-label">Mean PCIst</div></div>`;
    }}
    html += '</div>';
    card.innerHTML = html;
}}

// ── FULL EEG PLOT ──
function buildEEGPlot() {{
    const eeg = eegDataList[0];
    if (!eeg) return;
    updateEEGPlot();
}}

function updateEEGPlot() {{
    const eeg = eegDataList[0];
    if (!eeg) return;

    const chSelect = document.getElementById('ch-select').value;
    const scale = parseInt(document.getElementById('scale-slider').value);
    document.getElementById('scale-value').textContent = scale;

    let channelIndices = [];
    const keyChannels = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4', 'O1', 'O2'];

    if (chSelect === 'subset') {{
        keyChannels.forEach(name => {{
            const idx = eeg.ch_names.indexOf(name);
            if (idx >= 0) channelIndices.push(idx);
        }});
        if (channelIndices.length === 0) channelIndices = Array.from({{length: Math.min(10, eeg.ch_names.length)}}, (_, i) => i);
    }} else {{
        channelIndices = Array.from({{length: eeg.ch_names.length}}, (_, i) => i);
    }}

    const traces = [];
    const n_ch = channelIndices.length;

    // EEG traces with offset stacking
    channelIndices.forEach((chIdx, i) => {{
        const offset = (n_ch - 1 - i) * scale;
        const y = eeg.eeg_data[chIdx].map(v => v + offset);
        traces.push({{
            x: eeg.eeg_times,
            y: y,
            type: 'scattergl',
            mode: 'lines',
            line: {{ width: 0.5, color: '#333' }},
            name: eeg.ch_names[chIdx],
            hovertemplate: eeg.ch_names[chIdx] + ': %{{customdata:.1f}} µV<extra></extra>',
            customdata: eeg.eeg_data[chIdx],
            showlegend: false,
        }});
    }});

    // Stimulus markers as vertical lines
    const shapes = [];
    const annotations = [];

    // Session boundaries
    sessions.forEach((s, i) => {{
        if (s.start_time !== undefined) {{
            shapes.push({{
                type: 'rect',
                x0: s.start_time,
                x1: s.end_time,
                y0: -scale,
                y1: (n_ch) * scale,
                fillcolor: `rgba(11, 110, 138, 0.06)`,
                line: {{ color: 'rgba(11, 110, 138, 0.3)', width: 1, dash: 'dash' }},
            }});
            annotations.push({{
                x: (s.start_time + s.end_time) / 2,
                y: (n_ch) * scale + scale * 0.3,
                text: `<b>${{s.label}}</b>` + (s.pcist !== null && s.pcist !== undefined ? `<br>PCIst=${{s.pcist.toFixed(1)}}` : ''),
                showarrow: false,
                font: {{ size: 11, color: '#0b6e8a' }},
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: '#0b6e8a',
                borderwidth: 1,
                borderpad: 4,
            }});
        }}
    }});

    // Stimulus ticks
    eeg.stim_times.forEach(t => {{
        shapes.push({{
            type: 'line',
            x0: t, x1: t,
            y0: -scale * 0.5,
            y1: (n_ch) * scale,
            line: {{ color: 'rgba(198, 40, 40, 0.3)', width: 0.8 }},
        }});
    }});

    // Y-axis tick labels for channels
    const tickvals = channelIndices.map((_, i) => (n_ch - 1 - i) * scale);
    const ticktext = channelIndices.map(i => eeg.ch_names[i]);

    const layout = {{
        height: Math.max(400, n_ch * 18 + 80),
        margin: {{ l: 70, r: 30, t: 30, b: 50 }},
        xaxis: {{
            title: 'Time (s)',
            rangeslider: {{ visible: true, thickness: 0.06 }},
            showgrid: true,
            gridcolor: '#eee',
        }},
        yaxis: {{
            tickvals: tickvals,
            ticktext: ticktext,
            tickfont: {{ size: 8 }},
            showgrid: false,
            zeroline: false,
        }},
        shapes: shapes,
        annotations: annotations,
        dragmode: 'zoom',
        hovermode: 'x unified',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
    }};

    Plotly.newPlot('eeg-full-plot', traces, layout, {{
        responsive: true,
        scrollZoom: true,
        displayModeBar: true,
        modeBarButtonsToAdd: ['drawrect', 'eraseshape'],
    }});
}}

// ── SESSION TABS ──
function buildSessionTabs() {{
    const tabsDiv = document.getElementById('session-tabs');
    const panelsDiv = document.getElementById('session-panels');

    sessions.forEach((s, i) => {{
        // Tab
        const tab = document.createElement('div');
        tab.className = 'session-tab' + (i === 0 ? ' active' : '');
        tab.textContent = s.label;
        tab.onclick = () => switchSession(i);
        tabsDiv.appendChild(tab);

        // Panel
        const panel = document.createElement('div');
        panel.className = 'session-content' + (i === 0 ? ' active' : '');
        panel.id = `session-panel-${{i}}`;
        panel.innerHTML = buildSessionPanel(s, i);
        panelsDiv.appendChild(panel);
    }});

    // Render plots for first session
    setTimeout(() => renderSessionPlots(0), 100);
}}

function switchSession(idx) {{
    document.querySelectorAll('.session-tab').forEach((t, i) => {{
        t.className = 'session-tab' + (i === idx ? ' active' : '');
    }});
    document.querySelectorAll('.session-content').forEach((p, i) => {{
        p.className = 'session-content' + (i === idx ? ' active' : '');
    }});
    renderSessionPlots(idx);
}}

function buildSessionPanel(s, idx) {{
    if (s.error && s.pcist == null) {{
        return `<div class="metrics-grid">
            <div class="metric"><div class="metric-value">${{s.n_events}}</div><div class="metric-label">Stimuli</div></div>
            <div class="metric"><div class="metric-value">${{s.n_accepted || 0}}</div><div class="metric-label">Epochs</div></div>
            <div class="metric"><div class="metric-value" style="color:red;">ERROR</div><div class="metric-label">${{s.error}}</div></div>
        </div>`;
    }}

    const nComp = s.n_components || 0;
    const dNST = s.dNST || [];
    const dNSTstr = dNST.length > 0 ? dNST.map(d => d.toFixed(1)).join(', ') : 'N/A';

    return `
    <div class="metrics-grid">
        <div class="metric">
            <div class="metric-value" style="font-size:2rem;">${{s.pcist != null ? s.pcist.toFixed(1) : 'N/A'}}</div>
            <div class="metric-label">PCIst</div>
        </div>
        <div class="metric"><div class="metric-value">${{nComp}}</div><div class="metric-label">SVD Components</div></div>
        <div class="metric"><div class="metric-value">${{s.snr?.toFixed(2) || 'N/A'}}</div><div class="metric-label">SNR</div></div>
        <div class="metric"><div class="metric-value">${{s.n_accepted || 0}} / ${{s.n_events}}</div><div class="metric-label">Epochs</div></div>
        <div class="metric"><div class="metric-value">${{s.n_channels_used || '?'}}</div><div class="metric-label">Channels</div></div>
        <div class="metric"><div class="metric-value">${{s.duration?.toFixed(1) || '?'}}s</div><div class="metric-label">Duration</div></div>
    </div>
    <div style="margin:8px 0; font-size:0.85em; color:#666;">
        <strong>ΔNST per component:</strong> [${{dNSTstr}}]
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
        <div id="evoked-plot-${{idx}}" class="plot-container" style="min-height:280px;"></div>
        <div id="ss-plot-${{idx}}" class="plot-container" style="min-height:280px;"></div>
    </div>
    `;
}}

function renderSessionPlots(idx) {{
    const s = sessions[idx];
    if (!s || s.pcist == null) return;

    // Check if already rendered
    const evokedDiv = document.getElementById(`evoked-plot-${{idx}}`);
    if (!evokedDiv || evokedDiv.dataset.rendered === 'true') return;
    evokedDiv.dataset.rendered = 'true';

    // Evoked butterfly + GFP
    if (s.evoked_data && s.evoked_times) {{
        const traces = [];
        const n_ch = s.evoked_data.length;
        const alpha = Math.max(0.08, Math.min(0.5, 8 / n_ch));

        // Individual channels
        for (let i = 0; i < n_ch; i++) {{
            traces.push({{
                x: s.evoked_times,
                y: s.evoked_data[i],
                type: 'scatter',
                mode: 'lines',
                line: {{ width: 0.4, color: `rgba(60,60,60,${{alpha}})` }},
                showlegend: false,
                hoverinfo: 'skip',
            }});
        }}

        // GFP
        traces.push({{
            x: s.evoked_times,
            y: s.evoked_gfp,
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 2, color: '#000' }},
            name: 'GFP',
        }});

        Plotly.newPlot(`evoked-plot-${{idx}}`, traces, {{
            title: {{ text: 'TMS-Evoked Potentials', font: {{ size: 13 }} }},
            xaxis: {{ title: 'Time (ms)', zeroline: true }},
            yaxis: {{ title: 'Amplitude (µV)' }},
            shapes: [{{ type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#C62828', dash: 'dash', width: 1.5 }} }}],
            margin: {{ l: 50, r: 20, t: 40, b: 40 }},
            height: 280,
            plot_bgcolor: 'white',
            showlegend: true,
            legend: {{ x: 0.85, y: 0.95, font: {{ size: 9 }} }},
        }}, {{ responsive: true }});
    }}

    // ΔNST component bar chart (replaces SS matrix heatmap for PCIst)
    const ssDiv = document.getElementById(`ss-plot-${{idx}}`);
    if (ssDiv && s.dNST && s.dNST.length > 0) {{
        const compLabels = s.dNST.map((_, i) => `Comp ${{(s.components_kept || [])[i] || i + 1}}`);
        Plotly.newPlot(`ss-plot-${{idx}}`, [{{
            x: compLabels,
            y: s.dNST,
            type: 'bar',
            marker: {{ color: '#0b6e8a' }},
            text: s.dNST.map(d => d.toFixed(1)),
            textposition: 'outside',
        }}], {{
            title: {{ text: 'ΔNST per SVD Component', font: {{ size: 13 }} }},
            xaxis: {{ title: 'Component' }},
            yaxis: {{ title: 'ΔNST (state transitions)', rangemode: 'tozero' }},
            margin: {{ l: 50, r: 20, t: 40, b: 40 }},
            height: 280,
            plot_bgcolor: 'white',
        }}, {{ responsive: true }});
    }} else if (ssDiv) {{
        ssDiv.innerHTML = '<p style="text-align:center;color:#999;padding-top:80px;">No components survived SNR filter</p>';
    }}
}}

// ── COMPARISON TABLE ──
function buildComparisonTable() {{
    const div = document.getElementById('comparison-table');
    const validSessions = sessions.filter(s => s.pcist != null);

    if (validSessions.length === 0) {{
        div.innerHTML = '<p>No valid PCIst results to compare.</p>';
        return;
    }}

    let html = '<table class="summary-table"><tr>';
    html += '<th>Session</th><th>PCIst</th><th>SVD Comp.</th><th>SNR</th><th>Epochs</th><th>Channels</th><th>Duration</th><th>ISI (s)</th>';
    html += '</tr>';

    validSessions.forEach(s => {{
        html += `<tr>
            <td><strong>${{s.label}}</strong></td>
            <td><strong>${{s.pcist.toFixed(1)}}</strong></td>
            <td>${{s.n_components || 0}}</td>
            <td>${{s.snr?.toFixed(2) || '-'}}</td>
            <td>${{s.n_accepted}}/${{s.n_events}}</td>
            <td>${{s.n_channels_used || '-'}}</td>
            <td>${{s.duration?.toFixed(1) || '-'}}s</td>
            <td>${{s.median_isi?.toFixed(2) || '-'}}</td>
        </tr>`;
    }});

    html += '</table>';
    div.innerHTML = html;
}}

// ── START ──
document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return str(obj)


# ═══════════════════════════════════════════════════════════════════════════
# §7 CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TMS-EEG PCIst Analyzer (Comolatti 2019)")
    parser.add_argument("vhdr_files", nargs="+", help="BrainVision .vhdr file(s)")
    parser.add_argument("-o", "--output", default="pcist_report.html", help="Output HTML file")
    parser.add_argument("--gap", type=float, default=30.0, help="Gap threshold for session detection (seconds)")
    parser.add_argument("--reject", type=float, default=150.0, help="Artifact rejection threshold (µV)")
    parser.add_argument("--decimate-to", type=float, default=1000.0, help="Target sampling rate after decimation (Hz)")
    parser.add_argument("--pcist-k", type=float, default=1.2, help="PCIst baseline penalty factor")
    parser.add_argument("--pcist-min-snr", type=float, default=1.1, help="PCIst min component SNR")
    parser.add_argument("--pcist-max-var", type=float, default=99.0, help="PCIst max cumulative variance (%)")
    parser.add_argument("--pcist-n-steps", type=int, default=100, help="PCIst number of threshold steps")

    args = parser.parse_args()

    kwargs = dict(
        gap_seconds=args.gap,
        reject_uv=args.reject,
        decimate_to=args.decimate_to,
        pcist_k=args.pcist_k,
        pcist_min_snr=args.pcist_min_snr,
        pcist_max_var=args.pcist_max_var,
        pcist_n_steps=args.pcist_n_steps,
    )

    if len(args.vhdr_files) == 1:
        result = analyze_file(args.vhdr_files[0], **kwargs)
        analysis = {"results": [result]}
    else:
        analysis = analyze_multiple_files(args.vhdr_files, **kwargs)

    generate_html_report(analysis, args.output)
    print(f"\nReport saved to: {args.output}")
