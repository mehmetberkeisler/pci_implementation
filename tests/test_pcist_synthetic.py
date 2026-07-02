"""Ground-truth sanity checks for the reference-PCIst wrapper.

These tests guard against regressions if the vendored pci_st.py is ever
refreshed or the wrapper is changed. They are not a PCIst *validation* -
that belongs with the authors' paper - but they pin down a few
qualitative expectations:

1. Flat baseline-like noise across response and baseline → PCIst near 0.
2. A strong deterministic oscillatory burst in the response window → PCIst
   clearly above zero with at least one retained component.
3. NaN anywhere in the evoked matrix → wrapper returns shape-complete zero.
4. Result keys match the contract the pipeline depends on.
"""

import numpy as np
import pytest

from pcist import calc_PCIst


EXPECTED_KEYS = {
    "PCIst",
    "n_components",
    "dNST",
    "var_explained",
    "snrs",
    "cumvar",
    "components_kept",
    "max_thresholds",
    "NST_diff",
    "pcist_method",
}


def _make_times(sfreq: int = 1000) -> np.ndarray:
    return np.arange(-0.500, 0.351, 1 / sfreq)


def test_result_keys_contract():
    times = _make_times()
    rng = np.random.RandomState(1)
    evoked = 0.01 * rng.randn(6, len(times))
    result = calc_PCIst(evoked, times, n_steps=20, min_snr=1.01)
    assert EXPECTED_KEYS.issubset(result.keys()), (
        f"missing keys: {EXPECTED_KEYS - result.keys()}"
    )


def test_flat_noise_drops_out_under_strict_snr():
    """With a strict SNR gate, pure noise should yield PCIst ≈ 0.

    Note that with Comolatti's permissive default `min_snr=1.1`,
    PCIst on pure noise can still be non-trivial because weak components
    pass by chance. This test uses a stricter gate so that noise-driven
    components are filtered out entirely, which is the regime real
    analyses should aim for when the SNR on epochs is good.
    """
    times = _make_times()
    rng = np.random.RandomState(42)
    noise = 0.01 * rng.randn(6, len(times))
    result = calc_PCIst(noise, times, n_steps=30, min_snr=1.5)
    assert result["PCIst"] < 2.0, (
        f"Strict-SNR PCIst on pure noise should be near 0, got {result['PCIst']}"
    )


def test_burst_beats_noise_under_strict_snr():
    """With a strict SNR gate, a clean burst scores higher than pure noise."""
    times = _make_times()
    rng = np.random.RandomState(11)
    n_ch = 8

    noise = 0.01 * rng.randn(n_ch, len(times))
    noise_res = calc_PCIst(noise, times, n_steps=30, min_snr=1.5)

    burst = 0.01 * rng.randn(n_ch, len(times))
    resp = (times >= 0.0) & (times <= 0.300)
    t_r = times[resp]
    spatial = np.linspace(-1.0, 1.0, n_ch)[:, None]
    burst[:, resp] += spatial * (
        np.sin(2 * np.pi * 12 * t_r)
        + 0.4 * np.sin(2 * np.pi * 27 * t_r)
    )[None, :]
    burst_res = calc_PCIst(burst, times, n_steps=30, min_snr=1.5)

    assert burst_res["PCIst"] > noise_res["PCIst"] + 3.0, (
        f"burst={burst_res['PCIst']} should exceed noise={noise_res['PCIst']}"
    )
    assert burst_res["n_components"] >= 1


def test_strong_oscillation_produces_positive_pcist():
    """A clean sine burst in the response window should score > flat noise."""
    times = _make_times()
    rng = np.random.RandomState(7)
    evoked = 0.005 * rng.randn(8, len(times))

    resp_mask = (times >= 0.0) & (times <= 0.300)
    t_resp = times[resp_mask]
    spatial = np.array([1.0, 0.8, 0.6, 0.4, -0.3, -0.5, -0.7, 0.2])[:, None]
    burst = np.sin(2 * np.pi * 12 * t_resp) + 0.4 * np.sin(2 * np.pi * 27 * t_resp)
    evoked[:, resp_mask] += spatial * burst[None, :]

    result = calc_PCIst(evoked, times, n_steps=30, min_snr=1.01)
    assert result["n_components"] >= 1
    assert result["PCIst"] > 5.0, (
        f"Strong oscillatory burst produced suspiciously low PCIst: {result['PCIst']}"
    )
    # Every component contribution must be non-negative (matches reference).
    assert all(d >= 0 for d in result["dNST"])


def test_nan_input_returns_zero_result():
    times = _make_times()
    rng = np.random.RandomState(0)
    evoked = 0.01 * rng.randn(4, len(times))
    evoked[1, 50] = np.nan
    result = calc_PCIst(evoked, times, n_steps=30, min_snr=1.01)
    assert result["PCIst"] == 0.0
    assert result["n_components"] == 0
    assert result["dNST"] == []
    assert EXPECTED_KEYS.issubset(result.keys())


def test_bad_input_shape_raises():
    times = _make_times()
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError):
        calc_PCIst(rng.randn(len(times)), times)  # 1D evoked

    with pytest.raises(ValueError):
        calc_PCIst(rng.randn(4, len(times) - 3), times)  # length mismatch


def test_too_short_window_raises():
    times = np.linspace(-0.005, 0.005, 11)  # 11 samples total
    evoked = np.zeros((2, len(times)))
    with pytest.raises(ValueError):
        calc_PCIst(
            evoked, times,
            baseline_window=(-0.004, -0.003),   # only ~1-2 samples
            response_window=(0.001, 0.002),
        )
