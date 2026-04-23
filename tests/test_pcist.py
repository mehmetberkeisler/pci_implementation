import numpy as np

from analyze_pci import calc_PCIst


def test_pcist_uses_normalized_recurrence_transitions():
    """PCIst should be a non-negative normalized ST sum, not raw crossing counts."""
    sfreq = 1000
    times = np.arange(-0.5, 0.351, 1 / sfreq)
    rng = np.random.RandomState(7)
    evoked = 0.02 * rng.randn(6, len(times))

    response = (times >= 0.0) & (times <= 0.300)
    t_resp = times[response]
    spatial = np.array([1.0, 0.8, 0.45, -0.35, -0.7, 0.25])[:, None]
    evoked[:, response] += spatial * (
        np.sin(2 * np.pi * 12 * t_resp)[None, :]
        + 0.4 * np.sin(2 * np.pi * 27 * t_resp)[None, :]
    )

    result = calc_PCIst(evoked, times, n_steps=30, min_snr=1.01)
    n_response = int(np.sum(response))

    assert result["PCIst"] >= 0.0
    assert result["pcist_method"] == "recurrence_normalized_state_transitions"
    assert len(result["dNST"]) == result["n_components"]
    assert all(0.0 <= d <= n_response for d in result["dNST"])
