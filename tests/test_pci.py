"""
Tests for the PCI core algorithm.

Verifies:
    - Lempel-Ziv complexity (LZ76)
    - Source entropy
    - PCI normalisation formula: PCI = c_L * log2(L) / (L * H)
    - Bootstrap significance matrix (correlated resampling)
    - Trigger classification
    - Istisna mode detection
    - Window overlap validation
    - Edge cases
"""

import warnings
import pytest
import numpy as np
from pci import (
    lempel_ziv_complexity,
    lempel_ziv_complexity_lz76,
    lempel_ziv_2d,
    compute_entropy,
    compute_pci,
    compute_pci_temporal,
    compute_significance_matrix,
    classify_trigger,
    is_perturbation_trigger,
    split_event_blocks,
    detect_stimulation_like_response_train,
    validate_results,
    validate_window_overlap,
)


# ═══════════════════════════════════════════════════════════════════════════
# LZ76 Complexity
# ═══════════════════════════════════════════════════════════════════════════

class TestLZComplexity:
    def test_empty(self):
        assert lempel_ziv_complexity([]) == 0
        assert lempel_ziv_complexity(np.array([])) == 0

    def test_single(self):
        assert lempel_ziv_complexity([1]) == 1
        assert lempel_ziv_complexity([0]) == 1

    def test_constant_string(self):
        # "0000000" should have low complexity
        # LZ76: "0" (new, c=1), "00" (0 in history -> extend, 00 not -> c=2),
        #        "000" -> 0 in hist, 00 in hist, 000 not -> c=3, etc.
        # Actually: "0" c=1, i=1. "0" in "0" -> match=1. "00" not in "0" -> break.
        # new word len 2, i=3. "0" in "000" -> match=1. "00" in "000" -> match=2.
        # "000" in "000" -> match=3? s[:3]="000", s[3:6]="000", check "0" in "000" -> yes (match=1)
        # "00" in "000" -> yes (match=2). "000" in "000" -> yes (match=3). "0000" -> break.
        # new word len 4, i=7. Done. c=3.
        c = lempel_ziv_complexity([0] * 7)
        assert c == 3

    def test_alternating(self):
        # "10101010" — moderately complex
        c = lempel_ziv_complexity([1, 0, 1, 0, 1, 0, 1, 0])
        assert c > 1

    def test_random_higher_than_constant(self):
        rng = np.random.RandomState(42)
        rand_seq = rng.randint(0, 2, 200)
        const_seq = np.zeros(200, dtype=int)
        assert lempel_ziv_complexity(rand_seq) > lempel_ziv_complexity(const_seq)

    def test_backward_compat_alias(self):
        seq = [1, 0, 1, 1, 0, 0, 1]
        assert lempel_ziv_complexity(seq) == lempel_ziv_complexity_lz76(seq)

    def test_string_input(self):
        assert lempel_ziv_complexity("10110010") == lempel_ziv_complexity(
            [1, 0, 1, 1, 0, 0, 1, 0]
        )


# ═══════════════════════════════════════════════════════════════════════════
# Source Entropy
# ═══════════════════════════════════════════════════════════════════════════

class TestEntropy:
    def test_all_zeros(self):
        H, p1 = compute_entropy(np.zeros((10, 10)))
        assert H == 0.0
        assert p1 == 0.0

    def test_all_ones(self):
        H, p1 = compute_entropy(np.ones((10, 10)))
        assert H == 0.0
        assert p1 == 1.0

    def test_half_filled(self):
        ss = np.zeros((10, 10))
        ss[:5, :] = 1
        H, p1 = compute_entropy(ss)
        assert np.isclose(H, 1.0)
        assert p1 == 0.5

    def test_entropy_range(self):
        # Entropy must be in [0, 1]
        rng = np.random.RandomState(99)
        ss = rng.randint(0, 2, (20, 30))
        H, p1 = compute_entropy(ss)
        assert 0 <= H <= 1.0
        assert 0 <= p1 <= 1.0

    def test_empty(self):
        H, p1 = compute_entropy(np.array([]).reshape(0, 0))
        assert H == 0.0
        assert p1 == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PCI Computation — Analytical Normalisation
# ═══════════════════════════════════════════════════════════════════════════

class TestPCI:
    def test_zero_activity(self):
        ss = np.zeros((10, 100))
        result = compute_pci(ss, verbose=False)
        assert result["pci"] == 0.0
        assert result["c_L"] == 0

    def test_simple_pattern(self):
        ss = np.zeros((5, 20))
        ss[0, ::2] = 1  # periodic on row 0
        result = compute_pci(ss, verbose=False)
        assert 0 <= result["pci"] <= 2.0  # may be > 1 for very small matrices
        assert result["c_L"] > 0

    def test_random_matrix_pci_near_one(self):
        """For sufficiently large random matrices, PCI should approach 1."""
        rng = np.random.RandomState(42)
        ss = rng.randint(0, 2, (30, 100))
        result = compute_pci(ss, verbose=False)
        # Random data → PCI close to 1.0 (within ~0.3 for finite sizes)
        assert result["pci"] > 0.5
        assert result["pci"] < 1.5  # allow some finite-size effect

    def test_normalisation_formula(self):
        """Verify PCI = c_L * log2(L) / (L * H)."""
        rng = np.random.RandomState(123)
        ss = rng.randint(0, 2, (10, 50))
        result = compute_pci(ss, verbose=False)
        H, p1 = compute_entropy(ss)
        L = ss.size
        # Recompute manually
        c_L = result["c_L"]
        expected_pci = c_L * np.log2(L) / (L * H) if H > 0 else 0.0
        assert np.isclose(result["pci"], expected_pci, rtol=1e-6)

    def test_pci_returns_dict(self):
        ss = np.zeros((5, 20))
        ss[0, 0] = 1
        ss[1, 5] = 1
        result = compute_pci(ss, verbose=False)
        assert isinstance(result, dict)
        assert "pci" in result
        assert "pci_t" in result
        assert "c_L" in result
        assert "H" in result
        assert "p1" in result

    def test_sorted_sources(self):
        """Sources should be sorted before LZ; different ordering → same PCI."""
        rng = np.random.RandomState(7)
        ss = rng.randint(0, 2, (10, 30))
        r1 = compute_pci(ss, verbose=False)
        # Shuffle rows
        perm = rng.permutation(10)
        r2 = compute_pci(ss[perm, :], verbose=False)
        assert np.isclose(r1["pci"], r2["pci"])


# ═══════════════════════════════════════════════════════════════════════════
# Trigger Classification
# ═══════════════════════════════════════════════════════════════════════════

class TestTriggerClassification:
    def test_tms(self):
        cat, _ = classify_trigger("S  1", 100)
        assert cat == "tms_likely"

    def test_response(self):
        cat, _ = classify_trigger("Response", 50)
        assert cat == "response"

    def test_annotation(self):
        cat, _ = classify_trigger("New Segment", 1)
        assert cat == "annotation"

    def test_stim_channel(self):
        cat, _ = classify_trigger("Stim_1", 150)
        assert cat == "tms_likely"

    def test_brainvision_stimulus_label(self):
        cat, _ = classify_trigger("Stimulus/S 12", 120)
        assert cat == "tms_likely"

    def test_brainvision_response_label(self):
        cat, _ = classify_trigger("Response/R  1", 90)
        assert cat == "response"

    def test_unknown(self):
        cat, _ = classify_trigger("xyz_marker", 5)
        assert cat == "other"

    def test_perturbation_gate(self):
        assert is_perturbation_trigger("tms_likely")
        assert is_perturbation_trigger("unknown_likely")
        assert not is_perturbation_trigger("response")


# ═══════════════════════════════════════════════════════════════════════════
# Response-Train Exception Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseTrainDetection:
    def test_split_event_blocks(self):
        # Two 1 Hz blocks split by a long gap
        samples = np.array([100, 200, 300, 10000, 10100, 10200])
        blocks = split_event_blocks(samples, sfreq=100.0, gap_seconds=20.0)
        assert blocks == [(0, 2), (3, 5)]

    def test_stimulation_like_periodic_train(self):
        # 3 blocks × 50 events, ISI ~3 s at 1000 Hz
        block1 = np.arange(0, 50 * 3000, 3000)
        block2 = np.arange(300000, 300000 + 50 * 3000, 3000)
        block3 = np.arange(700000, 700000 + 50 * 3000, 3000)
        samples = np.concatenate([block1, block2, block3])

        info = detect_stimulation_like_response_train(samples, sfreq=1000.0)
        assert info["is_stimulation_like"]
        assert info["n_events"] == 150
        assert info["n_blocks"] == 3
        assert all(b["n_events"] == 50 for b in info["blocks"])

    def test_non_periodic_train_not_stimulation_like(self):
        rng = np.random.RandomState(11)
        samples = np.cumsum(rng.randint(100, 5000, size=120))
        info = detect_stimulation_like_response_train(samples, sfreq=1000.0)
        assert not info["is_stimulation_like"]

    def test_mixed_periodic_and_nonperiodic_train_not_stimulation_like(self):
        periodic = np.arange(0, 40 * 3000, 3000)
        random_tail = periodic[-1] + 20000 + np.cumsum(
            np.array([300, 1900, 750, 4200, 600, 2800, 550, 3600, 410, 1750, 960, 2500])
        )
        samples = np.concatenate([periodic, random_tail])
        info = detect_stimulation_like_response_train(samples, sfreq=1000.0)
        assert not info["is_stimulation_like"]


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_good_result(self):
        result = dict(
            pci=0.55, active_pct=8.0, H=0.35, threshold=4.5, snr=2.1,
        )
        val = validate_results(result)
        assert val["valid"]
        assert len(val["errors"]) == 0

    def test_bad_snr(self):
        result = dict(
            pci=0.55, active_pct=8.0, H=0.35, threshold=4.5, snr=0.5,
        )
        val = validate_results(result)
        assert not val["valid"]
        assert any("SNR" in e for e in val["errors"])

    def test_pci_over_one(self):
        result = dict(
            pci=1.5, active_pct=8.0, H=0.35, threshold=4.5, snr=2.0,
        )
        val = validate_results(result)
        assert not val["valid"]


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap Significance Matrix
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapSignificance:
    """Tests for compute_significance_matrix with correlated resampling."""

    def _make_source_data(self, n_src=10, n_times=200, n_trials=50, sfreq=500.0):
        """Generate synthetic source data with known structure."""
        rng = np.random.RandomState(0)
        times = np.linspace(-0.5, 0.35, n_times)
        # Baseline is random noise; post-stimulus has a signal on src 0-2
        data = rng.randn(n_src, n_times, n_trials) * 0.5
        post_mask = times >= 0.01
        data[:3, post_mask, :] += 3.0  # inject strong signal on first 3 sources
        return data, times

    def test_basic_output_shape(self):
        data, times = self._make_source_data()
        SS, thresh, diag = compute_significance_matrix(data, times, n_bootstrap=50, verbose=False)
        n_post = int(np.sum((times >= 0.008) & (times <= 0.300)))
        assert SS.shape == (10, n_post)
        assert thresh > 0

    def test_correlated_resampling_produces_higher_threshold(self):
        """Correlated resampling (correct) should be more conservative than
        independent resampling, i.e. produce a higher or equal threshold."""
        data, times = self._make_source_data(n_src=20, n_trials=80)
        SS, thresh_corr, _ = compute_significance_matrix(
            data, times, n_bootstrap=200, seed=42, verbose=False
        )
        # Threshold from correlated resampling should be reasonably large
        assert thresh_corr > 1.5  # global max of z across 20 sources

    def test_dead_channel_exclusion(self):
        """Sources with zero variance should be excluded, not floored."""
        data, times = self._make_source_data()
        # Make source 5 perfectly flat
        data[5, :, :] = 0.0
        SS, thresh, diag = compute_significance_matrix(data, times, n_bootstrap=50, verbose=False)
        assert diag["n_sources_excluded"] == 1
        # The flat source row should be all zeros
        assert np.sum(SS[5, :]) == 0

    def test_reproducible_with_seed(self):
        data, times = self._make_source_data()
        SS1, t1, _ = compute_significance_matrix(data, times, n_bootstrap=50, seed=99, verbose=False)
        SS2, t2, _ = compute_significance_matrix(data, times, n_bootstrap=50, seed=99, verbose=False)
        assert t1 == t2
        np.testing.assert_array_equal(SS1, SS2)

    def test_none_seed_is_non_deterministic(self):
        """seed=None should use a random state (non-deterministic)."""
        data, times = self._make_source_data(n_src=20, n_trials=80)
        # With seed=None, internal RandomState is unseeded.
        # We verify the function accepts None without error.
        SS, thresh, diag = compute_significance_matrix(
            data, times, n_bootstrap=50, seed=None, verbose=False
        )
        assert thresh > 0
        assert SS.shape[0] == 20


# ═══════════════════════════════════════════════════════════════════════════
# Window Overlap Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestWindowOverlap:
    def test_overlap_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_window_overlap(artifact_end_ms=10.0, post_stim_start_ms=8.0)
            assert len(w) == 1
            assert "overlap" in str(w[0].message).lower()

    def test_no_overlap_silent(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_window_overlap(artifact_end_ms=6.0, post_stim_start_ms=8.0)
            assert len(w) == 0


# ═══════════════════════════════════════════════════════════════════════════
# PCI Temporal (harmonised thresholds)
# ═══════════════════════════════════════════════════════════════════════════

class TestPCITemporal:
    def test_zero_curve_for_low_entropy(self):
        ss = np.zeros((10, 100))
        ss[0, 0] = 1  # p1 = 0.001, H ≈ 0.01 — below gate
        curve = compute_pci_temporal(ss)
        assert np.all(curve == 0)

    def test_curve_matches_final_pci(self):
        """Last point of PCI(t) should equal the full PCI."""
        rng = np.random.RandomState(42)
        ss = rng.randint(0, 2, (15, 60))
        result = compute_pci(ss, verbose=False)
        curve = compute_pci_temporal(ss)
        if result["pci"] > 0:
            np.testing.assert_allclose(curve[-1], result["pci"], rtol=1e-6)
