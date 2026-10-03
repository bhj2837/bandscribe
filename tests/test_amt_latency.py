"""Latency constant estimation and its pre-registered dev/test split (gtab.amt.latency)."""
from __future__ import annotations

import numpy as np
import pytest

from gtab.amt import latency as L


def _notes(times, pitches):
    return [{"onset_s": float(t), "offset_s": float(t) + 0.2, "pitch": int(p)} for t, p in zip(times, pitches)]


def test_shift_of_25ms_is_recovered():
    rng = np.random.default_rng(0)
    ref_t = np.sort(rng.uniform(0.5, 60.0, 400))
    ref_p = rng.integers(40, 80, 400)
    jitter = rng.normal(0.0, 0.004, 400)
    est = _notes(ref_t + 0.025 + jitter, ref_p)
    r = L.estimate_latency(est, (ref_t, ref_p))
    assert r["median_ms"] == pytest.approx(25.0, abs=1.0)
    assert r["n"] >= 390
    assert r["mad_ms"] < 5.0
    corr = L.median_abs_residual_ms(est, (ref_t, ref_p), r["median_ms"] / 1000.0)
    assert corr["median_abs_ms"] < 4.0


def test_pitch_matching_ignores_nearby_other_pitches():
    ref = (np.array([1.0, 1.01]), np.array([60, 72]))
    est = _notes([1.03], [72])
    assert L.estimate_latency(est, ref)["median_ms"] == pytest.approx(20.0, abs=1e-6)
    # without pitches the nearest onset wins
    assert L.estimate_latency((np.array([1.03]), None), np.array([1.0, 1.01]))["median_ms"] == pytest.approx(20.0)


def test_window_and_empty():
    assert L.estimate_latency(_notes([5.0], [60]), (np.array([1.0]), np.array([60])))["n"] == 0
    assert L.estimate_latency([], np.array([1.0]))["median_ms"] is None


def test_dev_test_split_never_overlaps():
    assert set(L.DEV_PLAYERS).isdisjoint(L.TEST_PLAYERS)
    assert [L.split_of_player(p) for p in ("00", "01", "02", "03", "04", "05")] == ["dev"] * 3 + ["test"] * 3
    with pytest.raises(ValueError):
        L.split_of_player("06")


def test_split_matches_the_guitarset_loader():
    try:
        from gtab.datasets.loaders import guitarset
    except ImportError:
        pytest.skip("DATA loader not present")
    assert tuple(guitarset.DEV_PLAYERS) == L.DEV_PLAYERS and tuple(guitarset.TEST_PLAYERS) == L.TEST_PLAYERS


def test_spectral_flux_finds_clicks():
    sr = 44100
    x = np.zeros(sr * 4, dtype=np.float64)
    rng = np.random.default_rng(1)
    onsets = [0.5, 1.25, 2.0, 2.6, 3.3]
    for t in onsets:
        i = int(t * sr)
        n = int(0.05 * sr)
        x[i:i + n] += rng.normal(0, 0.3, n) * np.exp(-np.arange(n) / (0.01 * sr))
    found = L.spectral_flux_onsets(x, sr)
    r = L.estimate_latency((found, None), np.array(onsets), max_ms=50)
    assert r["n"] == len(onsets)
    assert abs(r["median_ms"]) < 5.0  # measured bias ~ -2 ms (module comment)
