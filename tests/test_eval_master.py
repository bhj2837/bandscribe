"""Deterministic look-ahead limiter used for Tier B commercial-loudness remixes (E1 inputs)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from bandscribe.eval import remix


def _music(seconds: float = 12.0, sr: int = 44100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    x = 0.2 * np.sin(2 * np.pi * 110 * t) + 0.1 * np.sin(2 * np.pi * 440 * t + 1.0)
    clicks = np.zeros(n)
    clicks[:: sr // 2] = 1.0
    drums = signal.lfilter([1.0], [1, -0.995], clicks) * 0.6 * rng.uniform(0.5, 1.0)
    noise = rng.normal(0, 0.02, n)
    left = x + drums + noise
    right = 0.8 * x + drums + rng.normal(0, 0.02, n)
    return np.stack([left, right], axis=1)


def _loudness(x, sr=44100):
    import pyloudnorm

    return pyloudnorm.Meter(sr, filter_class="DeMan").integrated_loudness(np.asarray(x, dtype=float))


def test_master_bit_identical_and_float32():
    x = _music()
    a = remix.master(x, 44100)
    b = remix.master(x.copy(), 44100)
    assert a.dtype == np.float32
    assert a.tobytes() == b.tobytes()


def test_master_loudness_and_true_peak():
    x = _music()
    y = remix.master(x, 44100, lufs=-8.0, ceiling_dbtp=-1.0)
    assert _loudness(y) == pytest.approx(-8.0, abs=0.2)
    tp = remix.true_peak_per_sample(y).max()
    assert 20 * np.log10(tp) <= -1.0 + 0.1


def test_master_output_aligned_with_input():
    """No look-ahead delay left in the output: the cross-correlation peak is at lag 0."""
    x = _music(8.0, seed=3) * 0.5
    y = remix.master(x, 44100, lufs=-14.0)
    c = signal.correlate(y[:, 0], x[:, 0], mode="full", method="fft")
    lag = int(np.argmax(c)) - (len(x) - 1)
    assert lag == 0


def test_forward_min_window_and_release():
    g = np.ones(100)
    g[50] = 0.5
    m = remix._forward_min(g, 5)  # window [i, i+4]
    assert np.all(m[46:51] == 0.5) and m[45] == 1.0 and m[51] == 1.0
    env = remix._release(m, 0.9)
    assert env[46] == 0.5 and env[50] == 0.5
    assert 0.5 < env[51] < env[52] < 1.0
    ref = np.empty_like(m)
    prev = 1.0
    for i, v in enumerate(m):  # direct recursion
        prev = min(v, 1 - (1 - prev) * 0.9)
        ref[i] = prev
    assert np.allclose(env, ref, atol=1e-12)


def test_tierb_mix_sums_parts_and_masters(tmp_path):
    import soundfile as sf

    from bandscribe.schema.dataset import DatasetTrack, Part

    sr = 44100
    n = sr * 6
    t = np.arange(n) / sr
    sf.write(str(tmp_path / "gtr.wav"), (0.1 * np.sin(2 * np.pi * 196 * t)).astype(np.float32), sr)
    sf.write(str(tmp_path / "bass.wav"), np.stack([0.2 * np.sin(2 * np.pi * 55 * t)] * 2, 1).astype(np.float32), sr)
    tr = DatasetTrack(format="bandscribe.dataset_track/1", dataset="cambridge_mt", track_id="song", split=None, group="song",
                      tier="B", license="x", mix=None,
                      parts=[Part(id="g", kind="guitar", audio="gtr.wav", channels=1),
                             Part(id="b", kind="bass", audio="bass.wav", channels=2)],
                      lines=[], families_present=["guitar", "bass"], notes={}, string_fret=None, beats_s=None,
                      downbeats_s=None, tuning=None, meta={})
    y = remix.tierb_mix(tr, root=tmp_path, master_lufs=-10.0)
    assert y.shape == (n, 2) and y.dtype == np.float32
    assert _loudness(y) == pytest.approx(-10.0, abs=0.2)
    assert np.array_equal(y, remix.tierb_mix(tr, root=tmp_path, master_lufs=-10.0))
