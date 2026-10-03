"""Full-mix percussive onsets (M1_M2_SPEC 9.3 item 2)."""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest
import soundfile as sf

from gtab.analysis import onsets


@pytest.fixture(scope="module")
def click_track(tmp_path_factory):
    sr = 44100
    dur = 20.0
    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    x = np.zeros(n)
    for f in (110.0, 138.59, 164.81, 220.0):  # sustained chord: harmonic, must not produce onsets
        x += 0.1 * np.sin(2 * np.pi * f * t)
    grid = np.arange(0.5, dur - 0.5, 0.5)
    clicks = grid + rng.uniform(-0.01, 0.01, grid.size)
    for c in clicks:
        i = int(round(c * sr))
        L = int(0.01 * sr)
        x[i:i + L] += rng.normal(0, 0.5, L) * np.exp(-np.arange(L) / (0.002 * sr))
    path = tmp_path_factory.mktemp("onsets") / "clicks.wav"
    sf.write(str(path), np.stack([x, x], 1).astype(np.float32), sr, subtype="FLOAT")
    return path, clicks


def test_clicks_found_within_10ms(click_track):
    path, clicks = click_track
    res = onsets.compute(path)
    times = res["times"]
    assert res["fps"] == pytest.approx(22050 / 256)
    err = np.array([np.min(np.abs(times - c)) for c in clicks])
    assert np.all(err < 0.010), err.max()
    # no extra onsets from the sustained chord (at most one spurious detection)
    assert len(times) <= len(clicks) + 1
    assert res["env"].dtype == np.float32 and res["env"].size == 1 + int(20.0 * 22050) // 256


def test_window_boundaries_do_not_matter():
    rng = np.random.default_rng(5)
    y = rng.normal(0, 0.05, int(70 * 22050)).astype(np.float32)
    env = onsets.percussive_envelope(y)
    # a single-window computation over the region around a window boundary (30 s) gives the same values
    a, b = int(25 * onsets.FPS), int(35 * onsets.FPS)
    seg = y[a * onsets.HOP: b * onsets.HOP]
    import librosa

    mel = librosa.filters.mel(sr=onsets.SR, n_fft=onsets.N_FFT, n_mels=onsets.N_MELS).astype(np.float32)
    local = onsets._envelope_window(seg, mel)
    ctx = int(round(onsets.CONTEXT_S * onsets.FPS))
    mid = slice(ctx, (b - a) - ctx)
    np.testing.assert_allclose(env[a:b][mid], local[mid], rtol=1e-4, atol=1e-4)


def test_write_npz_is_byte_deterministic(tmp_path):
    arrays = onsets.to_arrays({"env": np.arange(10, dtype=np.float32), "fps": 86.1, "times": np.array([0.1, 0.25]),
                               "strength": np.array([1.0, 0.5], dtype=np.float32)})
    onsets.write_npz(tmp_path / "a.npz", arrays)
    time.sleep(2.1)  # zip timestamps have 2 s resolution; np.savez would differ now
    onsets.write_npz(tmp_path / "b.npz", arrays)
    h = [hashlib.sha256((tmp_path / f).read_bytes()).hexdigest() for f in ("a.npz", "b.npz")]
    assert h[0] == h[1]
    back = onsets.load(tmp_path / "a.npz")
    assert back["fps"] == 86.1 and list(back["times"]) == [0.1, 0.25]
