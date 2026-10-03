"""Vocal activity from the vocals stem (M1_M2_SPEC 9.3 item 5, §6.6)."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from gtab.analysis import vocal
from grid_testlib import make_grid


def _write(path, x, sr=44100):
    sf.write(str(path), np.stack([x, x], 1).astype(np.float32), sr, subtype="FLOAT")
    return path


def test_activity_and_per_bar(tmp_path):
    sr = 44100
    grid = make_grid(8, bpm=120.0, lead_s=1.0)  # bars of 2 s from 1.0 s to 17.0 s
    n = int(18 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    band = 0.1 * np.sin(2 * np.pi * 110 * t) + 0.02 * rng.normal(size=n)
    band[: int(1.0 * sr)] = 0.0  # silent intro
    voc = np.zeros(n)
    on = (t >= 5.0) & (t < 9.0)  # vocals over bars 3-4
    voc[on] = 0.08 * np.sin(2 * np.pi * 330 * t[on])
    voc[: int(1.0 * sr)] = 1e-6 * rng.normal(size=int(1.0 * sr))  # separation noise in the silent intro
    mix = band + voc
    doc = vocal.vocal_activity(_write(tmp_path / "v.wav", voc), _write(tmp_path / "m.wav", mix),
                               grid["bars"], -30.0)
    assert doc["format"] == "gtab.vocal/1" and doc["fps"] == 10 and doc["threshold_rel_db"] == -30.0
    assert len(doc["rel_db"]) == len(doc["active"]) == 180
    ratios = {b["bar"]: b["active_ratio"] for b in doc["per_bar"]}
    assert ratios[3] == 1.0 and ratios[4] == 1.0
    assert all(ratios[b] == 0.0 for b in (1, 6, 7, 8))
    assert not any(doc["active"][:10])  # silence is not "vocals"
    assert all(isinstance(v, float) and round(v, 1) == v for v in doc["rel_db"])


def test_deterministic(tmp_path):
    sr = 44100
    rng = np.random.default_rng(1)
    x = rng.normal(size=5 * sr) * 0.1
    grid = make_grid(2, lead_s=0.0)
    a = vocal.vocal_activity(_write(tmp_path / "v.wav", x * 0.3), _write(tmp_path / "m.wav", x), grid["bars"], -30.0)
    b = vocal.vocal_activity(tmp_path / "v.wav", tmp_path / "m.wav", grid["bars"], -30.0)
    assert a == b
    assert all(r == -10.5 or abs(r + 10.5) < 0.2 for r in a["rel_db"])
