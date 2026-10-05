"""M3 A4 reference: "[판정] A4 픽스처(+25, −30, +45 cents)를 ±5 cents 안에서 복원한다" (DESIGN 10 M3)."""

from __future__ import annotations

import numpy as np
import pytest

from bandscribe.tab import a4

pytest.importorskip("librosa")
SR = 22050


def tones(cents: float, *, seed: int = 0, seconds: float = 20.0, low: int = 40, high: int = 77,
          vibrato_share: float = 0.0, detuned_share: float = 0.0) -> np.ndarray:
    """Plucked harmonic tones (8 partials, 0.6 decay per partial, exponential envelope) on a scale whose A4 is
    440 * 2^(cents/1200); some notes with 6 Hz ±30 cent vibrato, some detuned by ±20-40 cents (outliers)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = np.zeros(n)
    ref = 440.0 * 2 ** (cents / 1200.0)
    pos = 0
    while pos < n:
        dur = min(int(rng.uniform(0.2, 0.8) * SR), n - pos)
        midi = int(rng.integers(low, high))
        off = rng.choice([-1, 1]) * rng.uniform(20, 40) if rng.random() < detuned_share else 0.0
        f0 = ref * 2 ** ((midi - 69 + off / 100.0) / 12.0)
        t = np.arange(dur) / SR
        if rng.random() < vibrato_share:
            inst = f0 * 2 ** (0.30 / 12.0 * np.sin(2 * np.pi * 6.0 * t))
            phase = 2 * np.pi * np.cumsum(inst) / SR
        else:
            phase = 2 * np.pi * f0 * t
        note = sum(0.6 ** (h - 1) * np.sin(h * phase) for h in range(1, 9) if f0 * h < SR / 2)
        y[pos:pos + dur] += np.exp(-3.0 * t) * note
        pos += dur
    y += 0.003 * rng.normal(size=n)
    return (y / np.max(np.abs(y))).astype(np.float32)


@pytest.mark.parametrize("cents", [25.0, -30.0, 45.0, 0.0])
def test_a4_fixture_recovered_within_five_cents(cents):
    doc = a4.estimate({"guitar_mono": (tones(cents, seed=1), SR), "bass_mono": (tones(cents, seed=2, low=28, high=52), SR)})
    assert doc["cents"] == pytest.approx(cents, abs=5.0)
    assert doc["confidence"] > 0.5
    assert doc["a4_hz"] == pytest.approx(440.0 * 2 ** (doc["cents"] / 1200.0), abs=1e-3)


@pytest.mark.parametrize("cents", [25.0, -30.0, 45.0])
def test_a4_robust_to_vibrato_and_out_of_tune_notes(cents):
    y = tones(cents, seed=3, vibrato_share=0.3, detuned_share=0.15)
    doc = a4.estimate({"guitar_mono": (y, SR)})
    assert doc["cents"] == pytest.approx(cents, abs=5.0)


def test_a4_wraps_around_the_semitone_boundary():
    """+48 cents and -48 cents are 4 cents apart on the circle: the estimate must not jump to 0."""
    for c in (48.0, -48.0):
        got = a4.estimate({"g": (tones(c, seed=4), SR)})["cents"]
        assert abs(((got - c) + 50) % 100 - 50) <= 5.0, (c, got)


def test_a4_warning_and_resampling():
    doc = a4.estimate({"g": (tones(45.0, seed=5, seconds=8.0), SR)})
    assert doc["warning"] and "E16" in doc["warning"]
    assert a4.estimate({"g": (tones(3.0, seed=5, seconds=8.0), SR)})["warning"] is None
    y44 = np.repeat(tones(-30.0, seed=6, seconds=8.0), 2)  # crude 44.1 kHz copy: resampling path
    assert a4.estimate({"g": (y44, 44100)})["cents"] == pytest.approx(-30.0, abs=5.0)


def test_a4_silence_gives_no_estimate():
    doc = a4.estimate({"g": (np.zeros(SR * 3, dtype=np.float32), SR)})
    assert doc["cents"] is None and doc["a4_hz"] is None and doc["warning"] is None
