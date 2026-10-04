"""Cause attribution and leakage (bandscribe.eval.attribution) on synthetic signals with known causes."""
from __future__ import annotations

import numpy as np
import pytest

from bandscribe.eval import attribution as A

SR = 44100
GUITAR = ("guitar_electric", "guitar_acoustic")


def hz(p: float) -> float:
    return 440.0 * 2 ** ((p - 69) / 12)


def tone(p: float, start: float, dur: float, total: float, amp: float = 0.3, harmonics: int = 4) -> np.ndarray:
    n = int(total * SR)
    x = np.zeros(n)
    a, z = int(start * SR), min(n, int((start + dur) * SR))
    t = np.arange(z - a) / SR
    for k in range(1, harmonics + 1):
        x[a:z] += amp / k * np.sin(2 * np.pi * k * hz(p) * t)
    return x


def bands_of(sigs: dict[str, np.ndarray]) -> tuple[dict[str, A.BandEnergy], float]:
    bands = {c: A.band_energy(x) for c, x in sigs.items()}
    mix = A.band_energy(sum(sigs.values()))
    MIX[0] = mix
    return bands, A.floor_power(bands, mix.frame_power_p95)


MIX: list = [None]


def test_pitch_band_matrix_covers_every_pitch():
    m1, m2 = A.pitch_band_matrix(1), A.pitch_band_matrix(2)
    assert m1.shape == (A.N_FFT // 2 + 1, A.N_PITCH)
    assert (m1.sum(axis=0) >= 1).all()      # low bands get the nearest bin
    assert (m2.sum(axis=0) >= 1).all()
    k = 69 - A.MIDI_LO  # A4 = 440 Hz: its band holds the bin nearest 440 Hz
    assert m1[int(round(440 * A.N_FFT / SR)), k] == 1


def test_band_energy_peaks_at_the_played_pitch():
    be = A.band_energy(tone(64, 0.5, 1.0, 2.0))
    e = [be.note(0.6, 1.2, p) for p in range(55, 75)]
    assert int(np.argmax(e)) + 55 == 64
    assert be.note(0.6, 1.2, 64) > 100 * be.note(0.6, 1.2, 61)
    assert be.note(0.6, 1.2, 10) == 0.0  # outside the pitch range


def test_fp_caused_by_a_synth_pad_note():
    total = 3.0
    sigs = {"guitar_electric": tone(52, 0.2, 2.5, total), "synth_pad": tone(75, 1.0, 1.0, total, amp=0.5),
            "drums": np.random.default_rng(0).normal(0, 0.01, int(total * SR))}
    bands, floor = bands_of(sigs)
    ref = np.array([[0.2, 2.7, 52]])
    c = A.attribute_fp((1.05, 1.6, 75), ref, bands, guitar=GUITAR, floor=floor, mix=MIX[0])
    assert c.label == "synth_pad" and c.dominant == "synth_pad" and c.dominant_share > 0.9


def test_fp_rules_in_order_timing_octave_near_unref_unknown():
    total = 3.0
    sigs = {"guitar_electric": tone(52, 0.2, 2.5, total) + tone(57, 0.2, 2.5, total, amp=0.2),
            "synth_pad": tone(76, 1.0, 1.0, total)}
    bands, floor = bands_of(sigs)
    ref = np.array([[0.2, 2.7, 52]])
    assert A.attribute_fp((0.35, 0.8, 52), ref, bands, guitar=GUITAR, floor=floor, mix=MIX[0]).label == "guitar_timing"
    assert A.attribute_fp((1.0, 1.5, 64), ref, bands, guitar=GUITAR, floor=floor, mix=MIX[0]).label == "guitar_octave"
    assert A.attribute_fp((1.0, 1.5, 71), ref, bands, guitar=GUITAR, floor=floor, mix=MIX[0]).label == "guitar_octave"  # +19
    assert A.attribute_fp((1.0, 1.5, 53), ref, bands, guitar=GUITAR, floor=floor, mix=MIX[0]).label == "guitar_near"
    # a real guitar tone the reference does not have (57) -> guitar_unref
    assert A.attribute_fp((1.0, 1.5, 57), ref, bands, guitar=GUITAR, floor=floor, mix=MIX[0]).label == "guitar_unref"
    # nothing sounds at 90 -> unknown
    assert A.attribute_fp((1.0, 1.5, 90), ref, bands, guitar=GUITAR, floor=floor, mix=MIX[0]).label == A.UNKNOWN
    # the octave rule needs time overlap: after the reference note ends it is the pad
    late = A.attribute_fp((1.1, 1.5, 76), np.array([[0.0, 0.5, 64]]), bands, guitar=GUITAR, floor=floor, mix=MIX[0])
    assert late.label == "synth_pad"


def test_a_louder_distractor_at_the_octave_wins_over_the_octave_rule():
    total = 3.0
    sigs = {"guitar_electric": tone(52, 0.2, 2.5, total, amp=0.1), "organ": tone(64, 0.8, 1.5, total, amp=0.6)}
    bands, floor = bands_of(sigs)
    ref = np.array([[0.2, 2.7, 52]])
    c = A.attribute_fp((1.0, 1.5, 64), ref, bands, guitar=GUITAR, floor=floor, mix=MIX[0])
    assert c.label == "organ" and c.dominant == "organ"
    # without the organ the same false note is the guitar's own octave error
    bands2, floor2 = bands_of({"guitar_electric": tone(52, 0.2, 2.5, total, amp=0.1)})
    c2 = A.attribute_fp((1.0, 1.5, 64), ref, bands2, guitar=GUITAR, floor=floor2, mix=MIX[0])
    assert c2.label == "guitar_octave"


def test_fn_masked_by_a_loud_pad_vs_clear_guitar():
    total = 3.0
    g = tone(60, 0.5, 1.0, total, amp=0.05)
    sigs = {"guitar_electric": g, "synth_pad": tone(60, 0.0, 3.0, total, amp=0.6),
            "vocals": tone(67, 0.0, 3.0, total, amp=0.05)}
    bands, floor = bands_of(sigs)
    c = A.attribute_fn((0.5, 1.5, 60), np.zeros((0, 3)), bands, guitar=GUITAR, floor=floor, mix=MIX[0])
    assert c.label == "synth_pad"
    # an estimated note one octave up in the same time -> est_octave, whatever masks it
    c3 = A.attribute_fn((0.5, 1.5, 60), np.array([[0.52, 1.4, 72]]), bands, guitar=GUITAR, floor=floor, mix=MIX[0])
    assert c3.label == "est_octave"
    loud_g = {"guitar_electric": tone(60, 0.5, 1.0, total, amp=0.6), "vocals": tone(67, 0.0, 3.0, total, amp=0.05)}
    bands2, floor2 = bands_of(loud_g)
    c2 = A.attribute_fn((0.5, 1.5, 60), np.zeros((0, 3)), bands2, guitar=GUITAR, floor=floor2, mix=MIX[0])
    assert c2.label == A.GUITAR_CLEAR


def test_leakage_reads_the_mixing_gain_of_a_known_leak():
    total = 4.0
    g = tone(52, 0.0, 4.0, total, amp=0.3)
    pad = tone(83, 0.0, 4.0, total, amp=0.3)   # B5: far from the guitar's harmonics
    sep = g + 0.1 * pad                          # the separator passes the pad at -20 dB
    out = A.leakage({"guitar_electric": g, "synth_pad": pad}, sep, group={"guitar_electric": "guitar"})
    assert set(out) == {"guitar", "synth_pad"}
    assert A.ratio_db(out["synth_pad"]["leak"]) == pytest.approx(-20.0, abs=1.0)
    assert A.ratio_db(out["guitar"]["leak"]) == pytest.approx(0.0, abs=0.5)
    assert out["guitar"]["share"] > 0.95 and out["synth_pad"]["share"] < 0.05
    assert out["synth_pad"]["bins"] > 0
