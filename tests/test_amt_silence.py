"""Silence gate of the notes stage (bandscribe.amt.silence, 2026-10-05)."""

from __future__ import annotations

import numpy as np
import pytest

from bandscribe.amt import silence as SG

FPS = 10.0


def _energy(n: int = 100, **silent: tuple[int, int]) -> dict[str, np.ndarray]:
    """Mix at -12 dB, stems at -20 dB; ``silent={"bass": (a, z)}`` puts frames a..z-1 of a stem at the floor."""
    e = {"mix": np.full(n, -12.0, np.float32)}
    for s in ("bass", "guitar", "other", "piano", "nonvox"):
        e[s] = np.full(n, -20.0, np.float32)
    for s, (a, z) in silent.items():
        e[s][a:z] = SG.FLOOR_DB
    return e


def _note(t: float, pitch: int = 31) -> dict:
    return {"onset_s": t, "offset_s": t + 0.2, "pitch": pitch, "instrument": "electric_bass"}


def test_reference_is_the_mix_p95_and_views_sum_their_stems():
    e = _energy()
    e["mix"][:10] = -60.0  # a quiet intro does not move the loud level
    assert SG.reference_db(e) == pytest.approx(-12.0)
    assert SG.view_level_db(e, "guitar_other_mono")[0] == pytest.approx(-20.0 + 10 * np.log10(2), abs=1e-6)
    assert SG.view_level_db(e, "bass_mono")[0] == pytest.approx(-20.0)
    assert SG.view_level_db(e, "who_knows") is None
    assert SG.reference_db({}) is None


def test_notes_over_a_silent_stem_are_dropped_and_the_edges_kept():
    """Frames 20-29 (2.0-3.0 s) silent; a note's level is its loudest frame over onset -30 ms .. +120 ms."""
    e = _energy(bass=(20, 30))
    onsets = [1.85, 2.0, 2.10, 2.5, 2.85, 2.95, 3.0, 3.10]
    kept, rep = SG.gate([_note(t) for t in onsets], e, "bass_mono", FPS)
    # 1.85: frames 18-19 (sound); 2.0: frames 19-21, frame 19 still sounds; 2.95: frames 29-30, 30 sounds again
    assert [n["onset_s"] for n in kept] == [1.85, 2.0, 2.95, 3.0, 3.10]
    assert rep["gated"] and rep["dropped"] == 3 and rep["floor_db"] == pytest.approx(-62.0)
    assert rep["spans_s"] == [[2.1, 2.85]]
    other_stem_silent = _energy(guitar=(0, 100))  # the guitar's silence does not touch the bass
    assert len(SG.gate([_note(t) for t in onsets], other_stem_silent, "bass_mono", FPS)[0]) == len(onsets)


def test_a_quiet_but_real_part_stays():
    """A soft bass 45 dB under the mix's loud level is kept with the default -50 dB gate."""
    e = _energy()
    e["bass"][:] = -12.0 - 45.0
    kept, rep = SG.gate([_note(1.0), _note(5.0)], e, "bass_mono", FPS)
    assert len(kept) == 2 and rep["dropped"] == 0


def test_past_the_end_off_and_unknown_views():
    e = _energy(n=50)
    kept, rep = SG.gate([_note(4.0), _note(6.0)], e, "bass_mono", FPS)  # 6.0 s is past the 5 s of energy
    assert [n["onset_s"] for n in kept] == [4.0] and rep["dropped"] == 1
    kept, rep = SG.gate([_note(6.0)], e, "bass_mono", FPS, gate_db=-200.0)
    assert len(kept) == 1 and rep == {"gated": False, "reason": "off", "dropped": 0}
    kept, rep = SG.gate([_note(6.0)], e, "piano_left", FPS)
    assert len(kept) == 1 and not rep["gated"] and "piano_left" in rep["reason"]
    kept, rep = SG.gate([_note(6.0)], {}, "bass_mono", FPS)
    assert len(kept) == 1 and rep["reason"] == "no mix level"
