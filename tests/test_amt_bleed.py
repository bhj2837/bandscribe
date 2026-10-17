"""Bleed flags (``bandscribe.amt.bleed``, E30): a guitar note the voice arrives at is flagged, a chord tone the
singer was already holding is not; a note under E2 on the bass's pitch class is flagged."""

from __future__ import annotations

import numpy as np

from bandscribe.amt import bleed as B
from bandscribe.parts import eval_e30 as E30

HOP = 0.01


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def track(segments, n=400):
    """[(start_s, end_s, midi)] -> (hz, voiced prob)."""
    f, v = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    for a, b, m in segments:
        f[int(a / HOP):int(b / HOP)] = hz(m)
        v[int(a / HOP):int(b / HOP)] = 0.9
    return f, v


def note(on, off, p):
    return {"onset_s": on, "offset_s": off, "pitch": p}


def test_the_voice_arriving_with_the_note_flags_it():
    f, v = track([(0.5, 1.2, 60), (1.2, 2.0, 64)])
    notes = [note(1.2, 1.8, 64), note(0.5, 1.0, 60), note(0.9, 1.1, 60), note(1.2, 1.8, 55)]
    # 0: the voice moves to 64 with the note; 1: arrives with the note too (silence before);
    # 2: the voice was already on 60 for 400 ms (a chord tone equal to the sung note); 3: another pitch
    assert B.vocal_flags(notes, f, v, HOP) == [0, 1]


def test_a_note_longer_than_the_sung_note_is_kept():
    f, v = track([(1.0, 1.25, 62)])
    assert B.vocal_flags([note(1.0, 2.0, 62)], f, v, HOP) == []  # the voice covers 25 % of it
    assert B.vocal_flags([note(1.0, 1.2, 62)], f, v, HOP) == [0]


def test_bass_flags_only_low_notes_on_the_bass_pitch_class():
    f, c = track([(0.0, 3.0, 26)])  # D1 on the bass
    notes = [note(1.0, 1.5, 38), note(1.0, 1.5, 39), note(1.0, 1.5, 50), note(3.5, 4.0, 38)]
    assert B.bass_flags(notes, f, c, HOP) == [0]  # D2 = D; D#2 another class; D3 is not under E2; 3.5 s unvoiced


def test_pooled_precision_and_the_hold_rule():
    prec, lo, hi = E30.pooled_precision({"a": (10, 9), "b": (10, 7)})
    assert prec == 0.8 and lo <= 0.8 <= hi
    rows = [{"song": "a", "vocal": {"flagged": 5, "false": 5, "f1_diff": 0.1}}]
    assert E30.decide(rows, "vocal")["verdict"] == "HOLD"  # under 20 flagged notes
    rows = [{"song": s, "vocal": {"flagged": 10, "false": 9, "f1_diff": 0.1}} for s in "abc"]
    assert E30.decide(rows, "vocal")["verdict"] == "PASS"
    rows[0]["vocal"]["false"] = 1  # 19 / 30
    assert E30.decide(rows, "vocal")["verdict"] == "FAIL"
