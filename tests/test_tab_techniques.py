"""Bends from the f0 (``bandscribe.tab.techniques``, M9 / E32)."""

from __future__ import annotations

import numpy as np

from bandscribe.tab import techniques as TQ

HOP = 0.01


def contour(segs, n=300):
    """[(start_s, end_s, midi_start, midi_end)] -> (hz, vprob); linear glides in MIDI."""
    hz, vp = np.zeros(n), np.zeros(n)
    for a, b, m0, m1 in segs:
        ia, ib = int(round(a / HOP)), int(round(b / HOP))
        m = np.linspace(m0, m1, max(1, ib - ia))
        hz[ia:ib] = 440.0 * 2 ** ((m - 69) / 12)
        vp[ia:ib] = 0.9
    return hz, vp


def note(on, off, p):
    return {"onset_s": on, "offset_s": off, "pitch": p}


def test_a_full_bend_with_release_and_one_written_at_its_target():
    hz, vp = contour([(0.5, 0.6, 64, 64), (0.6, 0.7, 64, 66), (0.7, 0.9, 66, 66), (0.9, 1.0, 66, 64.2),
                      (1.0, 1.2, 64.2, 64.2)])
    r = TQ.bends([note(0.5, 1.2, 64)], hz, vp, HOP)
    assert len(r) == 1 and r[0]["class"] == "full" and r[0]["release"] and 180 <= r[0]["rise_cents"] <= 210
    assert TQ.bends([note(0.5, 0.9, 66)], hz, vp, HOP)[0]["class"] == "full"  # the transcriber wrote the target


def test_half_bend_flat_note_chords_and_tracker_jumps():
    hz, vp = contour([(0.5, 0.6, 60, 60), (0.6, 0.7, 60, 61), (0.7, 1.0, 61, 61)])
    assert TQ.bends([note(0.5, 1.0, 60)], hz, vp, HOP)[0]["class"] == "half"
    flat, fvp = contour([(0.5, 1.0, 60, 60)])
    assert TQ.bends([note(0.5, 1.0, 60)], flat, fvp, HOP) == []
    assert TQ.bends([note(0.5, 1.0, 60), note(0.5, 1.0, 64)], hz, vp, HOP) == []  # a chord: f0 means nothing
    jump, jvp = contour([(0.5, 0.7, 60, 60), (0.7, 1.0, 72, 72)])
    assert TQ.bends([note(0.5, 1.0, 60)], jump, jvp, HOP) == []  # an octave jump of the tracker
