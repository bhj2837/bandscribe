"""Repeat consensus (``bandscribe.amt.consensus``, E29): a note most repeats hold is added where one repeat lacks
it; a note only one repeat plays is kept and marked; bars that are not the same music do not vote."""

from __future__ import annotations

import pytest

from bandscribe.amt import consensus as C

BEAT = 0.5  # 120 BPM, 4/4


def grid(n_bars=12, pickup=None):
    beats, bars, t = [], [], 0.0
    for b in range(n_bars):
        nb = pickup if (pickup is not None and b == 0) else 4
        bars.append({"bar": b, "start_s": t, "end_s": t + nb * BEAT, "n_beats": nb})
        for k in range(nb):
            beats.append({"t_s": t + k * BEAT})
        t += nb * BEAT
    beats.append({"t_s": t})
    return {"bars": bars, "beats": beats}


RIFF = [(0.0, 40), (1.0, 43), (2.0, 45), (3.0, 47)]  # (beat in bar, pitch)


def occurrence(bar, riff=RIFF, shift=0.0, start=0.0):
    return [{"onset_s": start + bar * 4 * BEAT + b * BEAT + shift, "offset_s": start + bar * 4 * BEAT + b * BEAT
             + shift + 0.4, "pitch": p, "instrument": "distorted_electric_guitar"} for b, p in riff]


def rmap(starts, n_bars=1, offsets=None):
    offsets = offsets or [0.0] * len(starts)
    return {"groups": [{"label": "A", "occurrences": [{"section": f"S{i}", "start_bar": s, "n_bars": n_bars,
                                                        "offset_s": o} for i, (s, o) in enumerate(zip(starts, offsets))]}]}


def test_a_note_one_repeat_misses_is_added_where_the_others_play_it():
    notes = occurrence(1) + occurrence(3)[:2] + occurrence(3)[3:] + occurrence(5)  # bar 3 lacks beat 2 (pitch 45)
    r = C.consensus(notes, grid(), rmap([1, 3, 5]))
    assert len(r["added"]) == 1
    a = r["added"][0]
    assert a["pitch"] == 45 and a["onset_s"] == pytest.approx(3 * 2.0 + 2 * BEAT) and a["added"] == C.ADDED
    assert a["instrument"] == "distorted_electric_guitar" and a["support"] == 1.0
    assert a["offset_s"] == pytest.approx(a["onset_s"] + 0.4, abs=1e-3)
    assert r["lone"] == [] and r["report"]["voted_cells"] == 3


def test_a_variation_is_kept_and_marked_lone_not_removed():
    extra = {"onset_s": 5 * 2.0 + 3.5 * BEAT, "offset_s": 5 * 2.0 + 3.9 * BEAT, "pitch": 50}
    notes = occurrence(1) + occurrence(3) + occurrence(5) + [extra]
    r = C.consensus(notes, grid(), rmap([1, 3, 5]))
    assert r["added"] == [] and r["lone"] == [len(notes) - 1]


def test_bars_that_are_not_the_same_music_do_not_vote():
    other = [(0.0, 52), (1.0, 55), (2.0, 57), (3.0, 59)]
    notes = occurrence(1) + occurrence(3)[:3] + occurrence(5, riff=other)
    r = C.consensus(notes, grid(), rmap([1, 3, 5]))
    assert r["added"] == []  # bar 5 shares none of bar 3's notes: one voter is not enough


def test_two_occurrences_or_different_beat_counts_never_vote():
    notes = occurrence(1) + occurrence(3)[:3]
    assert C.consensus(notes, grid(), rmap([1, 3]))["added"] == []
    g = grid(pickup=2)  # bar 0 has 2 beats: it cannot be laid against the full bars

    def at(bar, riff=RIFF):
        t0 = g["bars"][bar]["start_s"]
        return [{"onset_s": t0 + b * BEAT, "offset_s": t0 + b * BEAT + 0.4, "pitch": p} for b, p in riff]

    r = C.consensus(at(0, RIFF[:2] + [(1.5, 45)]) + at(2) + at(4), g, rmap([0, 2, 4]))
    assert r["report"]["voted_cells"] == 0 and r["added"] == []
    assert C.consensus(at(1) + at(2)[:3] + at(4), g, rmap([1, 2, 4]))["added"]  # the full bars do vote


def test_the_occurrence_offset_moves_the_added_note():
    notes = occurrence(1) + occurrence(3, shift=0.03)[:2] + occurrence(3, shift=0.03)[3:] + occurrence(5)
    r = C.consensus(notes, grid(), rmap([1, 3, 5], offsets=[0.0, 0.03, 0.0]))
    assert len(r["added"]) == 1 and r["added"][0]["onset_s"] == pytest.approx(6.0 + 2 * BEAT + 0.03)


def test_add_frac_needs_more_holders_and_small_bars_do_not_vote():
    four = occurrence(1) + occurrence(3)[:2] + occurrence(3)[3:] + occurrence(5) + occurrence(7)[:2] + occurrence(7)[3:]
    assert C.consensus(four, grid(), rmap([1, 3, 5, 7]), {"add_frac": 1.0})["added"] == []  # 2 of 3 voters
    assert len(C.consensus(four, grid(), rmap([1, 3, 5, 7]), {"add_frac": 2 / 3})["added"]) == 2
    tiny = occurrence(1) + occurrence(3)[:2] + occurrence(5)
    assert C.consensus(tiny, grid(), rmap([1, 3, 5]))["added"] == []  # bar 3 has 2 notes (< min_notes)


def test_e29_reference_union_and_decision_rules():
    from bandscribe.parts import eval_e29 as E29

    a = [{"onset_s": 1.0, "offset_s": 1.2, "pitch": 40}, {"onset_s": 2.0, "offset_s": 2.2, "pitch": 43}]
    b = [{"onset_s": 1.03, "offset_s": 1.2, "pitch": 40}, {"onset_s": 3.0, "offset_s": 3.2, "pitch": 45}]
    ref = E29.union_ref([a, b])
    assert [(n["pitch"], n["onset_s"]) for n in ref] == [(40, 1.0), (43, 2.0), (45, 3.0)]  # the double is one note
    assert E29.note_f1(ref, a).f1 == pytest.approx(0.8)
    rows = [{"song": s, "scenario": "P", "off": {"f1": 50.0}, "consensus": {"f1": 50.0 + d, "added": 3,
             "added_precision": 0.5, "voted_cells": 1}} for s, d in (("x", 3.0), ("y", 2.5), ("z", 4.0))]
    assert E29.decide(rows)["verdict"] == "PASS"
    rows[0]["consensus"]["f1"] = 40.0  # one song loses 10 points: the mean falls under +2
    assert E29.decide(rows)["verdict"] == "FAIL"


def test_a_note_already_there_within_50_ms_is_not_added_again():
    """Review 2026-10-18: the beat tolerance (here 1/16 beat = 31 ms) must not let a near note count as missing."""
    late = [dict(n, onset_s=n["onset_s"] + 0.04, offset_s=n["offset_s"] + 0.04) if n["pitch"] == 45 else n
            for n in occurrence(3)]
    notes = occurrence(1) + late + occurrence(5)
    assert C.consensus(notes, grid(), rmap([1, 3, 5]), {"tol_beats": 0.0625})["added"] == []
