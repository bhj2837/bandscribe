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


def test_e30b_drops_a_reference_ghost_an_octave_below_but_keeps_a_real_low_note():
    sr = 44100
    t = np.arange(int(2.0 * sr)) / sr
    x = np.zeros_like(t)
    a = t < 1.0  # 0-1 s: an A2 alone (its transcription also wrote A1 under it: a ghost)
    x[a] = 0.3 * np.sin(2 * np.pi * 110.0 * t[a])
    b = t >= 1.0  # 1-2 s: a real D2 with its octave
    x[b] = 0.3 * np.sin(2 * np.pi * 73.42 * t[b]) + 0.3 * np.sin(2 * np.pi * 146.83 * t[b])
    notes = [{"onset_s": 0.0, "offset_s": 0.9, "pitch": 33}, {"onset_s": 0.0, "offset_s": 0.9, "pitch": 45},
             {"onset_s": 1.0, "offset_s": 1.9, "pitch": 38}, {"onset_s": 1.01, "offset_s": 1.9, "pitch": 50}]
    assert E30.ghost_notes(notes, x.astype(np.float32), sr) == {0}


def test_the_notes_stage_moves_bass_bleed_out_and_the_viewer_names_it(tmp_path):
    from bandscribe.amt import stage as S
    from bandscribe.bass import f0 as F0
    from bandscribe.tab import viewer_media as VM

    f, c = track([(0.0, 3.0, 26)])  # the bass holds D1
    F0.save_npz(tmp_path / "f0.npz", F0.F0Track(hz={"pyin": f}, conf={"pyin": c}, hop_s=HOP))
    notes = [note(1.0, 1.5, 38), note(1.0, 1.5, 50)]
    kept, removed = S.bass_bleed(notes, tmp_path)
    assert [n["pitch"] for n in kept] == [50] and removed[0]["pitch"] == 38 and removed[0]["why"] == "bass"
    assert S.bass_bleed(notes, None) == (notes, [])
    nd, gd = tmp_path / "notes", tmp_path / "amt_gtr"
    (nd).mkdir()
    (gd / "raw").mkdir(parents=True)
    import json

    (nd / "bleed.json").write_text(json.dumps({"notes": removed}), encoding="utf-8")
    raw = {"notes": [{**n, "instrument": "distorted_electric_guitar"} for n in notes]}
    (gd / "raw" / "guitar_mono__muscriptor.json").write_text(json.dumps(raw), encoding="utf-8")
    cands = VM.candidates({"notes": nd, "amt_gtr": gd}, {"guitar_all": kept})
    assert [x["pitch"] for x in cands] == [38] and "베이스 블리드" in cands[0]["why"]
