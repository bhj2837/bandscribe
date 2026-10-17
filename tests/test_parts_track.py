"""M7b identity tracking (bandscribe.parts.track), repeat links, the ``part`` edit and its spread to repeats."""

from __future__ import annotations

import numpy as np
import pytest

from bandscribe.parts import idmetrics as IM
from bandscribe.parts import track as TR
from bandscribe.tab import edits as E
from bandscribe.tab import quantize as Q

from test_parts_tracks import make_parts_job


def _scene(swap_window: int = 2, p_right: float = 0.55, n_win: int = 5):
    """Two streams over ``n_win`` 4 s windows: a low chord part (Line 1) and a high single-note part (Line 2).
    v0's labels in ``swap_window`` are swapped, with ``p_right`` the posterior of the label it chose."""
    notes, feats, sigs, line0, post, truth = [], [], [], [], [], []
    for w in range(n_win):
        for k in range(8):
            t = w * 4.0 + k * 0.5
            for low, (pitch, chord) in enumerate(((64, False), (45, True))):
                true = 2 if low == 0 else 1
                notes.append({"onset_s": t + 0.01 * low, "offset_s": t + 0.4, "pitch": pitch + k % 3,
                              "instrument": "distorted_electric_guitar"})
                feats.append({"chord": chord, "lead": not chord, "r": -1 if chord else 1})
                sigs.append({"ild_db": None})
                lab = (3 - true) if w == swap_window else true
                line0.append(lab)
                row = np.full(2, 1.0 - p_right)
                row[lab - 1] = p_right
                post.append(row)
                truth.append({str(true)})
    wins = [{"start_s": w * 4.0, "end_s": w * 4.0 + 4.0, "label": "R0", "section": "S1"} for w in range(n_win)]
    return notes, feats, sigs, wins, np.array(post), line0, truth


def test_a_window_relabelled_by_v0_is_put_back_when_v0_is_unsure():
    notes, feats, sigs, wins, post, line0, truth = _scene(p_right=0.55)
    on = [n["onset_s"] for n in notes]
    before = IM.identity_switches(truth, [str(x) for x in line0], on, {"1": "1", "2": "2"})
    out = TR.track(notes, feats, sigs, wins, post, line0)
    after = IM.identity_switches(truth, [str(x) for x in out["line"]], on, {"1": "1", "2": "2"})
    assert before["total"] == 4 and after["total"] == 0
    assert out["changed"] == 16 and out["perms"][2] == [2, 1] and out["perms"][1] == [1, 2]


def test_a_sure_v0_window_is_kept_and_one_line_is_left_alone():
    notes, feats, sigs, wins, post, line0, _truth = _scene(p_right=0.999)
    out = TR.track(notes, feats, sigs, wins, post, line0)
    assert out["changed"] == 0  # the emission outweighs the links
    one = TR.track(notes, feats, sigs, wins, post[:, :1], [1] * len(notes))
    assert one["changed"] == 0 and one["line"] == [1] * len(notes)


def test_section_boundaries_make_a_role_change_cheaper():
    notes, feats, sigs, wins, post, line0, _truth = _scene(swap_window=4, p_right=0.62, n_win=5)
    assert TR.track(notes, feats, sigs, wins, post, line0)["changed"] == 16  # inside a section: put back
    for w in wins[4:]:
        w["section"] = "S2"
    out = TR.track(notes, feats, sigs, wins, post, line0, params={"boundary": 0.05})
    assert out["changed"] == 0  # at a section boundary the parts may swap roles


def test_repeat_pairs_and_links():
    bars = [{"bar": b, "start_s": 2.0 * b} for b in range(8)]
    repeat = {"bars": [{"bar": b, "matches": [{"bar": b + 4, "score": 0.9}] if b < 4 else []} for b in range(8)]}
    wins = [{"start_s": 0.0, "end_s": 8.0, "start_bar": 0, "end_bar": 4},
            {"start_s": 8.0, "end_s": 16.0, "start_bar": 4, "end_bar": 8}]
    assert TR.repeat_pairs(wins, repeat) == [(0, 1)]
    notes = [{"onset_s": 0.50, "pitch": 60}, {"onset_s": 8.52, "pitch": 60}, {"onset_s": 8.50, "pitch": 62},
             {"onset_s": 3.0, "pitch": 55}, {"onset_s": 11.2, "pitch": 55}]
    rep = TR.repeat_links(notes, bars, repeat)
    assert rep[0] == [1] and rep[1] == [0] and rep[2] == []  # same pitch, 20 ms off; the other pitch is not
    assert rep[3] == [] and rep[4] == []  # 200 ms off: no link
    assert TR.repeat_links(notes, bars, None) == [[]] * 5


def test_a_part_edit_in_verse_1_reaches_verse_2(tmp_path):
    """DESIGN 4.10 / M7b: an edit's must-link through the repeat map - and an own edit of the repeat wins."""
    job0, _d0 = make_parts_job(tmp_path / "a")
    g0 = Q.notes_of(E.apply(E.load_base(job0), []).quant, "guitar_all")
    i, j = next((a["i"], b["i"]) for a in g0 for b in g0
                if a["i"] < b["i"] and a["pitch"] == b["pitch"] and a["line"] == b["line"] == "1")
    job, _d = make_parts_job(tmp_path / "b", rep={i: [j], j: [i]})
    s = E.Session(job)
    try:
        uid = next(u for u, n in s.map["notes"].items() if n["track"] == "guitar_all"
                   and Q.notes_of(s.state.quant, "guitar_all")[int(u.split(":")[1])]["i"] == i)
        line_of = {int(n["i"]): n["line"] for n in Q.notes_of(s.state.quant, "guitar_all")}
        assert line_of[i] == line_of[j] == "1"
        r = s.command({"cmd": "part", "uid": uid, "to": "2"})
        assert r["applied"]
        line_of = {int(n["i"]): n["line"] for n in Q.notes_of(s.state.quant, "guitar_all")}
        assert line_of[i] == "2" and line_of[j] == "2" and s.state.report["propagated"] == 1
        # the repeat's own edit wins over the spread, whatever the order
        uj = next(u for u, n in s.map["notes"].items() if n["track"] == "guitar_all"
                  and Q.notes_of(s.state.quant, "guitar_all")[int(u.split(":")[1])]["i"] == j)
        s.command({"cmd": "part", "uid": uj, "to": "1"})
        line_of = {int(n["i"]): n["line"] for n in Q.notes_of(s.state.quant, "guitar_all")}
        assert line_of[i] == "2" and line_of[j] == "1"
        assert s.command({"cmd": "part", "uid": uj, "to": "3"})["applied"] is False  # no third part
        assert s.command({"cmd": "part", "uid": uj, "to": "u"})["applied"]
    finally:
        s.close()
    # the log replays to the same parts after a restart
    s2 = E.Session(job)
    try:
        line_of = {int(n["i"]): n["line"] for n in Q.notes_of(s2.state.quant, "guitar_all")}
        assert line_of[i] == "2" and line_of[j] == "u"
    finally:
        s2.close()


def _line_by_i(s):
    return {int(n["i"]): n["line"] for n in Q.notes_of(s.state.quant, "guitar_all")
            if n.get("from_track", "guitar_all") == "guitar_all" and not n.get("added")}


def _uid_of(s, i):
    return next(u for u, n in s.map["notes"].items() if n["track"] == "guitar_all"
                and (lambda q: q.get("i") == i and q.get("from_track", "guitar_all") == "guitar_all")(
                    Q.notes_of(s.state.quant, "guitar_all")[int(u.split(":")[1])]))


def _pair(tmp_path):
    job0, _ = make_parts_job(tmp_path / "a")
    g0 = Q.notes_of(E.apply(E.load_base(job0), []).quant, "guitar_all")
    return next((a["i"], b["i"]) for a in g0 for b in g0
                if a["i"] < b["i"] and a["pitch"] == b["pitch"] and a["line"] == b["line"] == "1")


def test_a_bass_note_moved_to_the_guitar_is_not_a_repeat_target(tmp_path):
    """Review 2026-10-17: a note assigned in from the bass keeps the bass list's index; the spread must not take it
    for the guitar note with that index."""
    i, j = _pair(tmp_path)
    job, _d = make_parts_job(tmp_path / "b", rep={i: [j], j: [i]})
    s = E.Session(job)
    try:
        bass = Q.notes_of(s.state.quant, "bass")
        b_uid = next(u for u, n in s.map["notes"].items() if n["track"] == "bass"
                     and Q.notes_of(s.state.quant, "bass")[int(u.split(":")[1])]["i"] == j) \
            if any(n["i"] == j for n in bass) else None
        if b_uid is None:  # the fixture's bass is shorter: any bass note, its index made j
            pytest.skip("no bass note with that index")
        s.command({"cmd": "assign", "uid": b_uid, "track": "guitar_all"})
        s.command({"cmd": "part", "uid": _uid_of(s, i), "to": "2"})
        assert _line_by_i(s)[j] == "2"  # the real guitar repeat followed
        moved = [n for n in Q.notes_of(s.state.quant, "guitar_all") if n.get("from_track") == "bass"]
        assert moved and moved[0].get("line") != "2"
    finally:
        s.close()


def test_a_part_edit_keeps_the_repeats_pin_and_fails_softly_on_a_missing_part(tmp_path):
    i, j = _pair(tmp_path)
    job, _d = make_parts_job(tmp_path / "b", rep={i: [j], j: [i]})
    E.append(job, {"op": "pin", "note": E.anchor("guitar_all", next(
        n for n in Q.notes_of(E.apply(E.load_base(job), []).quant, "guitar_all") if n["i"] == j)),
        "string": 2, "fret": 5})
    s = E.Session(job)
    try:
        s.command({"cmd": "part", "uid": _uid_of(s, i), "to": "2"})
        tj = next(n for n in Q.notes_of(s.state.quant, "guitar_all") if n["i"] == j)
        assert tj["line"] == "2" and tj.get("pin") == [2, 5]  # spread, the user's pin kept
    finally:
        s.close()
    # a log that names a part this run does not have (fewer parts after a re-run): reported, not a phantom track
    E.append(job, {"op": "part", "note": E.anchor("guitar_all", next(
        n for n in Q.notes_of(E.apply(E.load_base(job), []).quant, "guitar_all") if n["i"] == i)), "to": "3"})
    s = E.Session(job)
    try:
        assert any(u.get("op") == "part" for u in s.state.report["unmatched"])
        assert "gtr_3" not in s.state.tab["tracks"] or not s.state.tab["tracks"]["gtr_3"].get("notes")
        assert [x["why"] for x in s.info()["not_applied"] if x["op"] == "part"] == ["unmatched"]
    finally:
        s.close()


def test_repeat_pairs_pull_a_window_to_its_repeat():
    """ICM over repeat pairs (w_rep): with the chain links switched off, a window v0 relabelled follows the window
    the repeat map pairs it with - and stays as v0 left it without the repeat term."""
    notes, feats, sigs, wins, post, line0, _truth = _scene(swap_window=3, p_right=0.55, n_win=4)
    for k, w in enumerate(wins):
        w["start_bar"], w["end_bar"] = 4 * k, 4 * k + 4
    repeat = {"bars": [{"bar": b, "matches": [{"bar": b + 8, "score": 0.9}]} for b in range(4, 8)]
              + [{"bar": b, "matches": [{"bar": b - 8, "score": 0.9}]} for b in range(12, 16)]}
    alone = TR.track(notes, feats, sigs, wins, post, line0, repeat=repeat, params={"w_link": 0.0, "w_rep": 0.0})
    assert alone["pairs"] == [(1, 3)] and alone["changed"] == 0
    out = TR.track(notes, feats, sigs, wins, post, line0, repeat=repeat, params={"w_link": 0.0, "w_rep": 5.0})
    # the repeat term only asks the pair to agree (v0 labelled them the other way round): one of the two turns
    assert out["perms"][1] != out["perms"][3] and out["changed"] == 16
    linked = TR.track(notes, feats, sigs, wins, post, line0, repeat=repeat, params={"w_rep": 5.0})
    assert linked["perms"][:3] == [[1, 2]] * 3 and linked["perms"][3] == [2, 1]  # neighbours anchor window 1


def test_the_parts_stage_runs_the_tracker_when_asked():
    from bandscribe.parts import stage as PS

    notes, _feats, sigs, wins, _post, _line0, _truth = _scene(p_right=0.55)
    sigs = [{"ild_db": 8.0 if n["pitch"] >= 60 else -8.0} for n in notes]  # two sides: v0 finds two parts
    for w in wins:
        w["label"] = "R3"
    bars = [{"bar": k, "start_s": 4.0 * k, "end_s": 4.0 * k + 4.0} for k in range(5)]
    for p in ({"k": 2, "count": {}, "tau": 0.4, "track": False}, {"k": 2, "count": {}, "tau": 0.4, "track": True}):
        body = PS.parts_song(notes, sigs, wins, bars, p, repeat=None)
        assert len(body["notes"]) == len(notes) and all("rep" in n for n in body["notes"])
        assert ("track" in body["count"]) == p["track"]


def test_beat_time_matches_beat_to_time_at_the_edges():
    from bandscribe.schema.grid import TempoMap

    from test_tab_quantize import beat_times, grid_doc

    tm = TempoMap.from_dict(grid_doc(beat_times(12, warp=True)))
    for b in (-3.5, -0.0, 0.0, 0.25, 5.5, 10.999, 11.0, 11.5, 40.0):
        assert tm.beat_time(b) == float(tm.beat_to_time(b))
