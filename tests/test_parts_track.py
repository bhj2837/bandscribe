"""M7b identity tracking (bandscribe.parts.track), repeat links, the ``part`` edit and its spread to repeats."""

from __future__ import annotations

import numpy as np

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
