"""M7c in the pipeline and the viewer: the exported Role / Player views, the part count switch (``parts_k``), the
"to the end of the section" edit (``part_span``) and the view route."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from bandscribe import atomic
from bandscribe.parts import stage as PS
from bandscribe.tab import edits as E
from bandscribe.tab import quantize as Q

from test_parts_tracks import make_parts_job
from test_tab_edits import KEYS

NS_KEY, RO_KEY = "ns" * 32, "ro" * 32


def k_job(root):
    """``make_parts_job`` plus the parts stage's own inputs (signatures: part 1 left, part 2 right; R3 windows), so
    the parts stage can run again with another K."""
    job, d = make_parts_job(root)
    st = job / "stages"
    lines = atomic.read_json(d["parts"] / "guitar_lines.json")["notes"]
    sigs = [{"ild_db": 8.0 if n["line"] == "1" else -8.0, "coh": 0.6, "ms_db": 3.0} for n in lines]
    atomic.write_json(st / "notesig" / NS_KEY[:12] / PS.NOTESIG_FILE, {"format": PS.NOTESIG_FORMAT, "sigs": sigs})
    end = max(float(n["offset_s"]) for n in lines) + 1.0
    wins = [{"start_s": i * end / 4, "end_s": (i + 1) * end / 4, "label": "R3", "section": "S1",
             "start_bar": 4 * i, "end_bar": 4 * i + 4} for i in range(4)]
    atomic.write_json(st / "router" / RO_KEY[:12] / "router.json", {"windows": wins})
    atomic.write_json(d["parts"] / "manifest.json", {"stage": "parts", "params": PS.parts_params(None), "deps": {
        "notes": KEYS["notes"], "notesig": NS_KEY, "router": RO_KEY, "grid": KEYS["grid"], "sections": KEYS["sections"]}})
    return job, d


def _lines(s):
    return sorted({n.get("line") for n in Q.notes_of(s.state.quant, "guitar_all")})


def test_the_score_stage_writes_the_role_and_player_views(tmp_path):
    job, d = make_parts_job(tmp_path)
    views = d["score"] / "tab" / "views"
    for f in ("role.alphatex", "role.gp5", "player.alphatex", "player.gp5", "views.json"):
        assert (views / f).is_file(), f
    doc = json.loads((views / "views.json").read_text(encoding="utf-8"))
    assert doc["settings"] == {"players": 2, "idle_policy": "rest", "unison_policy": "duplicate"}
    assert doc["player"]["violations"] == 0 and [t["name"] for t in doc["player"]["tracks"]][-1] == "Bass"
    tex = (views / "player.alphatex").read_text(encoding="utf-8")
    assert '\\track ("Player 1"' in tex and '\\track ("Player 2"' in tex
    score = json.loads((d["score"] / "score.json").read_text(encoding="utf-8"))
    roles = {ln["id"]: [r["role"] for r in ln["role_segments"]] for ln in score["lines"]}
    assert roles["gtr_1"] and all(r in ("Rhythm", "Riff", "Lead", "Fill", "Arpeggio", "Tacet") for r in roles["gtr_1"])


def test_the_part_count_switch_runs_the_parts_stage_again_and_undoes(tmp_path):
    job, _d = k_job(tmp_path)
    base = E.load_base(job)
    assert {n["line"] for n in E.parts_with_k(base, 1)["notes"]} == {"1"}
    s = E.Session(job)
    try:
        assert _lines(s) == ["1", "2"] and s.info()["parts_k"] == 0
        r = s.command({"cmd": "parts_k", "k": 1})
        assert r["applied"] and _lines(s) == ["1"] and s.info()["parts_k"] == 1
        assert "gtr_2" not in s.state.tab["tracks"] and s.state.quant["parts"]["count"]["forced"] == 1
        v = s.view("player")
        assert [t["id"] for t in v["map"]["tracks"]] == ["player_1", "bass"] and v["doc"]["violations"] == 0
        assert s.command({"cmd": "parts_k", "k": 1})["applied"] is False  # already that count
        s.command({"cmd": "undo"})
        assert _lines(s) == ["1", "2"] and "gtr_2" in s.state.tab["tracks"]
        with pytest.raises(E.EditError):
            s.command({"cmd": "parts_k", "k": 7})
    finally:
        s.close()


def test_part_span_moves_one_part_to_the_end_of_the_section(tmp_path):
    job, _d = make_parts_job(tmp_path)
    s = E.Session(job)
    try:
        g = Q.notes_of(s.state.quant, "guitar_all")
        k = next(i for i, n in enumerate(g) if n["line"] == "1" and n["bar"] >= 3)
        t0 = g[k]["onset_s"]
        before = [(n["onset_s"], n["line"]) for n in g]
        r = s.command({"cmd": "part_span", "uid": f"guitar_all:{k}", "to": "2"})
        assert r["applied"] and r["select"] == f"guitar_all:{k}"
        after = {(n["onset_s"], n["pitch"]): n["line"] for n in Q.notes_of(s.state.quant, "guitar_all")}
        moved = [(t, ln) for t, ln in before if ln == "1" and t >= t0]
        assert moved and all(after[(n["onset_s"], n["pitch"])] == "2" for n in g if n["line"] == "1" and n["onset_s"] >= t0)
        assert any(n["line"] == "1" and n["onset_s"] < t0 for n in Q.notes_of(s.state.quant, "guitar_all"))  # earlier stay
        assert s.command({"cmd": "part_span", "uid": f"guitar_all:{k}", "to": "3"})["applied"] is False
    finally:
        s.close()


def test_the_view_route(tmp_path):
    from bandscribe.tab import viewer

    job, _d = make_parts_job(tmp_path)
    at = tmp_path / "at"
    at.mkdir()
    (at / "alphaTab.min.js").write_text("//", encoding="utf-8")
    httpd, t = viewer.serve(job, alphatab=at)
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        d = json.loads(urllib.request.urlopen(base + "/api/view?kind=player&players=1&idle=double").read())
        assert [x["id"] for x in d["map"]["tracks"]][0] == "player_1" and "\\track" in d["tex"]
        d = json.loads(urllib.request.urlopen(base + "/api/view?kind=role").read())
        assert d["map"]["tracks"][0]["id"].startswith("role_")
        for bad in ("/api/view?kind=part", "/api/view?kind=player&idle=nap", "/api/view?kind=player&players=x"):
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(base + bad)
            assert ei.value.code == 400
    finally:
        httpd.shutdown()
        t.join(timeout=5)
        httpd.session.close()


def test_view_tracks_keep_the_guitar_sound_and_the_extra_track_is_a_staff(tmp_path):
    from bandscribe.tab import stage as T

    job, d = make_parts_job(tmp_path)
    for view in ("role", "player"):
        tex = (d["score"] / "tab" / "views" / f"{view}.alphatex").read_text(encoding="utf-8")
        assert r"\instrument (distortionguitar)" in tex and "electricguitarclean" not in tex
    st = E.apply(E.load_base(job), [])
    lay, doc = T.view_layout(st.quant, st.tab, st.sections, "player", {"players": 1})
    extra = [e for e in lay if e["tid"] == "player_extra"]
    assert extra and extra[0]["fretted"] is False  # several parts at once: written as a staff, no note dropped
    parted = sum(1 for n in Q.notes_of(st.quant, "guitar_all") if n.get("line") in ("1", "2"))
    assert sum(len(e["notes"]) for e in lay if e["tid"].startswith("player")) == parted


def test_a_switch_drops_the_parts_own_tunings_and_the_runs_count_is_auto(tmp_path):
    job, d = k_job(tmp_path)
    tab = atomic.read_json(d["tab"] / "tab.json")
    tab["tuning"]["lines"] = {"gtr_1": {**tab["tuning"]["guitar"], "label": "Drop D"}}
    atomic.write_json(d["tab"] / "tab.json", tab)
    s = E.Session(job)
    try:
        assert s.command({"cmd": "parts_k", "k": 2})["applied"] is False  # the run chose 2: that is "auto"
        s.command({"cmd": "parts_k", "k": 1})
        assert s.state.tab["tuning"]["lines"] == {}
        # a part edit made under one split does not land under another
        uid = next(u for u, n in s.map["notes"].items() if n["track"] == "guitar_all" and u in s.map["at"])
        assert s.command({"cmd": "part", "uid": uid, "to": "u"})["applied"]
        s.command({"cmd": "parts_k", "k": 0})  # back to the run's split: the edit made with K=1 is set aside
        assert any(x["op"] == "part" for x in s.info()["not_applied"])
    finally:
        s.close()


def test_a_span_beats_an_earlier_part_edits_spread(tmp_path):
    """Review 2026-10-17: a part edit spreads at once (log order), so a later span sees the spread note and moves it."""
    job0, _ = make_parts_job(tmp_path / "a")
    g0 = Q.notes_of(E.apply(E.load_base(job0), []).quant, "guitar_all")
    i, j = next((a["i"], b["i"]) for a in g0 for b in g0
                if a["i"] < b["i"] and a["pitch"] == b["pitch"] and a["line"] == b["line"] == "1")
    job, _d = make_parts_job(tmp_path / "b", rep={i: [j], j: [i]})
    s = E.Session(job)

    def uid_of(idx):
        return next(f"guitar_all:{k}" for k, n in enumerate(Q.notes_of(s.state.quant, "guitar_all")) if n["i"] == idx)

    def line_of(idx):
        return next(n["line"] for n in Q.notes_of(s.state.quant, "guitar_all") if n["i"] == idx)

    try:
        s.command({"cmd": "part", "uid": uid_of(i), "to": "2"})
        assert line_of(j) == "2"  # spread
        assert s.command({"cmd": "part_span", "uid": uid_of(j), "to": "u"})["applied"]
        assert line_of(j) == "u" and line_of(i) == ("u" if g0[[n["i"] for n in g0].index(i)]["onset_s"]
                                                      >= g0[[n["i"] for n in g0].index(j)]["onset_s"] else "2")
    finally:
        s.close()
    s2 = E.Session(job)  # the log replays to the same
    try:
        assert next(n["line"] for n in Q.notes_of(s2.state.quant, "guitar_all") if n["i"] == j) == "u"
    finally:
        s2.close()


def test_gp5_keeps_the_markers(tmp_path):
    """Beat text reaches both writers (GP5 holds the fretted tracks only, as in the Part view)."""
    import guitarpro as gp

    from bandscribe.tab import alphatex as ATX
    from bandscribe.tab import gp5 as GP5
    from bandscribe.tab import stage as T

    job, _d = make_parts_job(tmp_path)
    st = E.apply(E.load_base(job), [])
    lay, _doc = T.view_layout(st.quant, st.tab, st.sections, "player", {"players": 2})
    first_bar = min(int(n["bar"]) for n in lay[0]["notes"])
    lay[0]["markers"] = [(first_bar, "plays Gtr 2 (Lead)")]
    score, problems = T.build_score("t", st.quant, st.tab, st.grid, st.sections, T.score_params(None), layout=lay)
    assert not problems and 'txt "plays Gtr 2 (Lead)"' in ATX.write(score)
    GP5.write(score, tmp_path / "v.gp5")
    song = gp.parse(str(tmp_path / "v.gp5"))
    texts = [str(b.text) for tr in song.tracks for m in tr.measures for v in m.voices for b in v.beats if b.text]
    assert any("plays Gtr 2 (Lead)" in x for x in texts)
