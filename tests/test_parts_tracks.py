"""M7a Part view: the notesig / parts stages, and the part tracks in tab / score / edits."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from bandscribe import atomic
from bandscribe.eval import remix as R
from bandscribe.parts import stage as PS
from bandscribe.router import notesig as NS
from bandscribe.tab import edits as E
from bandscribe.tab import quantize as Q
from bandscribe.tab import stage as T

from test_tab_edits import KEYS, _raw
from test_tab_quantize import beat_times, grid_doc
from test_tab_tuning import E_STD, bass_line, song

LOOSE = {"n_min": 2, "min_windows": 1, "min_sections": 1, "side_db": 0.0, "use_class": True, "use_symbolic": True}


def _tones(n, specs, sr=R.SR):
    x = np.zeros(n)
    for p, t0, dur, seed in specs:
        y = R.tone(440 * 2 ** ((p - 69) / 12), dur, 3.0, np.random.default_rng(seed))
        a = int(t0 * sr)
        x[a:a + len(y)] += y[: n - a]
    return x


@pytest.fixture(scope="module")
def two_parts():
    """A left part (E4 G4 B4 ...) and a right part (A3 C4 ...) played together, 6 s."""
    n = int(6 * R.SR)
    left_specs = [(64 + (k % 3) * 3, 0.5 + k, 0.8, k) for k in range(5)]
    right_specs = [(57 + (k % 2) * 3, 0.5 + k, 0.8, 10 + k) for k in range(5)]
    st = np.stack([_tones(n, left_specs), _tones(n, right_specs)], axis=1) * 0.3
    notes = sorted([{"onset_s": t, "offset_s": t + 0.8, "pitch": p, "instrument": "distorted_electric_guitar",
                     "side": "L"} for p, t, _d, _s in left_specs]
                   + [{"onset_s": t, "offset_s": t + 0.8, "pitch": p, "instrument": "distorted_electric_guitar",
                       "side": "R"} for p, t, _d, _s in right_specs], key=lambda x: (x["onset_s"], x["pitch"]))
    return notes, st, NS.signatures(st, R.SR, notes)


def _win(label, a=0.0, b=6.0):
    return [{"start_s": a, "end_s": b, "label": label, "section": "S1"}]


def test_parts_song_gives_each_side_its_own_part(two_parts):
    notes, _st, sigs = two_parts
    body = PS.parts_song(notes, sigs, _win("R3"), [], {"k": 0, "count": LOOSE, "tau": 0.4})
    assert body["count"]["chosen"] == 2 and body["count"]["evidence"] == "spatial"
    sides: dict[str, set] = {}
    for n, o in zip(notes, body["notes"]):
        sides.setdefault(o["line"], set()).add(n["side"])
    assert sorted(sides) == ["1", "2"] and all(len(v) == 1 for v in sides.values())
    assert all(o["ild_db"] is not None and not o["shared"] for o in body["notes"])
    assert {ln["position"] for ln in body["lines"]} == {"L", "R"} and body["unassigned"] == 0


def test_parts_song_keeps_one_part_without_spatial_evidence_and_a_forced_count_wins(two_parts):
    notes, _st, sigs = two_parts
    plain = {**LOOSE, "use_symbolic": False}  # (with it, the two peeled tops over the fifths would be a lead layer)
    one = PS.parts_song(notes, sigs, _win("R0"), [], {"k": 0, "count": plain, "tau": 0.4})
    assert one["count"]["chosen"] == 1 and {o["line"] for o in one["notes"]} == {"1"}
    assert one["lines"][0]["name"].startswith("Gtr - ")
    forced = PS.parts_song(notes, sigs, _win("R0"), [], {"k": 2, "count": plain, "tau": 0.4})
    assert forced["count"]["forced"] == 2 and forced["count"]["k"] == 1
    with pytest.raises(PS.PartsStageError):
        PS.parts_song(notes, sigs[:-1], _win("R3"), [], {"k": 0, "count": LOOSE, "tau": 0.4})


def test_parts_params_follow_the_config():
    from bandscribe.config import load_config

    p = PS.parts_params(load_config())
    assert p["k"] == 0 and p["tau"] == 0.4 and p["count"]["n_min"] >= 1
    p3 = PS.parts_params(load_config(overrides=["tab.guitar_parts=3", "parts.n_min=12"]))
    assert p3["k"] == 3 and p3["count"]["n_min"] == 12


def test_notesig_and_parts_stages_end_to_end(tmp_path, two_parts):
    from bandscribe import audio as AU

    notes, st, _sigs = two_parts
    d = {k: tmp_path / k for k in ("notes", "sep", "router", "grid", "notesig", "parts")}
    atomic.write_json(d["notes"] / "guitar_all.json", {"format": "x", "notes": notes})
    (d["sep"] / "stems").mkdir(parents=True)
    AU.write_f32(d["sep"] / "stems" / "guitar.wav", st.astype(np.float32), R.SR)
    atomic.write_json(d["router"] / "router.json", {"windows": _win("R3")})
    atomic.write_json(d["grid"] / "grid.json", {"bars": [{"bar": i + 1, "start_s": 2.0 * i, "end_s": 2.0 * i + 2}
                                                        for i in range(3)]})

    def ctx(stage, deps, params):
        d[stage].mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(out_dir=d[stage], dep_dirs={k: d[k] for k in deps}, params=params)

    PS.run_notesig(ctx("notesig", ("notes", "sep"), PS.notesig_params(None)))
    sig_doc = atomic.read_json(d["notesig"] / PS.NOTESIG_FILE)
    assert sig_doc["n_notes"] == len(notes) == len(sig_doc["sigs"])
    PS.run_parts(ctx("parts", ("notes", "notesig", "router", "grid"), {"k": 0, "count": LOOSE, "tau": 0.4}))
    doc = atomic.read_json(d["parts"] / PS.LINES_FILE)
    assert doc["format"] == T.PARTS_FORMAT and doc["count"]["chosen"] == 2 and len(doc["lines"]) == 2
    assert {n["line"] for n in doc["notes"]} == {"1", "2"} and all(lb["bars"] >= 1 for lb in doc["lines"])


# ------------------------------------------------------------------------------------------ part tracks

PARTS_META = {"count": {"chosen": 2}, "lines": [{"number": 1, "name": "Gtr 1 - Rhythm (dist., L)", "short": "Gtr 1"},
                                                {"number": 2, "name": "Gtr 2 - Lead (dist., R)", "short": "Gtr 2"}]}


def _quant_with_lines(n_bars=6):
    times = beat_times(n_bars * 4 + 8, warp=False)
    grid = grid_doc(times)
    g = _raw(song(E_STD, n_bars=n_bars), 0.5, "distorted_electric_guitar")
    for k, n in enumerate(g):  # alternate events between the parts; one note no part took
        n["line"], n["p"] = ("1" if k % 2 == 0 else "2"), 0.9
    g[-1]["line"], g[-1]["p"] = "u", 0.3
    b = _raw(bass_line((28, 33, 38, 43), n=24), 0.5, "electric_bass")
    q = Q.quantize({"guitar_all": g, "bass": b}, grid)
    q["parts"] = PARTS_META
    return q, grid


def test_derived_tracks_are_the_guitar_notes_of_each_line():
    q, _grid = _quant_with_lines()
    g = Q.notes_of(q, T.GUITAR)
    assert all("line" in n and "p" in n for n in g)  # the quantiser passes the parts' keys through
    one, two, un = (T.track_notes(q, t) for t in ("gtr_1", "gtr_2", T.UNASSIGNED))
    assert len(one) + len(two) + len(un) == len(g) and len(un) == 1
    assert all(n["uid"] == f"guitar_all:{n['src_k']}" and g[n["src_k"]]["line"] == "1" for n in one)
    assert T.split_on(q) and not T.split_on({"tracks": {"guitar_all": {"notes": [{"line": "all"}]}}})
    assert T.track_name(q, "gtr_2") == ("Gtr 2 - Lead (dist., R)", "Gtr 2")
    assert T.track_name({}, "gtr_3") == ("Gtr 3", "Gtr 3")


def test_a_shared_note_is_in_every_part_but_not_unassigned():
    q, _grid = _quant_with_lines()
    g = Q.notes_of(q, T.GUITAR)
    g[0]["shared"] = True
    assert any(n["src_k"] == 0 for n in T.track_notes(q, "gtr_2"))
    assert not any(n["src_k"] == 0 for n in T.track_notes(q, T.UNASSIGNED))
    assert T.track_notes(q, "gtr_3") == []  # only the parts the song has (review 2026-10-11: no phantom tracks)


def test_a_guitar_note_without_a_part_is_unassigned():
    q, _grid = _quant_with_lines()
    g = Q.notes_of(q, T.GUITAR)
    g[1]["line"] = None  # an added note with nothing selected, or a bass note moved to the guitar
    g[2].pop("line")
    un = {n["src_k"] for n in T.track_notes(q, T.UNASSIGNED)}
    assert {1, 2, len(g) - 1} <= un
    parts = sum(len(T.track_notes(q, t)) for t in ("gtr_1", "gtr_2"))
    assert parts + len(un) == len(g)  # parts + Unassigned = every guitar note
    no_parts = Q.quantize({"guitar_all": [{"onset_s": 1.0, "offset_s": 1.5, "pitch": 60}], "bass": []}, _grid)
    assert T.track_notes(no_parts, T.UNASSIGNED) == [] and not T.split_on(no_parts)


def _run_tab(tmp_path, q, name="tab", params=None):
    from test_tab_stages import _ctx

    d = {"quant": tmp_path / f"quant_{name}", "a4": tmp_path / "a4"}
    atomic.write_json(d["quant"] / "quant.json", q)
    atomic.write_json(d["a4"] / "a4.json", {"cents": 0.0, "a4_hz": 440.0, "confidence": 0.9})
    ctx = _ctx(tmp_path, d, {**T.tab_params(None), **(params or {})}, name)
    T.run_tab(ctx)
    return atomic.read_json(ctx.out_dir / "tab.json"), ctx.out_dir


def test_tab_frets_each_part_and_the_score_writes_the_part_view(tmp_path):
    q, grid = _quant_with_lines()
    tab, out = _run_tab(tmp_path, q)
    tracks = tab["tracks"]
    assert tracks["guitar_all"]["fretted"] is False and tracks[T.UNASSIGNED]["fretted"] is False
    for tid in ("gtr_1", "gtr_2"):
        assert tracks[tid]["fretted"] and tracks[tid]["check"]["bad_events"] == []
        assert all(n["uid"].startswith("guitar_all:") for n in tracks[tid]["notes"])
        assert (out / "text" / f"{tid}.txt").read_text(encoding="utf-8").startswith("Gtr ")
    assert tab["tuning"]["lines"] == {}  # too few notes for a part's own tuning
    score, problems = T.build_score("t", q, tab, grid, [], T.score_params(None))
    assert problems == []
    assert [t.name for t in score.tracks] == ["Gtr 1 - Rhythm (dist., L)", "Gtr 2 - Lead (dist., R)",
                                             "All guitars (reference)", "Unassigned (review)", "Bass"]
    srcs = {u for be in score.tracks[0].beats for b in be for wn in b.notes for u in wn.src}
    assert srcs and all(u.startswith("guitar_all:") for u in srcs)
    doc = T.score_doc("k", {"duration_s": 30.0}, "t", q, tab, grid)
    kinds = {ln.id: ln.kind for ln in doc.lines}
    assert kinds == {"gtr_1": "guitar", "gtr_2": "guitar", "guitar_all": "reference", "unassigned": "unassigned",
                     "bass": "bass"}
    part = next(ln for ln in doc.lines if ln.id == "gtr_1")
    assert {n.confidence.part for n in part.notes} == {0.9}
    T.write_score(tmp_path / "score_out", song_key="k", meta={}, title="t", quant=q, tab=tab, grid=grid,
                  sections=[], params=T.score_params(None))  # MIDI track names are Latin-1
    assert (tmp_path / "score_out" / "midi" / "gtr_2_quantized.mid").is_file()
    assert (tmp_path / "score_out" / "midi" / "unassigned_quantized.mid").is_file()


def test_quant_reads_the_guitar_and_the_part_names_from_the_parts_stage(tmp_path):
    from test_tab_stages import _ctx

    notes_dir, parts_dir = tmp_path / "notes", tmp_path / "parts"
    times = beat_times(12, warp=False)
    atomic.write_json(notes_dir / "guitar_all.json", {"notes": [{"onset_s": 1.0, "offset_s": 1.5, "pitch": 60}]})
    atomic.write_json(notes_dir / "bass_raw.json", {"notes": []})
    atomic.write_json(parts_dir / T.LINES_JSON, {"format": T.PARTS_FORMAT, **PARTS_META,
                                                 "notes": [{"onset_s": 1.0, "offset_s": 1.5, "pitch": 60,
                                                            "line": "2", "p": 0.8, "shared": False}]})
    assert T.quant_tracks({"notes": notes_dir})["guitar_all"][0].get("line") is None
    assert T.quant_tracks({"notes": notes_dir, "parts": parts_dir})["guitar_all"][0]["line"] == "2"
    d = {"notes": notes_dir, "parts": parts_dir, "grid": tmp_path / "grid", "sections": tmp_path / "sections"}
    atomic.write_json(d["grid"] / "grid.json", grid_doc(times))
    atomic.write_json(d["sections"] / "sections.json", {"sections": []})
    ctx = _ctx(tmp_path, d, T.quant_params(None), "quant")
    T.run_quant(ctx)
    q = atomic.read_json(ctx.out_dir / "quant.json")
    assert q["parts"]["lines"][1]["name"] == "Gtr 2 - Lead (dist., R)"
    assert T.parts_meta({"split": parts_dir}) is None  # an M6 run has no Part view


def test_a_part_gets_its_own_tuning_only_when_its_notes_clearly_want_one(monkeypatch):
    q, _grid = _quant_with_lines()
    tun = {"guitar": {"label": "E standard", "tuning": [40, 45, 50, 55, 59, 64], "capo": 0,
                      "top": [{"label": "E standard", "cost": 0.4}, {"label": "Drop D", "cost": 0.6}]},
           "bass": {"label": "E standard", "tuning": [28, 33, 38, 43], "capo": 0, "downtune": 0}}
    p = T.tab_params(None)
    assert T.line_tunings(q, tun, p, set()) == {}  # each part has fewer than LINE_TUNING_MIN_NOTES notes
    monkeypatch.setattr(T, "LINE_TUNING_MIN_NOTES", 1)
    calls = []

    def row(label, cost):
        return {"label": label, "tuning": [38 if label == "Drop D" else 40, 45, 50, 55, 59, 64], "capo": 0,
                "cost": cost, "per_note": cost, "unplayable": 0.0, "joint": cost, "prior": 0.0}

    def fake_rank(events, candidates, params, bass_downtune):
        calls.append((list(candidates), bass_downtune))
        if len(calls) == 1:
            return [row("Drop D", 1.0), row("E standard", 1.5)]
        return [row("Drop D", 1.0), row("E standard", 1.1)]

    monkeypatch.setattr(T.TU, "rank_guitar", fake_rank)
    got = T.line_tunings(q, tun, p, set())
    assert list(got) == ["gtr_1"] and got["gtr_1"]["label"] == "Drop D"  # gtr_2: the song's tuning is close enough
    assert got["gtr_1"]["song_cost"] == 1.5 and "joint" not in got["gtr_1"]
    # only the song's best few are searched (2026-10-17), against the song's bass
    assert calls[0] == (["E standard", "Drop D"], 0)
    assert T.line_tunings(q, tun, {**p, "guitar_tuning": "E standard"}, set()) == {}  # the user named the tuning
    tracks = T.fret_tracks(q, {**tun, "lines": got}, p)
    assert tracks["gtr_1"]["tuning"] == [38, 45, 50, 55, 59, 64] and tracks["gtr_2"]["tuning"][0] == 40


# --------------------------------------------------------------------------------------- edits


def make_parts_job(root):
    """A finished run whose quant read the guitar from a parts stage (two parts)."""
    job = root / "f-test"
    st = job / "stages"
    keys = dict(KEYS)
    d = {s: st / s / keys[s][:12] for s in keys}
    times = beat_times(12 * 4 + 4, warp=False)
    g = _raw(song(E_STD, n_bars=6), 0.5, "distorted_electric_guitar")
    for k, n in enumerate(g):
        n["line"], n["shared"], n["p"] = ("1" if (k // 3) % 2 == 0 else "2"), False, 0.9
    b = _raw(bass_line((28, 33, 38, 43), n=40), 0.5, "electric_bass")
    grid = grid_doc(times)
    sections = [{"id": "S1", "label": "A", "start_bar": 1, "end_bar": 14, "start_s": times[0],
                 "end_s": times[-1] + 0.5}]
    atomic.write_json(d["notes"] / "guitar_all.json", {"notes": [{k: v for k, v in n.items() if k not in
                                                                  ("line", "shared", "p")} for n in g]})
    atomic.write_json(d["notes"] / "bass_raw.json", {"notes": b})
    atomic.write_json(d["parts"] / T.LINES_JSON, {"format": T.PARTS_FORMAT, **PARTS_META, "notes": g})
    atomic.write_json(d["grid"] / "grid.json", grid)
    atomic.write_json(d["sections"] / "sections.json", {"sections": sections})
    qp = T.quant_params(None)
    qdirs = {"notes": d["notes"], "parts": d["parts"]}
    atomic.write_json(d["quant"] / "quant.json", {**Q.quantize(T.quant_tracks(qdirs), grid, sections, qp),
                                                  "parts": T.parts_meta(qdirs)})
    atomic.write_json(d["a4"] / "a4.json", {"cents": 0.0, "a4_hz": 440.0, "confidence": 0.9})
    tp = T.tab_params({"tab.guitar_parts": 0})

    def ctx(stage, deps, params):
        return SimpleNamespace(song_key="f-test", stage=stage, key=keys[stage], out_dir=d[stage],
                               dep_dirs={k: d[k] for k in deps}, params=params, config=None,
                               store=SimpleNamespace(job_dir=lambda key: job))

    T.run_tab(ctx("tab", ("quant", "a4"), tp))
    sp = T.score_params(None)
    atomic.write_json(job / "input" / "meta.json", {"source": {"filename": "song.wav"}, "duration_s": times[-1] + 1})
    T.run_score(ctx("score", ("tab", "quant", "grid", "sections"), sp))
    for stage, params, deps in (
            ("quant", qp, {"notes": keys["notes"], "parts": keys["parts"], "grid": keys["grid"],
                           "sections": keys["sections"]}),
            ("tab", tp, {"quant": keys["quant"], "a4": keys["a4"]}),
            ("score", sp, {k: keys[k] for k in ("tab", "quant", "grid", "sections")})):
        atomic.write_json(d[stage] / "manifest.json", {"stage": stage, "key": keys[stage], "song_key": "f-test",
                                                        "params": json.loads(json.dumps(params)), "deps": deps})
    exp = job / "export"
    files = {}
    for rel in ("tab/score.alphatex", "tab/score.gp5", "tab/gtr_1.txt", "tab/gtr_2.txt", "score.json"):
        if (d["score"] / rel).is_file():
            (exp / rel).parent.mkdir(parents=True, exist_ok=True)
            (exp / rel).write_bytes((d["score"] / rel).read_bytes())
            files[rel] = {"stage": "score", "key12": keys["score"][:12], "sha256": atomic.sha256_file(exp / rel)}
    atomic.write_json(exp / "export.json", {"format": "bandscribe.export/1", "files": files})
    return job, d


def test_a_click_on_a_part_track_edits_the_guitar_note_and_the_part_follows(tmp_path):
    job, d = make_parts_job(tmp_path)
    assert (d["score"] / "tab" / "gtr_1.txt").is_file() and (d["score"] / "tab" / "gtr_2.txt").is_file()
    base = E.load_base(job)
    assert base.parts and base.parts["lines"][0]["name"] == "Gtr 1 - Rhythm (dist., L)"
    built = E.build(base, [])
    assert {"gtr_1", "gtr_2"} <= set(built.state.tab["tracks"])
    assert built.state.quant["parts"] == base.parts  # the names survive a re-quantisation
    pos0 = [u for k, u in built.map["pos"].items() if k.startswith("0:")]
    assert pos0 and all(u.startswith("guitar_all:") for u in pos0)
    uid = pos0[0]
    n_one = len(built.state.tab["tracks"]["gtr_1"]["notes"])
    s = E.Session(job)
    r = s.command({"cmd": "delete", "uid": uid})
    assert r["applied"] and len(s.state.tab["tracks"]["gtr_1"]["notes"]) == n_one - 1
    r = s.command({"cmd": "barline", "bar_index": s.score_bars.index(3), "delta": 1})
    assert r["applied"] and {"gtr_1", "gtr_2"} <= set(s.state.tab["tracks"])  # re-quantised from the parts' notes
    assert s.close()
    assert (job / "export" / "tab" / "gtr_1.txt").is_file()  # the edited parts are exported too


def test_the_string_key_on_a_part_note_pins_it_in_its_part(tmp_path):
    job, _d = make_parts_job(tmp_path)
    s = E.Session(job)
    part_uids = [u for k, u in s.map["pos"].items() if k.startswith("0:")]  # written in Gtr 1
    uid = next(u for u in part_uids if s.map["notes"][u]["string"] is not None and s.map["notes"][u]["pitch"] >= 45)
    before = s.map["notes"][uid]
    assert before["track"] == "guitar_all" and before["fret"] is not None  # shown with its part's fret
    r = s.command({"cmd": "string", "uid": uid, "dir": 1})  # A2 and up: always another string
    assert r["applied"], r["message"]
    after = s.map["notes"][r["select"]]
    assert after["pinned"] and (after["string"], after["fret"]) != (before["string"], before["fret"])
    assert r["hash"] != E.state_hash(E.build(E.load_base(job), []).state)  # the part's new fret is in the hash
    assert s.close()


def test_the_viewer_track_list_follows_the_written_tracks(tmp_path):
    """The viewer looks up ``map.tracks[alphaTab track index]``: it must list every written track in order."""
    job, _d = make_parts_job(tmp_path)
    s = E.Session(job)
    assert [t["id"] for t in s.map["tracks"]] == ["gtr_1", "gtr_2", "guitar_all", "bass"]
    assert [t["fretted"] for t in s.map["tracks"]] == [True, True, False, True]
    for ti, t in enumerate(s.map["tracks"]):  # every position key of a track uses that track's key kind
        keys = [k.split(":")[3] for k in s.map["pos"] if k.startswith(f"{ti}:")]
        assert keys and all(k[0] == ("s" if t["fretted"] else "p") for k in keys)
    assert s.close()


def test_a_note_added_next_to_a_part_note_joins_that_part(tmp_path):
    job, _d = make_parts_job(tmp_path)
    s = E.Session(job)
    uid = next(u for k, u in s.map["pos"].items() if k.startswith("1:"))  # a note written in Gtr 2
    n_two = len(s.state.tab["tracks"]["gtr_2"]["notes"])
    r = s.command({"cmd": "add", "track": "guitar_all", "t_s": 3.0, "pitch": 64, "uid": uid})
    assert r["applied"]
    added = [n for n in Q.notes_of(s.state.quant, "guitar_all") if n.get("added")]
    assert added and added[0].get("line") == "2"
    assert len(s.state.tab["tracks"]["gtr_2"]["notes"]) == n_two + 1
    r = s.command({"cmd": "add", "track": "guitar_all", "t_s": 5.0, "pitch": 64})  # nothing selected: no part
    assert r["applied"]
    assert any(n.get("added") and n.get("line") is None for n in Q.notes_of(s.state.quant, "guitar_all"))
    assert s.close()


def test_an_added_note_follows_its_neighbours_part_after_a_rerun_renumbers_the_parts(tmp_path):
    job, d = make_parts_job(tmp_path)
    s = E.Session(job)
    uid = next(u for k, u in s.map["pos"].items() if k.startswith("1:"))  # a note written in Gtr 2
    r = s.command({"cmd": "add", "track": "guitar_all", "t_s": 3.0, "pitch": 64, "uid": uid})
    assert r["applied"]
    rec = [e for e in E.read_log(job) if e.get("op") == "add"][-1]
    assert "near" in rec and "line" not in rec  # the neighbour's anchor, not its (re-numbered) part
    assert s.close()
    # a re-run swaps the part numbers: the neighbour (and its part) is now "1"
    lines_path = d["parts"] / T.LINES_JSON
    doc = atomic.read_json(lines_path)
    for n in doc["notes"]:
        n["line"] = {"1": "2", "2": "1"}[n["line"]]
    atomic.write_json(lines_path, doc)
    base = E.load_base(job)
    st = E.build(base, E.active(E.read_log(job))).state
    added = [n for n in Q.notes_of(st.quant, "guitar_all") if n.get("added")]
    assert added and added[0]["line"] == "1"


def test_an_old_add_record_does_not_make_a_phantom_part(tmp_path):
    from test_tab_edits import make_job

    job = make_job(tmp_path)  # one part
    base = E.load_base(job)
    rec = {"id": "e1", "op": "add", "track": "guitar_all", "gtick": 96, "dur": 12, "pitch": 64, "line": "2"}
    st = E.build(base, [rec]).state
    assert "gtr_2" not in st.tab["tracks"]  # the old record's part does not exist: the note is Unassigned
    added = [n for n in Q.notes_of(st.quant, "guitar_all") if n.get("added")]
    assert added and added[0].get("line") is None
    assert any(n.get("added") for n in T.track_notes(st.quant, T.UNASSIGNED))


def test_a_run_from_before_the_part_view_asks_for_a_rerun(tmp_path):
    from test_tab_edits import make_job

    job = make_job(tmp_path, guitar_parts=0)  # no parts stage behind its quantisation
    with pytest.raises(E.EditError, match="bandscribe run"):
        E.load_base(job)


def test_the_removed_split_settings_are_ignored_with_a_warning(caplog):
    from bandscribe.config import load_config

    with caplog.at_level("WARNING"):
        cfg = load_config(overrides=["router.split=false", "router.split_min_notes=10"])
    assert not hasattr(cfg.router, "split") and "router.split" in caplog.text


def test_remix_experiments_read_the_pinned_e6_run(tmp_path, monkeypatch):
    from bandscribe import paths
    from bandscribe.router import split_eval as SE

    monkeypatch.setattr(paths, "RUNS", tmp_path)
    (tmp_path / "20991231-000000_abc_exp-e6").mkdir()
    assert SE.e6_run({}).name == "20991231-000000_abc_exp-e6"  # no pinned run here: the latest
    (tmp_path / SE.E6_RUN).mkdir()
    assert SE.e6_run({}).name == SE.E6_RUN  # the pinned one wins over a later E6 run
    assert SE.e6_run({"e6_run": str(tmp_path / "x")}).name == "x"


def test_an_empty_stem_is_silent_not_mixed():
    from bandscribe.router import route as RT

    x = np.zeros((2, 16 * R.SR))
    bars = [{"bar": i, "start_s": 2.0 * i, "end_s": 2.0 * i + 2} for i in range(8)]
    doc = RT.route_song(x, R.SR, bars, None)
    assert {w["label"] for w in doc["windows"]} == {"silent"}
