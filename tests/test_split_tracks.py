"""M6b part split: the split stage's decision and lines, and the derived part tracks in tab / score / edits."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from bandscribe import atomic
from bandscribe.eval import remix as R
from bandscribe.router import stage as RS
from bandscribe.tab import edits as E
from bandscribe.tab import quantize as Q
from bandscribe.tab import stage as T

from test_tab_edits import KEYS, _raw
from test_tab_quantize import beat_times, grid_doc
from test_tab_tuning import E_STD, bass_line, song

SPLIT_ON = {"enabled": True, "p_ild_db": 0.0, "a_split": False, "min_share": 0.5, "min_notes": 2}


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
    return notes, st


def _win(label, a=0.0, b=6.0):
    return [{"start_s": a, "end_s": b, "label": label}]


def test_split_song_sends_each_note_to_its_side(two_parts):
    notes, st = two_parts
    body = RS.split_song(notes, _win("R3"), st, R.SR, SPLIT_ON)
    assert body["decision"]["split"] is True and body["decision"]["p_share"] == 1.0
    assert [n["line"] for n in body["notes"]] == [n["side"] for n in notes]
    assert all(n["ild_db"] is not None for n in body["notes"]) and not any(n["shared"] for n in body["notes"])


def test_split_song_keeps_one_track_when_it_is_not_worth_two(two_parts):
    notes, st = two_parts
    off = RS.split_song(notes, _win("R0"), st, R.SR, SPLIT_ON)  # narrow windows only: no P evidence
    assert off["decision"]["split"] is False and any("P" in r for r in off["decision"]["reasons"])
    assert {n["line"] for n in off["notes"]} == {"all"}
    few = RS.split_song(notes, _win("R3"), st, R.SR, {**SPLIT_ON, "min_notes": 6})
    assert few["decision"]["split"] is False and any("한쪽 음" in r for r in few["decision"]["reasons"])
    disabled = RS.split_song(notes, _win("R3"), st, R.SR, {**SPLIT_ON, "enabled": False})
    assert disabled["decision"]["split"] is False and {n["line"] for n in disabled["notes"]} == {"all"}
    a_win = RS.split_song(notes, _win("R2"), st, R.SR, SPLIT_ON)  # A is held (E22b): merged
    assert a_win["decision"]["split"] is False


def test_split_params_follow_the_config_and_one_guitar_turns_it_off():
    from bandscribe.config import load_config

    p = RS.split_params(load_config())
    assert p["enabled"] is True and p["p_ild_db"] == 0.0 and p["a_split"] is False
    p1 = RS.split_params(load_config(overrides=["tab.guitar_parts=1"]))
    assert p1["enabled"] is False
    p2 = RS.split_params(load_config(overrides=["router.split=false"]))
    assert p2["enabled"] is False


# --------------------------------------------------------------------------------------- derived tracks


def _quant_with_lines(n_bars=6):
    times = beat_times(n_bars * 4 + 8, warp=False)
    grid = grid_doc(times)
    g = _raw(song(E_STD, n_bars=n_bars), 0.5, "distorted_electric_guitar")
    for k, n in enumerate(g):  # alternate events between the parts; one note in neither (a merged window)
        n["line"] = "L" if k % 2 == 0 else "R"
    g[-1]["line"] = "all"
    b = _raw(bass_line((28, 33, 38, 43), n=24), 0.5, "electric_bass")
    q = Q.quantize({"guitar_all": g, "bass": b}, grid)
    return q, grid


def test_derived_tracks_are_the_guitar_notes_of_each_line():
    q, _grid = _quant_with_lines()
    g = Q.notes_of(q, T.GUITAR)
    assert all("line" in n for n in g)  # the quantiser passes the split's keys through
    left = T.track_notes(q, "gtr_l")
    right = T.track_notes(q, "gtr_r")
    assert len(left) + len(right) == len(g) - 1
    assert all(n["uid"] == f"guitar_all:{n['src_k']}" and g[n["src_k"]]["line"] == "L" for n in left)
    assert T.split_on(q) and not T.split_on({"tracks": {"guitar_all": {"notes": [{"line": "all"}]}}})


def _run_tab(tmp_path, q, parts):
    from test_tab_stages import _ctx

    d = {"quant": tmp_path / "quant", "a4": tmp_path / "a4"}
    atomic.write_json(d["quant"] / "quant.json", q)
    atomic.write_json(d["a4"] / "a4.json", {"cents": 0.0, "a4_hz": 440.0, "confidence": 0.9})
    params = {**T.tab_params(None), "guitar_parts": parts}
    ctx = _ctx(tmp_path, d, params, f"tab{parts}")
    T.run_tab(ctx)
    return atomic.read_json(ctx.out_dir / "tab.json"), ctx.out_dir


def test_tab_frets_each_part_and_the_score_writes_them_first(tmp_path):
    q, grid = _quant_with_lines()
    tab, out = _run_tab(tmp_path, q, 0)
    tracks = tab["tracks"]
    assert tracks["guitar_all"]["fretted"] is False  # the merged reference stays a staff
    for tid in ("gtr_l", "gtr_r"):
        assert tracks[tid]["fretted"] and tracks[tid]["check"]["bad_events"] == []
        assert all(n["uid"].startswith("guitar_all:") for n in tracks[tid]["notes"])
        assert (out / "text" / f"{tid}.txt").is_file()
    score, problems = T.build_score("t", q, tab, grid, [], T.score_params(None))
    assert problems == []
    assert [t.name for t in score.tracks] == ["Gtr 1 (L)", "Gtr 2 (R)", "Guitar (all parts)", "Bass"]
    srcs = {u for be in score.tracks[0].beats for b in be for wn in b.notes for u in wn.src}
    assert srcs and all(u.startswith("guitar_all:") for u in srcs)
    doc = T.score_doc("k", {"duration_s": 30.0}, "t", q, tab, grid)
    kinds = {ln.id: ln.kind for ln in doc.lines}
    assert kinds == {"gtr_l": "guitar", "gtr_r": "guitar", "guitar_all": "reference", "bass": "bass"}


def test_one_guitar_part_frets_the_merged_guitar_and_has_no_part_tracks(tmp_path):
    q, _grid = _quant_with_lines()
    tab, _out = _run_tab(tmp_path, q, 1)
    assert tab["tracks"]["guitar_all"]["fretted"] and "gtr_l" not in tab["tracks"] and "gtr_r" not in tab["tracks"]


def test_quant_reads_the_guitar_from_the_split_stage(tmp_path):
    notes_dir, split_dir = tmp_path / "notes", tmp_path / "split"
    atomic.write_json(notes_dir / "guitar_all.json", {"notes": [{"onset_s": 1.0, "offset_s": 1.5, "pitch": 60}]})
    atomic.write_json(notes_dir / "bass_raw.json", {"notes": []})
    atomic.write_json(split_dir / T.LINES_JSON, {"notes": [{"onset_s": 1.0, "offset_s": 1.5, "pitch": 60,
                                                            "line": "R", "shared": False}]})
    assert T.quant_tracks({"notes": notes_dir})["guitar_all"][0].get("line") is None
    assert T.quant_tracks({"notes": notes_dir, "split": split_dir})["guitar_all"][0]["line"] == "R"


# --------------------------------------------------------------------------------------- edits


def make_split_job(root):
    """A finished run whose quant read the guitar from a split stage (lines L / R)."""
    job = root / "f-test"
    st = job / "stages"
    keys = {**KEYS, "split": ("sp" * 32)[:64]}
    d = {s: st / s / keys[s][:12] for s in keys}
    times = beat_times(12 * 4 + 4, warp=False)
    g = _raw(song(E_STD, n_bars=6), 0.5, "distorted_electric_guitar")
    for k, n in enumerate(g):
        n["line"], n["shared"] = ("L" if (k // 3) % 2 == 0 else "R"), False
    b = _raw(bass_line((28, 33, 38, 43), n=40), 0.5, "electric_bass")
    grid = grid_doc(times)
    sections = [{"id": "S1", "label": "A", "start_bar": 1, "end_bar": 14, "start_s": times[0],
                 "end_s": times[-1] + 0.5}]
    atomic.write_json(d["notes"] / "guitar_all.json", {"notes": [{k: v for k, v in n.items() if k not in
                                                                  ("line", "shared")} for n in g]})
    atomic.write_json(d["notes"] / "bass_raw.json", {"notes": b})
    atomic.write_json(d["split"] / T.LINES_JSON, {"notes": g})
    atomic.write_json(d["grid"] / "grid.json", grid)
    atomic.write_json(d["sections"] / "sections.json", {"sections": sections})
    qp = T.quant_params(None)
    atomic.write_json(d["quant"] / "quant.json",
                      Q.quantize(T.quant_tracks({"notes": d["notes"], "split": d["split"]}), grid, sections, qp))
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
            ("quant", qp, {"notes": keys["notes"], "split": keys["split"], "grid": keys["grid"],
                           "sections": keys["sections"]}),
            ("tab", tp, {"quant": keys["quant"], "a4": keys["a4"]}),
            ("score", sp, {k: keys[k] for k in ("tab", "quant", "grid", "sections")})):
        atomic.write_json(d[stage] / "manifest.json", {"stage": stage, "key": keys[stage], "song_key": "f-test",
                                                        "params": json.loads(json.dumps(params)), "deps": deps})
    exp = job / "export"
    files = {}
    for rel in ("tab/score.alphatex", "tab/score.gp5", "tab/gtr_l.txt", "tab/gtr_r.txt", "score.json"):
        if (d["score"] / rel).is_file():
            (exp / rel).parent.mkdir(parents=True, exist_ok=True)
            (exp / rel).write_bytes((d["score"] / rel).read_bytes())
            files[rel] = {"stage": "score", "key12": keys["score"][:12], "sha256": atomic.sha256_file(exp / rel)}
    atomic.write_json(exp / "export.json", {"format": "bandscribe.export/1", "files": files})
    return job, d


def test_a_click_on_a_part_track_edits_the_guitar_note_and_the_part_follows(tmp_path):
    job, d = make_split_job(tmp_path)
    assert (d["score"] / "tab" / "gtr_l.txt").is_file() and (d["score"] / "tab" / "gtr_r.txt").is_file()
    base = E.load_base(job)
    assert T.split_on(base.raw and {"tracks": {"guitar_all": {"notes": base.raw["guitar_all"]}}})
    built = E.build(base, [])
    assert {"gtr_l", "gtr_r"} <= set(built.state.tab["tracks"])
    # the first written note of the left part maps to a guitar note id
    pos0 = [u for k, u in built.map["pos"].items() if k.startswith("0:")]
    assert pos0 and all(u.startswith("guitar_all:") for u in pos0)
    uid = pos0[0]
    n_left = len(built.state.tab["tracks"]["gtr_l"]["notes"])
    s = E.Session(job)
    r = s.command({"cmd": "delete", "uid": uid})
    assert r["applied"] and len(s.state.tab["tracks"]["gtr_l"]["notes"]) == n_left - 1
    r = s.command({"cmd": "barline", "bar_index": s.score_bars.index(3), "delta": 1})
    assert r["applied"] and {"gtr_l", "gtr_r"} <= set(s.state.tab["tracks"])  # re-quantised from the split notes
    assert s.close()
    assert (job / "export" / "tab" / "gtr_l.txt").is_file()  # the edited parts are exported too


def test_the_string_key_on_a_part_note_pins_it_in_its_part(tmp_path):
    job, _d = make_split_job(tmp_path)
    s = E.Session(job)
    part_uids = [u for k, u in s.map["pos"].items() if k.startswith("0:")]  # written in Gtr 1 (L)
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
    job, _d = make_split_job(tmp_path)
    s = E.Session(job)
    assert [t["id"] for t in s.map["tracks"]] == ["gtr_l", "gtr_r", "guitar_all", "bass"]
    assert [t["fretted"] for t in s.map["tracks"]] == [True, True, False, True]
    for ti, t in enumerate(s.map["tracks"]):  # every position key of a track uses that track's key kind
        keys = [k.split(":")[3] for k in s.map["pos"] if k.startswith(f"{ti}:")]
        assert keys and all(k[0] == ("s" if t["fretted"] else "p") for k in keys)
    assert s.close()


# ------------------------------------------------------------------------------------------------ review fixes


def test_a_shared_note_is_written_in_both_parts():
    """Review 2026-10-10: a note the split writes to both sides (|ILD| under split_p_ild_db) is line "all" +
    shared; it must be in both part tracks, not in neither."""
    q, _grid = _quant_with_lines()
    g = Q.notes_of(q, T.GUITAR)
    g[0]["line"], g[0]["shared"] = "all", True
    left, right = T.track_notes(q, "gtr_l"), T.track_notes(q, "gtr_r")
    assert any(n["src_k"] == 0 for n in left) and any(n["src_k"] == 0 for n in right)


def test_split_song_with_a_shared_zone_writes_shared_notes(two_parts):
    notes, st = two_parts
    body = RS.split_song(notes, _win("R3"), st, R.SR, {**SPLIT_ON, "p_ild_db": 60.0})
    assert all(n["shared"] and n["line"] == "all" for n in body["notes"])  # every |ILD| is under 60 dB


def test_a_note_added_next_to_a_part_note_joins_that_part(tmp_path):
    job, _d = make_split_job(tmp_path)
    s = E.Session(job)
    uid = next(u for k, u in s.map["pos"].items() if k.startswith("1:"))  # a note written in Gtr 2 (R)
    n_right = len(s.state.tab["tracks"]["gtr_r"]["notes"])
    r = s.command({"cmd": "add", "track": "guitar_all", "t_s": 3.0, "pitch": 64, "uid": uid})
    assert r["applied"]
    added = [n for n in Q.notes_of(s.state.quant, "guitar_all") if n.get("added")]
    assert added and added[0].get("line") == "R"
    assert len(s.state.tab["tracks"]["gtr_r"]["notes"]) == n_right + 1
    r = s.command({"cmd": "add", "track": "guitar_all", "t_s": 5.0, "pitch": 64})  # nothing selected: no part
    assert r["applied"]
    assert any(n.get("added") and n.get("line") is None for n in Q.notes_of(s.state.quant, "guitar_all"))
    assert s.close()


def test_an_empty_stem_is_silent_not_mixed():
    from bandscribe.router import route as RT

    x = np.zeros((2, 16 * R.SR))
    bars = [{"bar": i, "start_s": 2.0 * i, "end_s": 2.0 * i + 2} for i in range(8)]
    doc = RT.route_song(x, R.SR, bars, None)
    assert {w["label"] for w in doc["windows"]} == {"silent"}
