"""M4a edits (bandscribe.tab.edits): the log, its replay, re-export and the viewer's edit API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from bandscribe import atomic
from bandscribe.tab import edits as E
from bandscribe.tab import fretting as F
from bandscribe.tab import quantize as Q
from bandscribe.tab import stage as T

from test_tab_quantize import beat_times, grid_doc
from test_tab_tuning import E_STD, bass_line, song

KEYS = {s: (s[:2] * 32)[:64] for s in ("notes", "grid", "sections", "quant", "a4", "tab", "score")}


def _raw(events, offset: float, instrument: str) -> list[dict]:
    return [{"onset_s": round(e.onset_s + offset, 4), "offset_s": round(off + offset, 4), "pitch": p,
             "instrument": instrument} for e in events for p, off in zip(e.pitches, e.offsets_s)]


def make_job(root: Path, *, guitar_parts: int = 1, shift_s: float = 0.0) -> Path:
    """A finished run of a synthetic song laid out like a real job (stage dirs with manifests, export/)."""
    job = root / "f-test"
    st = job / "stages"
    d = {s: st / s / KEYS[s][:12] for s in KEYS}
    times = beat_times(12 * 4 + 4, warp=False)
    raw = {"guitar_all": _raw(song(E_STD, n_bars=6), 0.5 + shift_s, "distorted_electric_guitar"),
           "bass_raw": _raw(bass_line((28, 33, 38, 43), n=40), 0.5 + shift_s, "electric_bass")}
    grid = grid_doc(times)
    sections = [{"id": "S1", "label": "A", "start_bar": 1, "end_bar": 7, "start_s": times[0], "end_s": times[24]},
                {"id": "S2", "label": "B", "start_bar": 7, "end_bar": 14, "start_s": times[24], "end_s": times[-1] + 0.5}]
    atomic.write_json(d["notes"] / "guitar_all.json", {"notes": raw["guitar_all"]})
    atomic.write_json(d["notes"] / "bass_raw.json", {"notes": raw["bass_raw"]})
    atomic.write_json(d["grid"] / "grid.json", grid)
    atomic.write_json(d["sections"] / "sections.json", {"sections": sections})
    qp = T.quant_params(None)
    atomic.write_json(d["quant"] / "quant.json", Q.quantize(T.quant_tracks({"notes": d["notes"]}), grid, sections, qp))
    atomic.write_json(d["a4"] / "a4.json", {"cents": 0.0, "a4_hz": 440.0, "confidence": 0.9})
    tp = T.tab_params({"tab.guitar_parts": guitar_parts})

    def ctx(stage: str, deps: dict, params: dict) -> SimpleNamespace:
        return SimpleNamespace(song_key="f-test", stage=stage, key=KEYS[stage], out_dir=d[stage],
                               dep_dirs={k: d[k] for k in deps}, params=params, config=None,
                               store=SimpleNamespace(job_dir=lambda key: job))

    T.run_tab(ctx("tab", ("quant", "a4"), tp))
    sp = T.score_params(None)
    atomic.write_json(job / "input" / "meta.json", {"source": {"filename": "song.wav"}, "duration_s": times[-1] + 1})
    T.run_score(ctx("score", ("tab", "quant", "grid", "sections"), sp))
    for stage, params, deps in (("quant", qp, {"notes": KEYS["notes"], "grid": KEYS["grid"], "sections": KEYS["sections"]}),
                                ("tab", tp, {"quant": KEYS["quant"], "a4": KEYS["a4"]}),
                                ("score", sp, {k: KEYS[k] for k in ("tab", "quant", "grid", "sections")})):
        atomic.write_json(d[stage] / "manifest.json", {"stage": stage, "key": KEYS[stage], "song_key": "f-test",
                                                        "params": json.loads(json.dumps(params)), "deps": deps})
    exp = job / "export"
    files = {}
    for rel in ("tab/score.alphatex", "tab/score.gp5", "tab/bass_raw.txt", "score.json"):
        if (d["score"] / rel).is_file():
            (exp / rel).parent.mkdir(parents=True, exist_ok=True)
            (exp / rel).write_bytes((d["score"] / rel).read_bytes())
            files[rel] = {"stage": "score", "key12": KEYS["score"][:12], "sha256": atomic.sha256_file(exp / rel)}
    atomic.write_json(exp / "export.json", {"format": "bandscribe.export/1", "files": files})
    return job


def _bass(s: E.Session) -> list[str]:
    return [u for u, n in s.map["notes"].items() if n["track"] == "bass_raw" and u in s.map["at"]]


def test_every_edit_replays_to_the_same_score_and_undo_restores_the_run(tmp_path):
    job = make_job(tmp_path)
    s = E.Session(job)
    base_hash, base_tex = s.hash, s.tex
    assert base_tex == (job / "export" / "tab" / "score.alphatex").read_text(encoding="utf-8")  # same writer, no edits
    b = _bass(s)
    r = s.command({"cmd": "pitch", "uid": b[5], "delta": 12})
    assert r["applied"] and s.map["notes"][r["select"]]["pitch"] == s.map["notes"][b[5]]["pitch"]
    r = s.command({"cmd": "string", "uid": r["select"], "dir": 1})
    pinned = s.map["notes"][r["select"]]
    assert pinned["pinned"]
    s.command({"cmd": "delete", "uid": b[10]})
    g = next(u for u, n in s.map["notes"].items() if n["track"] == "guitar_all" and n["pitch"] < 52)
    r = s.command({"cmd": "assign", "uid": g, "track": "bass_raw"})
    assert s.map["notes"][r["select"]]["track"] == "bass_raw"
    r = s.command({"cmd": "add", "track": "bass_raw", "t_s": 3.0, "pitch": 40})
    assert s.map["notes"][r["select"]]["added"]
    r = s.command({"cmd": "barline", "bar_index": s.score_bars.index(4), "delta": 1})
    assert r["applied"]
    s.command({"cmd": "undo"})
    s.command({"cmd": "redo"})
    assert s.info()["edits"] == 6 and s.state.report["unmatched"] == []
    h, tex = s.hash, s.tex
    assert s.close()
    # the log alone gives the same score (round trip through the file), and export/ is that score
    s2 = E.Session(job)
    assert (s2.hash, s2.tex) == (h, tex)
    assert (job / "export" / "tab" / "score.alphatex").read_text(encoding="utf-8") == tex
    exp = atomic.read_json(job / "export" / "export.json")
    assert exp["edits"]["state"] == h and exp["files"]["tab/score.gp5"]["stage"] == "edits"
    # every edit undone: the run's score again
    while s2.info()["undo"]:
        s2.command({"cmd": "undo"})
    assert (s2.hash, s2.tex) == (base_hash, base_tex)
    assert s2.close()
    exp = atomic.read_json(job / "export" / "export.json")  # every edit undone: the run's own files again
    assert "edits" not in exp and exp["files"]["tab/score.gp5"]["stage"] == "score"


def test_the_gp5_holds_the_edited_notes(tmp_path):
    import guitarpro as gp

    job = make_job(tmp_path)
    s = E.Session(job)
    b = _bass(s)
    s.command({"cmd": "delete", "uid": b[3]})
    s.command({"cmd": "pitch", "uid": b[6], "delta": 1})
    assert s.close()
    tr = s.state.tab["tracks"]["bass_raw"]
    want = sorted((n["string"], n["fret"]) for n in tr["notes"] if n["string"] is not None)
    song_ = gp.parse(str(job / "export" / "tab" / "score.gp5"))
    bass = next(t for t in song_.tracks if len(t.strings) == 4)
    got = sorted((n.string, n.value) for m in bass.measures for v in m.voices for be in v.beats for n in be.notes
                 if n.type != gp.NoteType.tie)
    assert got == want


def test_anchors_follow_a_rerun_and_report_what_is_gone(tmp_path):
    """A re-run that moved every onset 10 ms (and lost one note) still re-applies the other edits."""
    job = make_job(tmp_path / "a")
    s = E.Session(job)
    b = _bass(s)
    s.command({"cmd": "pitch", "uid": b[2], "delta": 1})
    s.command({"cmd": "delete", "uid": b[7]})
    assert s.close()
    rerun = make_job(tmp_path / "b", shift_s=0.01)
    (rerun / E.EDITS_FILE).write_bytes((job / E.EDITS_FILE).read_bytes())
    s2 = E.Session(rerun)
    assert s2.state.report["applied"] == 2 and s2.state.report["fuzzy"] == 2 and not s2.state.report["unmatched"]
    s2.close()
    gone = s.entries[0]["note"]
    notes = atomic.read_json(next((rerun / "stages" / "notes").iterdir()) / "bass_raw.json")
    notes["notes"] = [n for n in notes["notes"] if abs(n["onset_s"] - gone["onset_s"] - 0.01) > 1e-6]
    atomic.write_json(next((rerun / "stages" / "notes").iterdir()) / "bass_raw.json", notes)
    s3 = E.Session(rerun)
    assert [u["id"] for u in s3.state.report["unmatched"]] == ["e00001"] and s3.info()["unmatched"] == 1
    s3.close()


def test_a_bar_line_edit_moves_every_later_bar(tmp_path):
    grid = grid_doc(beat_times(5 * 4, warp=False))
    rep = {"applied": 0, "unmatched": []}
    g = E.edited_grid(grid, [{"id": "e1", "op": "barline", "at": 8, "to": 9}], rep)
    assert rep["applied"] == 1
    assert [b["n_beats"] for b in g["bars"]] == [4, 5, 4, 4, 3]  # bar 2 took the beat, the last bar gave it up
    assert E.bar_starts(g) == [0, 4, 9, 13, 17]
    assert (g["bars"][1]["numerator"], g["bars"][1]["denominator"]) == (5, 4)
    assert [b["beat"] for b in g["beats"][4:10]] == [1, 2, 3, 4, 5, 1]
    g2 = E.edited_grid(grid, [{"id": "e2", "op": "barline", "at": 6, "to": 7}], rep)  # no bar line at beat 6
    assert E.bar_starts(g2) == [0, 4, 8, 12, 16] and rep["unmatched"][-1]["id"] == "e2"


def test_a_tap_moves_the_nearest_bar_line_to_the_tapped_beat(tmp_path):
    job = make_job(tmp_path)
    s = E.Session(job)
    starts = E.bar_starts(s.state.grid)
    beat_t = [b["t_s"] for b in s.state.grid["beats"]]
    r = s.command({"cmd": "tap", "t_s": beat_t[starts[3]] + 0.01})
    assert not r["applied"] and "이미" in r["message"]
    s.command({"cmd": "tap", "t_s": beat_t[starts[3] - 1] - 0.02})  # one beat before a bar line: that line moves back
    assert starts[3] - 1 in E.bar_starts(s.state.grid) and starts[3] not in E.bar_starts(s.state.grid)
    s.close()


def test_pins_hold_and_the_rest_is_fretted_around_them():
    inst = F.Instrument((28, 33, 38, 43))
    evs = [F.Event([33 + i % 5], 0.25 * i, [0.25 * i + 0.2], [i]) for i in range(12)]
    free = {a.note_id: (a.string, a.fret) for a in F.assign(evs, inst)}
    alt = next(p for p in inst.positions(evs[5].pitches[0]) if p != free[5])
    pinned = {a.note_id: (a.string, a.fret) for a in F.assign(evs, inst, pins={5: alt})}
    assert pinned[5] == alt
    assert all(inst.open_pitch(s) + f == evs[i].pitches[0] for i, (s, f) in pinned.items())
    bad = F.assign(evs, inst, pins={5: (1, 0)})  # a position that cannot sound the pitch is ignored
    assert {a.note_id: (a.string, a.fret) for a in bad}[5] == free[5]


def test_a_torn_last_line_is_skipped_and_other_damage_refused(tmp_path):
    job = make_job(tmp_path)
    s = E.Session(job)
    s.command({"cmd": "delete", "uid": _bass(s)[0]})
    assert s.close()
    p = job / E.EDITS_FILE
    p.write_text(p.read_text(encoding="utf-8") + '{"id": "e00002", "op": "del', encoding="utf-8")
    assert len(E.read_log(job)) == 1
    p.write_text('not json\n' + p.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(E.EditError, match="1번째 줄"):
        E.read_log(job)


def test_reapply_after_a_run_rewrites_the_export(tmp_path):
    job = make_job(tmp_path)
    s = E.Session(job)
    s.command({"cmd": "delete", "uid": _bass(s)[1]})
    assert s.close()
    stage_tex = (next((job / "stages" / "score").iterdir()) / "tab" / "score.alphatex").read_text(encoding="utf-8")
    (job / "export" / "tab" / "score.alphatex").write_text(stage_tex, encoding="utf-8")  # what publish puts back
    info = E.reapply(job)
    assert info["applied"] == 1 and info["unmatched"] == 0
    assert (job / "export" / "tab" / "score.alphatex").read_text(encoding="utf-8") == s.tex
    assert E.reapply(make_job(tmp_path / "clean")) is None  # no log: nothing to do


def test_viewer_edit_api_needs_the_token_and_the_servers_host(tmp_path):
    from bandscribe.tab import viewer

    job = make_job(tmp_path)
    (job / "input" / "source.wav").write_bytes(b"RIFF")
    at = tmp_path / "at"
    at.mkdir()
    (at / "alphaTab.min.js").write_text("//", encoding="utf-8")
    httpd, t = viewer.serve(job, alphatab=at)
    try:
        assert httpd.server_address[0] == "127.0.0.1" and httpd.session is not None
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode("utf-8")
        token = page.split('const TOKEN = "', 1)[1].split('"', 1)[0]
        snap = json.loads(urllib.request.urlopen(base + "/api/score").read())
        uid = next(u for u, n in snap["map"]["notes"].items() if n["track"] == "bass_raw")

        def post(body: dict, headers: dict) -> tuple[int, dict]:
            req = urllib.request.Request(base + "/api/cmd", data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", **headers})
            try:
                r = urllib.request.urlopen(req)
                return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        cmd = {"cmd": "delete", "uid": uid, "version": snap["version"]}
        assert post(cmd, {})[0] == 403  # no token
        assert post(cmd, {"X-Bandscribe-Token": token, "Host": "evil.example:80"})[0] == 403
        code, res = post(cmd, {"X-Bandscribe-Token": token})
        assert code == 200 and res["applied"] and res["edits"] == 1
        assert post(cmd, {"X-Bandscribe-Token": token})[0] == 409  # the page's version is old now
        code, res = post({"cmd": "nonsense", "version": res["version"]}, {"X-Bandscribe-Token": token})
        assert code == 400 and "알 수 없는" in res["error"]
        req = urllib.request.Request(base + "/api/score", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(req)
        r = urllib.request.urlopen(base + "/")  # no other site may frame the editor
        assert r.headers["X-Frame-Options"] == "DENY" and "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
    finally:
        httpd.shutdown()
        t.join(timeout=5)
        httpd.session.close()


# ------------------------------------------------------------------------------- review 2026-10-05 (M4a)


def _pickup_grid(n_pickup: int = 1, n_bars: int = 4) -> dict:
    """grid.json like the grid stage writes it: a pickup bar with its own beat count and the full numerator."""
    times = beat_times(n_pickup + 4 * n_bars, warp=False)
    beats, bars = [], [{"bar": 0, "start_s": times[0], "end_s": times[n_pickup], "n_beats": n_pickup, "numerator": 4,
                        "denominator": 4, "beat_unit": "quarter", "pickup": True}]
    for i, t in enumerate(times):
        bar, pos = (0, i + 1) if i < n_pickup else (1 + (i - n_pickup) // 4, 1 + (i - n_pickup) % 4)
        beats.append({"t_s": t, "t_raw_s": t, "bar": bar, "beat": pos, "is_downbeat": pos == 1, "shift_ms": 0.0,
                      "activation": None})
    for b in range(n_bars):
        s = n_pickup + 4 * b
        e = times[s + 4] if s + 4 < len(times) else times[-1] + 0.5
        bars.append({"bar": b + 1, "start_s": times[s], "end_s": e, "n_beats": 4, "numerator": 4, "denominator": 4,
                     "beat_unit": "quarter", "pickup": False})
    return {"format": "bandscribe.grid/1", "tpb": 48, "duration_s": times[-1] + 1, "beats": beats, "bars": bars,
            "tempo": [{"start_bar": 0, "end_bar": n_bars + 1, "bpm": 120.0}], "meter": [], "beat_unit": "quarter",
            "pickup": {"present": True, "beats": n_pickup}, "hypotheses": {}, "confidence": {}, "params": {}}


def test_a_bar_line_edit_keeps_the_pickup_a_pickup():
    """Review: the pickup lost its flag and was written 4/4 (one beat in a 4/4 bar) after any bar-line edit."""
    from bandscribe.schema.grid import TempoMap
    from bandscribe.tab import notation as N

    grid = _pickup_grid()
    rep = {"applied": 0, "unmatched": []}
    g = E.edited_grid(grid, [{"id": "e1", "op": "barline", "at": 9, "to": 10}], rep)
    assert rep["applied"] == 1 and g["bars"][0]["pickup"] and g["bars"][0]["n_beats"] == 1
    q = Q.quantize({"guitar_all": [{"onset_s": grid["beats"][2]["t_s"], "offset_s": grid["beats"][2]["t_s"] + 0.2,
                                    "pitch": 52}]}, g)
    bars = N.bars_from_quant(q, TempoMap.from_dict(g), 0, 4, [])
    assert (bars[0].numerator, bars[0].denominator) == (1, 4)  # the pickup's own length, as before the edit
    g2 = E.edited_grid(grid, [{"id": "e2", "op": "barline", "at": 1, "to": 2}], rep)  # the pickup grows to 2 beats
    assert g2["bars"][0]["pickup"] and g2["bars"][0]["n_beats"] == 2


def test_bar_line_edits_follow_the_beat_times_not_the_indices():
    """Review: a re-run whose grid gained a beat at the start must not move another bar line."""
    import copy

    grid = grid_doc(beat_times(5 * 4, warp=False))
    t = [b["t_s"] for b in grid["beats"]]
    move = {"id": "e1", "op": "barline", "at": 8, "to": 9, "at_s": t[8], "to_s": t[9]}
    shifted = copy.deepcopy(grid)  # the same bar lines plus a pickup beat in front: every index is one higher
    t0 = t[0] - 0.5
    shifted["beats"].insert(0, {"t_s": t0, "t_raw_s": t0, "bar": 0, "beat": 1, "is_downbeat": True, "shift_ms": 0.0,
                                "activation": None})
    shifted["bars"].insert(0, {"bar": 0, "start_s": t0, "end_s": t[0], "n_beats": 1, "numerator": 4, "denominator": 4,
                               "beat_unit": "quarter", "pickup": True})
    rep = {"applied": 0, "unmatched": []}
    g = E.edited_grid(shifted, [move], rep)
    assert rep["applied"] == 1 and 10 in E.bar_starts(g) and 9 not in E.bar_starts(g)
    far = dict(move, at_s=t[8] + 0.2)  # no beat within a third of a beat: unmatched, not guessed
    rep2 = {"applied": 0, "unmatched": []}
    E.edited_grid(grid, [far], rep2)
    assert rep2["applied"] == 0 and rep2["unmatched"][0]["id"] == "e1"


def test_an_edit_that_cannot_be_written_is_refused_and_not_logged(tmp_path):
    """Review: deleting the last note logged the edit and then broke every later start of the viewer."""
    job = make_job(tmp_path)
    s = E.Session(job)
    for u in [u for u, n in s.map["notes"].items() if n["track"] == "guitar_all"][::-1]:
        s.command({"cmd": "assign", "uid": u, "track": "bass_raw"})
    n_log, version, h = len(E.read_log(job)), s.version, s.hash
    left = sorted(s.map["notes"], key=lambda u: int(u.split(":")[1]))
    for u in left[:-1][::-1]:
        s.command({"cmd": "delete", "uid": u})
    last = next(iter(s.map["notes"]))
    with pytest.raises(E.EditError, match="적용할 수 없습니다"):
        s.command({"cmd": "delete", "uid": last})
    assert len(E.read_log(job)) == n_log + len(left) - 1  # the refused delete is not in the log
    assert s.map["notes"] and s.close()
    s2 = E.Session(job)  # and the job still opens
    assert s2.info()["skipped"] == 0 and len(s2.map["notes"]) == 1
    assert (version, h) != (s2.version, s2.hash)
    s2.close()


def test_a_torn_line_is_repaired_before_the_next_edit_and_ids_never_repeat(tmp_path):
    job = make_job(tmp_path)
    s = E.Session(job)
    s.command({"cmd": "delete", "uid": _bass(s)[0]})
    s.close()
    p = job / E.EDITS_FILE
    p.write_text(p.read_text(encoding="utf-8") + '{"id": "e00002", "op": "del', encoding="utf-8")  # crash mid-append
    s = E.Session(job)
    s.command({"cmd": "delete", "uid": _bass(s)[0]})
    s.close()
    log_ = E.read_log(job)
    assert [e["id"] for e in log_] == ["e00001", "e00002"] and all(e["op"] == "delete" for e in log_)
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")  # a line removed by hand
    assert E.new_record({"op": "delete"}, E.read_log(job))["id"] == "e00003"


def test_one_viewer_edits_a_job_at_a_time(tmp_path):
    job = make_job(tmp_path)
    s = E.Session(job)
    with pytest.raises(E.EditError, match="다른 뷰어"):
        E.Session(job)
    s.close()
    E.Session(job).close()


def test_a_session_follows_a_newer_run_instead_of_writing_the_old_one(tmp_path):
    """Review: a viewer left open across `bandscribe run` wrote the old run's score over the new export."""
    import shutil

    job = make_job(tmp_path)
    s = E.Session(job)
    s.command({"cmd": "delete", "uid": _bass(s)[0]})
    assert s.flush()
    # a new run: another score stage dir, published to export/ (as publish does: the stage's own files)
    old = next((job / "stages" / "score").iterdir())
    new_key = "f" * 64
    new = old.parent / new_key[:12]
    shutil.copytree(old, new)
    exp = atomic.read_json(job / "export" / "export.json")
    exp.pop("edits")
    exp["files"] = {r: {"stage": "score", "key12": new_key[:12]} for r in ("score.json", "tab/score.alphatex")}
    atomic.write_json(job / "export" / "export.json", exp)
    s._export(s.state, s.hash)  # the background writer must not put the old run back
    assert E.export_score_key(job) == new_key[:12] and s.export_error  # refused, the new export untouched
    with pytest.raises(E.StaleVersion):
        s.command({"cmd": "delete", "uid": _bass(s)[0]})
    assert s.base.keys["score"] == new_key[:12] and s.info()["edits"] == 1  # reloaded, the log re-applied
    s.close()


def test_emptying_the_log_restores_the_runs_export(tmp_path):
    job = make_job(tmp_path)
    stage_tex = (next((job / "stages" / "score").iterdir()) / "tab" / "score.alphatex").read_bytes()
    s = E.Session(job)
    s.command({"cmd": "delete", "uid": _bass(s)[2]})
    s.close()
    assert (job / "export" / "tab" / "score.alphatex").read_bytes() != stage_tex
    (job / E.EDITS_FILE).unlink()
    E.Session(job).close()
    assert (job / "export" / "tab" / "score.alphatex").read_bytes() == stage_tex
    assert "edits" not in atomic.read_json(job / "export" / "export.json")
