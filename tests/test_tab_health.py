"""Song health (mid-term plan 1-4, 1-5, 1-8, 1-9): the checks, the document and the viewer route."""

from __future__ import annotations

import json
import urllib.request

import numpy as np

from bandscribe import atomic
from bandscribe.tab import health as H


def _notes(t0, t1, step, dur, pitch=40):
    t, out = t0, []
    while t < t1:
        out.append({"onset_s": round(t, 3), "offset_s": round(t + dur, 3), "pitch": pitch})
        t += step
    return out


def test_burst_spans_flags_tiny_note_floods_and_dense_windows_only():
    normal = _notes(0.0, 150.0, 0.5, 0.3)                 # 20 notes per 10 s window, normal length
    burst = _notes(150.0, 180.0, 0.05, 0.01)              # 200 tiny notes per window
    spans, stats = H.burst_spans(normal + burst, 200.0)
    assert spans == [[150.0, 180.0]] and stats["tiny_in_flagged"] >= 590
    dense = _notes(60.0, 70.0, 0.1, 0.08)                 # 100 notes in one window, 5x the median
    spans2, _ = H.burst_spans(normal + dense, 160.0)
    assert spans2 == [[60.0, 70.0]]
    assert H.burst_spans(normal, 160.0)[0] == [] and H.burst_spans([], 10.0)[0] == []


def test_bleed_risk_marks_bars_where_piano_other_is_as_loud_as_the_guitar():
    fps, n = 10.0, 200
    g = np.full(n, -20.0)
    piano = np.full(n, -60.0)
    other = np.full(n, -60.0)
    other[100:] = -19.0                                   # loud strings from 10 s
    grid = {"bars": [{"start_s": float(a), "end_s": float(a + 2)} for a in range(0, 20, 2)]}
    spans, share = H.bleed_risk({"fps": np.array([fps]), "guitar": g, "piano": piano, "other": other}, grid)
    assert spans == [[10.0, 20.0]] and share == 0.5


def _job(tmp_path, *, guitar_rel_db=-8.0, bass_burst=False, by="onsets", bass_dropped=40):
    job = tmp_path / "f-test"
    dirs = {k: tmp_path / "stages" / k for k in ("grid", "stems", "bass", "parts", "router")}
    for d in dirs.values():
        d.mkdir(parents=True)
    atomic.write_json(dirs["grid"] / "grid.json", {
        "duration_s": 200.0, "bars": [{"start_s": float(a), "end_s": float(a + 2)} for a in range(0, 200, 2)],
        "hypotheses": {"tempo_octave": {"decision": "keep", "by": by, "scores": {
            "keep": {"bpm": 120.0, "score": 0.30}, "double": {"bpm": 240.0, "score": 0.25}, "half": {"bpm": 60.0, "score": 0.1}}}}})
    atomic.write_json(dirs["stems"] / "stem_stats.json", {"stems": {"guitar": {"rel_db": guitar_rel_db}}})
    e = {"fps": np.array([10.0]), "guitar": np.full(2000, -20.0), "piano": np.full(2000, -60.0), "other": np.full(2000, -60.0)}
    e["other"][1000:] = -18.0
    np.savez(dirs["stems"] / "energy.npz", **e)
    bass = _notes(0.0, 150.0, 0.5, 0.3) + (_notes(150.0, 180.0, 0.05, 0.01) if bass_burst else [])
    atomic.write_json(dirs["bass"] / "bass.json", {"notes": bass})
    atomic.write_json(dirs["parts"] / "guitar_lines.json", {"notes": _notes(0.0, 190.0, 0.5, 0.3, 60)})
    atomic.write_json(dirs["router"] / "router.json", {"windows": [
        {"start_s": 0.0, "end_s": 100.0, "label": "R3"}, {"start_s": 100.0, "end_s": 200.0, "label": "R2"}]})
    atomic.write_json(job / "export" / "tab" / "tab.json", {"tuning": {
        "guitar": {"label": "Eb standard", "margin": -0.05, "top": [{"label": "E standard"}, {"label": "Eb standard"}],
                   "runner_up": {"label": "E standard"}},
        "bass": {"label": "E standard", "margin": 0.3, "top": [{"label": "E standard"}]}},
        "tracks": {"bass": {"check": {"notes": 100, "dropped": bass_dropped, "octave_shifted": 5}},
                   "gtr_1": {"check": {"notes": 100, "dropped": 2}}}})
    atomic.write_json(job / "input" / "meta.json", {"source": {"filename": "song_backing.mp4"}})
    return job, dirs


def test_song_health_items(tmp_path):
    job, dirs = _job(tmp_path, guitar_rel_db=-25.0, bass_burst=True)
    doc = H.song_health(job, results=dirs)
    ids = {i["id"]: i for i in doc["items"]}
    assert ids["guitar_removed"]["level"] == "warn" and doc["metrics"]["guitar_rel_db"] == -25.0
    assert ids["file_backing"]["level"] == "info"
    assert ids["burst_bass"]["level"] == "warn" and ids["burst_bass"]["spans"] == [[150.0, 180.0]]
    assert "burst_guitar_all" not in ids
    assert ids["unplaced_bass"]["level"] == "warn" and "unplaced_gtr_1" not in ids
    assert ids["tuning_guitar"]["level"] == "info" and "E standard" in ids["tuning_guitar"]["title"]
    assert "tuning_bass" not in ids
    assert ids["tempo"]["level"] == "info" and "확인 안 됨" in ids["tempo"]["title"]
    assert ids["bleed_risk"]["spans"] == [[100.0, 200.0]] and doc["metrics"]["bleed_risk_bar_share"] == 0.5
    assert ids["merged_parts"]["spans"] == [[100.0, 200.0]]
    assert [i["level"] for i in doc["items"]] == sorted((i["level"] for i in doc["items"]), key=lambda x: x != "warn")
    assert doc["summary"]["warn"] == 3
    # a normal original: no warning at all; a hinted tempo is plain information
    job2, dirs2 = _job(tmp_path / "ok", by="hint", bass_dropped=3)
    doc2 = H.song_health(job2, results=dirs2)
    assert doc2["summary"]["warn"] == 0
    assert next(i for i in doc2["items"] if i["id"] == "tempo")["title"].endswith("(힌트로 정함)")


def test_viewer_serves_health(tmp_path):
    from test_viewer_m4b import media_job

    from bandscribe.tab import viewer

    job = media_job(tmp_path)
    at = tmp_path / "at"
    at.mkdir()
    (at / "alphaTab.min.js").write_text("//", encoding="utf-8")
    httpd, t = viewer.serve(job, alphatab=at)
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        h = json.loads(urllib.request.urlopen(base + "/api/health").read())
        assert h["format"] == H.FORMAT and "items" in h
        assert any(i["id"] == "merged_parts" for i in h["items"])  # the fixture's router has an R0 window
        assert (job / "export" / H.HEALTH_FILE).is_file()           # written once, then reused
        page = urllib.request.urlopen(base + "/").read().decode("utf-8")
        assert 'id="health"' in page and '"api/health"' in page
    finally:
        httpd.shutdown()
        t.join(timeout=5)
        httpd.session.close()


def test_a_kept_automatic_tempo_reads_confirmed_and_hints_changes_rebuild_the_file(tmp_path, monkeypatch):
    from bandscribe.jobs import hints as HI
    from bandscribe.tab import edits as E

    job, dirs = _job(tmp_path)
    monkeypatch.setattr(H, "stage_dirs", lambda job_dir: dirs)
    monkeypatch.setattr(E, "export_score_key", lambda job_dir: None)
    doc = H.load_health(job)
    assert "확인 안 됨" in next(i for i in doc["items"] if i["id"] == "tempo")["title"]
    assert H.load_health(job)["hints_sha"] is None
    HI.set_hint(job, "hints", "tempo_confirmed", True)       # run asked, the user pressed Enter
    doc = H.load_health(job)                                  # same score: rebuilt because hints.toml changed
    assert next(i for i in doc["items"] if i["id"] == "tempo")["title"] == "템포 120 BPM (자동 판단, 확인함)"
