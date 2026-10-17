"""M4b viewer: media lookup (stems, CQT, unkept candidates, router windows), the note map's timing and part fields,
the review list of edits that did not land, the new routes, and the slowed-playback shifter's pitch (Node)."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest

from bandscribe import atomic
from bandscribe.tab import edits as E
from bandscribe.tab import viewer_media as VM

from test_tab_edits import make_job

ROOT = Path(__file__).resolve().parents[1]


def _wav(path: Path, x: np.ndarray, sr: int = 22050) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.atleast_2d(x).T, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(y.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((y * 32767).astype("<i2").tobytes())


def _tone(midi: int, seconds: float = 2.0, sr: int = 22050) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return 0.4 * np.sin(2 * np.pi * 440.0 * 2 ** ((midi - 69) / 12) * t)


def media_job(root: Path) -> Path:
    """``make_job`` plus the stages a real run has behind its notes: sep (stems), stems (practice mixes), amt raw
    notes and the router - wired through the notes manifest's deps."""
    job = make_job(root)
    st = job / "stages"
    keys = {k: (k[:2] * 40)[:64] for k in ("sep", "stems", "amt_gtr", "amt_bp", "amt_ms1", "router")}
    for k, key in keys.items():
        (st / k / key[:12]).mkdir(parents=True, exist_ok=True)
        atomic.write_json(st / k / key[:12] / "manifest.json", {"deps": {"sep": keys["sep"]} if k != "sep" else {}})
    _wav(st / "sep" / keys["sep"][:12] / "stems" / "guitar.wav", _tone(69))
    _wav(st / "sep" / keys["sep"][:12] / "stems" / "bass.wav", _tone(40))
    _wav(st / "stems" / keys["stems"][:12] / "practice" / "mix_minus_guitar.wav", _tone(40))
    notes_dir = st / "notes" / E.load_base(job).keys["notes"]
    kept = atomic.read_json(notes_dir / "guitar_all.json")["notes"]
    k0 = kept[0]
    atomic.write_json(st / "amt_gtr" / keys["amt_gtr"][:12] / "raw" / "guitar_mono__muscriptor.json", {"notes": [
        dict(k0), {"onset_s": 9.0, "offset_s": 9.2, "pitch": 70, "instrument": "piano"}]})
    atomic.write_json(st / "amt_bp" / keys["amt_bp"][:12] / "raw" / "guitar_mono__basicpitch.json", {"notes": [
        {**k0, "onset_s": k0["onset_s"] + 0.02}, {"onset_s": 9.5, "offset_s": 9.7, "pitch": 71}]})
    atomic.write_json(st / "router" / keys["router"][:12] / "router.json", {"windows": [
        {"start_s": 0.0, "end_s": 4.0, "label": "R3", "section": "S1"}, {"start_s": 4.0, "end_s": 8.0, "label": "R0"}]})
    atomic.write_json(notes_dir / "manifest.json", {"deps": {k: keys[k] for k in ("amt_gtr", "amt_bp", "amt_ms1",
                                                                                 "stems", "router")}})
    _wav(job / "input" / "source.wav", _tone(57))
    return job


def test_stage_dirs_sources_candidates_and_router(tmp_path):
    job = media_job(tmp_path)
    dirs = VM.stage_dirs(job, E.export_score_key(job))
    assert {"score", "tab", "quant", "notes", "sep", "stems", "amt_gtr", "amt_bp", "router"} <= set(dirs)
    srcs = VM.sources(job, dirs, job / "input" / "source.wav")
    assert [s["id"] for s in srcs] == ["mix", "stem_guitar", "stem_bass", "mix_minus_guitar"]
    assert [s["on"] for s in srcs] == [True, False, False, False]
    from bandscribe.tab import stage as T

    cands = VM.candidates(dirs, T.quant_tracks(dirs))
    # the kept note (exact for MuScriptor, within 50 ms for Basic Pitch) is not a candidate; the other two are
    assert sorted((c["src"], c["pitch"]) for c in cands) == [("basicpitch", 71), ("muscriptor", 70)]
    assert [w["label"] for w in VM.router_windows(dirs)] == ["R3", "R0"]


def test_cqt_puts_a_tone_on_its_row_and_is_cached(tmp_path):
    job = media_job(tmp_path)
    dirs = VM.stage_dirs(job, E.export_score_key(job))
    src = next(s for s in VM.sources(job, dirs, None) if s["id"] == "stem_guitar")
    data, head = VM.cqt(job, src)
    img = np.frombuffer(data, dtype=np.uint8).reshape(head["bins"], head["frames"])
    assert head["bins"] == VM.CQT_SEMITONES * VM.CQT_BINS_PER_SEMITONE and head["min_midi"] == VM.CQT_MIN_MIDI
    row = int(np.argmax(img[:, head["frames"] // 2]))
    assert row == (69 - VM.CQT_MIN_MIDI) * VM.CQT_BINS_PER_SEMITONE + 1  # bin 1 is centred on min_midi
    cached = list((job / "cache" / "viewer").glob("cqt_stem_guitar_*.u8"))
    assert len(cached) == 1 and VM.cqt(job, src)[0] == data


def test_note_map_has_times_parts_and_a_timeline(tmp_path):
    job = make_job(tmp_path)
    s = E.Session(job)
    try:
        m = s.map
        n = next(v for v in m["notes"].values() if v["track"] == "guitar_all")
        assert {"t", "d", "on", "line", "p", "amber", "fc"} <= set(n)
        assert n["line"] == "1" and n["p"] == 1.0 and n["d"] > 0 and abs(n["t"] - n["on"]) < 0.25
        tl = m["timeline"]
        assert tl["bars"] and tl["beats"] and [s_["label"] for s_ in tl["sections"]] == ["A", "B"]
        assert tl["lines"] == [{"number": "1", "name": "Gtr - Rhythm (dist.)", "short": "Gtr"}]
        assert all(b[2] > b[1] for b in tl["bars"]) and tl["beats"] == sorted(tl["beats"])
    finally:
        s.close()


def test_edits_that_do_not_land_are_listed_with_their_place(tmp_path):
    job = make_job(tmp_path)
    E.append(job, {"op": "pitch", "note": {"track": "bass", "onset_s": 77.7, "pitch": 40}, "to": 41})
    s = E.Session(job)
    try:
        info = s.info()
        assert info["unmatched"] == 1
        assert info["not_applied"] == [{"id": "e00001", "op": "pitch", "why": "unmatched", "t_s": 77.7, "pitch": 40,
                                        "track": "bass"}]
    finally:
        s.close()


def test_viewer_m4b_routes(tmp_path):
    from bandscribe.tab import viewer

    job = media_job(tmp_path)
    at = tmp_path / "at"
    at.mkdir()
    (at / "alphaTab.min.js").write_text("//", encoding="utf-8")
    httpd, t = viewer.serve(job, alphatab=at)
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        m = json.loads(urllib.request.urlopen(base + "/api/media").read())
        assert [s["id"] for s in m["sources"]] == ["mix", "stem_guitar", "stem_bass", "mix_minus_guitar"]
        assert all("file" not in s for s in m["sources"]) and m["cqt_default"] == "stem_guitar"
        assert [w["label"] for w in m["router"]] == ["R3", "R0"]
        r = urllib.request.urlopen(urllib.request.Request(base + "/media/stem_guitar", headers={"Range": "bytes=0-3"}))
        assert r.status == 206 and r.read() == b"RIFF"
        r = urllib.request.urlopen(base + "/api/cqt/stem_guitar")
        head = json.loads(r.headers["X-Cqt"])
        assert len(r.read()) == head["bins"] * head["frames"]
        assert len(json.loads(urllib.request.urlopen(base + "/api/candidates").read())["notes"]) == 2
        for name in ("engine.js", "synth.js", "roll.js", "stretch_core.js", "stretch.worklet.js"):
            assert urllib.request.urlopen(base + f"/static/{name}").headers["Content-Type"].startswith("text/javascript")
        for bad in ("/static/index.html", "/static/../viewer.py", "/static/../viewer_media.js", "/static/..%2Fedits.js",
                    "/static/%2E%2E/%2E%2E/alphatex.js", "/media/nope", "/media/../input/meta.json", "/api/cqt/nope"):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(base + bad)
        for route in ("/api/media", "/media/mix", "/api/cqt/mix", "/api/candidates", "/static/engine.js"):
            req = urllib.request.Request(base + route, headers={"Host": "evil.example"})
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req)
            assert ei.value.code == 403
        # the original is the decoded mix the stems come from, when there is one
        (job / "input" / "mix_44k_f32.wav").write_bytes((job / "input" / "source.wav").read_bytes())
        assert VM.original(job, job / "input" / "source.wav") == job / "input" / "mix_44k_f32.wav"
        page = urllib.request.urlopen(base + "/").read().decode("utf-8")
        assert 'import { Engine } from "/static/engine.js"' in page and '<audio id=' not in page
    finally:
        httpd.shutdown()
        t.join(timeout=5)
        httpd.session.close()


@pytest.mark.node
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js 없음")
def test_engine_position_logic():
    """engine.js on a fake AudioContext: a source decoded during the count-in starts with the others, the heard
    position, a rate change keeps the playing position, loops are clamped and never empty, removal stops a source,
    synth buffers are padded to the recording, a pause during play()'s setup wins (M4b review)."""
    out = subprocess.run([shutil.which("node"), str(ROOT / "tests" / "node" / "engine_test.mjs")],
                         capture_output=True, text=True, timeout=60, check=True).stdout
    checks = json.loads(out)
    assert len(checks) >= 11 and all(checks.values()), checks


@pytest.mark.node
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js 없음")
def test_slowed_playback_keeps_the_pitch():
    """DESIGN 10 M4b "50–100 % 감속에서 피치가 보존된다": tones played at 50/60/75/90 % through the shifter."""
    out = subprocess.run([shutil.which("node"), str(ROOT / "tests" / "node" / "stretch_test.mjs")],
                         capture_output=True, text=True, timeout=120, check=True).stdout
    rows = json.loads(out)
    assert {r["rate"] for r in rows} == {0.5, 0.6, 0.75, 0.9}
    for r in rows:
        assert abs(r["cents"]) <= 1.0, r
        assert 0.9 <= r["rms_min"] and r["rms_max"] <= 1.1, r
