"""Basic Pitch worker end-to-end in the bp310 env (Python 3.10, onnxruntime). Skipped without that env."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from gtab import atomic, gpu
from gtab.amt import stage as S

BP = S.bp310_python()
pytestmark = [pytest.mark.bp310, pytest.mark.slow,
              pytest.mark.skipif(not BP.is_file(), reason="bp310 env not installed")]
SR = 44100


def _run(req: dict, work: Path):
    base = {"format": "gtab.worker/1", "job": None, "out_dir": str(work.parent), "device": "cpu", "threads": 4}
    res = gpu.run_worker("amt_basicpitch", {**base, **req}, work, python=BP, use_lock=False, timeout_s=600)
    assert res.ok, (res.error, res.stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:])
    return res.output


def _melody_wav(path: Path) -> None:
    t = np.arange(int(SR * 5.0)) / SR
    x = np.zeros_like(t)

    def tone(midi: int, t0: float, dur: float, glide: float = 0.0) -> None:
        idx = (t >= t0) & (t < t0 + dur)
        tt = t[idx] - t0
        f = 440 * 2 ** ((midi - 69) / 12) * 2 ** (glide * tt / dur / 12)
        ph = 2 * np.pi * np.cumsum(f) / SR
        x[idx] += sum(np.sin(h * ph) / h for h in range(1, 6)) * np.exp(-tt * 3) * 0.3

    for k, m in enumerate([60, 64, 67, 72]):
        tone(m, k * 0.5, 0.45)
    tone(62, 2.2, 1.5, glide=2.0)  # a two-semitone bend
    sf.write(str(path), x.astype(np.float32), SR, subtype="FLOAT")


def test_probe_runs_on_python_310(tmp_path):
    out = _run({"mode": "probe"}, tmp_path / "_worker")
    assert out["python_version"][:2] == [3, 10]
    assert out["basic_pitch_version"] == "0.4.0"
    assert out["model_path"].endswith("nmp.onnx")


def test_melody_has_notes_with_bends_and_is_byte_identical(tmp_path):
    wav = tmp_path / "view.wav"
    _melody_wav(wav)
    views = [{"id": "guitar_mono", "wav": str(wav), "out": str(tmp_path / "a" / "raw.json"), "view_sha256": None}]
    out = _run({"mode": "transcribe", "views": views, "params": {"min_note_frames": 4}}, tmp_path / "a" / "_worker")
    assert out["min_note_len_frames"] == 3 and out["min_note_frames_kept"] == 4 and out["threads"] == 4
    doc = atomic.read_json(tmp_path / "a" / "raw.json")
    assert doc["format"] == "gtab.notes/1" and doc["backend"]["params"]["min_note_frames"] == 4
    pitches = {n["pitch"] for n in doc["notes"]}
    assert {60, 64, 67, 72} <= pitches
    assert any("bend" in n for n in doc["notes"])
    assert all(n["amplitude"] > 0 for n in doc["notes"])
    try:
        from gtab.schema.notes import NoteSet

        NoteSet.model_validate(doc)
    except ImportError:
        pass
    views[0]["out"] = str(tmp_path / "b" / "raw.json")
    _run({"mode": "transcribe", "views": views, "params": {"min_note_frames": 4}}, tmp_path / "b" / "_worker")
    assert (tmp_path / "a" / "raw.json").read_bytes() == (tmp_path / "b" / "raw.json").read_bytes()


def test_sweep_loads_once_and_matches_separate_runs(tmp_path):
    wav = tmp_path / "view.wav"
    _melody_wav(wav)
    settings = [{"min_note_frames": 3, "onset_threshold": 0.4}, {"min_note_frames": 5},
                {"min_note_frames": 12, "frame_threshold": 0.2}]
    views = [{"id": "g", "wav": str(wav), "out": str(tmp_path / "sw" / "g.json"), "view_sha256": None}]
    out = _run({"mode": "sweep", "views": views, "settings": settings, "allow_default_min_len": True},
               tmp_path / "sw" / "_worker")
    assert out["model_loads"] == 1 and out["inference_runs"] == 1
    assert len(out["views"][0]["outs"]) == 3
    for k, s in enumerate(settings):
        one = [{"id": "g", "wav": str(wav), "out": str(tmp_path / f"t{k}" / "g.json"), "view_sha256": None}]
        _run({"mode": "transcribe", "views": one, "params": s, "allow_default_min_len": True},
             tmp_path / f"t{k}" / "_worker")
        assert (tmp_path / "sw" / f"g__s{k}.json").read_bytes() == (tmp_path / f"t{k}" / "g.json").read_bytes()


def test_default_min_length_is_refused_by_the_worker(tmp_path):
    wav = tmp_path / "view.wav"
    _melody_wav(wav)
    views = [{"id": "g", "wav": str(wav), "out": str(tmp_path / "g.json"), "view_sha256": None}]
    res = gpu.run_worker("amt_basicpitch", {"mode": "transcribe", "views": views, "params": {"min_note_frames": 12},
                                            "out_dir": str(tmp_path)}, tmp_path / "_worker", python=BP,
                         use_lock=False, timeout_s=300)
    assert not res.ok and "127.7" in (res.error or "")


def _model_sha() -> str:
    return hashlib.sha256(S.bp_model_path().read_bytes()).hexdigest()


def test_min_note_frames_on_a_synthetic_model_output(tmp_path):
    """Via the model-output cache seam: a 4-frame and a 5-frame note; min_note_frames=5 keeps only the 5-frame one."""
    n_frames = 200
    note = np.zeros((n_frames, 88), np.float32)
    onset = np.zeros((n_frames, 88), np.float32)
    contour = np.zeros((n_frames, 264), np.float32)
    for start, length, pitch in ((50, 4, 60), (120, 5, 67)):
        b = pitch - 21
        note[start:start + length, b] = 0.8
        onset[start, b] = 0.9
        contour[start:start + length, 3 * b] = 1.0  # contour bin of the note's own pitch -> zero bend
    cache = tmp_path / "cache"
    vsha = "c" * 64
    buf = io.BytesIO()
    np.savez(buf, note=note, onset=onset, contour=contour)
    atomic.write_bytes(cache / f"{vsha}__{_model_sha()[:12]}.npz", buf.getvalue())
    wav = tmp_path / "dummy.wav"
    sf.write(str(wav), np.zeros(SR * 3, np.float32), SR, subtype="FLOAT")
    views = [{"id": "g", "wav": str(wav), "out": str(tmp_path / "g.json"), "view_sha256": vsha}]
    settings = [{"min_note_frames": 5, "melodia_trick": False}, {"min_note_frames": 4, "melodia_trick": False},
                {"min_note_frames": 5, "melodia_trick": True}]
    out = _run({"mode": "sweep", "views": views, "settings": settings, "model_output_cache": str(cache)},
               tmp_path / "_worker")
    assert out["inference_runs"] == 0 and out["views"][0]["model_output_cached"] is True
    kept = [[n["pitch"] for n in atomic.read_json(tmp_path / f"g__s{k}.json")["notes"]] for k in range(3)]
    assert kept[0] == [67]
    assert kept[1] == [60, 67]
    assert kept[2] == [67]  # the melodia pass measures one frame shorter, so it cannot resurrect the 4-frame note
