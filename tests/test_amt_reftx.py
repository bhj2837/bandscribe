"""Tier B reference transcription plumbing (gtab.amt.reftx) with a fake MuScriptor call."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gtab import audio
from gtab.amt import instruments as I
from gtab.amt import reftx as R
from gtab.amt import stage as S
from gtab.schema.dataset import DatasetTrack, Part
from gtab.schema.gt import GtLine

SR = 44100


def _track(root: Path) -> DatasetTrack:
    t = np.arange(SR) / SR
    parts = []
    for pid, line, kind, f, ch in (("g1", "L1", "guitar", 220.0, 2), ("g2", "L1", "guitar", 330.0, 1),
                                   ("b", "B", "bass", 55.0, 1), ("v", None, "vocals", 440.0, 1)):
        x = (0.1 * np.sin(2 * np.pi * f * t)).astype(np.float32)
        audio.write_f32(root / f"{pid}.wav", np.stack([x, -x], axis=1) if ch == 2 else x, SR)
        parts.append(Part(id=pid, kind=kind, audio=f"{pid}.wav", channels=ch, line=line))
    return DatasetTrack(format="gtab.dataset_track/1", dataset="cambridge_mt", track_id="Song A", split=None,
                        group="Song A", tier="B", license="test", mix=None, parts=parts,
                        lines=[GtLine(id="L1", name="Gtr", kind="guitar", takes=[1, 2]),
                               GtLine(id="B", name="Bass", kind="bass", takes=[3])],
                        families_present=["guitar", "bass", "vocals"], notes={}, string_fret=None, beats_s=None,
                        downbeats_s=None, tuning=None, meta={})


def test_reference_transcribe_sums_takes_masks_and_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "dist_version", lambda python, dist: "0.0-test")
    monkeypatch.setattr(S, "muscriptor_weights", lambda size="medium", revision=None: tmp_path / "w.safetensors")
    calls = []

    def fake(views, params, cfg, *, work_dir, cache, job, force_rung):
        calls.append({"views": views, "force_rung": force_rung, "cache": cache.root, "work_dir": work_dir})
        return {v["id"]: {"format": "gtab.notes/1", "view": v["id"], "notes": []} for v in views}, {}

    root = tmp_path / "ds"
    out = R.reference_transcribe(_track(root), tmp_path / "out", {}, root=root, cache_root=tmp_path / "c",
                                 transcribe=fake)
    assert sorted(out) == ["B", "L1"]
    c = calls[0]
    assert c["force_rung"] == "medium-lean" and c["cache"] == tmp_path / "c"  # top rung of the default (lean) ladder
    by = {v["id"]: v for v in c["views"]}
    assert by["L1"]["instruments"] == I.mask_guitar_only() and by["B"]["instruments"] == I.mask_bass()
    # the line's takes are summed and down-mixed: stereo take g1 is (x, -x) -> mono 0, so only g2 remains
    x, sr = audio.read_f32(by["L1"]["wav"])
    g2, _ = audio.read_f32(root / "g2.wav")
    assert sr == SR and np.allclose(x[:, 0], g2[:, 0], atol=1e-7)
    assert by["L1"]["pcm_sha256"] == audio.pcm_sha256(by["L1"]["wav"])
    assert (tmp_path / "out" / "l1__reftx.json").is_file() and (tmp_path / "out" / "b__reftx.json").is_file()
    assert Path(by["L1"]["wav"]).name == "l1_mono.wav"  # ASCII slugs


def test_line_without_audio_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "dist_version", lambda python, dist: "0.0-test")
    monkeypatch.setattr(S, "muscriptor_weights", lambda size="medium", revision=None: tmp_path / "w.safetensors")
    root = tmp_path / "ds"
    tr = _track(root)
    tr = tr.model_copy(update={"parts": [p for p in tr.parts if p.line != "B"]})
    seen = []

    def fake(views, params, cfg, **kw):
        seen.extend(v["id"] for v in views)
        return {v["id"]: {"format": "gtab.notes/1", "notes": []} for v in views}, {}

    assert sorted(R.reference_transcribe(tr, tmp_path / "o", {}, root=root, transcribe=fake,
                                         cache_root=tmp_path / "c")) == ["L1"]
    assert seen == ["L1"]
    assert R.line_audio(tr, "B", root) is None


def test_line_audio_leaves_out_a_di_next_to_an_amp_track(tmp_path):
    """Same rule as the Tier B remix (gtab.eval.remix.mix_parts): a DI is used only when it is the line's only
    source, so the reference transcribes the audio that is actually in the mix."""
    root = tmp_path / "ds"
    t = np.arange(SR) / SR
    amp = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    di = (0.1 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
    audio.write_f32(root / "amp.wav", amp, SR)
    audio.write_f32(root / "di.wav", di, SR)
    tr = _track(root).model_copy(update={"parts": [
        Part(id="amp", kind="guitar", audio="amp.wav", channels=1, line="L1"),
        Part(id="di", kind="di", audio="di.wav", channels=1, line="L1"),
        Part(id="di2", kind="di", audio="di.wav", channels=1, line="B")]})
    x, sr = R.line_audio(tr, "L1", root)
    assert sr == SR and np.allclose(x, amp, atol=1e-7)  # DI dropped
    y, _ = R.line_audio(tr, "B", root)
    assert np.allclose(y, di, atol=1e-7)  # a lone DI stays
