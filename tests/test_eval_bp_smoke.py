"""bp-smoke suite wiring: 20 GuitarSet tracks drawn with the seed, one Basic Pitch worker call (bp310, no GPU lock),
onset-only F1 at 50 ms, flagged as not a judgment (GuitarSet is in Basic Pitch's training data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bandscribe import config, paths
from bandscribe.eval import suites

pytest.importorskip("mir_eval")


def _tracks(n: int = 25):
    from bandscribe.schema.dataset import DatasetTrack, Part
    from bandscribe.schema.gt import GtLine
    from bandscribe.schema.notes import BackendInfo, NoteEvent, NoteSet

    out = []
    for i in range(n):
        rng = np.random.default_rng(i)
        on = np.cumsum(rng.uniform(0.15, 0.5, 40))
        notes = [NoteEvent(onset_s=float(t), offset_s=float(t) + 0.2, pitch=int(p))
                 for t, p in zip(on, rng.integers(40, 80, 40))]
        tid = f"0{i % 6}_track{i:02d}_comp"
        out.append(DatasetTrack(
            format="bandscribe.dataset_track/1", dataset="guitarset", track_id=tid, split=None, group=f"0{i % 6}",
            tier="C", license="CC BY 4.0", mix=None,
            parts=[Part(id="mic", kind="guitar", audio=f"audio_mono-mic/{tid}_mic.wav", channels=1)],
            lines=[GtLine(id="guitar", name="guitar", kind="guitar", takes=[1])], families_present=["guitar"],
            notes={"guitar": NoteSet(view="gt", backend=BackendInfo(name="gt"), notes=notes)},
            string_fret=None, beats_s=None, downbeats_s=None, tuning=None, meta={}))
    return out


class _Res:
    ok = True
    error = None


def test_bp_smoke_uses_worker_once_and_flags_result(tmp_path, monkeypatch):
    from bandscribe import datasets

    tracks = {t.track_id: t for t in _tracks()}
    monkeypatch.setattr(datasets, "iter_tracks", lambda name, split=None: iter(tracks.values()))
    monkeypatch.setattr(datasets, "track_ids", lambda name, split=None: list(tracks))
    monkeypatch.setattr(datasets, "load_track", lambda name, tid: tracks[tid])
    monkeypatch.setattr(datasets, "dataset_root", lambda name: tmp_path / "guitarset")
    monkeypatch.setattr(datasets, "is_available", lambda name: True)
    calls = []

    def fake_run_worker(name, request, work_dir, **kw):
        calls.append((name, request, kw))
        for v in request["views"]:
            tid = Path(v["wav"]).name.removesuffix("_mic.wav")
            gt = tracks[tid].notes["guitar"].notes
            notes = [{"onset_s": n.onset_s + 0.01, "offset_s": n.offset_s, "pitch": n.pitch} for n in gt[:-4]]
            Path(v["out"]).write_text(json.dumps({"format": "bandscribe.notes/1", "view": v["id"],
                                                  "backend": {"name": "basicpitch"}, "notes": notes}), encoding="utf-8")
        return _Res()

    res = suites.run_suite("bp-smoke", None, config.load_config(), out=tmp_path / "runs", gitsha="abcdef1",
                           run_worker=fake_run_worker)
    assert len(calls) == 1
    name, req, kw = calls[0]
    assert name == "amt_basicpitch" and kw["use_lock"] is False and kw["python"] == paths.BP310_PYTHON
    assert req["mode"] == "transcribe" and len(req["views"]) == 20
    assert req["params"]["min_note_frames"] != 12 and req["allow_default_min_len"] is False
    s = res.summary
    assert s["system"] == "basicpitch" and s["n_items"] == 20
    assert "판정용 아님" in s["note"]
    f1 = s["aggregates"]["note_f1_onset50"]["value"]
    assert 0.9 < f1 < 1.0 and s["aggregates"]["note_f1_onset50"]["ci"][0] <= f1
    text = (res.run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "판정용 아님" in text
    # the draw is deterministic (seeded) and does not depend on iteration order
    items_a = suites.items_for_suite("bp-smoke", config.load_config())
    monkeypatch.setattr(datasets, "iter_tracks", lambda name, split=None: iter(reversed(list(tracks.values()))))
    monkeypatch.setattr(datasets, "track_ids", lambda name, split=None: list(reversed(list(tracks))))
    items_b = suites.items_for_suite("bp-smoke", config.load_config())
    assert [i.item for i in items_a] == [i.item for i in items_b]
