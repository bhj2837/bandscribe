"""Transcription cache: key sensitivity, instrument-order invariance, rung label, and "send only missing views"."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bandscribe import atomic
from bandscribe.amt import cache as C
from bandscribe.amt import stage as S

BASE = dict(backend="muscriptor", backend_version="0.3.0", model_sha256="m" * 64, view_pcm_sha256="v" * 64,
            instruments=["acoustic_guitar", "clean_electric_guitar"], params={"dtype": "upstream", "beam": 1},
            code_version="1", rung_label="medium")


def key(**kw):
    return C.amt_key(**{**BASE, **kw})


def test_key_is_stable_and_hex():
    assert key() == key()
    assert len(key()) == 64 and int(key(), 16) >= 0


@pytest.mark.parametrize("field,value", [
    ("backend", "basicpitch"), ("backend_version", "0.3.1"), ("model_sha256", "n" * 64),
    ("view_pcm_sha256", "w" * 64), ("instruments", ["acoustic_guitar"]), ("instruments", None),
    ("params", {"dtype": "fp16", "beam": 1}), ("code_version", "2"), ("rung_label", "small"),
    ("rung_label", "medium-lean"),
])
def test_key_sensitivity(field, value):
    assert key(**{field: value}) != key()


def test_instrument_order_and_duplicates_do_not_matter():
    assert key(instruments=["clean_electric_guitar", "acoustic_guitar"]) == key()
    assert key(instruments=["clean_electric_guitar", "acoustic_guitar", "acoustic_guitar"]) == key()


def test_params_dict_order_does_not_matter():
    assert key(params={"beam": 1, "dtype": "upstream"}) == key()


def test_missing_view_hash_rejected():
    with pytest.raises(ValueError):
        key(view_pcm_sha256="")


def test_cache_roundtrip_and_corrupt_entry_is_a_miss(tmp_path):
    c = C.AmtCache(tmp_path)
    k = key()
    assert c.get(k) is None
    doc = {"format": "bandscribe.notes/1", "view": "v", "notes": []}
    p = c.put(k, doc)
    assert p == tmp_path / k[:2] / f"{k}.json"
    assert c.get(k) == doc
    p.write_text("{not json", encoding="utf-8")
    assert c.get(k) is None
    atomic.write_json(p, {"format": "other"})
    assert c.get(k) is None


# ------------------------------------------------------------------ transcribe_views_ms with a fake worker


PARAMS = {"muscriptor_model": "medium", "muscriptor_revision": "rev0", "muscriptor_fallback": [],
          "muscriptor_loader": "upstream", "muscriptor_version": "0.3.0", "dtype": "upstream",
          "decode": {"prelude_forcing": True, "beam_size": 1, "cfg_coef": 1.0, "use_sampling": False, "batch_size": 1},
          "latency_s": 0.02}


@pytest.fixture
def fake_weights(monkeypatch, tmp_path):
    w = tmp_path / "hf" / "model.safetensors"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"weights")
    monkeypatch.setattr(S, "muscriptor_weights", lambda size="medium", revision=None: w if size == "medium" else None)
    return w


class FakeRunner:
    """Stands in for bandscribe.vram.run_gpu_stage: records requests, writes raw NoteSets like the worker."""

    def __init__(self, rung="medium"):
        self.calls = []
        self.rung = rung

    def __call__(self, backend, labels, req, cfg, work_dir, notify=None):
        self.calls.append((backend, list(labels), json.loads(json.dumps(req, default=str))))
        for v in req["views"]:
            atomic.write_json(Path(v["out"]), {
                "format": "bandscribe.notes/1", "view": v["id"], "view_sha256": v["view_sha256"],
                "backend": {"name": "muscriptor", "instruments": v["instruments"]},
                "time_offset_applied_s": 0.0, "audio_duration_s": 2.0,
                "notes": [{"onset_s": 0.5, "offset_s": 1.0, "pitch": 60, "instrument": "acoustic_guitar"}]})
        out = {"status": "ok", "vram": {"rung_label": self.rung, "attempts": []},
               "backend": {"device_used": "cuda", "model_sha256": "m" * 64}}
        return SimpleNamespace(ok=True, output=out, waited_s=0.0, stderr_path=work_dir / "stderr.log")


def _views(tmp_path):
    return [{"id": "mix_mono", "wav": tmp_path / "a.wav", "pcm_sha256": "a" * 64, "instruments": None},
            {"id": "bass_mono", "wav": tmp_path / "b.wav", "pcm_sha256": "b" * 64,
             "instruments": ["electric_bass", "acoustic_bass"]}]


def test_only_missing_views_are_sent(tmp_path, fake_weights):
    cache = C.AmtCache(tmp_path / "cache")
    run = FakeRunner()
    raws, info = S.transcribe_views_ms(_views(tmp_path), PARAMS, {}, work_dir=tmp_path / "w1" / "_worker",
                                       cache=cache, model_sha256="m" * 64, job=None, run_gpu_stage=run)
    assert sorted(raws) == ["bass_mono", "mix_mono"] and len(run.calls) == 1
    backend, labels, req = run.calls[0]
    assert backend == "amt_muscriptor" and labels == ["medium"]
    assert [v["id"] for v in req["views"]] == ["mix_mono", "bass_mono"]
    # the mask is sent sorted by MT3 id (acoustic_bass=7 before electric_bass=8)
    assert req["views"][1]["instruments"] == ["acoustic_bass", "electric_bass"]
    assert req["model"]["weights_path"] == str(fake_weights) and req["model"]["sha256"] == "m" * 64

    # second pass: one view already cached -> only the new one goes to the worker
    views = _views(tmp_path) + [{"id": "piano_other_mono", "wav": tmp_path / "c.wav", "pcm_sha256": "c" * 64,
                                 "instruments": ["organ"]}]
    run2 = FakeRunner()
    raws2, info2 = S.transcribe_views_ms(views, PARAMS, {}, work_dir=tmp_path / "w2" / "_worker", cache=cache,
                                         model_sha256="m" * 64, job=None, run_gpu_stage=run2)
    assert [v["id"] for v in run2.calls[0][2]["views"]] == ["piano_other_mono"]
    assert info2["cached_views"] == ["bass_mono", "mix_mono"]

    # everything cached -> no worker at all
    run3 = FakeRunner()
    _r, info3 = S.transcribe_views_ms(views, PARAMS, {}, work_dir=tmp_path / "w3" / "_worker", cache=cache,
                                      model_sha256="m" * 64, job=None, run_gpu_stage=run3)
    assert run3.calls == [] and info3["worker_output"] is None and info3["rung_label"] == "medium"


def test_degraded_rung_never_answers_for_the_top_rung(tmp_path, fake_weights):
    cache = C.AmtCache(tmp_path / "cache")
    run = FakeRunner(rung="small")
    _r, info = S.transcribe_views_ms(_views(tmp_path)[:1], PARAMS, {}, work_dir=tmp_path / "w1" / "_worker",
                                     cache=cache, model_sha256="m" * 64, job=None, run_gpu_stage=run)
    assert info["rung_label"] == "small"
    run2 = FakeRunner()
    S.transcribe_views_ms(_views(tmp_path)[:1], PARAMS, {}, work_dir=tmp_path / "w2" / "_worker", cache=cache,
                          model_sha256="m" * 64, job=None, run_gpu_stage=run2)
    assert len(run2.calls) == 1  # the small-rung entry did not satisfy a medium lookup


def test_force_rung_is_the_lookup_label_and_reaches_the_request(tmp_path, fake_weights):
    cache = C.AmtCache(tmp_path / "cache")
    run = FakeRunner()
    S.transcribe_views_ms(_views(tmp_path)[:1], PARAMS, {}, work_dir=tmp_path / "w1" / "_worker", cache=cache,
                          model_sha256="m" * 64, job=None, run_gpu_stage=run, force_rung="medium")
    assert run.calls[0][2]["vram"]["force_rung"] == "medium"


def test_apply_latency_clamps_and_records_offset():
    doc = {"format": "bandscribe.notes/1", "time_offset_applied_s": 0.0, "notes": [
        {"onset_s": 0.01, "offset_s": 0.02, "pitch": 60, "instrument": "organ"},
        {"onset_s": 1.0, "offset_s": 1.5, "pitch": 62, "instrument": "organ"}]}
    out = S.apply_latency(doc, 0.025)
    assert out["time_offset_applied_s"] == -0.025
    assert out["notes"][0]["onset_s"] == 0.0 and out["notes"][0]["offset_s"] == 0.0
    assert out["notes"][1]["onset_s"] == pytest.approx(0.975) and out["notes"][1]["offset_s"] == pytest.approx(1.475)
    assert doc["notes"][1]["onset_s"] == 1.0  # input untouched


def test_segmented_view_has_its_own_cache_entry_and_sends_its_segments(tmp_path, fake_weights):
    """Presence windows (2026-10-04): segments enter that view's key; whole-view keys keep the M2 form."""
    cache = C.AmtCache(tmp_path / "cache")
    whole = {"id": "piano_other_mono", "wav": tmp_path / "c.wav", "pcm_sha256": "c" * 64, "instruments": ["organ"]}
    seg_a = dict(whole, id="piano_other_presence", segments=[[1.0, 11.0], [30.0, 40.0]])
    seg_b = dict(whole, id="piano_other_presence", segments=[[1.0, 11.0], [50.0, 60.0]])
    run = FakeRunner()
    S.transcribe_views_ms([whole, seg_a], PARAMS, {}, work_dir=tmp_path / "w1" / "_worker", cache=cache,
                          model_sha256="m" * 64, job=None, run_gpu_stage=run)
    sent = {v["id"]: v for v in run.calls[0][2]["views"]}
    assert "segments" not in sent["piano_other_mono"]
    assert sent["piano_other_presence"]["segments"] == [[1.0, 11.0], [30.0, 40.0]]
    run2 = FakeRunner()
    _r, info = S.transcribe_views_ms([seg_a, seg_b], PARAMS, {}, work_dir=tmp_path / "w2" / "_worker", cache=cache,
                                     model_sha256="m" * 64, job=None, run_gpu_stage=run2)
    assert [v["segments"] for v in run2.calls[0][2]["views"]] == [[[1.0, 11.0], [50.0, 60.0]]]  # only the new windows
    # the whole-view key is unchanged by the segments feature (M2 cache entries stay valid)
    cp = S._ms_cache_params(PARAMS)
    assert S._view_cparams(cp, whole) == cp
    assert S._view_cparams(cp, seg_a) == {**cp, "segments": [[1.0, 11.0], [30.0, 40.0]]}
