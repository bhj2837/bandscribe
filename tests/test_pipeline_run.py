"""Runner, publish and CPU-stage determinism (M1_M2_SPEC 3.3, 4; GRID)."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bandscribe import atomic
from bandscribe.jobs.store import JobStore
from bandscribe.pipeline import graph, publish, runner
from grid_testlib import Fakes, synth_song, write_gpu_run


@pytest.fixture
def job(tmp_path):
    store = JobStore(tmp_path / "jobs")
    key = "f-0123456789abcdef"
    inp = store.job_dir(key) / "input"
    inp.mkdir(parents=True)
    atomic.write_json(inp / "meta.json", {"decode": {"pcm_sha256": "ab" * 32}, "params": {"version": 2}})
    return store, key


@pytest.fixture
def fakes(monkeypatch):
    f = Fakes()
    f.install(monkeypatch)
    return f


def events_of(ev, kind):
    return [i["stage"] for k, i in ev if k == kind]


def test_cache_hit_on_rerun_and_from_cascades(job, fakes):
    store, key = job
    ev: list = []
    res = runner.run(key, None, store=store, progress=lambda k, i: ev.append((k, i)))
    assert set(res) == set(graph.NAMES)
    assert events_of(ev, "stage_start") == [n for n in graph.NAMES]
    ev.clear()
    fakes.calls.clear()
    runner.run(key, None, store=store, progress=lambda k, i: ev.append((k, i)))
    assert fakes.calls == [] and set(events_of(ev, "stage_cached")) == set(graph.NAMES)
    fakes.calls.clear()
    runner.run(key, None, store=store, force={"sections"})
    # amt_ms1 follows sections (presence windows); its unchanged views come from the transcription cache
    assert set(fakes.calls) == {"sections", "resid1", "amt_ms1", "instr", "amt_gtr", "s35", "notes", "quant", "tab",
                                "score"}
    items = runner.plan(key, None, store=store)
    assert all(it.cached for it in items)


def test_plan_reports_all_missing_models_before_running(job, fakes):
    store, key = job
    fakes.missing = {"beats", "amt_ms1"}
    with pytest.raises(runner.ModelsMissing) as ei:
        runner.run(key, None, store=store)
    assert ei.value.names == ["model.beats", "model.amt_ms1"]
    assert "bandscribe models fetch model.beats" in str(ei.value) and "model.amt_ms1" in str(ei.value)
    assert fakes.calls == []
    assert not (store.job_dir(key) / "stages").exists()
    items = runner.plan(key, None, store=store, missing_ok=True)
    by = {it.stage: it for it in items}
    assert by["beats"].missing_models == ["model.beats"] and by["beats"].key == ""
    assert by["grid"].key == "" and by["sep"].key  # downstream of a missing model has no key
    assert by["notes"].key == ""


def test_degraded_cached_gpu_stage_warns(job, fakes):
    store, key = job
    fakes.degraded = {"sep"}
    runner.run(key, None, store=store, until="stems")
    ev: list = []
    runner.run(key, None, store=store, until="stems", progress=lambda k, i: ev.append((k, i)))
    warn = [i for k, i in ev if k == "stage_degraded_cache"]
    assert [w["stage"] for w in warn] == ["sep"]
    assert "낮은 칸(low)" in warn[0]["message"] and f"bandscribe run {key} --force sep" in warn[0]["message"]


def test_data_outputs_ignore_worker_and_manifest(job, fakes):
    store, key = job
    res = runner.run(key, None, store=store, until="grid")
    a = {s: runner.data_outputs(res[s]) for s in res}
    man_a = (res["beats"] / "manifest.json").read_bytes()
    runner.run(key, None, store=store, until="grid", force={"beats"})
    b = {s: runner.data_outputs(res[s]) for s in res}
    assert a == b
    assert "beats.json" in a["beats"] and "gpu_run.json" in a["beats"]
    assert not any(k.startswith("_worker") or k == "manifest.json" for k in a["beats"])
    assert (res["beats"] / "_worker" / "result.json").exists()
    assert (res["beats"] / "manifest.json").read_bytes() != man_a  # provenance differs, data does not


def test_publish_copies_links_and_is_idempotent(job, fakes):
    store, key = job
    res = runner.run(key, None, store=store)
    export = store.job_dir(key) / "export"
    doc = atomic.read_json(export / "export.json")
    assert doc["format"] == "bandscribe.export/1"
    # (the fake M3 stages write only <stage>.json: of the optional score files, tab/tab.json and score.json)
    assert set(doc["files"]) == {"midi/guitar_all.mid", "midi/bass_raw.mid", "practice/mix_minus_guitar.wav",
                                 "practice/mix_minus_bass.wav", "practice/bass_stem.wav", "instrumentation.json",
                                 "tab/tab.json", "score.json"}
    assert doc["files"]["practice/bass_stem.wav"]["stage"] == "sep"
    assert doc["files"]["midi/guitar_all.mid"]["key12"] == res["notes"].name
    # WAVs: read-only hardlinks; MIDI: independent copies
    wav = export / "practice" / "bass_stem.wav"
    assert not os.access(wav, os.W_OK)
    assert os.path.samefile(wav, res["sep"] / "stems" / "bass.wav")
    mid = export / "midi" / "guitar_all.mid"
    mid.write_bytes(b"edited in a DAW")
    assert (res["notes"] / "midi" / "guitar_all.mid").read_bytes() == b"MThd-guitar-v1"
    # idempotent: same keys -> nothing rewritten (the edited MIDI has another size -> republished)
    r1 = publish.publish(store.job_dir(key), res)
    assert r1["changed"] is True and mid.read_bytes() == b"MThd-guitar-v1"
    mtime = (export / "export.json").stat().st_mtime_ns
    r2 = publish.publish(store.job_dir(key), res)
    assert r2["changed"] is False and (export / "export.json").stat().st_mtime_ns == mtime
    # `--force sep` with the same key: the stage file is replaced by identical bytes (new inode) and deleting the
    # old one cleared the read-only bit the export shared -> publish re-links instead of keeping a writable orphan
    src = res["sep"] / "stems" / "bass.wav"
    data = src.read_bytes()
    os.chmod(src, stat.S_IWRITE)
    src.unlink()
    src.write_bytes(data)
    assert not os.path.samefile(wav, src) and os.access(wav, os.W_OK)
    r3 = publish.publish(store.job_dir(key), res)
    assert r3["changed"] is True and os.path.samefile(wav, src) and not os.access(wav, os.W_OK)


def test_publish_with_locked_old_export(job, fakes):
    store, key = job
    runner.run(key, None, store=store)
    export = store.job_dir(key) / "export"
    fakes.variant = "v2"  # new params -> new keys -> a different export
    held = open(export / "midi" / "guitar_all.mid", "rb")  # a DAW holding the old file open
    try:
        with pytest.raises(publish.PublishError, match="다른 프로그램에서 열려 있습니다"):
            runner.run(key, None, store=store)
        assert (export / "midi" / "guitar_all.mid").read_bytes() == b"MThd-guitar-v1"  # old export kept
        leftovers = [p.name for p in store.job_dir(key).iterdir() if p.name.startswith(".tmp-export")]
        assert leftovers and all(n.startswith(".tmp-export-") for n in leftovers)
    finally:
        held.close()
    runner.run(key, None, store=store)  # stages cached, publish succeeds now
    assert (export / "midi" / "guitar_all.mid").read_bytes() == b"MThd-guitar-v2"
    assert not os.access(export / "practice" / "mix_minus_bass.wav", os.W_OK)


def test_run_warnings_and_summary(job, fakes):
    store, key = job
    items = runner.plan(key, None, store=store)
    res = runner.run(key, None, store=store)
    rows = runner.stage_summary(res, items)
    assert [r["stage"] for r in rows] == list(graph.NAMES)
    assert all(r["cached"] is False and r["duration_s"] is not None for r in rows)
    beats = next(r for r in rows if r["stage"] == "beats")
    assert beats["device"] == "gpu" and beats["rung"] == "top" and not beats["degraded"]
    w = runner.run_warnings(res)
    assert len(w) == 1 and "leftover" in w[0] and "10s" in w[0]
    atomic.write_json(res["grid"] / "grid.json", {"hypotheses": {"warnings": ["반 마디로 보고 묶었습니다"]}})
    w = runner.run_warnings(res)
    assert w[0] == "박자 격자: 반 마디로 보고 묶었습니다" and "leftover" in w[1]


def test_resolve_job_by_prefix(job):
    store, key = job
    r = runner.resolve_job("f-0123", store, None)
    assert r.song_key == key and r.cache_hit
    with pytest.raises(FileNotFoundError, match="찾을 수 없습니다"):
        runner.resolve_job("nope", store, None)


# ------------------------------------------------------------------------------ real GRID CPU stages


@pytest.fixture
def real_grid_job(tmp_path, monkeypatch):
    """A job whose beats/sep/stems stages are fakes writing synthetic data; grid/sections/vocal/resid1 are real."""
    from bandscribe.analysis import stage as gstage

    sr = 44100
    y, grid, hits = synth_song("ABAB", n_bars=4, sr=sr, seed=1)
    store = JobStore(tmp_path / "jobs")
    key = "f-00000000000000aa"
    inp = store.job_dir(key) / "input"
    inp.mkdir(parents=True)
    stereo = np.stack([y, y], 1)
    sf.write(str(inp / "mix_44k_f32.wav"), stereo, sr, subtype="FLOAT")
    atomic.write_json(inp / "meta.json", {"decode": {"pcm_sha256": "cd" * 32}, "params": {"version": 2}})
    beat_times = [b["t_s"] for b in grid["beats"]]

    def fake_beats(ctx):
        out = Path(ctx.out_dir)
        atomic.write_json(out / "beats.json", {
            "format": "bandscribe.beats/1", "tracker": "beat_this/final0", "checkpoint_sha256": None, "dbn": False,
            "fps": 50.0, "beats_s": [round(t * 50) / 50 for t in beat_times],
            "downbeats_s": [round(t * 50) / 50 for t in beat_times[::4]], "duration_s": len(y) / sr})
        (out / "_worker").mkdir()
        (out / "_worker" / "request.json").write_text(str(out))  # temp path: differs between runs
        write_gpu_run(out, "final0", False)

    def fake_sep(ctx):
        out = Path(ctx.out_dir) / "stems"
        out.mkdir()
        voc = np.zeros_like(stereo)
        voc[int(9 * sr):int(17 * sr)] = 0.05 * np.sin(2 * np.pi * 300 * np.arange(int(8 * sr)) / sr)[:, None]
        sf.write(str(out / "vocals.wav"), voc, sr, subtype="FLOAT")
        write_gpu_run(Path(ctx.out_dir), "chunk=588800", False)

    def fake_stems(ctx):
        out = Path(ctx.out_dir) / "views"
        out.mkdir()
        sf.write(str(out / "guitar_mono.wav"), y, sr, subtype="FLOAT")
        atomic.write_json(Path(ctx.out_dir) / "views.json",
                          {"format": "bandscribe.views/1", "views": {"guitar_mono": {"path": "views/guitar_mono.wav"}}})

    real = dict(gstage.STAGES)
    fake = {"run": None, "code_version": "1", "params": lambda cfg: {}, "models": lambda cfg: {}}
    impls = {"beats": {**fake, "run": fake_beats, "device": "gpu"}, "sep": {**fake, "run": fake_sep, "device": "gpu"},
             "stems": {**fake, "run": fake_stems, "device": "cpu"},
             **{n: real[n] for n in ("grid", "sections", "vocal", "resid1")}}
    monkeypatch.setattr(graph, "_provider",
                        lambda module: {n: v for n, v in impls.items() if graph.PROVIDER[n] == module})
    return store, key


def test_real_cpu_stages_are_byte_deterministic(real_grid_job):
    store, key = real_grid_job
    cfg = {"s35": {"resid_min_occurrences": 2}}
    res = {**runner.run(key, cfg, store=store, until="resid1"), **runner.run(key, cfg, store=store, until="vocal")}
    first = {s: runner.data_outputs(res[s]) for s in ("grid", "sections", "vocal", "resid1")}
    grid = atomic.read_json(res["grid"] / "grid.json")
    assert len([b for b in grid["bars"] if not b["pickup"]]) == 16
    secs = atomic.read_json(res["sections"] / "sections.json")
    assert [s["label"] for s in secs["sections"]] == ["A", "B", "A", "B"]
    resid = atomic.read_json(res["resid1"] / "resid1.json")
    assert set(resid["groups_used"]) == {"A", "B"}
    voc = atomic.read_json(res["vocal"] / "vocal_activity.json")
    act = {b["bar"]: b["active_ratio"] for b in voc["per_bar"]}
    assert act[5] == 1.0 and act[1] == 0.0  # vocals in bars 5-8 (9..17 s)
    runner.run(key, cfg, store=store, until="resid1", force={"beats"})
    runner.run(key, cfg, store=store, until="vocal", force={"beats"})
    second = {s: runner.data_outputs(res[s]) for s in ("grid", "sections", "vocal", "resid1")}
    assert first == second
    assert set(first["grid"]) == {"grid.json", "onsets.npz"}
    assert set(first["resid1"]) == {"repeat_resid_pass1.wav", "resid1.json"}


def test_run_warnings_surface_cached_gpu_flags(job, fakes):
    """Review 2026-10-05: cached GPU stages keep their lower rung, slowness and the worker's warnings in
    gpu_run.json (older sep dirs: sep.json); run_warnings shows them, except for stages already shown live."""
    store, key = job
    res = runner.run(key, None, store=store, until="amt_gtr")
    slow = "chunk=352256: 실시간 배수 1.4x 로 기준보다 느림(다른 GPU 작업과 경합했을 수 있음)"
    atomic.write_json(res["sep"] / "sep.json", {"warnings": [slow]})  # an old sep dir: no "warnings" in gpu_run
    gr = atomic.read_json(res["amt_gtr"] / "gpu_run.json")
    eos = "guitar_mono: MuScriptor 가 1개 구간에서 끝 토큰(EOS) 없이 생성 한도에 닿았습니다(12 s 부근)."
    atomic.write_json(res["amt_gtr"] / "gpu_run.json", {**gr, "degraded": True, "slow": True, "warnings": [eos],
                                                         "rung": {"index": 1, "label": "small", "device": "cuda"}})
    atomic.write_json(res["beats"] / "gpu_run.json", {**atomic.read_json(res["beats"] / "gpu_run.json"),
                                                       "slow": True, "warnings": []})
    w = runner.run_warnings(res)
    assert f"sep: {slow}" in w
    assert f"amt_gtr: 낮은 칸(small)으로 만든 캐시입니다. 여유가 있을 때 'bandscribe run {key} --force amt_gtr' 로 다시 만드세요." in w
    assert f"amt_gtr: {eos}" in w
    assert any(x.startswith("amt_gtr: 기준보다 느리게") for x in w) and any(x.startswith("beats: 기준보다 느리게") for x in w)
    assert not any(x.startswith("sep: 기준보다 느리게") for x in w)  # the worker's own slow text is enough
    w2 = runner.run_warnings(res, exclude_stages={"sep", "amt_gtr", "beats"})
    assert not any(x.split(":")[0] in ("sep", "amt_gtr", "beats") for x in w2)


def test_fresh_degraded_gpu_stage_warns_live(job, fakes):
    store, key = job
    fakes.degraded = {"beats"}
    ev: list = []
    runner.run(key, None, store=store, until="grid", progress=lambda k, i: ev.append((k, i)))
    warn = [i for k, i in ev if k == "warning"]
    assert [w["stage"] for w in warn] == ["beats"]
    assert warn[0]["message"].startswith("낮은 칸(low)으로 실행했습니다")
    assert f"bandscribe run {key} --force beats" in warn[0]["message"]
