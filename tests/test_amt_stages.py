"""AMT pipeline stages end to end with fake workers (gtab.amt.stage, M1_M2_SPEC 3.2 rows 8-13).

amt_ms1 -> instr -> amt_gtr -> amt_bp (skipped and faked) -> s35 -> notes on a tiny synthetic job: outputs exist,
validate against the P0 schemas, the latency constant is applied, CPU stage data outputs are byte-identical across
runs, and the config is read strictly.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mido
import numpy as np
import pytest

from gtab import atomic, models, paths, vram
from gtab.amt import instruments as I
from gtab.amt import stage as S
from gtab.config import Config
from gtab.schema.instrumentation import Instrumentation
from gtab.schema.notes import NoteSet

SONG = "song0"
LAT = 0.02
BEAT = 0.5  # 120 BPM
N_BARS = 8
T0 = 0.37  # first beat: exercises the MIDI lead-in


# ----------------------------------------------------------------------------------------------- fixtures


class FakeStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def job_dir(self, song_key: str) -> Path:
        return self.root / "jobs" / song_key


def _grid() -> dict:
    beats, bars = [], []
    for b in range(N_BARS):
        start = T0 + b * 4 * BEAT
        bars.append({"bar": b + 1, "start_s": start, "end_s": start + 4 * BEAT, "n_beats": 4, "numerator": 4,
                     "denominator": 4, "beat_unit": "quarter", "pickup": False})
        for k in range(4):
            beats.append({"t_s": start + k * BEAT, "t_raw_s": start + k * BEAT, "bar": b + 1, "beat": k + 1,
                          "is_downbeat": k == 0, "shift_ms": 0.0, "activation": None})
    return {"format": "gtab.grid/1", "tpb": 48, "duration_s": T0 + N_BARS * 4 * BEAT, "beats": beats, "bars": bars,
            "tempo": [{"start_bar": 1, "end_bar": N_BARS + 1, "bpm": 120.0}], "meter": [], "beat_unit": "quarter",
            "pickup": {"present": False, "beats": 0}, "hypotheses": {}, "confidence": {}, "params": {}}


def _notes(cls: str, pitch0: int, n: int) -> list[dict]:
    out = []
    for k in range(n):
        t = round(T0 + 0.25 * k, 6)
        out.append({"onset_s": t, "offset_s": round(t + 0.2, 6), "pitch": pitch0 + k % 5, "instrument": cls})
    return out


@pytest.fixture
def job(tmp_path, monkeypatch):
    """Dep dirs of a finished stems/grid/sections/vocal/resid1 run plus a fake MuScriptor worker."""
    store = FakeStore(tmp_path)
    deps = {name: tmp_path / "deps" / name for name in ("stems", "grid", "sections", "vocal", "resid1")}
    for d in deps.values():
        d.mkdir(parents=True)
    views = {}
    for i, vid in enumerate(("mix_mono", "guitar_mono", "bass_mono", "piano_other_mono", "nonvox_mono",
                             "guitar_other_mono")):
        p = deps["stems"] / "views" / f"{vid}.wav"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"RIFF")  # never read: the worker is faked
        views[vid] = {"path": f"views/{vid}.wav", "pcm_sha256": f"{i:x}" * 64, "source": "x", "mono": "(L+R)/2"}
    atomic.write_json(deps["stems"] / "views.json", {"format": "gtab.views/1", "views": views})
    n_fr = int((T0 + N_BARS * 2.0) * 10) + 1
    energy = {"mix": np.full(n_fr, -20.0, np.float32)}
    for stem, rel in (("guitar", -6.0), ("bass", -6.0), ("piano", -10.0), ("other", -60.0), ("vocals", -60.0),
                      ("drums", -8.0), ("nonvox", -3.0), ("leftover", -50.0)):
        energy[stem] = np.full(n_fr, -20.0 + rel, np.float32)
    energy["fps"] = np.array([10.0], np.float32)
    np.savez(deps["stems"] / "energy.npz", **energy)
    atomic.write_json(deps["grid"] / "grid.json", _grid())
    atomic.write_json(deps["sections"] / "sections.json", {
        "format": "gtab.sections/1", "method": "test", "params": {},
        "sections": [{"id": "S1", "label": "A", "start_bar": 1, "end_bar": 5, "start_s": T0, "end_s": T0 + 8.0,
                      "confidence": 1.0},
                     {"id": "S2", "label": "B", "start_bar": 5, "end_bar": 9, "start_s": T0 + 8.0,
                      "end_s": T0 + 16.0, "confidence": 1.0}]})
    atomic.write_json(deps["vocal"] / "vocal_activity.json", {"format": "gtab.vocal/1"})
    atomic.write_json(deps["resid1"] / "resid1.json", {"format": "gtab.resid1/1"})
    (deps["resid1"] / "repeat_resid_pass1.wav").write_bytes(b"RIFF")

    w = tmp_path / "hf" / "model.safetensors"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"weights")
    monkeypatch.setattr(S, "muscriptor_weights", lambda size="medium", revision=None: w if size == "medium" else None)
    monkeypatch.setattr(S, "_sha256", lambda p: "m" * 64)  # keep the real data/models/.sha_cache.json untouched
    monkeypatch.setattr(S, "dist_version", lambda python, dist: "0.0-test")
    fake = FakeMs()
    monkeypatch.setattr(vram, "run_gpu_stage", fake)
    return SimpleNamespace(store=store, deps=deps, fake=fake, root=tmp_path)


class FakeMs:
    """gtab.vram.run_gpu_stage stand-in: writes raw NoteSets for the requested views like amt_muscriptor."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, backend, labels, req, cfg, work_dir, notify=None):
        self.calls.append(json.loads(json.dumps(req, default=str)))
        for v in req["views"]:
            mask = v["instruments"]
            if v["id"] == "bass_mono":
                notes = _notes("electric_bass", 36, 16)
            elif v["id"] == "mix_mono" and mask is None:
                notes = sorted(_notes("distorted_electric_guitar", 52, 24) + _notes("acoustic_piano", 60, 24)
                               + _notes("electric_bass", 36, 8),
                               key=lambda n: (n["onset_s"], n["pitch"], n["offset_s"], n["instrument"]))
            else:
                classes = mask or list(I.GUITAR_CLASSES)
                notes = _notes(classes[-1] if "distorted_electric_guitar" in classes else classes[0], 52, 20)
            atomic.write_json(Path(v["out"]), {
                "format": "gtab.notes/1", "view": v["id"], "view_sha256": v["view_sha256"],
                "backend": {"name": "muscriptor", "version": "0.3.0", "model": "muscriptor-medium",
                            "model_sha256": "m" * 64, "params": {}, "instruments": mask, "device": "cuda",
                            "dtype": "fp32+autocast-fp16"},
                "time_offset_applied_s": 0.0, "audio_duration_s": 20.0, "notes": notes})
        out = {"status": "ok", "vram": {"rung_label": labels[0], "attempts": []},
               "backend": {"device_used": "cuda", "model_sha256": "m" * 64}}
        return SimpleNamespace(ok=True, output=out, waited_s=0.0, stderr_path=Path(work_dir) / "stderr.log")


def _cfg(**amt) -> dict:
    return {"amt": {"latency_s": LAT, **amt}}


def _ctx(job, stage: str, out: Path, dep_dirs: dict, cfg: dict) -> SimpleNamespace:
    out.mkdir(parents=True, exist_ok=True)
    params = S.STAGES[stage]["params"](cfg)
    params = json.loads(json.dumps(params))  # what the DAG hands over (JSON round trip of the key params)
    return SimpleNamespace(song_key=SONG, stage=stage, key=f"{stage}-key".ljust(64, "0"), out_dir=out,
                           dep_dirs=dep_dirs, config=cfg, store=job.store, params=params)


def _run_chain(job, cfg: dict, tag: str) -> dict[str, Path]:
    d = {k: v for k, v in job.deps.items()}
    out = {}
    for stage, deps in (("amt_ms1", ("stems",)), ("amt_bp", ("stems",)),
                        ("instr", ("amt_ms1", "stems", "sections", "grid")), ("amt_gtr", ("stems", "instr")),
                        ("s35", ("instr", "vocal", "resid1", "amt_ms1", "amt_gtr", "amt_bp")),
                        ("notes", ("amt_gtr", "amt_ms1", "amt_bp", "grid", "s35"))):
        o = job.store.job_dir(SONG) / "stages" / stage / tag
        ctx = _ctx(job, stage, o, {k: d[k] for k in deps}, cfg)
        S.STAGES[stage]["run"](ctx)
        d[stage] = out[stage] = o
    return out


def _data_files(stage_dir: Path) -> dict[str, bytes]:
    return {p.relative_to(stage_dir).as_posix(): p.read_bytes() for p in sorted(stage_dir.rglob("*"))
            if p.is_file() and not p.relative_to(stage_dir).as_posix().startswith("_worker/")}


# --------------------------------------------------------------------------------------------------- tests


def test_full_chain_without_basic_pitch(job, monkeypatch):
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    out = _run_chain(job, _cfg(), "a")

    # amt_ms1: three pass-1 views (quality), latency applied, gpu_run.json
    ms1 = out["amt_ms1"]
    names = sorted(p.name for p in (ms1 / "raw").iterdir())
    assert names == ["bass_mono__muscriptor.json", "mix_mono__muscriptor.json", "piano_other_mono__muscriptor.json"]
    mix = NoteSet.model_validate(atomic.read_json(ms1 / "raw" / "mix_mono__muscriptor.json"))
    assert mix.time_offset_applied_s == -LAT and mix.notes[0].onset_s == pytest.approx(T0 - LAT)
    gr = vram.read_gpu_run(ms1)
    assert gr["backend"] == "amt_muscriptor" and gr["rung"]["label"] == "medium-lean" and gr["degraded"] is False
    req = job.fake.calls[0]
    assert [v["id"] for v in req["views"]] == ["mix_mono", "bass_mono", "piano_other_mono"]
    assert req["views"][0]["instruments"] is None and req["views"][1]["instruments"] == I.mask_bass()

    # amt_bp skipped (no bp310 env)
    assert atomic.read_json(out["amt_bp"] / "skipped.json") == {"reason": "bp310 env missing"}

    # instr: schema-valid, keys present (stem + mix evidence), guitar mask keeps the guitar classes
    inst = Instrumentation.model_validate(atomic.read_json(out["instr"] / "instrumentation.json"))
    assert inst.classes["acoustic_piano"].present and inst.classes["distorted_electric_guitar"].present
    gp = inst.masks["guitar_pass"]
    assert set(I.GUITAR_CLASSES) <= set(gp) and "acoustic_piano" in gp and gp == I.sort_mt3(gp)
    assert [s["section"] for s in inst.per_section] == ["S1", "S2"]

    # amt_gtr: the guitar view with the instrumentation's mask
    gtr_req = job.fake.calls[1]
    assert [v["id"] for v in gtr_req["views"]] == ["guitar_mono"] and gtr_req["views"][0]["instruments"] == gp
    assert (out["amt_gtr"] / "raw" / "guitar_mono__muscriptor.json").is_file()

    # s35: job-relative paths grouped by stage, Basic Pitch null
    s35 = atomic.read_json(out["s35"] / "s35.json")
    assert s35["raw"]["amt_bp"] == {"bass_mono__basicpitch": None, "guitar_mono__basicpitch": None}
    assert s35["guitar_pass"] == "stages/amt_gtr/a/raw/guitar_mono__muscriptor.json"
    assert s35["b0"] == "stages/amt_ms1/a/raw/mix_mono__muscriptor.json"
    assert s35["basic_pitch"] == {"skipped": True, "reason": "bp310 env missing"}
    for rel in (s35["instrumentation"], s35["vocal_activity"], s35["guitar_pass"]):
        assert not Path(rel).is_absolute()

    # notes: guitar_all.json + MIDI files + summary
    n = out["notes"]
    ga = NoteSet.model_validate(atomic.read_json(n / "guitar_all.json"))
    assert ga.notes and {x.instrument for x in ga.notes} <= set(I.GUITAR_CLASSES)
    assert sorted(p.name for p in (n / "midi").iterdir()) == ["b0_guitar.mid", "bass_raw.mid", "guitar_all.mid"]
    summ = atomic.read_json(n / "notes_summary.json")
    assert summ["midi_bar_offset"] == 1 and summ["basic_pitch"] is False  # one lead-in bar (T0 = 0.37 s)
    assert summ["time_offset_applied_s"] == -LAT
    mid = mido.MidiFile(str(n / "midi" / "bass_raw.mid"))
    assert [t.name for t in mid.tracks] == ["tempo", "electric_bass"]
    assert mid.tracks[1][1].program == 33


def test_cpu_stage_outputs_are_byte_identical(job, monkeypatch):
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    a = _run_chain(job, _cfg(), "a")
    b = _run_chain(job, _cfg(), "b")  # second run: MuScriptor views come from the job's transcription cache
    assert len(job.fake.calls) == 2  # the cache answered every view of the second run
    for stage in ("amt_ms1", "amt_gtr", "instr", "notes"):
        assert _data_files(a[stage]) == _data_files(b[stage]), stage
    sa, sb = atomic.read_json(a["s35"] / "s35.json"), atomic.read_json(b["s35"] / "s35.json")
    assert json.dumps(sa).replace("/a/", "/b/") == json.dumps(sb)  # only the stage-dir names differ


def test_fast_profile_drops_the_mix_view_and_b0(job, monkeypatch):
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    out = _run_chain(job, {"amt": {"latency_s": LAT}, "run": {"profile": "fast"}}, "f")
    assert [v["id"] for v in job.fake.calls[0]["views"]] == ["bass_mono", "piano_other_mono"]
    inst = atomic.read_json(out["instr"] / "instrumentation.json")
    assert inst["profile"] == "fast"
    assert all("mix_transcription" not in ev["sources"] for ev in inst["classes"].values())
    assert not (out["notes"] / "midi" / "b0_guitar.mid").exists()


def test_mix_view_as_guitar_view_does_not_collide_in_s35(job, monkeypatch):
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    out = _run_chain(job, _cfg(guitar_view="mix_mono"), "m")
    s35 = atomic.read_json(out["s35"] / "s35.json")
    assert s35["raw"]["amt_ms1"]["mix_mono__muscriptor"] != s35["raw"]["amt_gtr"]["mix_mono__muscriptor"]
    gtr_mask = job.fake.calls[1]["views"][0]["instruments"]
    assert gtr_mask is not None and set(I.GUITAR_CLASSES) <= set(gtr_mask)


def test_basic_pitch_stage_with_a_fake_worker(job, monkeypatch):
    py = job.root / "bp310" / "python.exe"
    py.parent.mkdir()
    py.write_bytes(b"")
    onnx = job.root / "bp310" / "nmp.onnx"
    onnx.write_bytes(b"onnx")
    monkeypatch.setattr(S, "bp310_python", lambda: py)
    monkeypatch.setattr(S, "bp_model_path", lambda python=None: onnx)
    recorded = []
    monkeypatch.setattr(S, "record_bp_model", lambda: recorded.append(1))
    calls = []

    def fake_run_worker(name, req, work_dir, *, timeout_s=None, python=None, use_lock=True, lock_timeout_s=None):
        calls.append((name, req, python, use_lock))
        for v in req["views"]:
            atomic.write_json(Path(v["out"]), {
                "format": "gtab.notes/1", "view": v["id"], "view_sha256": v["view_sha256"],
                "backend": {"name": "basicpitch", "version": "0.4.0", "model": "icassp_2022/nmp.onnx",
                            "model_sha256": "o" * 64, "params": req["params"], "instruments": None,
                            "device": "cpu", "dtype": "fp32"},
                "time_offset_applied_s": 0.0, "audio_duration_s": 20.0,
                "notes": [{"onset_s": 1.0, "offset_s": 1.2, "pitch": 64, "amplitude": 0.5,
                           "bend": {"t_s": [1.0, 1.0116], "cents": [0.0, 33.3333]}}]})
        return SimpleNamespace(ok=True, output={"status": "ok"}, error=None, stderr_path=Path(work_dir) / "e.log")

    import gtab.gpu

    monkeypatch.setattr(gtab.gpu, "run_worker", fake_run_worker)
    out = _run_chain(job, _cfg(), "p")
    name, req, python, use_lock = calls[0]
    assert name == "amt_basicpitch" and python == py and use_lock is False
    assert req["params"]["min_note_frames"] == 7 and req["threads"] == 4 and req["allow_default_min_len"] is False
    assert [v["id"] for v in req["views"]] == ["guitar_mono", "bass_mono"]
    assert recorded == [1]
    bp = NoteSet.model_validate(atomic.read_json(out["amt_bp"] / "raw" / "guitar_mono__basicpitch.json"))
    assert bp.time_offset_applied_s == 0.0 and bp.notes[0].onset_s == 1.0  # Basic Pitch is not latency-corrected
    s35 = atomic.read_json(out["s35"] / "s35.json")
    assert s35["basic_pitch"]["skipped"] is False and s35["raw"]["amt_bp"]["guitar_mono__basicpitch"]
    assert (out["notes"] / "midi" / "guitar_basicpitch.mid").is_file()
    assert atomic.read_json(out["notes"] / "notes_summary.json")["basic_pitch"] is True


def test_bp_models_and_params_follow_the_env(job, monkeypatch):
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "missing" / "python.exe")
    assert S.bp_models({}) == {"basic_pitch": "absent"}
    p = S.bp_params({"amt": {"bp_min_note_frames": 5, "bp_onset_threshold": 0.4}})
    assert p["params"]["min_note_frames"] == 5 and p["params"]["onset_threshold"] == 0.4
    assert p["views"] == ["guitar_mono", "bass_mono"]


def test_ms_models_missing_weights_is_model_missing(monkeypatch):
    monkeypatch.setattr(S, "muscriptor_weights", lambda size="medium", revision=None: None)
    with pytest.raises(models.ModelMissing) as ei:
        S.ms_models({})
    assert ei.value.name == "muscriptor-medium" and "gtab run 중에는 내려받지 않습니다" in ei.value.hint_ko


# ------------------------------------------------------------------------------------------- config reads


def test_amt_defaults_mirror_the_config_defaults():
    cfg = Config()
    for key, value in S.AMT_DEFAULTS.items():
        assert cfg.get(key) == value, key


def test_config_is_read_strictly(monkeypatch):
    cfg = Config()
    assert S.amt_value(cfg, "amt.bp_threads") == 4
    monkeypatch.setitem(S.AMT_DEFAULTS, "amt.no_such_key", 1)
    with pytest.raises(KeyError):
        S.amt_value(cfg, "amt.no_such_key")  # a Config never falls back to the mirror
    assert S.amt_value({}, "amt.no_such_key") == 1  # plain dicts (tests, experiments) do


def test_stage_params_from_a_real_config():
    cfg = Config.model_validate({"amt": {"guitar_mask": "guitar_only", "latency_s": 0.012}})
    p = S.amt_gtr_params(cfg)
    assert p["guitar_mask"] == "guitar_only" and p["latency_s"] == 0.012 and p["guitar_view"] == "guitar_mono"
    assert S.amt_ms1_params(cfg)["views"] == ["mix_mono", "bass_mono", "piano_other_mono"]
    ip = S.instr_params(cfg)
    assert ip["hints_instruments"] == "auto" and ip["threshold"] == 0.5


def test_unknown_hint_family_fails_at_plan_time():
    from gtab.config import ConfigError

    with pytest.raises(ConfigError):
        S.instr_params({"hints": {"instruments": "guitar,kazoo"}})


# ------------------------------------------------------------------------------------- model registry


def test_record_bp_model_is_idempotent(tmp_path, monkeypatch):
    root = tmp_path / "root"
    py = root / "envs" / "bp310" / ".venv" / "Scripts" / "python.exe"
    onnx = root / "envs" / "bp310" / ".venv" / "Lib" / "site-packages" / "basic_pitch" / "saved_models" / \
        "icassp_2022" / "nmp.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"onnx-model")
    py.parent.mkdir(parents=True)
    py.write_bytes(b"")
    monkeypatch.setattr(paths, "GTAB_ROOT", root)
    monkeypatch.setattr(paths, "BP310_PYTHON", py)
    monkeypatch.setattr(paths, "MODELS", root / "data" / "models")
    monkeypatch.setattr(paths, "MODELS_LOCK", root / "data" / "models" / "models.lock.json")
    monkeypatch.setattr(S, "dist_version", lambda python, dist: "0.4.0")
    e = S.record_bp_model()
    assert e is not None and e.name == "basic_pitch.icassp2022"
    assert e.path == "envs/bp310/.venv/Lib/site-packages/basic_pitch/saved_models/icassp_2022/nmp.onnx"
    assert e.bytes == len(b"onnx-model") and "Apache-2.0" in e.license and "0.4.0" in e.url
    assert models.get("basic_pitch.icassp2022").sha256 == e.sha256 and models.verify("basic_pitch.icassp2022")
    stamp = (root / "data" / "models" / "models.lock.json").read_bytes()
    assert S.record_bp_model().recorded_utc == e.recorded_utc  # unchanged file -> no rewrite
    assert (root / "data" / "models" / "models.lock.json").read_bytes() == stamp
    onnx.write_bytes(b"onnx-model-v2")
    assert S.record_bp_model().sha256 != e.sha256


def test_record_bp_model_without_env(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "BP310_PYTHON", tmp_path / "missing" / "python.exe")
    assert S.record_bp_model() is None
