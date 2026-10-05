"""AMT pipeline stages end to end with fake workers (bandscribe.amt.stage, M1_M2_SPEC 3.2 rows 8-13).

amt_ms1 -> instr -> amt_gtr -> amt_bp (skipped and faked) -> s35 -> notes on a tiny synthetic job: outputs exist,
validate against the P0 schemas, the latency constant is applied, CPU stage data outputs are byte-identical across
runs, and the config is read strictly. Profiles (2026-10-04): every profile runs the bass pass and a sampled
piano+other presence pass and estimates the instrumentation alike; eval adds the B0 mix pass and the full-length
piano+other pass as extra outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mido
import numpy as np
import pytest

from bandscribe import atomic, models, paths, vram
from bandscribe.amt import instruments as I
from bandscribe.amt import stage as S
from bandscribe.config import Config
from bandscribe.schema.instrumentation import Instrumentation
from bandscribe.schema.notes import NoteSet

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
    return {"format": "bandscribe.grid/1", "tpb": 48, "duration_s": T0 + N_BARS * 4 * BEAT, "beats": beats, "bars": bars,
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
    atomic.write_json(deps["stems"] / "views.json", {"format": "bandscribe.views/1", "views": views})
    n_fr = int((T0 + N_BARS * 2.0) * 10) + 1
    energy = {"mix": np.full(n_fr, -20.0, np.float32)}
    for stem, rel in (("guitar", -6.0), ("bass", -6.0), ("piano", -10.0), ("other", -60.0), ("vocals", -60.0),
                      ("drums", -8.0), ("nonvox", -3.0), ("leftover", -50.0)):
        energy[stem] = np.full(n_fr, -20.0 + rel, np.float32)
    energy["fps"] = np.array([10.0], np.float32)
    np.savez(deps["stems"] / "energy.npz", **energy)
    atomic.write_json(deps["grid"] / "grid.json", _grid())
    atomic.write_json(deps["sections"] / "sections.json", {
        "format": "bandscribe.sections/1", "method": "test", "params": {},
        "sections": [{"id": "S1", "label": "A", "start_bar": 1, "end_bar": 5, "start_s": T0, "end_s": T0 + 8.0,
                      "confidence": 1.0},
                     {"id": "S2", "label": "B", "start_bar": 5, "end_bar": 9, "start_s": T0 + 8.0,
                      "end_s": T0 + 16.0, "confidence": 1.0}]})
    atomic.write_json(deps["vocal"] / "vocal_activity.json", {"format": "bandscribe.vocal/1"})
    atomic.write_json(deps["resid1"] / "resid1.json", {"format": "bandscribe.resid1/1"})
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


def _sorted(notes: list[dict]) -> list[dict]:
    return sorted(notes, key=lambda n: (n["onset_s"], n["pitch"], n["offset_s"], n["instrument"]))


class FakeMs:
    """bandscribe.vram.run_gpu_stage stand-in: writes raw NoteSets for the requested views like amt_muscriptor.

    piano+other (full or presence segments): ``po_class`` notes (default synth_pad) inside the first segment.
    Guitar view: distorted guitar notes; with ``gtr_bleed`` also ``po_class`` notes when the mask allows that
    class (MuScriptor labelling stem bleed non-guitar). Off by default: on the real 怪獣の花唄 run the guitar view
    labelled no note non-guitar even with strings/keys in its mask (review 2026-10-04)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.po_class = "synth_pad"
        self.gtr_bleed = False
        self.warnings: list[str] = []  # the worker's own (Korean) warnings, as amt_muscriptor reports them

    def __call__(self, backend, labels, req, cfg, work_dir, notify=None):
        self.calls.append(json.loads(json.dumps(req, default=str)))
        for v in req["views"]:
            mask = v["instruments"]
            segs = v.get("segments")
            if v["id"] == "bass_mono":
                notes = _notes("electric_bass", 36, 16)
            elif v["id"] == "mix_mono" and mask is None:
                notes = _sorted(_notes("distorted_electric_guitar", 52, 24) + _notes("acoustic_piano", 60, 24)
                                + _notes("electric_bass", 36, 8))
            elif v["id"].startswith("piano_other"):
                t0 = segs[0][0] if segs else T0
                notes = [dict(n, onset_s=round(n["onset_s"] - T0 + t0, 6), offset_s=round(n["offset_s"] - T0 + t0, 6))
                         for n in _notes(self.po_class, 60, 20)]
            else:
                classes = mask or list(I.GUITAR_CLASSES)
                notes = _notes("distorted_electric_guitar", 52, 20)
                if self.gtr_bleed and self.po_class in classes:
                    notes = _sorted(notes + _notes(self.po_class, 72, 6))
            doc = {"format": "bandscribe.notes/1", "view": v["id"], "view_sha256": v["view_sha256"],
                   "backend": {"name": "muscriptor", "version": "0.3.0", "model": "muscriptor-medium",
                               "model_sha256": "m" * 64, "params": {}, "instruments": mask, "device": "cuda",
                               "dtype": "fp32+autocast-fp16"},
                   "time_offset_applied_s": 0.0, "audio_duration_s": 20.0}
            if segs:
                doc["segments"] = segs
            doc["notes"] = notes
            atomic.write_json(Path(v["out"]), doc)
        out = {"status": "ok", "vram": {"rung_label": labels[0], "attempts": []},
               "backend": {"device_used": "cuda", "model_sha256": "m" * 64}, "warnings": list(self.warnings)}
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
    for stage, deps in (("amt_ms1", ("stems", "grid", "sections")), ("amt_bp", ("stems",)),
                        ("instr", ("amt_ms1", "stems", "sections", "grid")), ("amt_gtr", ("stems", "instr")),
                        ("s35", ("instr", "vocal", "resid1", "amt_ms1", "amt_gtr", "amt_bp")),
                        ("notes", ("amt_gtr", "amt_ms1", "amt_bp", "grid", "s35", "stems"))):
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
    """Default (quality) profile: no B0 mix pass; the piano+other presence pass on sampled windows."""
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    job.fake.gtr_bleed = True  # exercise the non-guitar drop path of notes
    out = _run_chain(job, _cfg(), "a")

    # amt_ms1: bass + presence windows (no mix view), latency applied, gpu_run.json, presence.json
    ms1 = out["amt_ms1"]
    names = sorted(p.name for p in (ms1 / "raw").iterdir())
    assert names == ["bass_mono__muscriptor.json", "piano_other_presence__muscriptor.json"]
    pres = NoteSet.model_validate(atomic.read_json(ms1 / "raw" / "piano_other_presence__muscriptor.json"))
    assert pres.segments == [(T0, T0 + 10.0)]  # 16.37 s song: one 10 s window, starting on bar 1
    assert pres.time_offset_applied_s == -LAT and pres.notes[0].onset_s == pytest.approx(T0 - LAT)
    gr = vram.read_gpu_run(ms1)
    assert gr["backend"] == "amt_muscriptor" and gr["rung"]["label"] == "medium-lean" and gr["degraded"] is False
    req = job.fake.calls[0]
    assert [v["id"] for v in req["views"]] == ["bass_mono", "piano_other_presence"]
    assert req["views"][0]["instruments"] == I.mask_bass() and "segments" not in req["views"][0]
    pv = req["views"][1]
    assert pv["instruments"] == I.mask_piano_other("keys+guitar") and pv["segments"] == [[T0, T0 + 10.0]]
    assert pv["wav"].endswith("piano_other_mono.wav")  # windows are cut from the piano+other view
    pj = atomic.read_json(ms1 / "presence.json")
    assert pj["mode"] == "sampled" and pj["view"] == "piano_other_presence" and pj["skipped"] is None
    assert pj["file"] == "raw/piano_other_presence__muscriptor.json"
    assert pj["selection"]["windows"][0]["section"] == "S1" and pj["selection"]["n_target"] == 1
    assert "string_ensemble" in pj["evidence_classes"] and "acoustic_guitar" not in pj["evidence_classes"]

    # amt_bp skipped (no bp310 env)
    assert atomic.read_json(out["amt_bp"] / "skipped.json") == {"reason": "bp310 env missing"}

    # instr: schema-valid; synth_pad present through the presence pass (agreeing with the active piano stem);
    # the piano stem term goes to synth_pad, the class the pass found, not to the keys classes; the bass pass
    # decides the bass classes; the guitar mask keeps the guitar classes and adds the present keys/synth classes
    inst = Instrumentation.model_validate(atomic.read_json(out["instr"] / "instrumentation.json"))
    assert inst.classes["distorted_electric_guitar"].present
    pad = inst.classes["synth_pad"]
    assert pad.present and pad.sources["presence_transcription"]["n_notes"] == 20
    assert pad.sources["presence_transcription"]["stem_agree"] == 1.0
    assert "mix_transcription" not in pad.sources
    ap = inst.classes["acoustic_piano"]
    assert not ap.present and ap.sources["stem_energy"]["stems"]["piano"]["attributed"] is False
    assert inst.classes["electric_bass"].present and not inst.classes["acoustic_bass"].present
    assert inst.classes["electric_bass"].sources["bass_transcription"]["n_notes"] == 16
    gp = inst.masks["guitar_pass"]
    assert gp == I.sort_mt3([*I.GUITAR_CLASSES, "synth_pad"])
    assert [s["section"] for s in inst.per_section] == ["S1", "S2"]
    assert inst.presence["mode"] == "sampled" and inst.presence["class_notes"] == {"synth_pad": 20}
    assert inst.presence["windows"] == [[T0, T0 + 10.0]] and inst.presence["window_sections"] == ["S1"]
    assert inst.presence["rejected"] == {}
    assert inst.warnings == ["건반(piano) 스템 에너지는 있는데 전사에 건반 음이 거의 없음: 건반이 있는지 확인하세요."]

    # amt_gtr: the guitar view with the instrumentation's mask
    gtr_req = job.fake.calls[1]
    assert [v["id"] for v in gtr_req["views"]] == ["guitar_mono"] and gtr_req["views"][0]["instruments"] == gp
    assert (out["amt_gtr"] / "raw" / "guitar_mono__muscriptor.json").is_file()

    # s35: job-relative paths grouped by stage, Basic Pitch null, no B0 in quality
    s35 = atomic.read_json(out["s35"] / "s35.json")
    assert s35["raw"]["amt_bp"] == {"bass_mono__basicpitch": None, "guitar_mono__basicpitch": None}
    assert s35["guitar_pass"] == "stages/amt_gtr/a/raw/guitar_mono__muscriptor.json"
    assert s35["guitar_pass_mask"] == gp and s35["guitar_pass_non_guitar"] == {"synth_pad": 6}  # raw, unfiltered
    assert s35["b0"] is None and s35["presence"] == "stages/amt_ms1/a/presence.json"
    assert s35["basic_pitch"] == {"skipped": True, "reason": "bp310 env missing"}
    for rel in (s35["instrumentation"], s35["vocal_activity"], s35["guitar_pass"]):
        assert not Path(rel).is_absolute()

    # notes: guitar_all.json + MIDI files + summary; the synth_pad notes of the guitar view are dropped
    n = out["notes"]
    ga = NoteSet.model_validate(atomic.read_json(n / "guitar_all.json"))
    assert ga.notes and {x.instrument for x in ga.notes} == {"distorted_electric_guitar"}
    assert sorted(p.name for p in (n / "midi").iterdir()) == ["bass_raw.mid", "guitar_all.mid"]
    summ = atomic.read_json(n / "notes_summary.json")
    assert summ["midi_bar_offset"] == 1 and summ["basic_pitch"] is False  # one lead-in bar (T0 = 0.37 s)
    assert summ["time_offset_applied_s"] == -LAT
    assert summ["guitar_classes"] == {"distorted_electric_guitar": 20}
    assert summ["non_guitar_dropped"] == {"synth_pad": 6} and summ["guitar_view_notes"] == 26
    gmid = mido.MidiFile(str(n / "midi" / "guitar_all.mid"))
    assert [t.name for t in gmid.tracks] == ["tempo", "distorted_electric_guitar"]
    assert sum(1 for m in gmid.tracks[1] if m.type == "note_on" and m.velocity > 0) == 20
    mid = mido.MidiFile(str(n / "midi" / "bass_raw.mid"))
    assert [t.name for t in mid.tracks] == ["tempo", "electric_bass"]
    assert mid.tracks[1][1].program == 33
    bass = NoteSet.model_validate(atomic.read_json(n / "bass_raw.json"))  # what quant reads
    assert len(bass.notes) == 16 and bass.view == "bass_mono"
    gate = summ["silence_gate"]  # every stem sounds in the fixture: nothing dropped
    assert gate["db"] == -50.0 and gate["guitar_all"]["gated"] and gate["guitar_all"]["view"] == "guitar_mono"
    assert gate["guitar_all"]["dropped"] == 0 and gate["bass_raw"]["dropped"] == 0


def test_notes_drop_what_starts_over_a_silent_stem(job, monkeypatch):
    """2026-10-05: MuScriptor wrote 81 G1 over the silent bass stem of seisyun complex (bars 5-17). Notes that start
    where their stem is silent leave guitar_all, bass_raw.json and both MIDI files; the other stem's notes stay."""
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    with np.load(job.deps["stems"] / "energy.npz") as z:
        e = {k: np.array(z[k]) for k in z.files}
    e["bass"][20:30] = -200.0  # 2.0-3.0 s: bass onsets 2.10 2.35 2.60 2.85 (after the 20 ms latency)
    e["guitar"][40:] = -200.0  # from 4.0 s: guitar onsets 4.10 .. 5.10
    np.savez(job.deps["stems"] / "energy.npz", **e)
    out = _run_chain(job, _cfg(), "g")
    n = out["notes"]
    bass = atomic.read_json(n / "bass_raw.json")["notes"]
    assert len(bass) == 12 and not [x for x in bass if 2.0 < x["onset_s"] < 3.0]
    gtr = atomic.read_json(n / "guitar_all.json")["notes"]
    assert len(gtr) == 15 and max(x["onset_s"] for x in gtr) < 4.0
    gate = atomic.read_json(n / "notes_summary.json")["silence_gate"]
    assert gate["bass_raw"]["dropped"] == 4 and gate["bass_raw"]["spans_s"] == [[2.1, 2.85]]
    assert gate["guitar_all"]["dropped"] == 5
    for name, k in (("bass_raw.mid", 12), ("guitar_all.mid", 15)):
        mid = mido.MidiFile(str(n / "midi" / name))
        assert sum(1 for m in mid.tracks[1] if m.type == "note_on" and m.velocity > 0) == k
    off = _run_chain(job, _cfg(silence_gate_db=-200.0), "o")  # amt.silence_gate_db = -200: the gate is off
    assert len(atomic.read_json(off["notes"] / "bass_raw.json")["notes"]) == 16
    assert atomic.read_json(off["notes"] / "notes_summary.json")["silence_gate"]["bass_raw"]["reason"] == "off"


def test_eval_profile_adds_b0_and_the_full_pass_but_estimates_like_quality(job, monkeypatch):
    """Review 2026-10-04: eval's extra passes are outputs only. Its instrumentation (and so the guitar mask, i.e.
    the system under test) equals quality's; the B0 mix pass is reported with weight 0."""
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    out = _run_chain(job, {"amt": {"latency_s": LAT}, "run": {"profile": "eval"}}, "e")
    req = job.fake.calls[0]
    assert [v["id"] for v in req["views"]] == ["mix_mono", "bass_mono", "piano_other_presence", "piano_other_mono"]
    assert req["views"][0]["instruments"] is None and "segments" not in req["views"][3]
    ms1 = out["amt_ms1"]
    mix = NoteSet.model_validate(atomic.read_json(ms1 / "raw" / "mix_mono__muscriptor.json"))
    assert mix.time_offset_applied_s == -LAT and mix.notes[0].onset_s == pytest.approx(T0 - LAT)
    assert (ms1 / "raw" / "piano_other_mono__muscriptor.json").is_file()  # extra full-length output
    pj = atomic.read_json(ms1 / "presence.json")
    assert pj["mode"] == "sampled" and pj["view"] == "piano_other_presence"  # the presence source is quality's
    ev_inst = atomic.read_json(out["instr"] / "instrumentation.json")
    pad = ev_inst["classes"]["synth_pad"]["sources"]
    assert pad["presence_transcription"]["mode"] == "sampled"
    ap = ev_inst["classes"]["acoustic_piano"]["sources"]["mix_transcription"]
    assert ap["active_bars"] >= 2 and ap["w"] == 0.0 and ap["used"] is False
    q = _run_chain(job, _cfg(), "q")
    q_inst = atomic.read_json(q["instr"] / "instrumentation.json")
    assert {c: (v["p"], v["present"]) for c, v in ev_inst["classes"].items()} == \
        {c: (v["p"], v["present"]) for c, v in q_inst["classes"].items()}
    assert ev_inst["masks"] == q_inst["masks"] and ev_inst["profile"] == "eval"
    gtr_e = atomic.read_json(out["amt_gtr"] / "raw" / "guitar_mono__muscriptor.json")
    gtr_q = atomic.read_json(q["amt_gtr"] / "raw" / "guitar_mono__muscriptor.json")
    assert gtr_e == gtr_q  # the same guitar pass (cache hit with the same mask)
    s35 = atomic.read_json(out["s35"] / "s35.json")
    assert s35["b0"] == "stages/amt_ms1/e/raw/mix_mono__muscriptor.json"
    assert sorted(p.name for p in (out["notes"] / "midi").iterdir()) == ["b0_guitar.mid", "bass_raw.mid",
                                                                         "guitar_all.mid"]


def test_quiet_piano_other_skips_the_presence_pass(job, monkeypatch):
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    with np.load(job.deps["stems"] / "energy.npz") as z:
        e = {k: np.array(z[k]) for k in z.files}
    e["piano"][:] = -200.0  # piano and other silent
    np.savez(job.deps["stems"] / "energy.npz", **e)
    out = _run_chain(job, _cfg(), "q")
    assert [v["id"] for v in job.fake.calls[0]["views"]] == ["bass_mono"]
    pj = atomic.read_json(out["amt_ms1"] / "presence.json")
    assert pj["skipped"] == "piano_other_inactive" and pj["file"] is None
    inst = atomic.read_json(out["instr"] / "instrumentation.json")
    assert inst["presence"]["skipped"] == "piano_other_inactive"
    # the skipped pass covered its classes with e = 0: nominal weights, no renormalisation (review 2026-10-04)
    src = inst["classes"]["organ"]["sources"]
    assert src["presence_transcription"]["e"] == 0.0 and src["presence_transcription"]["skipped"]
    assert "presence_transcription" not in inst["classes"]["trumpet"]["sources"]
    assert inst["masks"]["guitar_pass"] == I.mask_guitar_only()


def test_cpu_stage_outputs_are_byte_identical(job, monkeypatch):
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    a = _run_chain(job, _cfg(), "a")
    b = _run_chain(job, _cfg(), "b")  # second run: MuScriptor views come from the job's transcription cache
    assert len(job.fake.calls) == 2  # the cache answered every view (presence windows included) of the second run
    for stage in ("amt_ms1", "amt_gtr", "instr", "notes"):
        assert _data_files(a[stage]) == _data_files(b[stage]), stage
    sa, sb = atomic.read_json(a["s35"] / "s35.json"), atomic.read_json(b["s35"] / "s35.json")
    assert json.dumps(sa).replace("/a/", "/b/") == json.dumps(sb)  # only the stage-dir names differ


def test_fast_profile_drops_the_mix_view_and_b0(job, monkeypatch):
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    out = _run_chain(job, {"amt": {"latency_s": LAT}, "run": {"profile": "fast"}}, "f")
    assert [v["id"] for v in job.fake.calls[0]["views"]] == ["bass_mono", "piano_other_presence"]
    inst = atomic.read_json(out["instr"] / "instrumentation.json")
    assert inst["profile"] == "fast"
    assert all("mix_transcription" not in ev["sources"] for ev in inst["classes"].values())
    assert inst["classes"]["synth_pad"]["present"]  # the presence pass still runs in fast
    assert not (out["notes"] / "midi" / "b0_guitar.mid").exists()


def test_presence_windows_follow_the_profile_budget(job, monkeypatch):
    """A longer song: quality samples up to 6 x 10 s within 20 % of the song, fast half the windows."""
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    n_bars, bar_s = 120, 2.0  # 240 s song, 12 sections of 10 bars (labels A, B, C alternating)
    grid = _grid()
    grid["bars"] = [{"bar": b + 1, "start_s": round(T0 + b * bar_s, 6), "end_s": round(T0 + (b + 1) * bar_s, 6),
                     "n_beats": 4, "numerator": 4, "denominator": 4, "beat_unit": "quarter", "pickup": False}
                    for b in range(n_bars)]
    grid["duration_s"] = T0 + n_bars * bar_s
    atomic.write_json(job.deps["grid"] / "grid.json", grid)
    secs = [{"id": f"S{i + 1}", "label": "ABC"[i % 3], "start_bar": 1 + 10 * i, "end_bar": 11 + 10 * i,
             "start_s": T0 + 20.0 * i, "end_s": T0 + 20.0 * (i + 1), "confidence": 1.0} for i in range(12)]
    atomic.write_json(job.deps["sections"] / "sections.json",
                      {"format": "bandscribe.sections/1", "method": "test", "params": {}, "sections": secs})
    n_fr = int(grid["duration_s"] * 10) + 1
    e = {k: np.full(n_fr, -20.0 + rel, np.float32) for k, rel in (("mix", 0.0), ("guitar", -6.0), ("bass", -6.0),
                                                                  ("piano", -10.0), ("other", -60.0))}
    e["fps"] = np.array([10.0], np.float32)
    np.savez(job.deps["stems"] / "energy.npz", **e)
    for prof, n_win in (("quality", 4), ("fast", 3)):  # quality: 20 % of 240 s = 48 s -> 4 windows; fast: 6 // 2
        job.fake.calls.clear()
        cfg = {"amt": {"latency_s": LAT}, "run": {"profile": prof}}
        o = job.store.job_dir(SONG) / "stages" / "amt_ms1" / prof
        S.STAGES["amt_ms1"]["run"](_ctx(job, "amt_ms1", o, {k: job.deps[k] for k in ("stems", "grid", "sections")},
                                        cfg))
        # (the bass view of the second profile comes from the job's transcription cache)
        segs = next(v for v in job.fake.calls[0]["views"] if v["id"] == "piano_other_presence")["segments"]
        assert len(segs) == n_win and all(b - a == pytest.approx(10.0) for a, b in segs)
        assert segs == sorted(segs) and all(b <= a2 for (_a, b), (a2, _b) in zip(segs, segs[1:]))
        labels = [w["label"] for w in atomic.read_json(o / "presence.json")["selection"]["windows"]]
        assert set(labels) == {"A", "B", "C"}  # every distinct section label is sampled


def test_mix_view_as_guitar_view_does_not_collide_in_s35(job, monkeypatch):
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    out = _run_chain(job, {"amt": {"latency_s": LAT, "guitar_view": "mix_mono"}, "run": {"profile": "eval"}}, "m")
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
                "format": "bandscribe.notes/1", "view": v["id"], "view_sha256": v["view_sha256"],
                "backend": {"name": "basicpitch", "version": "0.4.0", "model": "icassp_2022/nmp.onnx",
                            "model_sha256": "o" * 64, "params": req["params"], "instruments": None,
                            "device": "cpu", "dtype": "fp32"},
                "time_offset_applied_s": 0.0, "audio_duration_s": 20.0,
                "notes": [{"onset_s": 1.0, "offset_s": 1.2, "pitch": 64, "amplitude": 0.5,
                           "bend": {"t_s": [1.0, 1.0116], "cents": [0.0, 33.3333]}}]})
        return SimpleNamespace(ok=True, output={"status": "ok"}, error=None, stderr_path=Path(work_dir) / "e.log")

    import bandscribe.gpu

    monkeypatch.setattr(bandscribe.gpu, "run_worker", fake_run_worker)
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
    assert ei.value.name == "muscriptor-medium" and "bandscribe run 중에는 내려받지 않습니다" in ei.value.hint_ko


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
    ms1 = S.amt_ms1_params(cfg)
    assert ms1["views"] == ["bass_mono", "piano_other_presence"]
    assert ms1["presence"] == {"mode": "sampled", "window_s": 10.0, "max_windows": 6, "max_total_s": 60.0,
                               "max_fraction": 0.2, "min_active": 0.5, "active_rel_db": -40.0}
    ip = S.instr_params(cfg)
    assert ip["hints_instruments"] == "auto" and ip["threshold"] == 0.5 and ip["presence_min_notes"] == 8


@pytest.mark.parametrize("profile,presence,views", [
    ("quality", "auto", ["bass_mono", "piano_other_presence"]),
    ("fast", "auto", ["bass_mono", "piano_other_presence"]),
    ("eval", "auto", ["mix_mono", "bass_mono", "piano_other_presence", "piano_other_mono"]),
    ("quality", "full", ["bass_mono", "piano_other_mono"]),
    ("eval", "sampled", ["mix_mono", "bass_mono", "piano_other_presence", "piano_other_mono"]),
    ("eval", "full", ["mix_mono", "bass_mono", "piano_other_mono"]),
])
def test_profile_view_lists(profile, presence, views):
    assert S.pass1_views(profile, presence) == views
    p = S.amt_ms1_params({"run": {"profile": profile}, "amt": {"presence_pass": presence}})
    assert p["views"] == views and p["profile"] == profile
    assert p["presence"]["mode"] == ("full" if presence == "full" else "sampled")


def test_profiles_change_the_amt_ms1_key_params_and_fast_halves_the_budget():
    q, f, e = (S.amt_ms1_params({"run": {"profile": x}}) for x in ("quality", "fast", "eval"))
    assert q != f != e != q
    assert (f["presence"]["max_windows"], f["presence"]["max_total_s"]) == (3, 30.0)
    assert S.amt_ms1_params({"amt": {"presence_window_s": 12.0}}) != q  # a budget knob changes the stage key
    from bandscribe.config import ConfigError

    with pytest.raises(ConfigError):
        S.amt_ms1_params({"run": {"profile": "max"}})
    with pytest.raises(ConfigError):
        S.amt_ms1_params({"amt": {"presence_pass": "some"}})


def test_muscriptor_stages_forward_the_workers_warnings(job, monkeypatch):
    """Review 2026-10-05: amt_ms1 / amt_gtr ignored the worker's warnings (slow rung, a view that never emitted
    EOS); they now go to the console through the stage notifier and stay in gpu_run.json for cached runs."""
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    eos = "bass_mono: MuScriptor 가 2개 구간에서 끝 토큰(EOS) 없이 생성 한도에 닿았습니다(12, 40 s 부근). ..."
    job.fake.warnings = [eos]
    events: list[tuple[str, dict]] = []
    out = job.store.job_dir(SONG) / "stages" / "amt_ms1" / "w"
    ctx = _ctx(job, "amt_ms1", out, {k: job.deps[k] for k in ("stems", "grid", "sections")}, _cfg())
    ctx.progress = lambda e, i: events.append((e, i))
    S.STAGES["amt_ms1"]["run"](ctx)
    assert ("warning", {"stage": "amt_ms1", "message": eos}) in events
    assert json.loads((out / "gpu_run.json").read_text(encoding="utf-8"))["warnings"] == [eos]
    # a rerun whose views all come from the transcription cache has no worker, hence nothing to forward
    events.clear()
    out2 = job.store.job_dir(SONG) / "stages" / "amt_ms1" / "w2"
    ctx = _ctx(job, "amt_ms1", out2, {k: job.deps[k] for k in ("stems", "grid", "sections")}, _cfg())
    ctx.progress = lambda e, i: events.append((e, i))
    S.STAGES["amt_ms1"]["run"](ctx)
    assert not [e for e in events if e[0] == "warning"]
    assert json.loads((out2 / "gpu_run.json").read_text(encoding="utf-8"))["warnings"] == []


def test_stage_code_versions_were_bumped_for_the_profile_change():
    assert {n: S.STAGES[n]["code_version"] for n in ("amt_ms1", "instr", "amt_gtr", "s35", "notes")} == {
        "amt_ms1": "3", "instr": "4", "amt_gtr": "2", "s35": "4", "notes": "5"}


def test_load_presence_falls_back_to_an_m2_full_pass(tmp_path):
    """A pre-presence amt_ms1 dir (raw/piano_other_mono, no presence.json) still feeds instr as a full pass."""
    d = tmp_path / "ms1"
    atomic.write_json(d / "raw" / "piano_other_mono__muscriptor.json", {
        "format": "bandscribe.notes/1", "view": "piano_other_mono", "view_sha256": None,
        "backend": {"name": "muscriptor"}, "time_offset_applied_s": 0.0, "audio_duration_s": 9.0,
        "notes": _notes("organ", 60, 9)})
    pres = S.load_presence(d)
    assert pres["mode"] == "full" and pres["spans"] is None and len(pres["notes"]) == 9
    assert S.load_presence(tmp_path / "empty") is None


def test_load_presence_reads_an_empty_segment_list_as_a_sampled_pass(tmp_path):
    """Review 2026-10-04: ``segments: []`` (every window clipped away) used to read as a full pass with the
    song-wide saturation."""
    d = tmp_path / "ms1"
    atomic.write_json(d / "presence.json", {"format": "bandscribe.presence/1", "mode": "sampled",
                                            "view": "piano_other_presence", "skipped": None,
                                            "file": "raw/piano_other_presence__muscriptor.json",
                                            "selection": {"windows": [], "transcribed_s": 0.0}})
    atomic.write_json(d / "raw" / "piano_other_presence__muscriptor.json", {
        "format": "bandscribe.notes/1", "view": "piano_other_presence", "view_sha256": None, "segments": [],
        "backend": {"name": "muscriptor"}, "time_offset_applied_s": 0.0, "audio_duration_s": 9.0, "notes": []})
    pres = S.load_presence(d)
    assert pres["mode"] == "sampled" and pres["spans"] == []


def test_notes_warns_when_many_guitar_pass_notes_are_dropped(job, monkeypatch):
    monkeypatch.setattr(S, "bp310_python", lambda: job.root / "no-bp310" / "python.exe")
    job.fake.gtr_bleed = True  # 6 synth_pad of 26 notes = 23 %
    seen: list[tuple[str, dict]] = []
    cfg = {**_cfg(), "progress": None}
    out = _run_chain(job, cfg, "w")
    o = job.store.job_dir(SONG) / "stages" / "notes" / "w2"
    ctx = _ctx(job, "notes", o, {"amt_gtr": out["amt_gtr"], "amt_ms1": out["amt_ms1"], "amt_bp": out["amt_bp"],
                                 "grid": job.deps["grid"], "s35": out["s35"], "stems": job.deps["stems"]}, cfg)
    ctx.progress = lambda event, data: seen.append((event, data))
    S.STAGES["notes"]["run"](ctx)
    msgs = [d["message"] for e, d in seen if e == "warning"]
    assert len(msgs) == 1 and "23%(6/26)" in msgs[0] and "synth_pad 6" in msgs[0]
    job.fake.gtr_bleed = False
    seen.clear()
    out2 = _run_chain(job, _cfg(guitar_mask="guitar_only"), "n")
    ctx2 = _ctx(job, "notes", job.store.job_dir(SONG) / "stages" / "notes" / "n2",
                {"amt_gtr": out2["amt_gtr"], "amt_ms1": out2["amt_ms1"], "amt_bp": out2["amt_bp"],
                 "grid": job.deps["grid"], "s35": out2["s35"], "stems": job.deps["stems"]}, _cfg())
    ctx2.progress = lambda event, data: seen.append((event, data))
    S.STAGES["notes"]["run"](ctx2)
    assert not seen


def test_unknown_hint_family_fails_at_plan_time():
    from bandscribe.config import ConfigError

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
    monkeypatch.setattr(paths, "ROOT", root)
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
