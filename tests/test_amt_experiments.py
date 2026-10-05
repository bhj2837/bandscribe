"""AMT experiments with fake backends (bandscribe.amt.experiments, M1_M2_SPEC 9.2 item 10, §10).

The GPU/worker seams (``_separate``, ``_ms``, ``_reftx``, ``_bp_sweep``) are faked; the data plumbing is real:
pydantic ``DatasetTrack`` objects with small WAV parts, ``bandscribe.eval.remix.tierb_mix`` / ``make_scene``,
``bandscribe.amt.instrumentation`` and EVAL's metric code. Every experiment's rows are then fed to EVAL's pre-registered
decision code (``bandscribe.eval.experiments.evaluate`` with the docs/decisions.md spec) so arm, metric and column
names must line up with the registry.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bandscribe import datasets
from bandscribe.amt import experiments as E
from bandscribe.amt import instruments as I
from bandscribe.audio import write_f32
from bandscribe.eval import experiments as EX
from bandscribe.eval.runs import METRIC_COLUMNS
from bandscribe.schema.dataset import DatasetTrack, Part
from bandscribe.schema.gt import GtLine
from bandscribe.schema.notes import BackendInfo, NoteEvent, NoteSet

SR = 44100
DUR = 4.0


def _gt_notes(pitch0: int, n: int = 12, step: float = 0.3) -> list[NoteEvent]:
    return [NoteEvent(onset_s=round(0.2 + step * k, 6), offset_s=round(0.4 + step * k, 6), pitch=pitch0 + k % 4)
            for k in range(n)]


def _noteset(view: str, notes: list[NoteEvent]) -> NoteSet:
    return NoteSet(view=view, backend=BackendInfo(name="gt"), notes=notes)


def _tone(f: float, dur: float = DUR) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    return (0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _track(root: Path, dataset: str, tid: str, *, kinds=("guitar", "bass"), families=("guitar", "bass"),
           split: str | None = None, group: str | None = None, meta: dict | None = None) -> DatasetTrack:
    parts, lines, notes = [], [], {}
    for i, kind in enumerate(kinds):
        rel = f"{tid}/{kind}.wav"
        write_f32(root / rel, np.stack([_tone(110.0 * (i + 2)), _tone(110.0 * (i + 2))], axis=1), SR)
        lid = "L1" if kind == "guitar" else ("B" if kind == "bass" else f"X{i}")
        parts.append(Part(id=f"p{i}", kind=kind, audio=rel, channels=2, line=lid if kind in ("guitar", "bass") else None,
                          take=i + 1))
        if kind in ("guitar", "bass"):
            lines.append(GtLine(id=lid, name=lid, kind=kind, takes=[i + 1]))
            notes[lid] = _noteset(lid, _gt_notes(52 if kind == "guitar" else 36))
    return DatasetTrack(format="bandscribe.dataset_track/1", dataset=dataset, track_id=tid, split=split, group=group or tid,
                        tier="B" if dataset == "cambridge_mt" else "C", license="test", mix=None, parts=parts,
                        lines=lines, families_present=list(families), notes=notes, string_fret=None, beats_s=None,
                        downbeats_s=None, tuning=None, meta=meta or {})


class Fakes:
    """Stand-ins for the GPU / worker seams; MuScriptor 'transcribes' a view as the GT of its item shifted 10 ms."""

    def __init__(self, gt_by_wav: dict[str, list[NoteEvent]] | None = None) -> None:
        self.sep_calls: list[dict] = []
        self.ms_calls: list[dict] = []
        self.bp_calls: list[dict] = []
        self.gt_by_wav = gt_by_wav or {}
        self.default_notes = _gt_notes(52)

    def separate(self, mix_wav, out_dir, cfg, *, force_rung, input_lufs=None, dtype=None):
        self.sep_calls.append({"force_rung": force_rung, "input_lufs": input_lufs, "dtype": dtype})
        from bandscribe.audio import read_f32

        x, sr = read_f32(mix_wav)
        stems = {}
        for k, name in enumerate(("bass", "drums", "other", "vocals", "guitar", "piano")):
            p = Path(out_dir) / "stems" / f"{name}.wav"
            write_f32(p, x * (0.1 + 0.1 * k), sr)
            stems[name] = p
        return SimpleNamespace(stems=stems, rung={"index": 0, "label": force_rung, "device": "cuda"},
                               sep_json={"rtf": 6.0 if dtype != "fp16" else 7.5,
                                         "vram": {"attempts": [{"status": "ok", "max_reserved_mb":
                                                                1800.0 if dtype != "fp16" else 1300.0}]}})

    def ms(self, views, params, cfg, *, work_dir, cache, force_rung):
        self.ms_calls.append({"views": [v["id"] for v in views], "masks": [v.get("instruments") for v in views],
                              "segments": [v.get("segments") for v in views],
                              "force_rung": force_rung, "dtype": params["dtype"]})
        raws = {}
        for v in views:
            classes = v.get("instruments") or list(I.GUITAR_CLASSES)
            cls = classes[-1] if any(c in I.GUITAR_CLASSES for c in classes) else classes[0]
            base = self.gt_by_wav.get(Path(v["wav"]).name, self.default_notes)
            raws[v["id"]] = {"format": "bandscribe.notes/1", "view": v["id"], "notes": [
                {"onset_s": round(n.onset_s + 0.01, 6), "offset_s": round(n.offset_s + 0.01, 6), "pitch": n.pitch,
                 "instrument": cls} for n in base]}
        info = {"rung_label": force_rung, "worker_output": {
            "vram": {"attempts": [{"status": "ok", "max_reserved_mb": 2400.0 if params["dtype"] == "upstream"
                                   else 1700.0}]},
            "timing": {"rtf": 3.0}}}
        return raws, info

    def reftx(self, track, out_dir, cfg, *, force_rung):
        out = {}
        for line in track.lines:
            out[line.id] = {"format": "bandscribe.notes/1", "view": line.id,
                            "notes": [n.model_dump() for n in track.notes[line.id].notes]}
        return out

    def bp_sweep(self, views, settings, cfg, *, work_dir, model_output_cache):
        self.bp_calls.append({"views": [v["id"] for v in views], "n_settings": len(settings)})
        out = {}
        for v in views:
            base = self.gt_by_wav.get(Path(v["wav"]).name, self.default_notes)
            docs = []
            for s in settings:
                keep = base if s["min_note_frames"] < 12 else base[::2]  # the 12-frame arm drops short notes
                docs.append({"format": "bandscribe.notes/1", "view": v["id"], "notes": [
                    {"onset_s": n.onset_s, "offset_s": n.offset_s, "pitch": n.pitch} for n in keep]})
            out[v["id"]] = docs
        return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    roots = {}

    def root_of(name: str) -> Path:
        return roots.setdefault(name, tmp_path / "datasets" / name)

    tracks: dict[str, list[DatasetTrack]] = {}
    fakes = Fakes()
    monkeypatch.setattr(E, "_available", lambda name: name in tracks)
    monkeypatch.setattr(E, "_tracks", lambda name, split=None: [t for t in tracks.get(name, [])
                                                                if split is None or t.split == split])
    monkeypatch.setattr(E, "_root", root_of)
    monkeypatch.setattr(datasets, "dataset_root", root_of)
    monkeypatch.setattr(E, "_separate", fakes.separate)
    monkeypatch.setattr(E, "_ms", fakes.ms)
    monkeypatch.setattr(E, "_reftx", fakes.reftx)
    monkeypatch.setattr(E, "_bp_sweep", fakes.bp_sweep)
    monkeypatch.setattr(E, "ms_rung", lambda params: "medium")  # no HF weights needed
    monkeypatch.setattr(E.S, "dist_version", lambda python, dist: "0.0-test")
    return SimpleNamespace(tracks=tracks, fakes=fakes, root=root_of, out=tmp_path / "run")


def _check_rows(rows: list[dict], exp_id: str) -> None:
    assert rows, exp_id
    for r in rows:
        assert set(r) <= set(METRIC_COLUMNS), set(r) - set(METRIC_COLUMNS)
        json.dumps(r)  # plain values only
        if r["metric"] == E.METRIC_F1:
            assert r["rung"], r
            assert 0.0 <= r["value"] <= 1.0
    entry = EX.load_registry()[exp_id]
    rows = [dict(r, suite=f"exp:{exp_id}") for r in rows]
    summary = EX.evaluate(rows, entry.spec, n=200, seed=1)
    for role in ("baseline", "candidate"):
        if entry.spec.get(role):
            assert entry.spec[role] in summary["arms"], (role, summary["arms"])
    assert not [w for w in summary["warnings"] if "rung" in w], summary["warnings"]


def _tierb(env, n: int = 2) -> None:
    root = env.root("cambridge_mt")
    env.tracks["cambridge_mt"] = [_track(root, "cambridge_mt", f"song{i}") for i in range(n)]


def test_e1_arms_rows_and_pinned_rungs(env):
    _tierb(env)
    rows = E.run_e1(env.out, {}, {})
    assert {r["system"] for r in rows} == {"off", "-14", "-16"}
    assert {r["line"] for r in rows} == {"guitar", "bass"}
    assert {r["rung"] for r in rows} == {"chunk=588800|medium"}
    assert [c["input_lufs"] for c in env.fakes.sep_calls[:3]] == ["off", -14.0, -16.0]
    assert all(c["force_rung"] == "chunk=588800" for c in env.fakes.sep_calls)
    assert all(c["force_rung"] == "medium" for c in env.fakes.ms_calls)
    assert env.fakes.ms_calls[0]["masks"] == [I.mask_guitar_only(), I.mask_bass()]
    assert (env.out / "work" / "cambridge_mt-song0" / "mix.wav").is_file()  # tierb_mix ran for real
    _check_rows(rows, "E1")


def test_tier_b_data_arg_and_missing_data(env):
    with pytest.raises(E.ExperimentDataMissing):
        E.run_e1(env.out, {}, {})
    _tierb(env)
    with pytest.raises(E.ExperimentDataMissing):
        E.run_e2(env.out, {}, {"data": "medleydb"})
    assert E.run_e2(env.out, {}, {"data": "cambridge_mt"})


def test_e2_views_use_the_guitar_only_mask(env):
    _tierb(env)
    rows = E.run_e2(env.out, {}, {})
    assert {r["system"] for r in rows} == set(E.E2_VIEWS)
    call = env.fakes.ms_calls[0]
    assert call["views"] == list(E.E2_VIEWS) and all(m == I.mask_guitar_only() for m in call["masks"])
    _check_rows(rows, "E2")


def test_reference_copies_across_guitar_lines_count_once(env):
    """Review 2026-10-05: two guitar lines playing the same note (double-tracked, 20 ms apart) are one note to a
    transcription of the mixed guitar stem. The old pooling counted both, so recall was capped at 50 % here and
    every arm's F1 was biased; references and estimates are now merged like eval.suites (50 ms, same pitch)."""
    root = env.root("cambridge_mt")
    tr = _track(root, "cambridge_mt", "dbl")
    l1 = list(tr.notes["L1"].notes)
    l2 = [NoteEvent(onset_s=round(n.onset_s + 0.02, 6), offset_s=round(n.offset_s + 0.02, 6), pitch=n.pitch)
          for n in l1]
    tr = tr.model_copy(update={"lines": tr.lines + [GtLine(id="L2", name="L2", kind="guitar", takes=[9])],
                               "notes": {**tr.notes, "L2": _noteset("L2", l2)}})
    env.tracks["cambridge_mt"] = [tr]
    # the estimate also carries an exact duplicate of every note (one onset labelled with two guitar classes)
    env.fakes.gt_by_wav["guitar_mono.wav"] = l1 + l1
    rows = E.run_e2(env.out, {}, {})
    r = next(r for r in rows if r["system"] == "guitar_mono")
    assert (r["tp"], r["fp"], r["fn"], r["n"]) == (12, 0, 0, 12) and r["value"] == pytest.approx(1.0)
    # what unmerged scoring gives: 24 reference notes for 12 played (recall capped at 0.5), and duplicate
    # estimates as false positives once the reference is merged
    ref_raw = E._concat(E.note_arrays(l1), E.note_arrays(l2))
    one = [{"onset_s": n.onset_s + 0.01, "offset_s": n.offset_s + 0.01, "pitch": n.pitch} for n in l1]
    old = E._score(ref_raw, E.note_arrays(one))
    assert (old.tp, old.fp, old.fn) == (12, 0, 12)
    half = E._score(E.merged(ref_raw), E.note_arrays(one + one))
    assert (half.tp, half.fp, half.fn) == (12, 12, 0)
    new = E._score_merged(ref_raw, E.note_arrays(one + one))
    assert (new.tp, new.fp, new.fn) == (12, 0, 0)
    # merging is idempotent and never joins different pitches or notes further apart than 50 ms
    m = E.merged(ref_raw)
    assert len(m[1]) == 12 and len(E.merged(m)[1]) == 12
    far = E._concat(E.note_arrays(l1), E.note_arrays([{"onset_s": n.onset_s + 0.06, "offset_s": n.offset_s + 0.06,
                                                         "pitch": n.pitch} for n in l1]))
    assert len(E.merged(far)[1]) == 24
    _check_rows(rows, "E2")


def test_latency_leaves_egdb_out_unless_asked(env):
    """The registration amendment excludes EGDB dev (cost): the default must not add it (review 2026-10-05)."""
    _guitarset(env)
    _egdb(env, n_train=1, n_test=0)
    rows = E.run_latency(env.out, {}, {})
    assert not any(str(r["dataset"]) == "egdb" for r in rows)
    rows = E.run_latency(env.out / "with", {}, {"egdb_dev": True})
    assert any(str(r["dataset"]) == "egdb" and r["section"] == "dev" for r in rows)


def test_e3_synthetic_scene_policies_omissions_and_families(env):
    rows = E.run_e3(env.out, {}, {"scenes": ["A"], "seeds": [0], "thresholds": [0.3, 0.5]})
    systems = {r["system"] for r in rows}
    assert systems == {"guitar+present", "guitar+present|t=0.3", "guitar_only", "all"}
    om = [r for r in rows if r["metric"] == "guitar_class_omissions"]
    assert om and all(r["value"] == 0 for r in om)
    fam = [r for r in rows if r["metric"].endswith("_presence_correct")]
    assert {r["metric"] for r in fam} == {"keys_presence_correct", "synth_presence_correct",
                                          "strings_presence_correct"}
    assert all(r["section"] in ("dev", "test") for r in rows)
    assert env.fakes.ms_calls[0]["views"] == ["mix_mono"]  # E3: presence from the full-mix pass (M2 estimator)
    masks = [m[0] for m in (c["masks"] for c in env.fakes.ms_calls[1:])]
    assert None in masks and I.mask_guitar_only() in masks  # 'all' and 'guitar_only' arms
    _check_rows(rows, "E3")


def test_e3b_uses_the_production_presence_pass_and_the_strings_policy(env):
    """Review 2026-10-04: E3 estimated presence from the mix pass, which bandscribe run no longer runs."""
    rows = E.EXPERIMENTS["E3b"](env.out, {}, {"scenes": ["A"], "seeds": [0], "thresholds": [0.5]})
    assert {r["system"] for r in rows} == {"guitar+present", "guitar_only", "all", "guitar+present+strings"}
    first = env.fakes.ms_calls[0]
    assert first["views"] == ["piano_other_presence"] and first["segments"][0]  # windows, not the whole view
    assert first["masks"][0] == I.mask_piano_other("keys+guitar")
    assert all("mix_mono" not in c["views"] for c in env.fakes.ms_calls)
    om = [r for r in rows if r["metric"] == "guitar_class_omissions"]
    assert om and all(r["value"] == 0 for r in om)
    assert "strings_presence_correct" in {r["metric"] for r in rows}
    _check_rows(rows, "E3b")


def _egdb(env, n_train: int = 3, n_test: int = 3) -> None:
    root = env.root("egdb")
    tracks = []
    for i in range(n_train + n_test):
        split = "train" if i < n_train else "test"
        for timbre, drive in (("plexi", "distorted"), ("jcjazz", "clean")):
            tracks.append(_track(root, "egdb", f"{timbre}_{i:03d}", kinds=("guitar",), families=("guitar",),
                                 split=split, group=f"{i:03d}", meta={"timbre": timbre, "drive_guess": drive}))
    tracks.append(_track(root, "egdb", "di_000", kinds=("di",), families=("guitar",), split="train",
                         meta={"timbre": "DI", "drive_guess": "clean"}))
    env.tracks["egdb"] = tracks


def test_e23_sweep_one_pass_per_batch_and_dev_test(env):
    _egdb(env)
    rows = E.run_e23(env.out, {}, {})
    assert len(E.e23_settings()) == 72 and E.e23_arm(E.e23_settings()[0]) == "f3_o0.3_fr0.2"
    assert "f4_o0.5_fr0.3" in {r["system"] for r in rows}
    assert {r["section"] for r in rows} == {"dev", "test"}
    assert all(c["n_settings"] == 72 for c in env.fakes.bp_calls)
    assert len(env.fakes.bp_calls) == 2  # one sweep worker per split (batch of <= 16 tracks)
    assert not any("di_000" in r["item"] for r in rows)  # DI renders excluded
    assert {r["rung"] for r in rows} == {"onnx-cpu"}
    _check_rows(rows, "E23")


def test_gate_m2_rows_and_subsets(env):
    _egdb(env)
    rows = E.run_gate_m2(env.out, {}, {})
    assert {r["system"] for r in rows} == {"muscriptor", "basicpitch"}
    assert {r["scenario"] for r in rows} == {"distortion", "clean"}
    assert {r["rung"] for r in rows} == {"medium", "onnx-cpu"}
    assert all(r["section"] == "test" for r in rows)
    _check_rows(rows, "gate-m2")


def _guitarset(env) -> None:
    root = env.root("guitarset")
    env.tracks["guitarset"] = [
        _track(root, "guitarset", f"{p}_x{i}", kinds=("guitar",), families=("guitar",), group=p,
               meta={"player": p}) for p in ("00", "01", "02", "03", "04", "05") for i in range(2)]


def test_latency_dev_constant_and_held_out_report(env):
    _guitarset(env)
    rows = E.run_latency(env.out, {}, {"egdb_dev": False})
    const = [r for r in rows if r["metric"] == "latency_ms"]
    assert len(const) == 1 and const[0]["value"] == pytest.approx(10.0, abs=0.01)  # the fake's 10 ms lag
    test_rows = [r for r in rows if r["metric"] == "median_abs_residual_ms" and r["section"] == "test"]
    assert len(test_rows) == 6 and all(r["value"] == pytest.approx(0.0, abs=0.01) for r in test_rows)
    dev_items = {r["item"] for r in rows if r["section"] == "dev" and r["item"] != "dev-pooled"}
    test_items = {r["item"] for r in rows if r["section"] == "test"}
    assert dev_items and test_items and not dev_items & test_items
    assert all(i.split(":")[1][:2] in ("03", "04", "05") for i in test_items)
    lc = json.loads((env.out / "latency_constant.json").read_text(encoding="utf-8"))
    assert lc["latency_s"] == pytest.approx(0.01) and "latency_s = 0.01" in lc["integration_request"]
    _check_rows(rows, "latency")


def test_e24_sep_and_amt_resource_rows(env):
    _tierb(env)
    rows = E.run_e24_sep(env.out, {}, {})
    assert {r["system"] for r in rows} == {"upstream", "fp16"}
    assert {"reserved_peak_mb", "rtf"} <= {r["metric"] for r in rows}
    assert [c["dtype"] for c in env.fakes.sep_calls] == ["upstream", "fp16", "upstream", "fp16"]
    _check_rows(rows, "E24-sep")

    _guitarset(env)
    rows = E.run_e24_amt(env.out, {}, {"max_tracks": 4})
    assert {r["system"] for r in rows} == {"upstream", "fp16"}
    assert [c["dtype"] for c in env.fakes.ms_calls[-2:]] == ["upstream", "fp16"]
    res = {(r["system"], r["metric"]): r["value"] for r in rows if r["metric"] in ("reserved_peak_mb", "rtf")}
    assert res[("upstream", "reserved_peak_mb")] == 2400.0 and res[("fp16", "reserved_peak_mb")] == 1700.0
    _check_rows(rows, "E24-amt")


def test_registry_ids_match():
    ids = set(EX.load_registry())
    assert set(E.EXPERIMENTS) <= ids
