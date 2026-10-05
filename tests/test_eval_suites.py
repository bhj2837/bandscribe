"""Suite plumbing: Tier B items scored against reference transcriptions, job stage file choice, provenance,
the Tier B remix part selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bandscribe import atomic, config, paths
from bandscribe.eval import remix, suites

pytest.importorskip("mir_eval")


def _cmt_track(track_id: str = "song_a", *, with_di: bool = True):
    from bandscribe.schema.dataset import DatasetTrack, Part
    from bandscribe.schema.gt import GtLine

    parts = [Part(id="g1", kind="guitar", audio=f"{track_id}/gtr1.wav", channels=1, line="L1", take=1),
             Part(id="g2", kind="guitar", audio=f"{track_id}/gtr2.wav", channels=2, line="L2", take=1),
             Part(id="b", kind="bass", audio=f"{track_id}/bass.wav", channels=1, line="B1", take=1),
             Part(id="dr", kind="drums", audio=f"{track_id}/drums.wav", channels=2)]
    if with_di:
        parts += [Part(id="g1di", kind="di", audio=f"{track_id}/gtr1_di.wav", channels=1, line="L1", take=2),
                  Part(id="g3di", kind="di", audio=f"{track_id}/gtr3_di.wav", channels=1, line="L3", take=1)]
    lines = [GtLine(id="L1", name="rhythm", kind="guitar", takes=[1]), GtLine(id="L2", name="lead", kind="guitar", takes=[2]),
             GtLine(id="B1", name="bass", kind="bass", takes=[3])]
    return DatasetTrack(format="bandscribe.dataset_track/1", dataset="cambridge_mt", track_id=track_id, split=None,
                        group=track_id, tier="B", license="x", mix=None, parts=parts, lines=lines,
                        families_present=["guitar", "bass", "drums"], notes={}, string_fret=None, beats_s=None,
                        downbeats_s=None, tuning=None, meta={})


def _noteset(view: str, pitches, start: float = 0.5):
    from bandscribe.schema.notes import BackendInfo, NoteEvent, NoteSet

    notes = [NoteEvent(onset_s=round(start + 0.25 * i, 6), offset_s=round(start + 0.25 * i + 0.2, 6), pitch=int(p),
                       instrument="distorted_electric_guitar") for i, p in enumerate(pitches)]
    return NoteSet(view=view, backend=BackendInfo(name="muscriptor"), notes=notes)


def test_mix_parts_drop_di_doubled_by_an_amp():
    ids = [p.id for p in remix.mix_parts(_cmt_track())]
    assert ids == ["g1", "g2", "b", "dr", "g3di"]  # L1's DI goes (amp exists), L3's DI is its only source


def test_tier_b_items_use_stored_reference_transcriptions(tmp_path, monkeypatch):
    from bandscribe import datasets
    from bandscribe.eval.runs import slug

    tracks = [_cmt_track("song_a"), _cmt_track("song_b")]
    monkeypatch.setattr(datasets, "iter_tracks", lambda name, split=None: iter(tracks))
    monkeypatch.setattr(datasets, "track_ids", lambda name, split=None: [t.track_id for t in tracks])
    monkeypatch.setattr(datasets, "load_track", lambda name, tid: next(t for t in tracks if t.track_id == tid))
    monkeypatch.setattr(datasets, "is_available", lambda name: name == "cambridge_mt")
    root = tmp_path / "tierB"
    d = root / "cambridge_mt" / slug("song_a")
    atomic.write_text(d / "L1__reftx.json", _noteset("L1", [52, 55, 57, 59]).model_dump_json())
    atomic.write_text(d / "L2__reftx.json", _noteset("L2", [64, 67, 69], start=0.6).model_dump_json())
    monkeypatch.setattr(paths, "EVAL", tmp_path)
    items = suites.items_for_suite("tierB", config.load_config(), {"datasets": ["cambridge_mt"]})
    assert [it.item for it in items] == ["cambridge_mt:song_a"]  # song_b has no reference -> skipped
    it = items[0]
    assert it.lines == ["L1", "L2"] and "reftx" in it.meta["reference"]
    res = suites.run_suite("tierB", "oracle", config.load_config(), out=tmp_path / "runs", gitsha="abcdef1",
                           args={"datasets": ["cambridge_mt"]})
    agg = res.summary["aggregates"]
    assert agg["note_f1_onset50"]["value"] == pytest.approx(1.0) and agg["assign_macro"]["value"] == pytest.approx(1.0)
    run_json = json.loads((res.run_dir / "run.json").read_text(encoding="utf-8"))
    assert {"models", "models_lock_sha256", "datasets_lock_sha256"} <= set(run_json)


def test_tier_b_without_any_item_is_a_korean_error(monkeypatch):
    from bandscribe import datasets

    monkeypatch.setattr(datasets, "is_available", lambda name: False)
    with pytest.raises(suites.SuiteError, match="Tier B 항목이 없습니다"):
        suites.items_for_suite("tierB", config.load_config(), {})


def test_job_stage_picks_the_guitar_file(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("bass_mono__muscriptor.json", "mix_mono__muscriptor.json", "piano_other_mono__muscriptor.json"):
        (raw / name).write_text("{}", encoding="utf-8")
    assert suites.JobStage._guitar_file(tmp_path).name == "mix_mono__muscriptor.json"
    (raw / "guitar_mono__muscriptor.json").write_text("{}", encoding="utf-8")
    assert suites.JobStage._guitar_file(tmp_path).name == "guitar_mono__muscriptor.json"
    (tmp_path / "guitar_all.json").write_text("{}", encoding="utf-8")
    assert suites.JobStage._guitar_file(tmp_path).name == "guitar_all.json"
    assert suites.JobStage._guitar_file(tmp_path / "missing") is None


def test_job_stage_prediction_and_rung_from_gpu_run(tmp_path, monkeypatch):
    """job:amt_gtr reads the raw guitar view and puts both GPU rungs (sep, amt_gtr) into the rung column."""
    monkeypatch.setattr(paths, "JOBS", tmp_path / "jobs")
    key = "f-0123456789abcdef"
    for stage, label in (("sep", "chunk=588800"), ("amt_gtr", "medium")):
        sd = paths.JOBS / key / "stages" / stage / "abcdef012345"
        sd.mkdir(parents=True)
        (sd / "manifest.json").write_text("{}", encoding="utf-8")
        atomic.write_json(sd / "gpu_run.json", {"format": "bandscribe.gpu_run/1", "rung": {"index": 0, "label": label}})
    gtr = paths.JOBS / key / "stages" / "amt_gtr" / "abcdef012345"
    atomic.write_text(gtr / "raw" / "guitar_mono__muscriptor.json", _noteset("guitar_mono", [60, 62]).model_dump_json())
    it = suites.EvalItem(item="song:verse", dataset="tierA", group="song", scenario="A", gt=None, lines=["L1"],
                         song_key=key)
    sysobj = suites.make_system("job:amt_gtr", config.load_config())
    pred = sysobj.predict(it)
    assert pred is not None and [n.pitch for n in pred.notes] == [60, 62]
    assert sysobj.rung_for(it) == "sep=chunk=588800|amt_gtr=medium"


def test_bp_line_named_after_single_gt_line(tmp_path, monkeypatch):
    """bp-smoke names its prediction after the item's one GT line, so the identity mapping is meaningful."""
    from bandscribe.schema.dataset import DatasetTrack, Part
    from bandscribe.schema.gt import GtLine
    from bandscribe.schema.notes import BackendInfo, NoteEvent, NoteSet

    notes = [NoteEvent(onset_s=0.5 + 0.3 * i, offset_s=0.7 + 0.3 * i, pitch=50 + i) for i in range(10)]
    tr = DatasetTrack(format="bandscribe.dataset_track/1", dataset="guitarset", track_id="00_X1-100-C_comp", split="dev",
                      group="00", tier="C", license="x", mix=None,
                      parts=[Part(id="mic", kind="guitar", audio="a/00_X1-100-C_comp_mic.wav", channels=1, line="gtr")],
                      lines=[GtLine(id="gtr", name="gtr", kind="guitar", takes=[1])], families_present=["guitar"],
                      notes={"gtr": NoteSet(view="mic", backend=BackendInfo(name="gt"), notes=notes)},
                      string_fret=None, beats_s=None, downbeats_s=None, tuning=None, meta={})
    from bandscribe import datasets

    monkeypatch.setattr(datasets, "iter_tracks", lambda name, split=None: iter([tr]))
    monkeypatch.setattr(datasets, "track_ids", lambda name, split=None: [tr.track_id])
    monkeypatch.setattr(datasets, "load_track", lambda name, tid: tr)
    monkeypatch.setattr(datasets, "dataset_root", lambda name: tmp_path)

    class _Res:
        ok = True

    def fake(name, request, work_dir, **kw):
        for v in request["views"]:
            atomic.write_text(v["out"], NoteSet(view=v["id"], backend=BackendInfo(name="basicpitch"),
                                                notes=notes).model_dump_json())
        return _Res()

    res = suites.run_suite("bp-smoke", None, config.load_config(), out=tmp_path / "runs", gitsha="abcdef1",
                           run_worker=fake, args={"limit": 20})
    assert res.summary["aggregates"]["label_swap_raw"]["value"] == pytest.approx(1.0)
    assert np.isfinite(res.summary["aggregates"]["note_f1_onset50"]["value"])


def test_b0_plans_with_the_eval_profile_and_finds_the_mix_pass(tmp_path, monkeypatch):
    """The default profile has no mix pass (2026-10-04): B0 asks for the eval profile's amt_ms1 and, without a
    planned match, takes the newest amt_ms1 dir that has the mix file."""
    monkeypatch.setattr(paths, "JOBS", tmp_path / "jobs")
    cfg = config.load_config()
    assert cfg.run.profile == "quality"
    bsys = suites.make_system("b0", cfg)
    assert bsys.cfg.run.profile == "eval" and cfg.run.profile == "quality"  # a copy, the caller's cfg untouched
    assert suites.make_system("no_split", cfg).cfg.run.profile == "quality"
    assert suites.with_profile({"run": {"profile": "fast"}, "amt": {}}, "eval") == {"run": {"profile": "eval"},
                                                                                     "amt": {}}
    key = "f-00aa11bb22cc33ee"
    d = paths.JOBS / key / "stages" / "amt_ms1"
    import os

    for name, with_mix, mtime in (("aaaaaaaaaaaa", True, 1000), ("bbbbbbbbbbbb", False, 2000)):
        sd = d / name
        (sd / "raw").mkdir(parents=True)
        (sd / "manifest.json").write_text("{}", encoding="utf-8")
        atomic.write_text(sd / "raw" / "bass_mono__muscriptor.json", _noteset("bass_mono", [40]).model_dump_json())
        if with_mix:
            atomic.write_text(sd / "raw" / "mix_mono__muscriptor.json", _noteset("mix_mono", [52, 55]).model_dump_json())
        os.utime(sd, (mtime, mtime))
    bsys._planned[key] = {}  # nothing planned for this fake job
    assert bsys._stage_dir(key, "amt_ms1", suites.B0_FILE).name == "aaaaaaaaaaaa"  # older, but has the mix pass
    assert bsys._stage_dir(key, "amt_ms1").name == "bbbbbbbbbbbb"
    item = suites.EvalItem(item="song:verse", dataset="tierA", group="song", scenario="A", gt=None, lines=["L1"],
                           song_key=key)
    pred = bsys.predict(item)
    assert pred is not None and len(pred.notes) == 2
    (d / "aaaaaaaaaaaa" / "raw" / "mix_mono__muscriptor.json").unlink()
    assert suites.make_system("b0", cfg).predict(item) is None  # no eval run: B0 skipped (Korean hint logged)


def _stage(key: str, stage: str, key12: str, rel: str, view: str, pitches: list[int], mtime: int) -> Path:
    import os

    sd = paths.JOBS / key / "stages" / stage / key12
    sd.mkdir(parents=True)
    (sd / "manifest.json").write_text("{}", encoding="utf-8")
    atomic.write_text(sd / rel, _noteset(view, pitches).model_dump_json())
    os.utime(sd, (mtime, mtime))
    return sd


def test_job_stage_never_silently_scores_another_settings_result(tmp_path, monkeypatch, caplog):
    """Review 2026-10-05: when the planned (current settings) stage dir was missing, the newest dir made with other
    settings was scored under the same system name with only a log line. Now the item is skipped with a Korean
    warning; ``allow_stale`` opts in, and the metrics note says which stage key (and that it is stale)."""
    monkeypatch.setattr(paths, "JOBS", tmp_path / "jobs")
    key = "f-00aa11bb22cc33ff"
    _stage(key, "amt_gtr", "aaaaaaaaaaaa", "raw/guitar_mono__muscriptor.json", "guitar_mono", [60, 62], 1000)
    _stage(key, "amt_gtr", "bbbbbbbbbbbb", "raw/guitar_mono__muscriptor.json", "guitar_mono", [60, 62, 64], 2000)
    it = suites.EvalItem(item="song:verse", dataset="tierA", group="song", scenario="A", gt=None, lines=["L1"],
                         song_key=key)
    cfg = config.load_config()
    planned = {key: {"amt_gtr": "cccccccccccc", "sep": "dddddddddddd"}}  # the current settings' keys: not on disk
    sysobj = suites.make_system("job:amt_gtr", cfg)
    sysobj._planned = dict(planned)
    with caplog.at_level("WARNING"):
        assert sysobj.predict(it) is None
    assert len(sysobj.warnings) == 1 and "cccccccccccc" in sysobj.warnings[0] and "allow_stale" in sysobj.warnings[0]
    assert "bandscribe run f-00aa11bb22cc33ff --until amt_gtr" in sysobj.warnings[0]
    stale = suites.make_system("job:amt_gtr", cfg, allow_stale=True)
    stale._planned = dict(planned)
    pred = stale.predict(it)
    assert pred is not None and [n.pitch for n in pred.notes] == [60, 62, 64]  # the newest of the old dirs
    assert stale.note_for(it) == "stage=bbbbbbbbbbbb (stale)"
    assert stale.stage_keys() == {"song:verse": {"amt_gtr": "bbbbbbbbbbbb (stale)"}}
    # the planned dir exists: used as is, no warning, note = its key
    _stage(key, "amt_gtr", "cccccccccccc", "raw/guitar_mono__muscriptor.json", "guitar_mono", [67], 500)
    ok = suites.make_system("job:amt_gtr", cfg)
    ok._planned = dict(planned)
    assert [n.pitch for n in ok.predict(it).notes] == [67] and not ok.warnings
    assert ok.note_for(it) == "stage=cccccccccccc"
    # a job that cannot be planned at all (no config, foreign layout): the newest dir, as before
    unplanned = suites.make_system("job:amt_gtr", cfg)
    unplanned._planned = {key: {}}
    assert [n.pitch for n in unplanned.predict(it).notes] == [60, 62, 64] and not unplanned.warnings


def test_suite_records_stage_keys_and_skips_stale_items(tmp_path, monkeypatch):
    from bandscribe.schema.gt import GtNote, GtNotes

    monkeypatch.setattr(paths, "JOBS", tmp_path / "jobs")
    key = "f-00aa11bb22cc33ee"
    pitches = [52, 55, 57, 59]
    _stage(key, "notes", "111111111111", "guitar_all.json", "guitar_all", pitches, 1000)
    gt = GtNotes(format="bandscribe.gtnotes/1", song_id="song", tpb=48, bars=[], notes=[
        GtNote(line="L1", ref_track=1, bar=1, tick=float(i), dur_ticks=10.0, pitch=p, string=None, fret=None,
               onset_s=round(0.5 + 0.25 * i, 6), offset_s=round(0.7 + 0.25 * i, 6)) for i, p in enumerate(pitches)])
    item = suites.EvalItem(item="song:verse", dataset="tierA", group="song", scenario="A", gt=gt, lines=["L1"],
                           song_key=key)
    monkeypatch.setattr(suites, "items_for_suite", lambda suite, cfg, args=None: [item])
    plans = {"notes": "222222222222"}
    monkeypatch.setattr(suites.JobStage, "_planned_keys", lambda self, song_key: dict(plans))
    with pytest.raises(suites.SuiteError, match="222222222222"):
        suites.run_suite("tierA", "job:notes", config.load_config(), out=tmp_path / "runs", gitsha="abcdef1")
    res = suites.run_suite("tierA", "job:notes", config.load_config(), out=tmp_path / "runs", gitsha="abcdef1",
                           args={"allow_stale": True})
    run_json = json.loads((res.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["stage_keys"]["job:notes"] == {"song:verse": {"notes": "111111111111 (stale)"}}
    assert run_json["allow_stale"] is True and any("allow_stale" in w for w in res.summary["warnings"])
    text = (res.run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "stage=111111111111 (stale)" in text
    plans["notes"] = "111111111111"  # now the current settings' dir exists
    res = suites.run_suite("tierA", "job:notes", config.load_config(), out=tmp_path / "runs2", gitsha="abcdef1")
    assert not any("notes" in w for w in res.summary.get("warnings", []))  # only B0's "run --profile eval" hint
    assert all("B0" in w for w in res.summary.get("warnings", []))
    assert json.loads((res.run_dir / "run.json").read_text(encoding="utf-8"))["stage_keys"] == {
        "job:notes": {"song:verse": {"notes": "111111111111"}}}
    assert "stage=111111111111" in (res.run_dir / "metrics.csv").read_text(encoding="utf-8")


def test_tier_a_report_includes_b0_and_no_split_baselines(tmp_path, monkeypatch):
    """DESIGN 8.5: B0 and no-split come from the same job's amt_ms1 / amt_gtr stages and sit next to the system."""
    from bandscribe.schema.gt import GtNote, GtNotes

    monkeypatch.setattr(paths, "JOBS", tmp_path / "jobs")
    key = "f-00aa11bb22cc33dd"
    pitches = [52, 55, 57, 59, 60, 62]
    files = {"notes": ("guitar_all.json", "guitar_all", pitches),
             "amt_gtr": ("raw/guitar_mono__muscriptor.json", "guitar_mono", pitches[:4]),
             "amt_ms1": ("raw/mix_mono__muscriptor.json", "mix_mono", pitches[:2])}
    for stage, (rel, view, ps) in files.items():
        sd = paths.JOBS / key / "stages" / stage / "0123456789ab"
        (sd).mkdir(parents=True)
        (sd / "manifest.json").write_text("{}", encoding="utf-8")
        atomic.write_text(sd / rel, _noteset(view, ps).model_dump_json())
    gt = GtNotes(format="bandscribe.gtnotes/1", song_id="song", tpb=48, bars=[], notes=[
        GtNote(line="L1", ref_track=1, bar=1, tick=float(i), dur_ticks=10.0, pitch=p, string=None, fret=None,
               onset_s=round(0.5 + 0.25 * i, 6), offset_s=round(0.7 + 0.25 * i, 6)) for i, p in enumerate(pitches)])
    item = suites.EvalItem(item="song:verse", dataset="tierA", group="song", scenario="A", gt=gt, lines=["L1"],
                           song_key=key)
    monkeypatch.setattr(suites, "items_for_suite", lambda suite, cfg, args=None: [item])
    res = suites.run_suite("tierA", "job:notes", config.load_config(), out=tmp_path / "runs", gitsha="abcdef1")
    b = res.summary["baselines"]
    assert res.summary["aggregates"]["note_f1_onset50"]["value"] == pytest.approx(1.0)
    assert b["no_split:note_f1_onset50"]["value"] == pytest.approx(2 * 4 / (4 + 6))
    assert b["b0:note_f1_onset50"]["value"] == pytest.approx(2 * 2 / (2 + 6))
    text = (res.run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert ",b0,tierA,song:verse," in text and ",no_split,tierA,song:verse," in text
