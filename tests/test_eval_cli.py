"""`bandscribe eval …` and `bandscribe gt …` command wiring (Korean output, exit codes, files written)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from bandscribe import paths

pytest.importorskip("mir_eval")
sys.path.insert(0, str(Path(__file__).parent / "fixtures"))


@pytest.fixture
def app():
    from bandscribe.commands import eval as eval_cmds
    from bandscribe.commands import gt as gt_cmds

    a = typer.Typer()
    eval_cmds.register(a)
    gt_cmds.register(a)
    return a


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    return CliRunner()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every data path the commands touch into tmp (works with or without a reload of bandscribe.paths)."""
    for name, sub in (("EVAL", "eval"), ("RUNS", "runs"), ("JOBS", "jobs"), ("DOWNLOADS", "downloads")):
        monkeypatch.setattr(paths, name, tmp_path / "data" / sub)
    return tmp_path


def test_help_lists_subcommands(app, runner):
    r = runner.invoke(app, ["eval", "--help"])
    assert r.exit_code == 0
    for cmd in ("run", "compare", "import-external", "exp", "reftx"):
        assert cmd in r.output
    r = runner.invoke(app, ["gt", "--help"])
    assert r.exit_code == 0
    for cmd in ("import", "render", "align", "taps", "approve"):
        assert cmd in r.output


def test_eval_run_and_compare(app, runner, sandbox):
    r1 = runner.invoke(app, ["eval", "run", "--suite", "quick", "--system", "oracle"])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["eval", "run", "--suite", "quick", "--system", "oracle-noisy"])
    assert r2.exit_code == 0, r2.output
    runs_ = sorted(p for p in paths.RUNS.iterdir() if p.is_dir())
    assert len(runs_) == 2 and all((p / "metrics.csv").is_file() for p in runs_)
    assert "note_f1_onset50" in r1.output and "결과 폴더" in r1.output
    a = next(p for p in runs_ if p.name.endswith("_oracle"))
    b = next(p for p in runs_ if p.name.endswith("oracle-noisy"))
    rc = runner.invoke(app, ["eval", "compare", str(a), str(b)])
    assert rc.exit_code == 0, rc.output
    assert "note_f1_onset50" in rc.output and "B − A" in rc.output


def test_eval_run_bad_suite_exits_2(app, runner, sandbox):
    r = runner.invoke(app, ["eval", "run", "--suite", "bogus"])
    assert r.exit_code == 2 and "알 수 없는 평가 세트" in r.output


def test_eval_exp_dry_run_and_unknown(app, runner, monkeypatch, tmp_path):
    from bandscribe.eval import experiments

    mod = types.ModuleType("bandscribe.amt.experiments")
    mod.EXPERIMENTS = {"E1": lambda out, cfg, args: []}
    monkeypatch.setitem(sys.modules, "bandscribe.amt.experiments", mod)
    from tests.fixtures.registry import fresh_registry

    reg = fresh_registry(tmp_path / "decisions.md")
    monkeypatch.setattr(experiments, "decisions_path", lambda: reg)
    before = reg.read_bytes()
    r = runner.invoke(app, ["eval", "exp", "E1", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "주 지표" in r.output
    assert reg.read_bytes() == before
    r = runner.invoke(app, ["eval", "exp", "E99"])
    assert r.exit_code == 1 and "사전 등록" in r.output


def test_import_external_midi(app, runner, sandbox, tmp_path):
    import mido

    mid = mido.MidiFile(ticks_per_beat=480)
    for notes in ([60, 64], [40]):
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        for n in notes:
            tr += [mido.Message("note_on", note=n, velocity=90, time=0), mido.Message("note_off", note=n, velocity=0, time=480)]
    f = tmp_path / "ext.mid"
    mid.save(str(f))
    r = runner.invoke(app, ["eval", "import-external", "song_a", str(f), "--system", "klangio", "--map", "1=L1"])
    assert r.exit_code == 0, r.output
    from bandscribe.eval.external import load_external

    pred = load_external("klangio", "song_a:verse")
    assert pred.item == "song_a:verse" and pred.system == "external:klangio"
    assert {n.line for n in pred.notes} == {"L1", "T2"}
    first = [n for n in pred.notes if n.line == "L1"]
    assert first[0].onset_s == pytest.approx(0.0) and first[1].onset_s == pytest.approx(0.5)  # 120 BPM default
    assert all(":" not in p.name for p in (paths.EVAL / "external").rglob("*"))
    bad = runner.invoke(app, ["eval", "import-external", "song_a", str(f), "--system", "기타"])
    assert bad.exit_code == 1


def test_gt_import_render_approve(app, runner, sandbox, tmp_path):
    pytest.importorskip("guitarpro")
    import gp_make
    import soundfile as sf

    gp5 = gp_make.make_fixture_gp5(tmp_path / "fx.gp5")
    wav = tmp_path / "song.wav"
    t = np.arange(44100 * 3) / 44100
    sf.write(str(wav), np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2, 1).astype(np.float32), 44100)
    r = runner.invoke(app, ["gt", "import", "fx_song", str(gp5), "--audio", str(wav)])
    assert r.exit_code == 0, r.output
    d = paths.EVAL / "tierA" / "fx_song"
    for name in ("gt.yaml", "ref.gp5", "refscore.json", "gt_notes.json"):
        assert (d / name).is_file(), name
    assert "펼친 마디 7개" in r.output
    text = (d / "gt.yaml").read_text(encoding="utf-8")
    assert "f-" in text  # song_key from ingest
    (d / "gt.yaml").write_text(text + "# user edit\n", encoding="utf-8")
    r = runner.invoke(app, ["gt", "import", "fx_song", str(gp5), "--audio", str(wav)])
    assert r.exit_code == 0 and "기존 gt.yaml 을 유지" in r.output
    assert (d / "gt.yaml").read_text(encoding="utf-8").endswith("# user edit\n")
    r = runner.invoke(app, ["gt", "render", "fx_song"])
    assert r.exit_code == 0, r.output
    x, sr = sf.read(str(d / "render.wav"))
    assert sr == 44100 and len(x) / sr > 12.0
    r = runner.invoke(app, ["gt", "approve", "fx_song"])
    assert r.exit_code == 1 and "정렬되지 않았습니다" in r.output
    r = runner.invoke(app, ["gt", "import", "Bad Id", str(gp5), "--audio", str(wav)])
    assert r.exit_code == 2


def test_eval_reftx_writes_into_the_item_folder(app, runner, sandbox, monkeypatch):
    from bandscribe import datasets
    from bandscribe.amt import reftx

    tracks = [types.SimpleNamespace(dataset="cambridge_mt", track_id="Song A", lines=[]),
              types.SimpleNamespace(dataset="cambridge_mt", track_id="song_b", lines=[])]
    monkeypatch.setattr(datasets, "is_available", lambda name: True)
    monkeypatch.setattr(datasets, "iter_tracks", lambda name, split=None: iter(tracks))
    calls = []

    def fake(track, out_dir, cfg, **kw):
        calls.append(Path(out_dir))
        if track.track_id == "song_b":
            raise RuntimeError("VRAM 부족")
        return {"L1": {"notes": [{}, {}]}}

    monkeypatch.setattr(reftx, "reference_transcribe", fake)
    r = runner.invoke(app, ["eval", "reftx", "cambridge_mt"])
    assert r.exit_code == 1, r.output  # one track failed -> exit 1 after trying all
    assert calls == [paths.EVAL / "tierB" / "cambridge_mt" / "song_a", paths.EVAL / "tierB" / "cambridge_mt" / "song_b"]
    assert "L1=2음" in r.output and "VRAM 부족" in r.output


def test_eval_exp_data_missing_suggests_record(app, runner, monkeypatch, tmp_path):
    import shutil

    from bandscribe.eval import experiments

    class ExperimentDataMissing(RuntimeError):
        pass

    def no_data(out, cfg, args):
        raise ExperimentDataMissing("Tier B 데이터가 없습니다")

    from tests.fixtures.registry import fresh_registry

    reg = fresh_registry(tmp_path / "decisions.md")
    monkeypatch.setattr(experiments, "decisions_path", lambda: reg)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    mod = types.ModuleType("bandscribe.amt.experiments")
    mod.EXPERIMENTS = {"E1": no_data}
    monkeypatch.setitem(sys.modules, "bandscribe.amt.experiments", mod)
    r = runner.invoke(app, ["eval", "exp", "E1", "--arg", "max_tracks=2"])
    assert r.exit_code == 1
    assert "Tier B 데이터가 없습니다" in r.output and "판단 보류 (데이터 대기)" in r.output
    assert experiments.load_registry(reg)["E1"].status == "사전 등록"
    bad = runner.invoke(app, ["eval", "exp", "E1", "--arg", "novalue"])
    assert bad.exit_code == 2
