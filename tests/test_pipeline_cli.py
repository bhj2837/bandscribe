"""``gtab run`` command (M1_M2_SPEC 5; GRID) with fake stages."""
from __future__ import annotations

import typer
from typer.testing import CliRunner

from gtab import atomic, paths
from gtab.commands import run as run_cmd
from gtab.jobs.store import JobStore
from grid_testlib import Fakes

KEY = "f-0123456789abcdef"


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _root() -> None:  # keep "run" a subcommand
        pass

    run_cmd.register(app)
    return app


def _job(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "JOBS", tmp_path / "jobs")
    store = JobStore(tmp_path / "jobs")
    inp = store.job_dir(KEY) / "input"
    inp.mkdir(parents=True)
    atomic.write_json(inp / "meta.json", {"decode": {"pcm_sha256": "ab" * 32}, "params": {"version": 2}})
    f = Fakes()
    f.install(monkeypatch)
    return f


def test_help_is_korean():
    r = CliRunner().invoke(_app(), ["run", "--help"])
    assert r.exit_code == 0
    assert "파이프라인을 실행한다" in r.output and "--dry-run" in r.output


def test_dry_run_and_run(tmp_path, monkeypatch):
    _job(tmp_path, monkeypatch)
    r = CliRunner().invoke(_app(), ["run", "f-0123", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "실행 계획" in r.output and "beats" in r.output and "실행" in r.output
    r = CliRunner().invoke(_app(), ["run", "f-0123"])
    assert r.exit_code == 0, r.output
    assert "결과 파일" in r.output and "guitar_all.mid" in r.output
    assert "leftover" in r.output  # warning from stem_stats.json
    r = CliRunner().invoke(_app(), ["run", KEY, "--dry-run"])
    assert "캐시" in r.output


def test_missing_model_fails_before_running(tmp_path, monkeypatch):
    f = _job(tmp_path, monkeypatch)
    f.missing = {"beats"}
    r = CliRunner().invoke(_app(), ["run", KEY])
    assert r.exit_code == 1
    assert "gtab models fetch model.beats" in r.output
    assert f.calls == []
    r = CliRunner().invoke(_app(), ["run", KEY, "--dry-run"])
    assert r.exit_code == 0 and "모델 없음" in r.output


def test_bad_stage_names_are_usage_errors(tmp_path, monkeypatch):
    _job(tmp_path, monkeypatch)
    r = CliRunner().invoke(_app(), ["run", KEY, "--until", "tabs"])
    assert r.exit_code == 2 and "알 수 없는" in r.output
    r = CliRunner().invoke(_app(), ["run", KEY, "--from", "nope"])
    assert r.exit_code == 2


def test_from_reruns_downstream(tmp_path, monkeypatch):
    f = _job(tmp_path, monkeypatch)
    assert CliRunner().invoke(_app(), ["run", KEY, "--until", "sections"]).exit_code == 0
    f.calls.clear()
    r = CliRunner().invoke(_app(), ["run", KEY, "--until", "sections", "--from", "grid"])
    assert r.exit_code == 0, r.output
    assert f.calls == ["grid", "sections"]


def test_profile_and_instruments_flags_reach_the_config(tmp_path, monkeypatch):
    from gtab.pipeline import runner

    _job(tmp_path, monkeypatch)
    seen = {}
    real_plan = runner.plan

    def spy(song_key, cfg, **kw):
        seen["cfg"] = cfg
        return real_plan(song_key, cfg, **kw)

    monkeypatch.setattr(runner, "plan", spy)
    r = CliRunner().invoke(_app(), ["run", KEY, "--dry-run", "--profile", "fast", "--instruments", 'guitar, keys"x'])
    assert r.exit_code == 0, r.output
    assert seen["cfg"].run.profile == "fast" and seen["cfg"].hints.instruments == 'guitar, keys"x'
    r = CliRunner().invoke(_app(), ["run", KEY, "--profile", "max"])
    assert r.exit_code == 2 and "fast 또는 quality" in r.output
    r = CliRunner().invoke(_app(), ["run", KEY, "--set", "grid.refine_windw_ms=3"])
    assert r.exit_code == 2


def test_config_error_at_plan_time_is_a_usage_error(tmp_path, monkeypatch):
    from gtab.config import ConfigError
    from gtab.pipeline import runner

    _job(tmp_path, monkeypatch)

    def bad_plan(*a, **kw):
        raise ConfigError("hints.instruments 에 알 수 없는 악기 계열이 있습니다: kazoo")

    monkeypatch.setattr(runner, "plan", bad_plan)
    r = CliRunner().invoke(_app(), ["run", KEY, "--instruments", "kazoo"])
    assert r.exit_code == 2 and "kazoo" in r.output
    assert "Traceback" not in r.output
