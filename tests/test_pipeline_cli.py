"""``bandscribe run`` command (M1_M2_SPEC 5; GRID) with fake stages."""
from __future__ import annotations

import typer
from typer.testing import CliRunner

from bandscribe import atomic, paths
from bandscribe.commands import run as run_cmd
from bandscribe.jobs.store import JobStore
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
    assert "bandscribe models fetch model.beats" in r.output
    assert f.calls == []
    r = CliRunner().invoke(_app(), ["run", KEY, "--dry-run"])
    assert r.exit_code == 0 and "모델 없음" in r.output


def test_bad_stage_names_are_usage_errors(tmp_path, monkeypatch):
    _job(tmp_path, monkeypatch)
    r = CliRunner().invoke(_app(), ["run", KEY, "--until", "tabs"])
    assert r.exit_code == 2 and "알 수 없는" in r.output
    r = CliRunner().invoke(_app(), ["run", KEY, "--from", "nope"])
    assert r.exit_code == 2


def test_parts_takes_auto_or_one_to_four(tmp_path, monkeypatch):
    _job(tmp_path, monkeypatch)
    for bad in ("5", "0", "two"):
        r = CliRunner().invoke(_app(), ["run", KEY, "--parts", bad, "--dry-run"])
        assert r.exit_code == 2 and "--parts" in r.output
    for ok in ("auto", "1", "3"):
        assert CliRunner().invoke(_app(), ["run", KEY, "--parts", ok, "--dry-run"]).exit_code == 0


def test_from_reruns_downstream(tmp_path, monkeypatch):
    f = _job(tmp_path, monkeypatch)
    assert CliRunner().invoke(_app(), ["run", KEY, "--until", "sections"]).exit_code == 0
    f.calls.clear()
    r = CliRunner().invoke(_app(), ["run", KEY, "--until", "sections", "--from", "grid"])
    assert r.exit_code == 0, r.output
    assert f.calls == ["grid", "sections"]


def test_profile_and_instruments_flags_reach_the_config(tmp_path, monkeypatch):
    from bandscribe.pipeline import runner

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
    r = CliRunner().invoke(_app(), ["run", KEY, "--dry-run", "--profile", "eval"])
    assert r.exit_code == 0, r.output
    assert seen["cfg"].run.profile == "eval"
    r = CliRunner().invoke(_app(), ["run", KEY, "--profile", "max"])
    assert r.exit_code == 2 and "fast, quality, eval" in r.output
    r = CliRunner().invoke(_app(), ["run", KEY, "--set", "grid.refine_windw_ms=3"])
    assert r.exit_code == 2


def test_config_error_at_plan_time_is_a_usage_error(tmp_path, monkeypatch):
    from bandscribe.config import ConfigError
    from bandscribe.pipeline import runner

    _job(tmp_path, monkeypatch)

    def bad_plan(*a, **kw):
        raise ConfigError("hints.instruments 에 알 수 없는 악기 계열이 있습니다: kazoo")

    monkeypatch.setattr(runner, "plan", bad_plan)
    r = CliRunner().invoke(_app(), ["run", KEY, "--instruments", "kazoo"])
    assert r.exit_code == 2 and "kazoo" in r.output
    assert "Traceback" not in r.output


def test_degraded_and_worker_warnings_reach_the_console_once(tmp_path, monkeypatch):
    """Review 2026-10-05: a freshly run lower rung, and the cached flags (gpu_run.json / sep.json), are printed;
    a cached degraded stage, also announced live, is printed once."""
    f = _job(tmp_path, monkeypatch)
    f.degraded = {"sep"}
    r = CliRunner().invoke(_app(), ["run", KEY, "--until", "stems"])
    assert r.exit_code == 0, r.output
    assert "sep: 낮은 칸(low)으로 실행했습니다" in r.output  # fresh run: said now, not only next time
    sep_dir = next((tmp_path / "jobs" / KEY / "stages" / "sep").iterdir())
    atomic.write_json(sep_dir / "sep.json", {"warnings": ["chunk=352256: 실시간 배수 1.4x 로 기준보다 느림"]})
    r = CliRunner().invoke(_app(), ["run", KEY, "--until", "stems"])
    assert r.exit_code == 0, r.output
    flat = " ".join(r.output.split())
    assert flat.count("낮은 칸(low)으로 만든 캐시입니다") == 1
    assert "sep: chunk=352256: 실시간 배수 1.4x 로 기준보다 느림" in flat


def test_long_input_warning_on_run(tmp_path, monkeypatch):
    _job(tmp_path, monkeypatch)
    meta = tmp_path / "jobs" / KEY / "input" / "meta.json"
    atomic.write_json(meta, {**atomic.read_json(meta), "duration_s": 3000.0})
    r = CliRunner().invoke(_app(), ["run", KEY, "--until", "grid"])
    assert r.exit_code == 0, r.output
    assert "50분으로 깁니다" in " ".join(r.output.split())


def test_tempo_flag_sets_the_grid_hint(tmp_path, monkeypatch):
    from bandscribe.pipeline import runner

    _job(tmp_path, monkeypatch)
    seen = {}
    real_plan = runner.plan

    def spy(song_key, cfg, **kw):
        seen["cfg"] = cfg
        return real_plan(song_key, cfg, **kw)

    monkeypatch.setattr(runner, "plan", spy)
    r = CliRunner().invoke(_app(), ["run", KEY, "--dry-run", "--tempo", "185"])
    assert r.exit_code == 0, r.output
    assert seen["cfg"].grid.tempo_bpm == 185.0
    for bad in ("0", "500"):
        r = CliRunner().invoke(_app(), ["run", KEY, "--dry-run", "--tempo", bad])
        assert r.exit_code == 2 and "--tempo" in r.output, r.output


# ---- mid-term plan 1-6: the tempo question before the GPU stages, and song flags kept in hints.toml

def _octave_grid(monkeypatch, decision="keep"):
    """The fake grid stage also writes an automatic tempo-octave decision (lulu.: 176 tracked, 88 right)."""
    from pathlib import Path

    from bandscribe.pipeline import graph

    real = graph._provider

    def provider(module):
        impls = real(module)
        if "grid" in impls:
            inner = impls["grid"]["run"]

            def run(ctx):
                inner(ctx)
                atomic.write_json(Path(ctx.out_dir) / "grid.json", {
                    "meter": [{"start_bar": 0, "end_bar": 90, "numerator": 4, "denominator": 4}],
                    "hypotheses": {"tempo_octave": {"decision": decision, "by": "onsets", "scores": {
                        "keep": {"bpm": 176.47, "score": 0.29}, "half": {"bpm": 88.24, "score": 0.24},
                        "double": {"bpm": 352.94, "score": 0.0}}}}})

            impls["grid"] = dict(impls["grid"], run=run)
        return impls

    monkeypatch.setattr(graph, "_provider", provider)


def _console(monkeypatch, f, answers):
    asked = []

    def ask(prompt, timeout_s):
        asked.append((prompt, list(f.calls)))
        return answers.pop(0)

    monkeypatch.setattr(run_cmd, "_is_tty", lambda: True)
    monkeypatch.setattr(run_cmd, "_ask", ask)
    return asked


def _hints(tmp_path):
    from bandscribe.jobs import hints as HI

    return HI.read_hints(tmp_path / "jobs" / KEY)


def test_tempo_is_asked_after_the_grid_before_any_gpu_transcription(tmp_path, monkeypatch):
    f = _job(tmp_path, monkeypatch)
    _octave_grid(monkeypatch)
    asked = _console(monkeypatch, f, ["x", "h"])          # a wrong key is asked again
    r = CliRunner().invoke(_app(), ["run", KEY])
    assert r.exit_code == 0, r.output
    assert [a[1] for a in asked] == [["beats", "grid"]] * 2
    assert "176.5 BPM, 4/4" in asked[0][0] and "88.2" in asked[0][0] and "352.9" in asked[0][0]
    assert _hints(tmp_path)["grid"]["tempo_bpm"] == 88.2
    assert f.calls.count("beats") == 1 and "amt_gtr" in f.calls
    assert "88.2 BPM 으로 정했습니다" in " ".join(r.output.split())
    asked.clear()
    assert CliRunner().invoke(_app(), ["run", KEY, "--force", "amt_gtr"]).exit_code == 0
    assert asked == []                                     # the song has a tempo now


def test_enter_confirms_without_touching_the_cache_and_a_timeout_leaves_it_unconfirmed(tmp_path, monkeypatch):
    f = _job(tmp_path, monkeypatch)
    _octave_grid(monkeypatch)
    _console(monkeypatch, f, [None])
    r = CliRunner().invoke(_app(), ["run", KEY, "--until", "amt_ms1"])
    assert r.exit_code == 0, r.output
    assert "답이 없어 자동 템포로 진행합니다" in " ".join(r.output.split()) and _hints(tmp_path) == {}
    asked = _console(monkeypatch, f, [""])
    r = CliRunner().invoke(_app(), ["run", KEY, "--from", "beats"])  # forced beats: run once, not twice
    assert r.exit_code == 0, r.output
    assert len(asked) == 1 and _hints(tmp_path) == {"hints": {"tempo_confirmed": True}}
    assert f.calls.count("beats") == 2 and f.calls.count("amt_ms1") == 2  # first run + the forced one
    asked.clear()
    f.calls.clear()
    assert CliRunner().invoke(_app(), ["run", KEY, "--force", "amt_ms1"]).exit_code == 0
    assert asked == [] and "beats" not in f.calls


def test_no_question_without_a_terminal_or_with_no_ask(tmp_path, monkeypatch):
    f = _job(tmp_path, monkeypatch)
    _octave_grid(monkeypatch)
    asked = _console(monkeypatch, f, [])
    monkeypatch.setattr(run_cmd, "_is_tty", lambda: False)
    assert CliRunner().invoke(_app(), ["run", KEY, "--until", "amt_ms1"]).exit_code == 0
    monkeypatch.setattr(run_cmd, "_is_tty", lambda: True)
    assert CliRunner().invoke(_app(), ["run", KEY, "--force", "amt_ms1", "--no-ask"]).exit_code == 0
    assert asked == []


def test_tempo_answers():
    assert run_cmd.parse_tempo_answer("", 176.47) == "keep"
    assert run_cmd.parse_tempo_answer(" H ", 176.47) == 88.2 and run_cmd.parse_tempo_answer("ㅗ", 176.47) == 88.2
    assert run_cmd.parse_tempo_answer("d", 68.0) == 136.0 and run_cmd.parse_tempo_answer("ㅇ", 68.0) == 136.0
    assert run_cmd.parse_tempo_answer("88", 176.47) == 88.0
    assert run_cmd.parse_tempo_answer("d", 250.0) is None and run_cmd.parse_tempo_answer("abc", 1.0) is None
    assert run_cmd.tempo_choices({"hypotheses": {"tempo_octave": {"by": "hint", "scores": {"keep": {"bpm": 9}}}}}) is None


def test_song_flags_are_kept_in_hints_toml_but_not_on_a_dry_run_or_a_rejected_plan(tmp_path, monkeypatch):
    from bandscribe.config import ConfigError
    from bandscribe.pipeline import runner

    _job(tmp_path, monkeypatch)
    r = CliRunner().invoke(_app(), ["run", KEY, "--dry-run", "--tempo", "185", "--title", "青と夏"])
    assert r.exit_code == 0 and _hints(tmp_path) == {}
    r = CliRunner().invoke(_app(), ["run", KEY, "--until", "grid", "--tempo", "185", "--title", "青と夏", "--parts", "2",
                                    "--set", "tab.guitar_tuning=Eb standard", "--set", "run.profile=fast"])
    assert r.exit_code == 0, r.output
    assert _hints(tmp_path) == {"grid": {"tempo_bpm": 185.0}, "hints": {"title": "青と夏"},
                                "tab": {"guitar_parts": 2, "guitar_tuning": "Eb standard"}}
    assert "다음 실행에도 씀" in r.output
    r = CliRunner().invoke(_app(), ["run", KEY, "--until", "grid", "--tempo", "185"])
    assert "다음 실행에도 씀" not in r.output                # unchanged values are not rewritten

    def bad_plan(*a, **kw):
        raise ConfigError("hints.instruments 에 알 수 없는 악기 계열이 있습니다: kazoo")

    monkeypatch.setattr(runner, "plan", bad_plan)
    assert CliRunner().invoke(_app(), ["run", KEY, "--instruments", "kazoo"]).exit_code == 2
    assert "instruments" not in _hints(tmp_path).get("hints", {})


# ---- mid-term plan 1-7: refresh (CPU stages only, every song) and status --songs

def _bump(monkeypatch, stages):
    """A new code version of ``stages`` (their cache keys change, as after a code or defaults change)."""
    from bandscribe.pipeline import graph

    real = graph._provider

    def provider(module):
        impls = real(module)
        return {n: (dict(i, code_version="2") if n in stages else i) for n, i in impls.items()}

    monkeypatch.setattr(graph, "_provider", provider)


def _refresh_app():
    from bandscribe.commands import refresh

    app = typer.Typer()

    @app.callback()
    def _root():
        pass

    refresh.register(app)
    return app


def test_refresh_recomputes_cpu_stages_and_skips_songs_that_need_the_gpu(tmp_path, monkeypatch):
    f = _job(tmp_path, monkeypatch)
    assert CliRunner().invoke(_app(), ["run", KEY]).exit_code == 0
    tex = tmp_path / "jobs" / KEY / "export" / "tab" / "score.alphatex"

    def scored():  # the fake score stage writes no alphaTex, and a publish replaces export/
        tex.parent.mkdir(parents=True, exist_ok=True)
        tex.write_text("\title x", encoding="utf-8")

    scored()
    r = CliRunner().invoke(_refresh_app(), ["refresh", "--dry-run"])
    assert r.exit_code == 0 and "최신" in r.output
    _bump(monkeypatch, {"score"})
    f.calls.clear()
    r = CliRunner().invoke(_refresh_app(), ["refresh", "--dry-run"])
    assert "다시 계산: score" in r.output and f.calls == []
    r = CliRunner().invoke(_refresh_app(), ["refresh"])
    assert r.exit_code == 0, r.output
    assert f.calls == ["score"] and "다시 계산 1곡" in r.output
    scored()
    _bump(monkeypatch, {"amt_gtr"})
    f.calls.clear()
    r = CliRunner().invoke(_refresh_app(), ["refresh"])
    flat = " ".join(r.output.split())
    assert r.exit_code == 0 and "건너뜀" in flat and "amt_gtr" in flat and f.calls == []

