from __future__ import annotations

import importlib
import io
import json
import logging
import os
import subprocess
import sys
import tomllib
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from bandscribe import atomic, cli, config, log, paths

MANAGED_ENV = ("BANDSCRIBE_ROOT", "HF_HOME", "TORCH_HOME", "UV_CACHE_DIR", "PYTHONUTF8", "PYTHONIOENCODING", "PYTHONNOUSERSITE",
               "MPLBACKEND")

META = {
    "bandscribe_version": "0.1.0",
    "created_utc": "2026-09-30T00:00:00Z",
    "song_key": "f-0123456789abcdef",
    "source": {
        "kind": "file",
        "ref": "song [live].flac",
        "filename": "song [live].flac",
        "sha256": "aaa",
        "probe": {"format": {"duration": "215.5"}},
    },
    "youtube": None,
    "decode": {"sr": 44100, "pcm_sha256": "0123456789abcdef" * 4},
    "loudness": {"integrated_lufs": -7.96, "true_peak_dbfs": 1.02, "sample_peak_dbfs": 0.9, "gain_db": -2.02},
    "flags": {
        "quasi_mono": True,
        "side_mid_db": -35.0,
        "lowpass_hz_est": 16000.0,
        "lossy_cutoff": False,
        "nonstandard_sr": False,
        "drc": False,
        "mono_source": False,
    },
}


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """BANDSCRIBE_ROOT in tmp: bandscribe.paths is reloaded so every paths.X points below it, then restored."""
    r = tmp_path / "bandscribe-root"
    r.mkdir()
    root_logger = logging.getLogger()
    level = root_logger.level
    with monkeypatch.context() as mp:
        for key in MANAGED_ENV:
            # set-then-delete records the original value (or its absence) so main()'s
            # paths.apply_process_env() is fully undone when the context exits
            mp.setenv(key, "x")
            mp.delenv(key)
        mp.setenv("BANDSCRIBE_ROOT", str(r))
        mp.setenv("COLUMNS", "200")  # rich tables: no wrapping of what the tests look for
        importlib.reload(paths)
        try:
            yield r
        finally:
            log.remove_handlers()
            root_logger.setLevel(level)
    importlib.reload(paths)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def install(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    """Stand-in for a backend module; the CLI imports backends lazily, so sys.modules wins."""
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@dataclass
class FakeCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass
class FakeReport:
    checks: list[FakeCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.required)


class FakeStore:
    instances: list[FakeStore] = []

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self.removed = 3
        FakeStore.instances.append(self)

    def gc_temp(self) -> int:
        return self.removed


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> type[FakeStore]:
    FakeStore.instances = []
    install(monkeypatch, "bandscribe.jobs.store", JobStore=FakeStore)
    return FakeStore


# ------------------------------------------------------------------------------------------ basics


def test_help_lists_commands(root: Path, runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in ("doctor", "ingest", "status", "config", "gc"):
        assert name in result.output


def test_version(root: Path, runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"bandscribe {cli.__version__}"


def test_verbose_logs_debug_to_console_and_file(root: Path, runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["-v", "config", "show", "--set", "ingest.lowpass_detect_hz=15000"])
    assert result.exit_code == 0, result.output
    assert "config layer: --set ingest.lowpass_detect_hz=15000" in result.stderr
    assert "config layer: --set" in (paths.LOGS / "bandscribe.log").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------- config


def test_config_show_defaults(root: Path, runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert "(없음)" in result.stdout  # no user config yet: main() was not run
    parsed = tomllib.loads(result.stdout)
    assert parsed["youtube"]["enabled"] is True
    assert parsed["youtube"]["format"] == "ba[format_id!*=drc][acodec=opus]/ba[format_id!*=drc]/ba"


def test_config_show_layers_user_file_and_set(root: Path, runner: CliRunner) -> None:
    user = paths.USER_CONFIG
    user.parent.mkdir(parents=True)
    user.write_text("[ingest]\nlowpass_detect_hz = 15000\n[separator]\nbackend = \"sw\"\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["config", "show", "--set", "youtube.enabled=false", "-s", "gpu.vram_margin_gb=1.5"])
    assert result.exit_code == 0, result.output
    parsed = tomllib.loads(result.stdout)
    assert parsed["youtube"]["enabled"] is False
    assert parsed["ingest"]["lowpass_detect_hz"] == 15000
    assert parsed["gpu"]["vram_margin_gb"] == 1.5
    assert parsed["separator"] == {"backend": "sw"}
    assert "# --set youtube.enabled=false" in result.stdout


def test_config_show_bad_set_exits_2(root: Path, runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["config", "show", "--set", "youtube.enabled=maybe"])
    assert result.exit_code == 2
    assert "오류" in result.stderr and "youtube.enabled" in result.stderr
    result = runner.invoke(cli.app, ["config", "show", "--set", "novalue"])
    assert result.exit_code == 2
    assert "--set" in result.stderr


def test_set_typo_in_section_exits_2_before_anything_runs(
    root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, fake_store
) -> None:
    """Regression: `--set youtub.enabled=false` only warned, left YouTube on and downloaded."""
    calls = _fake_ingest(monkeypatch, meta=META)
    result = runner.invoke(cli.app, ["ingest", "https://youtu.be/YE7VzlLtp-4", "--set", "youtub.enabled=false"])
    assert result.exit_code == 2
    assert "'youtub'" in result.stderr and "'youtube'" in result.stderr and "뜻했나요" in result.stderr
    assert calls == []
    result = runner.invoke(cli.app, ["config", "show", "--set", "ingest.sample_rate=48000"])
    assert result.exit_code == 2 and "44100" in result.stderr


# ---------------------------------------------------------------------------------------- doctor


def test_doctor_all_ok(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def run_doctor(cfg: Any) -> FakeReport:
        seen.append(cfg)
        return FakeReport([FakeCheck("ffmpeg", True, "8.0.1 [soxr]"), FakeCheck("node", True, "v24")])

    install(monkeypatch, "bandscribe.doctor", run_doctor=run_doctor)
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert isinstance(seen[0], config.Config)
    assert "ffmpeg" in result.stdout and "8.0.1 [soxr]" in result.stdout  # brackets printed literally
    assert "통과" in result.stdout


def test_doctor_required_failure_exits_1(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    report = FakeReport([FakeCheck("cuda", False, "CUDA 없음"), FakeCheck("py310", False, "info", required=False)])
    install(monkeypatch, "bandscribe.doctor", run_doctor=lambda cfg: report)
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "실패" in result.stdout and "경고" in result.stdout and "CUDA 없음" in result.stdout


def test_doctor_warning_only_exits_0(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    report = FakeReport([FakeCheck("cuda", True, "ok"), FakeCheck("py310", False, "info", required=False)])
    install(monkeypatch, "bandscribe.doctor", run_doctor=lambda cfg: report)
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "경고 1개" in result.stdout


def test_doctor_json(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    report = FakeReport([FakeCheck("cuda", False, "없음"), FakeCheck("node", True, "v24")])
    install(monkeypatch, "bandscribe.doctor", run_doctor=lambda cfg: report)
    result = runner.invoke(cli.app, ["doctor", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["checks"][0] == {"name": "cuda", "ok": False, "detail": "없음", "required": True}


def test_doctor_broken_module_is_a_clean_error(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, "bandscribe.doctor")  # no run_doctor attribute
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "bandscribe.doctor.run_doctor" in result.stderr and "불러오지 못했습니다" in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_doctor_exception_is_a_clean_error(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(cfg: Any) -> None:
        raise RuntimeError("nvml exploded")

    install(monkeypatch, "bandscribe.doctor", run_doctor=boom)
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "환경 점검 실패: nvml exploded" in result.stderr


# ---------------------------------------------------------------------------------------- ingest


def _fake_ingest(monkeypatch: pytest.MonkeyPatch, *, meta: dict, cache_hit: bool = False, error: Exception | None = None):
    calls: list[dict[str, Any]] = []

    def ingest(source: str, *, store: Any, cfg: Any) -> types.SimpleNamespace:
        calls.append({"source": source, "store": store, "cfg": cfg})
        if error is not None:
            raise error
        key = meta["song_key"]
        return types.SimpleNamespace(song_key=key, job_dir=paths.JOBS / key, meta=meta, cache_hit=cache_hit)

    install(monkeypatch, "bandscribe.ingest", ingest=ingest)
    return calls


def test_ingest_file(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, fake_store, tmp_path: Path) -> None:
    src = tmp_path / "song [live].flac"
    src.write_bytes(b"fake")
    calls = _fake_ingest(monkeypatch, meta=META)
    result = runner.invoke(cli.app, ["ingest", str(src), "--set", "ingest.true_peak_ceiling_dbfs=-2.0"])
    assert result.exit_code == 0, result.output

    call = calls[0]
    assert call["source"] == str(src)
    assert call["cfg"].ingest.true_peak_ceiling_dbfs == -2.0
    assert call["store"].root == paths.JOBS and str(root) in str(paths.JOBS)

    out = result.stdout
    assert "song_key: f-0123456789abcdef" in out
    assert "캐시 적중: 아니오" in out
    assert "제목: song [live].flac" in out  # brackets not eaten as rich markup
    assert "3:35.5 (215.5 s)" in out
    assert "-7.96 LUFS" in out and "+1.02 dBTP" in out and "-2.02 dB" in out
    assert "준모노(S/M -35.0 dB)" in out
    assert str(paths.JOBS / "f-0123456789abcdef") in out


def test_ingest_url_cache_hit(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    url = "https://www.youtube.com/watch?v=YE7VzlLtp-4"
    meta = {
        **META,
        "song_key": "yt-YE7VzlLtp-4",
        "source": {"kind": "youtube", "ref": url},
        "youtube": {"id": "YE7VzlLtp-4", "title": "Big Buck Bunny", "format_id": "251", "acodec": "opus", "duration": 33},
        "flags": {"drc": False, "lossy_cutoff": True, "lowpass_hz_est": 16000.0},
    }
    calls = _fake_ingest(monkeypatch, meta=meta, cache_hit=True)
    result = runner.invoke(cli.app, ["ingest", url])
    assert result.exit_code == 0, result.output
    assert calls[0]["source"] == url  # URLs skip the local-file check
    assert calls[0]["cfg"].youtube.enabled is True  # default on
    out = result.stdout
    assert "캐시 적중: 예" in out and "Big Buck Bunny" in out and "251 opus" in out
    assert "손실 압축 컷오프(~16.0 kHz)" in out
    assert "0:33.0" in out


def test_ingest_missing_file(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    calls = _fake_ingest(monkeypatch, meta=META)
    result = runner.invoke(cli.app, ["ingest", "nope.flac"])
    assert result.exit_code == 2
    assert "파일을 찾을 수 없습니다" in result.stderr
    result = runner.invoke(cli.app, ["ingest", "youtu.be/YE7VzlLtp-4"])
    assert "https://" in result.stderr
    assert calls == []


def test_ingest_backend_error(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    _fake_ingest(monkeypatch, meta=META, error=RuntimeError("YouTube 입력이 꺼져 있습니다"))
    result = runner.invoke(cli.app, ["ingest", "https://youtu.be/YE7VzlLtp-4", "--set", "youtube.enabled=false"])
    assert result.exit_code == 1
    assert "입력 처리 실패: YouTube 입력이 꺼져 있습니다" in result.stderr


# ---------------------------------------------------------------------------------------- status


def make_jobs(jobs: Path) -> None:
    job = jobs / "f-0123456789abcdef"
    atomic.write_json(job / "input" / "meta.json", META)
    atomic.write_json(
        job / "stages" / "sep" / "abcdef012345" / "manifest.json",
        {"stage": "sep", "key": "abcdef012345" * 5, "status": "complete", "created_utc": "2026-09-30T01:02:03Z", "duration_s": 12.3},
    )
    (job / "stages" / "sep" / ".tmp-abcdef012345-999999-zz").mkdir()
    (job / "stages" / "amt" / "0123456789ab").mkdir(parents=True)
    (jobs / "f-0123aaaaaaaaaaaa").mkdir()  # input never completed
    atomic.write_json(
        jobs / "index.json",
        {"file:aaa": "f-0123456789abcdef", "file:bbb": "f-deadbeefdeadbeef"},  # second points at a deleted job
    )


def test_status_empty(root: Path, runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "작업이 없습니다" in result.stdout


def test_status_list(root: Path, runner: CliRunner) -> None:
    make_jobs(paths.JOBS)
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "작업 3개" in out
    assert "f-0123456789abcdef" in out and "song [live].flac" in out and "3:35.5" in out
    assert "f-0123aaaaaaaaaaaa" in out and "없음" in out
    assert "f-deadbeefdeadbeef" in out and "폴더 없음" in out


def test_status_one_job_by_prefix(root: Path, runner: CliRunner) -> None:
    make_jobs(paths.JOBS)
    result = runner.invoke(cli.app, ["status", "f-012345"])
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "작업 키: f-0123456789abcdef" in out
    assert "색인: file:aaa" in out
    assert "입력: 완료" in out and "-7.96 LUFS" in out
    assert "sep" in out and "abcdef012345" in out and "완료" in out and "12.3 s" in out
    assert "임시(미완료)" in out and "manifest 없음" in out


def test_status_unknown_or_ambiguous(root: Path, runner: CliRunner) -> None:
    make_jobs(paths.JOBS)
    result = runner.invoke(cli.app, ["status", "f-zzz"])
    assert result.exit_code == 1 and "작업을 찾을 수 없습니다" in result.stderr
    result = runner.invoke(cli.app, ["status", "f-0123"])
    assert result.exit_code == 1 and "여러 개" in result.stderr
    result = runner.invoke(cli.app, ["status", "../bandscribe-root"])
    assert result.exit_code == 1


def test_status_job_without_input(root: Path, runner: CliRunner) -> None:
    make_jobs(paths.JOBS)
    result = runner.invoke(cli.app, ["status", "f-0123aaaaaaaaaaaa"])
    assert result.exit_code == 0, result.output
    assert "입력: 없음" in result.stdout and "단계 결과: 없음" in result.stdout


# -------------------------------------------------------------------------------------------- gc


def test_gc(root: Path, runner: CliRunner, fake_store) -> None:
    result = runner.invoke(cli.app, ["gc"])
    assert result.exit_code == 0, result.output
    assert "임시 폴더 3개를 지웠습니다" in result.stdout
    assert fake_store.instances[0].root == paths.JOBS


def test_gc_nothing(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, fake_store) -> None:
    monkeypatch.setattr(FakeStore, "gc_temp", lambda self: 0)
    result = runner.invoke(cli.app, ["gc"])
    assert result.exit_code == 0 and "지울 임시 폴더가 없습니다" in result.stdout


def test_gc_without_jobs_module(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, "bandscribe.jobs.store")  # module present but JobStore missing
    result = runner.invoke(cli.app, ["gc"])
    assert result.exit_code == 1 and "bandscribe.jobs.store.JobStore" in result.stderr


# ------------------------------------------------------------------------------------------ main


def test_main_bootstraps_everything(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["config", "show"])
    assert exc.value.code == 0
    out = capsys.readouterr().out

    data = root / "data"
    for sub in ("jobs", "logs", "downloads", "models", "hf", "torch", "uv-cache"):
        assert (data / sub).is_dir(), sub
    user = data / "config.toml"
    assert tomllib.loads(user.read_text(encoding="utf-8"))["consent"]["youtube"] == config.YOUTUBE_CONSENT_TEXT
    assert config.YOUTUBE_CONSENT_TEXT in out  # the effective config now includes the consent
    assert os.environ["HF_HOME"] == str(data / "hf")
    assert os.environ["PYTHONUTF8"] == "1"
    assert os.environ["MPLBACKEND"] == "Agg"  # synctoolbox/matplotlib never look for a GUI backend
    for sub in ("eval", "runs", "datasets", "bench", "npm-cache"):  # M1a/M2 layout (M1_M2_SPEC 1.2)
        assert (data / sub).is_dir(), sub
    assert log.get_logfile() == data / "logs" / "bandscribe.log"
    assert "argv=['config', 'show']" in (data / "logs" / "bandscribe.log").read_text(encoding="utf-8")


def test_utf8_stdio_reconfigures_in_place_once(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp949")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(cli, "_stdio_done", False)
    cli._utf8_stdio()
    assert sys.stdout is stream  # same object: no second wrapper around the buffer
    assert stream.encoding == "utf-8"
    stream.write("한글 ✓")
    stream.flush()
    assert raw.getvalue() == "한글 ✓".encode()

    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp949"))
    cli._utf8_stdio()  # already done for this process: no-op
    assert sys.stdout.encoding == "cp949"


# ------------------------------------------------------------------------------- M1a/M2 commands


def test_help_lists_m2_commands(root: Path, runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in cli.OPTIONAL_COMMANDS:
        assert name in result.output
    assert "(불러오지 못함)" not in result.output  # every owner module registered for real


def test_managed_env_matches_paths(root: Path) -> None:
    assert set(paths._managed_env()) == set(MANAGED_ENV)
    assert paths.child_env()["MPLBACKEND"] == "Agg"


def test_m2_paths_layout(root: Path) -> None:
    data = root / "data"
    assert paths.EVAL == data / "eval" and paths.RUNS == data / "runs" and paths.DATASETS == data / "datasets"
    assert paths.BENCH == data / "bench" and paths.NPM_CACHE == data / "npm-cache"
    assert paths.MODELS_LOCK == data / "models" / "models.lock.json"
    assert paths.VRAM_TABLE == data / "bench" / "vram_table.json"
    assert paths.NODE_DIR == root / "node"
    assert paths.BP310_PYTHON == root / "envs" / "bp310" / ".venv" / "Scripts" / "python.exe"


def test_broken_command_module_becomes_a_placeholder(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    app = typer.Typer()

    @app.command()
    def hello() -> None:
        """기존 명령"""

    install(monkeypatch, "bandscribe.commands.fakebroken")  # no register() -> AttributeError
    assert cli._register_optional(app, "bandscribe.commands.fakebroken") is False
    assert cli._register_optional(app, "bandscribe.commands.does_not_exist_xyz") is False
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0 and "hello" in result.output and "fakebroken" in result.output
    for args in (["fakebroken"], ["fakebroken", "run", "--suite", "quick"], ["fakebroken", "--help"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1, (args, result.output)
        assert "fakebroken 명령을 불러오지 못했습니다: AttributeError" in result.stderr
    result = runner.invoke(app, ["does_not_exist_xyz"])
    assert result.exit_code == 1 and "ModuleNotFoundError" in result.stderr
    assert runner.invoke(app, ["hello"]).exit_code == 0  # the rest of the CLI still works


def test_half_registered_module_is_rolled_back(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    def register(app: typer.Typer) -> None:
        sub = typer.Typer()

        @sub.command("x")
        def _x() -> None:
            pass

        app.add_typer(sub, name="halfway")
        raise RuntimeError("register 도중 실패")

    install(monkeypatch, "bandscribe.commands.halfway", register=register)
    app = typer.Typer()

    @app.command()
    def hello() -> None:
        pass

    assert cli._register_optional(app, "bandscribe.commands.halfway") is False
    assert len(app.registered_groups) == 0 and [c.name for c in app.registered_commands] == [None, "halfway"]
    result = runner.invoke(app, ["halfway", "x"])
    assert result.exit_code == 1 and "register 도중 실패" in result.stderr


def test_working_module_is_registered(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    def register(app: typer.Typer) -> None:
        @app.command("ok-cmd")
        def _ok() -> None:
            print("실행됨")

    install(monkeypatch, "bandscribe.commands.okmod", register=register)
    app = typer.Typer()

    @app.command()
    def hello() -> None:
        pass

    assert cli._register_optional(app, "bandscribe.commands.okmod") is True
    result = runner.invoke(app, ["ok-cmd"])
    assert result.exit_code == 0 and "실행됨" in result.output


def test_optional_commands_load_lazily(root: Path) -> None:
    """`bandscribe ingest` / `status` must not pay for the M2 command modules (cache hit < 1 s, DESIGN 10)."""
    code = "import sys, bandscribe.cli; print(sorted(m for m in sys.modules if m.startswith('bandscribe.commands.')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_broken_owner_module_does_not_break_the_cli(root: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, "bandscribe.commands.eval")  # imports fine but has no register()
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0 and "(불러오지 못함)" in result.output and "doctor" in result.output
    result = runner.invoke(cli.app, ["eval", "run", "--suite", "quick"])
    assert result.exit_code == 1
    assert "eval 명령을 불러오지 못했습니다: AttributeError" in result.stderr
    assert runner.invoke(cli.app, ["config", "show"]).exit_code == 0  # the rest still works
    result = runner.invoke(cli.app, ["models", "--help"])  # other owner modules are unaffected
    assert result.exit_code == 0 and "verify" in result.output
