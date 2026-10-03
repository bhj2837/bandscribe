"""gtab command line (typer). Entry points: the ``gtab`` script, ``gtab.cmd`` and ``python -m gtab``.

Backends (gtab.doctor, gtab.ingest, gtab.jobs.store) are imported inside the commands, so ``--help``,
``config show`` and ``status`` keep working even when one of them is broken or missing a dependency.
User-facing text is Korean; this module and ``gtab/commands/*`` are the only ones that print.

The M1a/M2 commands (``run``, ``bench``, ``eval``, ``gt``, ``data``, ``models``) live in ``gtab/commands/<name>.py``
(``register(app)``, M1_M2_SPEC 1.7) and are loaded lazily by the root group: always listed, imported only when
that command runs or a help screen lists them. Building them eagerly costs ~50 ms per start (typer converts every
command signature), which ``gtab ingest``'s cache hit cannot afford (< 1 s with CLI start-up, DESIGN 10).
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from typer.core import TyperGroup

from gtab import __version__, atomic, config, log, paths

logger = logging.getLogger(__name__)

# Command modules owned by the M1a/M2 work packages (M1_M2_SPEC 1.7), loaded lazily by _RootGroup.
OPTIONAL_COMMANDS: tuple[str, ...] = ("run", "bench", "eval", "gt", "data", "models")


class _RootGroup(TyperGroup):
    """Root command group that adds ``gtab.commands.<name>`` on first use (see module docstring)."""

    def list_commands(self, ctx: Any) -> list[str]:
        names = list(super().list_commands(ctx))
        return names + [n for n in OPTIONAL_COMMANDS if n not in self.commands]

    def get_command(self, ctx: Any, cmd_name: str) -> Any:
        if cmd_name not in self.commands and cmd_name in OPTIONAL_COMMANDS:
            for name, cmd in _load_optional(cmd_name).items():
                self.commands.setdefault(name, cmd)
        return super().get_command(ctx, cmd_name)


app = typer.Typer(
    name="gtab",
    help="밴드 녹음 -> 기타 파트별 탭 + 베이스 탭 (개인용).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,  # locals can hold tokens or huge arrays
    cls=_RootGroup,
)
config_app = typer.Typer(help="설정 보기.", no_args_is_help=True)
app.add_typer(config_app, name="config")

SetOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        metavar="KEY=VALUE",
        help="설정 덮어쓰기(여러 번 가능). 예: --set youtube.enabled=false",
    ),
]

_FLAG_LABELS = {
    "quasi_mono": "준모노",
    "mono_source": "모노 원본",
    "lossy_cutoff": "손실 압축 컷오프",
    "nonstandard_sr": "비표준 샘플레이트",
    "drc": "DRC 스트림",
}


# ------------------------------------------------------------------------------------------- helpers


def _out() -> Console:
    # Built per call: terminal detection happens at construction, and main() or a test runner may have
    # reconfigured / swapped the stream after import.
    return Console(highlight=False)


def _err() -> Console:
    return Console(stderr=True, highlight=False)


def _fail(message: str, code: int = 1, *, show_log: bool = False) -> NoReturn:
    _err().print(f"[bold red]오류:[/] {escape(message)}", soft_wrap=True)
    logfile = log.get_logfile()
    if show_log and logfile is not None:
        _err().print(f"자세한 내용: {escape(str(logfile))}", soft_wrap=True)
    raise typer.Exit(code)


def _load_cfg(overrides: list[str] | None) -> config.Config:
    try:
        return config.load_config(overrides=overrides or ())
    except config.ConfigError as e:
        _fail(str(e), code=2)


def _backend(module: str, attr: str) -> Any:
    """Import ``module.attr`` lazily; any failure becomes a clean CLI error instead of a traceback."""
    try:
        return getattr(importlib.import_module(module), attr)
    except Exception as e:  # missing module or dependency, syntax error, missing attribute
        logger.debug("cannot load %s.%s", module, attr, exc_info=True)
        _fail(f"{module}.{attr} 을(를) 불러오지 못했습니다: {type(e).__name__}: {e}", show_log=True)


def _call(what: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # backend errors carry their own (Korean) message; the traceback goes to the log
        logger.debug("%s failed", what, exc_info=True)
        _fail(f"{what} 실패: {str(e) or type(e).__name__}", show_log=True)


def _dig(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _num(v: Any, fmt: str = ".2f", signed: bool = False) -> str:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "?"
    return format(v, ("+" if signed else "") + fmt)


def _duration_s(meta: dict[str, Any], job_dir: Path) -> float | None:
    """Best available duration: explicit meta fields, then the decoded WAV header, then ffprobe/yt-dlp."""
    for keys in (("duration_s",), ("decode", "duration_s")):
        v = _dig(meta, *keys)
        if isinstance(v, (int, float)):
            return float(v)
    wav = job_dir / "input" / "mix_44k_f32.wav"
    if wav.is_file():
        try:
            import soundfile

            return float(soundfile.info(str(wav)).duration)
        except Exception:
            logger.debug("soundfile.info failed for %s", wav, exc_info=True)
    for keys in (("source", "probe", "format", "duration"), ("youtube", "duration")):
        try:
            return float(_dig(meta, *keys))
        except (TypeError, ValueError):
            continue
    return None


def _fmt_duration(sec: float | None, short: bool = False) -> str:
    if sec is None:
        return "?"
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:04.1f}" + ("" if short else f" ({sec:.1f} s)")


def _fmt_flags(flags: dict[str, Any]) -> str:
    parts = []
    for key, value in flags.items():
        if value is not True:
            continue
        label = _FLAG_LABELS.get(key, key)
        if key == "quasi_mono" and isinstance(flags.get("side_mid_db"), (int, float)):
            label += f"(S/M {flags['side_mid_db']:.1f} dB)"
        elif key == "lossy_cutoff" and isinstance(flags.get("lowpass_hz_est"), (int, float)):
            label += f"(~{flags['lowpass_hz_est'] / 1000:.1f} kHz)"
        parts.append(label)
    return ", ".join(parts) or "없음"


def _print_meta(c: Console, meta: dict[str, Any], job_dir: Path) -> None:
    title = _dig(meta, "youtube", "title") or _dig(meta, "source", "filename")
    if title:
        c.print(f"제목: {escape(str(title))}", soft_wrap=True)
    kind, ref = _dig(meta, "source", "kind"), _dig(meta, "source", "ref")
    if kind or ref:
        c.print(f"출처: {escape(str(kind or '?'))} {escape(str(ref or ''))}".rstrip(), soft_wrap=True)
    yt = meta.get("youtube")
    if isinstance(yt, dict):
        fmt = " ".join(str(yt[k]) for k in ("format_id", "acodec", "abr", "asr") if yt.get(k) is not None)
        if fmt:
            c.print(f"YouTube 형식: {escape(fmt)}")
    c.print(f"길이: {_fmt_duration(_duration_s(meta, job_dir))}")
    loud = meta.get("loudness") or {}
    c.print(
        f"라우드니스: {_num(loud.get('integrated_lufs'))} LUFS, "
        f"true peak {_num(loud.get('true_peak_dbfs'), signed=True)} dBTP, "
        f"적용 게인 {_num(loud.get('gain_db'), signed=True)} dB"
    )
    c.print(f"플래그: {escape(_fmt_flags(meta.get('flags') or {}))}")


def _is_url(source: str) -> bool:
    return source.strip().lower().startswith(("http://", "https://"))


def _jsonable_check(check: Any) -> dict[str, Any]:
    if hasattr(check, "model_dump"):
        return check.model_dump(mode="json")
    if is_dataclass(check) and not isinstance(check, type):
        return asdict(check)
    return {k: getattr(check, k, None) for k in ("name", "ok", "detail", "required")}


# ---------------------------------------------------------------------------------------- job files
# Read-only views of the JobStore layout (M0_SPEC 6). Reading directly, instead of through JobStore,
# keeps `status` usable when gtab.jobs is broken and never creates directories as a side effect.


def _job_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))


def _read_index(root: Path) -> dict[str, str]:
    path = root / "index.json"
    if not path.is_file():
        return {}
    from filelock import FileLock, Timeout

    # Same lock as JobStore: on Windows an open reader makes a concurrent os.replace() of the index fail.
    try:
        with FileLock(root / "index.lock", timeout=5):
            data = atomic.read_json(path)
    except (OSError, ValueError, Timeout) as e:
        logger.warning("index.json 을 읽지 못했습니다: %s (%s)", path, e)
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str)} if isinstance(data, dict) else {}


def _read_meta(job_dir: Path) -> dict[str, Any] | None:
    path = job_dir / "input" / "meta.json"
    if not path.is_file():
        return None
    try:
        data = atomic.read_json(path)
    except (OSError, ValueError) as e:
        logger.warning("meta.json 을 읽지 못했습니다: %s (%s)", path, e)
        return {}
    return data if isinstance(data, dict) else {}


def _stage_rows(job_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stages = job_dir / "stages"
    if not stages.is_dir():
        return rows
    for stage_dir in sorted(d for d in stages.iterdir() if d.is_dir()):
        for key_dir in sorted(d for d in stage_dir.iterdir() if d.is_dir()):
            row: dict[str, Any] = {"stage": stage_dir.name, "key": key_dir.name, "complete": False}
            if key_dir.name.startswith(".tmp-"):
                row["status"] = "임시(미완료)"
            else:
                manifest: Any = None
                try:
                    manifest = atomic.read_json(key_dir / "manifest.json")
                except (OSError, ValueError):
                    pass
                if isinstance(manifest, dict):
                    row["complete"] = manifest.get("status") == "complete"
                    row["status"] = "완료" if row["complete"] else str(manifest.get("status", "?"))
                    row["created_utc"] = manifest.get("created_utc")
                    row["duration_s"] = manifest.get("duration_s")
                else:
                    row["status"] = "manifest 없음"
            rows.append(row)
    return rows


def _resolve_job(root: Path, key: str) -> Path | None:
    if not key or any(ch in key for ch in "/\\:") or key.startswith("."):
        return None
    exact = root / key
    if exact.is_dir():
        return exact
    matches = [d for d in _job_dirs(root) if d.name.startswith(key)]
    if len(matches) > 1:
        _fail(f"'{key}'(으)로 시작하는 작업이 여러 개입니다: {', '.join(d.name for d in matches)}")
    return matches[0] if matches else None


# ------------------------------------------------------------------------------------------ commands


def _version_callback(value: bool) -> None:
    if value:
        print(f"gtab {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="자세한 로그(DEBUG)를 콘솔에 표시")] = False,
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="버전 표시")
    ] = False,
) -> None:
    if verbose:
        log.setup_logging("DEBUG")


@app.command()
def doctor(as_json: Annotated[bool, typer.Option("--json", help="JSON으로 출력")] = False) -> None:
    """환경 점검: GPU/CUDA, ffmpeg, Node, 디스크, NVML/PDH, yt-dlp. 필수 항목이 실패하면 종료 코드 1."""
    cfg = _load_cfg(None)
    run_doctor = _backend("gtab.doctor", "run_doctor")
    if not as_json:
        _err().print("환경 점검 중... (GPU 셀프테스트 포함, 1분쯤 걸릴 수 있습니다)")
    report = _call("환경 점검", run_doctor, cfg)
    checks = list(report.checks)
    ok = bool(report.ok)

    if as_json:
        payload = {"ok": ok, "checks": [_jsonable_check(c) for c in checks]}
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    else:
        table = Table(title="gtab doctor", title_justify="left")
        table.add_column("결과", no_wrap=True)
        table.add_column("항목", no_wrap=True)
        table.add_column("내용", overflow="fold")
        for c in checks:
            if c.ok:
                mark = "[green]OK[/]"
            elif c.required:
                mark = "[bold red]실패[/]"
            else:
                mark = "[yellow]경고[/]"
            table.add_row(mark, escape(str(c.name)), escape(str(c.detail)))
        out = _out()
        out.print(table)
        failed = sum(1 for c in checks if not c.ok and c.required)
        warned = sum(1 for c in checks if not c.ok and not c.required)
        tail = f" 경고 {warned}개." if warned else ""
        if ok:
            out.print(f"[green]필수 점검을 모두 통과했습니다.[/]{tail}")
        else:
            out.print(f"[bold red]필수 점검 {failed}개가 실패했습니다.[/]{tail}")
    if not ok:
        raise typer.Exit(1)


@app.command()
def ingest(
    source: Annotated[str, typer.Argument(help="오디오 파일 경로 또는 YouTube URL", show_default=False)],
    set_: SetOpt = None,
) -> None:
    """오디오 파일이나 YouTube 링크로 작업 폴더(input/)를 만든다. 이미 처리한 입력이면 캐시를 쓴다."""
    cfg = _load_cfg(set_)
    if not _is_url(source) and not Path(source).is_file():
        hint = " (URL이면 https:// 로 시작해야 합니다)" if "youtu" in source.lower() else ""
        _fail(f"파일을 찾을 수 없습니다: {source}{hint}", code=2)
    ingest_fn = _backend("gtab.ingest", "ingest")
    store_cls = _backend("gtab.jobs.store", "JobStore")
    store = _call("작업 저장소 열기", store_cls, paths.JOBS)
    _err().print(f"입력 처리 중: {escape(source)}", soft_wrap=True)
    result = _call("입력 처리", ingest_fn, source, store=store, cfg=cfg)

    c = _out()
    c.print(f"song_key: [bold]{escape(str(result.song_key))}[/]")
    c.print("캐시 적중: " + ("예 (이미 처리한 입력, 다시 계산하지 않음)" if result.cache_hit else "아니오 (새로 처리함)"))
    _print_meta(c, result.meta or {}, Path(result.job_dir))
    c.print(f"작업 폴더: {escape(str(result.job_dir))}", soft_wrap=True)


@app.command()
def status(
    song_key: Annotated[str | None, typer.Argument(help="작업 키(앞부분만 써도 됨). 생략하면 전체 목록")] = None,
) -> None:
    """작업 목록, 또는 작업 하나의 입력 정보와 단계별 결과."""
    root = Path(paths.JOBS)
    index = _read_index(root)
    sources: dict[str, list[str]] = {}
    for src_key, key in index.items():
        sources.setdefault(key, []).append(src_key)
    c = _out()

    if song_key is None:
        dirs = {d.name: d for d in _job_dirs(root)}
        names = sorted(set(dirs) | set(sources))
        if not names:
            c.print(f"작업이 없습니다. ({escape(str(root))})", soft_wrap=True)
            return
        table = Table(title=f"작업 {len(names)}개", title_justify="left")
        for col in ("작업 키", "제목", "출처", "길이", "입력", "완료 단계"):
            table.add_column(col, overflow="fold")
        for name in names:
            job_dir = dirs.get(name)
            if job_dir is None:
                table.add_row(escape(name), "", escape(", ".join(sources[name])), "", "[red]폴더 없음[/]", "")
                continue
            meta = _read_meta(job_dir)
            title = _dig(meta, "youtube", "title") or _dig(meta, "source", "filename") or ""
            kind = _dig(meta, "source", "kind") or ", ".join(k.split(":", 1)[0] for k in sources.get(name, []))
            done = sum(1 for r in _stage_rows(job_dir) if r["complete"])
            table.add_row(
                escape(name),
                escape(str(title)),
                escape(str(kind)),
                _fmt_duration(_duration_s(meta, job_dir), short=True) if meta is not None else "",
                "완료" if meta is not None else "[yellow]없음[/]",
                str(done),
            )
        c.print(table)
        return

    job_dir = _resolve_job(root, song_key)
    if job_dir is None:
        _fail(f"작업을 찾을 수 없습니다: {song_key} (목록: gtab status)")
    meta = _read_meta(job_dir)
    c.print(f"작업 키: [bold]{escape(job_dir.name)}[/]")
    c.print(f"작업 폴더: {escape(str(job_dir))}", soft_wrap=True)
    if job_dir.name in sources:
        c.print(f"색인: {escape(', '.join(sorted(sources[job_dir.name])))}", soft_wrap=True)
    if meta is None:
        c.print("[yellow]입력: 없음 (input/meta.json 이 없습니다. 입력 처리가 끝나지 않았을 수 있습니다)[/]")
    else:
        c.print("입력: 완료")
        _print_meta(c, meta, job_dir)

    rows = _stage_rows(job_dir)
    if not rows:
        c.print("단계 결과: 없음")
        return
    table = Table(title="단계", title_justify="left")
    for col in ("단계", "키", "상태", "생성(UTC)", "소요"):
        table.add_column(col, overflow="fold")
    for r in rows:
        status_text = escape(r["status"]) if r["complete"] else f"[yellow]{escape(r['status'])}[/]"
        dur = r.get("duration_s")
        table.add_row(
            escape(r["stage"]),
            escape(r["key"]),
            status_text,
            escape(str(r.get("created_utc") or "")),
            f"{dur:.1f} s" if isinstance(dur, (int, float)) else "",
        )
    c.print(table)


@config_app.command("show")
def config_show(set_: SetOpt = None) -> None:
    """지금 적용되는 설정(기본값 -> data/config.toml -> --set)을 TOML로 보여 준다."""
    cfg = _load_cfg(set_)
    user = Path(paths.USER_CONFIG)
    header = [
        f"# 기본값: {config.DEFAULTS_PATH}",
        f"# 사용자 설정: {user}" + ("" if user.is_file() else " (없음)"),
        *(f"# --set {item}" for item in set_ or ()),
    ]
    # Plain print, not rich: TOML's [section] headers would be eaten as rich markup.
    print("\n".join(header))
    print(config.dump_effective(cfg), end="")


@app.command()
def gc() -> None:
    """죽은 프로세스가 남긴 임시 폴더(.tmp-*)를 지운다."""
    store_cls = _backend("gtab.jobs.store", "JobStore")
    store = _call("작업 저장소 열기", store_cls, paths.JOBS)
    removed = _call("임시 폴더 정리", store.gc_temp)
    _out().print(f"임시 폴더 {removed}개를 지웠습니다." if removed else "지울 임시 폴더가 없습니다.")


# --------------------------------------------------------------------------------- M1a/M2 commands
# Each module in gtab/commands exports register(app) and imports only typer/rich/stdlib at module level
# (M1_M2_SPEC 0, 1.7). They are owned by other work packages, so one broken module must never break `gtab`:
# a failed import or register() leaves a placeholder that explains the problem when used.


def _register_optional(target: typer.Typer, module: str) -> bool:
    """``module.register(target)``; on any failure register a placeholder named after the module instead.

    Returns True when the real commands were registered.
    """
    name = module.rsplit(".", 1)[-1]
    n_cmds, n_groups = len(target.registered_commands), len(target.registered_groups)
    try:
        importlib.import_module(module).register(target)
        return True
    except Exception as e:  # import error, missing register(), a bug inside register()
        logger.debug("cannot register %s", module, exc_info=True)
        # Drop anything a half-finished register() added, so the placeholder is the only entry of that name.
        del target.registered_commands[n_cmds:]
        del target.registered_groups[n_groups:]
        error = f"{type(e).__name__}: {e}"

    def _placeholder(ctx: typer.Context) -> None:
        _fail(f"{name} 명령을 불러오지 못했습니다: {error}", show_log=True)

    target.command(
        name,
        help=f"(불러오지 못함) {error}",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        add_help_option=False,
    )(_placeholder)
    return False


def _load_optional(name: str) -> dict[str, Any]:
    """Click commands that ``gtab.commands.<name>`` registers (or its placeholder), by command name."""
    sub = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

    @sub.callback()
    def _group() -> None:
        # A callback forces typer's group mode, so a module that registers one command keeps its name.
        pass

    _register_optional(sub, f"gtab.commands.{name}")
    return dict(typer.main.get_command(sub).commands)


# -------------------------------------------------------------------------------------------- main

_stdio_done = False


def _utf8_stdio() -> None:
    """Switch stdout/stderr to UTF-8, once per process (a cp949 pipe cannot encode everything we print).

    reconfigure() changes the existing wrapper in place. Wrapping sys.stdout.buffer in a second
    TextIOWrapper instead breaks as soon as the old wrapper is collected and closes the shared buffer.
    """
    global _stdio_done
    if _stdio_done:
        return
    _stdio_done = True
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if isinstance(stream, io.TextIOWrapper) and encoding != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                logger.debug("could not reconfigure %r to UTF-8", stream, exc_info=True)


def main(argv: list[str] | None = None) -> None:
    _utf8_stdio()
    paths.apply_process_env()
    paths.ensure_dirs()
    config.ensure_user_config()
    log.setup_logging()
    logger.debug("gtab %s argv=%s", __version__, sys.argv[1:] if argv is None else argv)
    app(args=argv, prog_name="gtab")


if __name__ == "__main__":
    main()
