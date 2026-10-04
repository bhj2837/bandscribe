"""``bandscribe run`` (M1_M2_SPEC 5; GRID). Imports only typer/rich/stdlib at module level (CLI start-up stays fast);
the pipeline (``bandscribe.pipeline.runner``) is imported inside the command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

RUN_HELP = ("파이프라인을 실행한다. 기본은 --until notes: 6 스템, nonvox, leftover, 편성(instrumentation.json), "
            "guitar_all.mid, bass_raw.mid, 연습용 반주(기타 뺀 믹스, 베이스 뺀 믹스). 끝난 단계는 캐시를 쓴다.")

_DEVICE_KO = {"gpu": "GPU", "cpu": "CPU"}


def _split(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for v in values or []:
        out += [x.strip() for x in v.split(",") if x.strip()]
    return out


def _dur(v: Any) -> str:
    if not isinstance(v, (int, float)):
        return "-"
    m, s = divmod(float(v), 60)
    return f"{int(m)}:{s:04.1f}" if m else f"{s:.1f} s"


def run_cmd(
    source: Annotated[str, typer.Argument(help="오디오 파일, YouTube URL 또는 작업 키(앞부분만 써도 됨)",
                                          show_default=False)],
    until: Annotated[str, typer.Option("--until", help="이 단계까지만 실행한다")] = "notes",
    from_: Annotated[str | None, typer.Option("--from", help="이 단계부터 다시 계산한다(뒤 단계도 모두)")] = None,
    force: Annotated[list[str] | None, typer.Option(
        "--force", help="캐시가 있어도 다시 계산할 단계(쉼표로 여러 개, 뒤 단계도 모두)")] = None,
    profile: Annotated[str | None, typer.Option(
        "--profile", help="fast | quality(기본) | eval. eval 은 평가용: 믹스 전체 전사(B0)와 piano+other 전체 전사를 "
                          "추가 결과로 더해 훨씬 느리다(편성·기타 전사는 quality 와 같다).")] = None,
    instruments: Annotated[str | None, typer.Option(
        "--instruments", help='편성 힌트: auto 또는 "guitar,keys" 처럼 악기군 목록')] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="실행하지 않고 단계별 캐시 여부만 보여 준다.")] = False,
    set_: Annotated[list[str] | None, typer.Option(
        "--set", "-s", metavar="KEY=VALUE", help="설정 덮어쓰기(여러 번 가능). 예: --set grid.refine_window_ms=30")] = None,
) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from bandscribe import config, paths

    out, err = Console(highlight=False), Console(stderr=True, highlight=False)

    def fail(msg: str, code: int = 1) -> None:
        err.print(f"[bold red]오류:[/] {escape(msg)}", soft_wrap=True)
        raise typer.Exit(code)

    overrides = list(set_ or [])
    if profile is not None:
        if profile not in config.RUN_PROFILES:
            fail(f"--profile 은 fast, quality, eval 중 하나입니다: {profile}", 2)
        overrides.append(f"run.profile={profile}")
    if instruments is not None:
        # a JSON string literal is a valid TOML basic string, so quotes/backslashes in the value cannot break
        # the override; the family names are checked by the instr stage's params (Korean ConfigError, exit 2)
        overrides.append(f"hints.instruments={json.dumps(instruments.strip(), ensure_ascii=False)}")
    try:
        cfg0 = config.load_config(overrides=overrides)
    except config.ConfigError as e:
        fail(str(e), 2)

    try:
        from bandscribe.jobs.store import JobStore
        from bandscribe.pipeline import graph, runner
    except Exception as e:  # a broken backend must not become a traceback
        fail(f"파이프라인을 불러오지 못했습니다: {type(e).__name__}: {e}")

    forced = set(_split(force))
    if from_:
        forced.add(from_.strip())
    try:
        graph.check_name(until, "--until 단계")
        for f in forced:
            graph.check_name(f, "--force/--from 단계")
    except graph.UnknownStage as e:
        fail(str(e), 2)

    store = JobStore(paths.JOBS)
    err.print(f"입력 확인 중: {escape(source)}", soft_wrap=True)
    try:
        job = runner.resolve_job(source, store, cfg0)
    except Exception as e:
        fail(f"입력 처리 실패: {e}")
    song_key = job.song_key
    hints = Path(job.job_dir) / "hints.toml"
    try:
        cfg = config.load_config(job_hints=hints, overrides=overrides)
    except config.ConfigError as e:
        fail(str(e), 2)
    out.print(f"작업: [bold]{escape(song_key)}[/]")

    try:
        items = runner.plan(song_key, cfg, until=until, force=forced, store=store, missing_ok=dry_run)
    except config.ConfigError as e:  # e.g. an unknown family in --instruments (checked by the stage params)
        fail(str(e), 2)
    except runner.ModelsMissing as e:
        fail(str(e))
    except (graph.StageNotImplemented, runner.JobInputMissing) as e:
        fail(str(e))
    except Exception as e:
        fail(f"실행 계획을 만들지 못했습니다: {str(e) or type(e).__name__}")

    if dry_run:
        t = Table(title="실행 계획 (--dry-run)", title_justify="left")
        for col in ("단계", "키", "캐시/실행", "장치"):
            t.add_column(col)
        for it in items:
            if it.missing_models:
                state = "[red]모델 없음[/] " + escape(", ".join(it.missing_models))
            elif not it.key:
                state = "[yellow]앞 단계 모델 없음[/]"
            elif it.cached:
                state = "캐시"
            else:
                state = "실행(강제)" if it.forced else "실행"
            t.add_row(it.stage, it.key12 or "-", state, _DEVICE_KO.get(it.device, it.device))
        out.print(t)
        return

    notices: list[str] = []
    warned_live: set[str] = set()  # stages whose "warning" events were shown during this run

    def progress(event: str, info: dict) -> None:
        stage = info.get("stage", "")
        if event == "stage_start":
            err.print(f"▶ {escape(stage)} 실행 중... ({_DEVICE_KO.get(info.get('device'), '')})")
        elif event == "stage_done":
            err.print(f"  {escape(stage)} 완료 ({_dur(info.get('duration_s'))})")
        elif event in ("vram_wait", "stage_degraded_cache", "model_missing", "warning"):
            msg = str(info.get("message") or "")
            if event == "warning" and stage:
                warned_live.add(str(stage))
                msg = f"{stage}: {msg}"
            err.print(f"[yellow]경고:[/] {escape(msg)}", soft_wrap=True)
            notices.append(msg)

    try:
        results = runner.run(song_key, cfg, until=until, force=forced, progress=progress, store=store)
    except runner.ModelsMissing as e:
        fail(str(e))
    except KeyboardInterrupt:
        fail("중단했습니다. 끝난 단계는 캐시에 남아 있어 다시 실행하면 이어서 진행합니다.", 130)
    except Exception as e:
        from bandscribe.pipeline.publish import PublishError

        if isinstance(e, PublishError):
            fail(str(e))
        fail(f"실행 실패: {str(e) or type(e).__name__}\n(끝난 단계는 캐시에 남아 있습니다. 로그: {paths.LOGS})")

    rows = runner.stage_summary(results, items)
    t = Table(title=f"단계 ({len(rows)}개)", title_justify="left")
    for col in ("단계", "키", "캐시/실행", "소요", "장치"):
        t.add_column(col)
    for r in rows:
        state = "캐시" if r["cached"] else "실행"
        dev = _DEVICE_KO.get(r["device"] or "", r["device"] or "-")
        if r["rung"]:
            dev += f" ({escape(str(r['rung']))}{', 낮은 칸' if r['degraded'] else ''})"
        t.add_row(escape(r["stage"]), r["key12"], state, _dur(r["duration_s"]), dev)
    out.print(t)

    export = Path(job.job_dir) / "export"
    if "notes" in results and export.is_dir():
        out.print("결과 파일:")
        for p in sorted(x for x in export.rglob("*") if x.is_file()):
            out.print(f"  {escape(str(p))}", soft_wrap=True)
    else:
        out.print("단계 결과 폴더:")
        for stage, d in results.items():
            out.print(f"  {escape(stage)}: {escape(str(d))}", soft_wrap=True)
    for w in runner.run_warnings(results, exclude_stages=warned_live):
        out.print(f"[yellow]경고:[/] {escape(w)}", soft_wrap=True)


def register(app: typer.Typer) -> None:
    app.command("run", help=RUN_HELP)(run_cmd)
