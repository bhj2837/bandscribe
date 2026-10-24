"""``bandscribe refresh`` (mid-term plan 1-7): bring every song's score up to date after the code or the defaults
changed, without asking anything. Only CPU stages are recomputed: a song whose plan needs a GPU stage again is skipped
and named (``bandscribe run <job>`` does it, with the GPU rules). The user's edits are replayed and the song health is
rewritten, as after ``run``."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer


def refresh_cmd(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="실행하지 않고 곡마다 다시 계산할 단계만 보여 준다.")] = False,
) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from bandscribe import config, paths
    from bandscribe.jobs.store import JobStore
    from bandscribe.pipeline import runner
    from bandscribe.tab import edits as E
    from bandscribe.tab import health as HL
    from bandscribe.tab import viewer

    out, err = Console(highlight=False), Console(stderr=True, highlight=False)
    root = Path(paths.JOBS)
    store = JobStore(root)
    keys = viewer.song_keys(root)
    if not keys:
        out.print(f"악보가 있는 곡이 없습니다. ({escape(str(root))})", soft_wrap=True)
        return
    t = Table(title=f"곡 {len(keys)}개" + (" (--dry-run)" if dry_run else ""), title_justify="left")
    for col in ("제목", "작업 키", "결과"):
        t.add_column(col, overflow="fold")
    n_done = n_skip = n_fail = 0
    for key in keys:
        job_dir = root / key
        try:
            title = viewer.job_files(job_dir)["title"]
        except (viewer.ViewerError, OSError, ValueError):
            title = key
        try:
            cfg = config.load_config(job_hints=job_dir / "hints.toml")
            items = runner.plan(key, cfg, store=store)
        except Exception as e:  # noqa: BLE001 - one song must not stop the others
            n_fail += 1
            t.add_row(escape(title), key, f"[red]계획 실패:[/] {escape(str(e) or type(e).__name__)}")
            continue
        stale = [it.stage for it in items if not it.cached]
        gpu = [it.stage for it in items if not it.cached and it.device == "gpu"]
        if not stale:
            t.add_row(escape(title), key, "최신")
            continue
        if gpu:
            n_skip += 1
            t.add_row(escape(title), key, f"[yellow]건너뜀[/]: GPU 단계도 다시 필요({escape(', '.join(gpu))}) — "
                                          f"bandscribe run {key}")
            continue
        if dry_run:
            t.add_row(escape(title), key, f"다시 계산: {escape(', '.join(stale))}")
            continue
        err.print(f"▶ {escape(title)}: {escape(', '.join(stale))}")
        t0 = time.monotonic()
        try:
            runner.run(key, cfg, store=store)
        except Exception as e:  # noqa: BLE001 - the finished stages stay cached
            n_fail += 1
            t.add_row(escape(title), key, f"[red]실패:[/] {escape(str(e) or type(e).__name__)}")
            continue
        note = f"다시 계산함({len(stale)}단계, {time.monotonic() - t0:.0f} s)"
        try:
            ed = E.reapply(job_dir)
            if ed:
                lost = len(ed["unmatched_edits"]) + len(ed["skipped"])
                note += f", 편집 {ed['applied']}개 다시 적용" + (f"([yellow]{lost}개는 빠짐[/])" if lost else "")
        except Exception as e:  # noqa: BLE001 - the export stays without the edits; run reports it the same way
            note += f", [yellow]편집을 다시 적용하지 못함: {escape(str(e))}[/]"
        try:
            doc = HL.write_health(job_dir)
            if doc["summary"]["warn"]:
                note += f", 경고 {doc['summary']['warn']}"
        except Exception:  # noqa: BLE001 - the viewer rebuilds it on its next request
            pass
        n_done += 1
        t.add_row(escape(title), key, note)
    out.print(t)
    if not dry_run:
        out.print(f"다시 계산 {n_done}곡, 건너뜀 {n_skip}곡, 실패 {n_fail}곡.")
    if n_fail:
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    app.command("refresh", help="코드나 기본 설정이 바뀐 뒤 모든 곡의 악보를 새로 만든다(CPU 단계만, 묻지 않음). "
                                "GPU 단계가 다시 필요한 곡은 건너뛰고 알려 준다.")(refresh_cmd)
