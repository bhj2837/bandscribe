"""``bandscribe backup [<dir>]`` (mid-term plan 1-1): mirror what cannot be made again (``bandscribe.backup``)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def backup_cmd(
    dest: Annotated[str | None, typer.Argument(
        help="백업 폴더(다른 디스크 권장). 안 주면 data/config.toml 의 [backup] dir", show_default=False)] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="복사하지 않고 무엇을 넣을지만 센다.")] = False,
) -> None:
    from rich.console import Console
    from rich.markup import escape

    from bandscribe import backup as B
    from bandscribe import config

    out, err = Console(highlight=False), Console(stderr=True, highlight=False)
    try:
        cfg = config.load_config()
    except config.ConfigError as e:
        err.print(f"[bold red]오류:[/] {escape(str(e))}", soft_wrap=True)
        raise typer.Exit(2) from e
    target = dest or cfg.backup.dir
    if not target:
        err.print("[bold red]오류:[/] 백업 폴더를 주세요: bandscribe backup <폴더> (또는 data/config.toml 의 [backup] dir)")
        raise typer.Exit(2)
    files, missing = B.plan_files(cfg)
    for m in missing:
        err.print(f"[yellow]경고:[/] 없는 폴더는 건너뜁니다: {escape(m)}", soft_wrap=True)
    if not cfg.backup.extra:
        err.print("[yellow]알림:[/] [backup] extra 가 비어 있어 원본 음원·악보 폴더는 들어가지 않습니다(data/config.toml).",
                  soft_wrap=True)
    size = sum(p.stat().st_size for _, p in files)
    if dry_run:
        out.print(f"넣을 파일 {len(files):,}개, {size / 1e9:.2f} GB → {escape(str(Path(target)))}", soft_wrap=True)
        return
    out.print(f"백업 중: 파일 {len(files):,}개, {size / 1e9:.2f} GB → {escape(str(Path(target)))}", soft_wrap=True)
    try:
        rec = B.backup(Path(target), cfg, progress=lambda m: err.print(f"  {escape(m)}"))
    except (B.BackupError, OSError) as e:
        err.print(f"[bold red]오류:[/] 백업 실패: {escape(str(e))}", soft_wrap=True)
        raise typer.Exit(1) from e
    out.print(f"끝: {rec['files']:,}개 중 {rec['copied']:,}개 복사({rec['copied_bytes'] / 1e9:.2f} GB, {rec['seconds']:.0f} s). "
              f"목록: {escape(str(Path(target) / B.MANIFEST))}", soft_wrap=True)
    if not rec["sw_sha256"]:
        err.print(f"[yellow]경고:[/] SW 체크포인트({B.SW_NAME})가 백업에 없습니다. 다시 받을 곳이 없는 파일입니다.",
                  soft_wrap=True)


def register(app: typer.Typer) -> None:
    app.command("backup", help="다시 만들 수 없는 것(모델, 정답 악보 추출, 곡별 힌트·편집, 설정, [backup] extra 의 원본 음원·악보 "
                               "폴더)을 다른 폴더로 증분 복사하고 sha256 으로 확인한다.")(backup_cmd)
