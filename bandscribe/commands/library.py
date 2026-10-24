"""``bandscribe library`` (mid-term plan 1-7): every valid song's Guitar Pro files under its title (``bandscribe.library``)."""

from __future__ import annotations

import typer


def library_cmd() -> None:
    from rich.console import Console
    from rich.markup import escape

    from bandscribe import library as L

    out = Console(highlight=False)
    try:
        res = L.build(progress=lambda m: out.print(f"  {escape(m)}"))
    except OSError as e:
        Console(stderr=True, highlight=False).print(f"[bold red]오류:[/] 서고를 만들지 못했습니다: {escape(str(e))}")
        raise typer.Exit(1) from e
    out.print(f"서고: {escape(res['dir'])}  ({len(res['songs'])}곡; 목록은 README.md)", soft_wrap=True)
    for s in res["skipped"]:
        out.print(f"  넣지 않음: {escape(str(s.get('title') or s['key']))} — {escape(s['why'])}", soft_wrap=True)


def register(app: typer.Typer) -> None:
    app.command("library", help="악보가 있는 곡을 data/library/<제목>/ 에 모은다(.gp · .gp5 · 코드표 · README: 템포·튜닝·"
                                "점검). 기타가 빠진 음원은 넣지 않는다. 바뀐 것만 복사하고 지우지 않는다.")(library_cmd)
