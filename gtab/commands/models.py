"""`gtab models list|fetch|verify` (DATA owner, M1_M2_SPEC §5, §9.5 item 7; DESIGN §9.4).

Module level imports only typer/rich/stdlib; gtab.model_fetch is imported inside the commands.
"""

from __future__ import annotations

from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

STATUS_KO = {"absent": "없음", "recorded": "받음", "missing_file": "파일 없음", "size_mismatch": "크기 다름",
             "unrecorded": "파일 있음(기록 안 됨: gtab models fetch 로 기록)"}


def _out() -> Console:
    return Console(highlight=False)


def _fail(msg: str, code: int = 1) -> NoReturn:
    Console(stderr=True, highlight=False).print(f"[bold red]오류:[/] {escape(msg)}", soft_wrap=True)
    raise typer.Exit(code)


def cmd_list() -> None:
    from gtab import model_fetch as MF

    try:
        rows = MF.list_models()
    except MF.ModelFetchError as e:
        _fail(str(e))
    t = Table(title="모델")
    for col in ("이름", "크기", "라이선스", "상태", "위치"):
        t.add_column(col, overflow="fold")
    for r in rows:
        st = STATUS_KO.get(r["status"], r["status"]) + (" (로컬 전용)" if r["local_only"] else "")
        n = r["bytes"]
        size = "-" if not n else (f"{n / 1e6:,.1f} MB" if n >= 1e6 else f"{n / 1e3:,.1f} KB")
        t.add_row(r["name"], size, escape(r["license"]), st, escape(r["path"]))
    _out().print(t)
    _out().print("검증(SHA256 재계산)은 `gtab models verify` 로 합니다.")


def cmd_fetch(name: str, yes: bool) -> None:
    from gtab import model_fetch as MF
    from gtab.datasets import fetch as F

    out = _out()

    def confirm(text: str) -> bool:
        out.print(escape(text), soft_wrap=True)
        return typer.confirm("받을까요?", default=False)

    try:
        entry = MF.fetch(name, yes=yes, confirm=confirm, progress=True)
    except F.FetchCancelled:
        _fail("취소했습니다.")
    except (MF.ModelFetchError, F.FetchError) as e:
        _fail(str(e))
    out.print(f"기록: {entry.name} {entry.bytes / 1e6:,.1f} MB sha256 {entry.sha256[:16]}… -> models.lock.json")


def cmd_verify(name: str | None) -> None:
    from gtab import model_fetch as MF

    try:
        res = MF.verify(name)
    except MF.ModelFetchError as e:
        _fail(str(e), code=2)
    if not res:
        _out().print("기록된 모델이 없습니다.")
        return
    bad = 0
    for n, ok in res:
        _out().print(f"[green]OK[/] {n}" if ok else f"[red]불일치/없음[/] {n}")
        bad += not ok
    if bad:
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    models_app = typer.Typer(help="모델 파일 관리: 목록, 받기, 검증 (models.lock.json).", no_args_is_help=True)

    @models_app.command("list")
    def _list() -> None:
        """모델 목록: 이름, 크기, 라이선스, 상태(없음/받음/검증됨), 위치"""
        cmd_list()

    @models_app.command("fetch")
    def _fetch(
        name: Annotated[str, typer.Argument(metavar="이름", help="예: beat_this.final0")],
        yes: Annotated[bool, typer.Option("--yes", "-y", help="확인 없이 받는다")] = False,
    ) -> None:
        """URL·크기·라이선스·저장 위치를 보여 주고 y/N 을 받은 뒤 받는다."""
        cmd_fetch(name, yes)

    @models_app.command("verify")
    def _verify(name: Annotated[str | None, typer.Argument(metavar="[이름]", help="생략하면 전부")] = None) -> None:
        """모델 파일의 SHA256 을 다시 계산해 models.lock.json 과 비교한다."""
        cmd_verify(name)

    app.add_typer(models_app, name="models")
