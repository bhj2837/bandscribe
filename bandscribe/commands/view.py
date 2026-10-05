"""``bandscribe view <source>``: the score of a job in the browser, playing along with the recording (M3)."""

from __future__ import annotations

import time
from typing import Annotated

import typer


def view_cmd(
    source: Annotated[str, typer.Argument(help="작업 키(앞부분만 써도 됨), 오디오 파일 또는 YouTube URL", show_default=False)],
    port: Annotated[int, typer.Option("--port", help="포트(0 = 빈 포트 자동)")] = 0,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="브라우저를 연다")] = True,
) -> None:
    from rich.console import Console
    from rich.markup import escape

    from bandscribe import paths
    from bandscribe.jobs.store import JobStore
    from bandscribe.tab import viewer

    out, err = Console(highlight=False), Console(stderr=True, highlight=False)
    store = JobStore(paths.JOBS)
    job_dir = None
    root = paths.JOBS
    matches = [d for d in sorted(root.glob("*")) if d.is_dir() and d.name.startswith(source)] if source else []
    if len(matches) == 1:
        job_dir = matches[0]
    elif len(matches) > 1:
        err.print(f"[bold red]오류:[/] '{escape(source)}'(으)로 시작하는 작업이 여러 개입니다: "
                  f"{', '.join(d.name for d in matches)}")
        raise typer.Exit(1)
    else:
        try:
            from bandscribe import config
            from bandscribe.pipeline import runner

            job_dir = runner.resolve_job(source, store, config.load_config()).job_dir
        except Exception as e:  # noqa: BLE001 - shown in Korean
            err.print(f"[bold red]오류:[/] 작업을 찾지 못했습니다: {escape(str(e))}")
            raise typer.Exit(1) from e
    try:
        httpd, _t = viewer.serve(job_dir, port=port)
    except (viewer.ViewerError, OSError) as e:
        err.print(f"[bold red]오류:[/] {escape(str(e))}")
        raise typer.Exit(1) from e
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    out.print(f"악보 보기: {url}  (이 컴퓨터에서만 열림, 끝내려면 Ctrl+C)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()


def register(app: typer.Typer) -> None:
    app.command("view", help="악보를 브라우저로 본다(원곡에 맞춰 커서가 움직임). 먼저 bandscribe run 으로 악보를 만든다.")(view_cmd)
