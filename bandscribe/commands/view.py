"""``bandscribe view [<song>]``: the scores in the browser, playing along with the recording (M3). One server shows
every song (mid-term plan 1-7): without a song it opens the home page (the songs by title), with one that song."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer


def find_jobs(source: str, root: Path) -> list[Path]:
    """Jobs with a score whose key starts with ``source``, else whose title is ``source`` (case and spaces ignored),
    else whose title contains it."""
    from bandscribe.tab import viewer

    dirs = [d for d in sorted(root.glob("*")) if d.is_dir()]
    by_key = [d for d in dirs if d.name.startswith(source)]
    if by_key:
        return by_key
    titles = {}
    for d in dirs:
        try:
            titles[d] = viewer.job_files(d)["title"]
        except (viewer.ViewerError, OSError, ValueError):
            continue
    norm = lambda t: "".join(str(t).split()).casefold()  # noqa: E731
    want = norm(source)
    exact = [d for d, t in titles.items() if norm(t) == want]
    return exact or [d for d, t in titles.items() if want and want in norm(t)]


def view_cmd(
    source: Annotated[str | None, typer.Argument(
        help="곡: 작업 키(앞부분만 써도 됨), 제목, 오디오 파일 또는 YouTube URL. 안 주면 곡 목록이 열린다", show_default=False)] = None,
    port: Annotated[int, typer.Option("--port", help="포트(0 = 빈 포트 자동)")] = 0,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="브라우저를 연다")] = True,
    edit: Annotated[bool, typer.Option("--edit/--read-only", help="키보드 편집(기록은 작업 폴더 edits.jsonl)")] = True,
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
    matches = find_jobs(source, root) if source else []
    if len(matches) == 1:
        job_dir = matches[0]
    elif len(matches) > 1:
        err.print(f"[bold red]오류:[/] '{escape(source)}' 에 맞는 곡이 여러 개입니다: "
                  f"{', '.join(d.name for d in matches)} (작업 키로 고르거나, 곡 목록에서 고르세요: bandscribe view)")
        raise typer.Exit(1)
    elif source:
        try:
            from bandscribe import config
            from bandscribe.pipeline import runner

            job_dir = runner.resolve_job(source, store, config.load_config()).job_dir
        except Exception as e:  # noqa: BLE001 - shown in Korean
            err.print(f"[bold red]오류:[/] 작업을 찾지 못했습니다: {escape(str(e))}")
            raise typer.Exit(1) from e
    try:
        httpd, _t = viewer.serve(job_dir, port=port, edit=edit, jobs_root=root)
    except (viewer.ViewerError, OSError) as e:
        err.print(f"[bold red]오류:[/] {escape(str(e))}")
        raise typer.Exit(1) from e
    home = f"http://127.0.0.1:{httpd.server_address[1]}/home"
    url = f"http://127.0.0.1:{httpd.server_address[1]}/s/{job_dir.name}/" if job_dir is not None else home
    out.print(f"악보 보기: {url}  (이 컴퓨터에서만 열림, 끝내려면 Ctrl+C)")
    if job_dir is not None:
        out.print(f"곡 목록: {home}")
    session = getattr(httpd, "session", None)
    if session is not None:
        n = session.info()["edits"]
        out.print(f"편집: 음을 누르고 키보드로 고칩니다(키 목록은 페이지의 ? 버튼). 기록: "
                  f"{escape(str(job_dir / 'edits.jsonl'))}" + (f" (지금 {n}개 적용)" if n else ""), soft_wrap=True)
    elif edit and getattr(httpd, "readonly", ""):
        err.print(f"[yellow]경고:[/] 편집은 꺼졌습니다: {escape(str(httpd.readonly))}", soft_wrap=True)
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
        for s in httpd.sessions():  # every song opened on this server, not only the first
            done = s.close(30.0)
            if not done or s.export_error:
                why = s.export_error or "시간 안에 끝나지 않았습니다"
                err.print(f"[yellow]경고:[/] 마지막 편집을 export 에 쓰지 못했습니다({escape(why)}). 편집 기록은 남아 있어 "
                          "다시 view 를 열거나 bandscribe run 을 하면 다시 씁니다.", soft_wrap=True)


def register(app: typer.Typer) -> None:
    app.command("view", help="악보를 브라우저로 보고 키보드로 고친다(원곡에 맞춰 커서가 움직임). 곡을 안 주면 곡 목록이 "
                             "열리고, 서버 하나로 모든 곡을 본다. 먼저 bandscribe run 으로 악보를 만든다.")(view_cmd)
