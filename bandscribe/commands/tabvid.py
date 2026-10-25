"""``bandscribe tabvid`` (docs/TAB_VIDEO.md): read the tab a YouTube tab video shows into silver sheet parts, and score
one part against a sheet. Videos, frames and results stay under ``data/eval/tabvid/`` (a transcriber's work)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

SetOpt = Annotated[list[str] | None, typer.Option("--set", "-s", metavar="KEY=VALUE", help="설정 덮어쓰기(여러 번 가능)")]


def _fail(msg: str, code: int = 1) -> None:
    Console(stderr=True, highlight=False).print(f"[bold red]오류:[/] {escape(msg)}", soft_wrap=True)
    raise typer.Exit(code)


def _frames(source: str, fps: float, overrides: list[str] | None) -> tuple[str, list[Path]]:
    """(name, frame paths) of a frame folder, a video file, or a YouTube URL / id (downloaded once)."""
    from bandscribe.tabvid import fetch as F
    from bandscribe.tabvid import video as V

    p = Path(source)
    if p.is_dir():
        frames = sorted(p.glob("f*.png"))
        if not frames:
            _fail(f"프레임(f*.png)이 없는 폴더입니다: {p}")
        return p.name, frames
    if p.is_file():
        vid, video = p.stem, p
    else:
        vid = F.video_id(source)
        if vid is None:
            _fail(f"영상 파일, 프레임 폴더, YouTube 주소나 id 가 아닙니다: {source}", 2)
        from bandscribe import config
        from bandscribe.ingest.decode import IngestError

        try:
            cfg = config.load_config(overrides=overrides or ())
            video = F.download(vid, cfg)
        except config.ConfigError as e:
            _fail(str(e), 2)
        except IngestError as e:
            _fail(str(e))
    try:
        return vid, V.extract_frames(video, F.root() / "frames" / vid, fps=fps)
    except (OSError, ValueError) as e:
        _fail(f"프레임을 뽑지 못했습니다(ffmpeg): {e}")
    raise AssertionError  # _fail raised


def read_cmd(
    source: Annotated[str, typer.Argument(help="YouTube 주소나 id, 영상 파일, 또는 프레임 폴더(f*.png)")],
    strings: Annotated[list[int] | None, typer.Option("--strings", "-n", help="탭 줄 수(여러 번 가능): 기타 6, 베이스 4·5")] = None,
    fps: Annotated[float, typer.Option("--fps", help="초당 프레임(처음 뽑을 때만)")] = 2.0,
    set_: SetOpt = None,
) -> None:
    """탭 영상을 읽어 파트(시스템 안 순서: 기타1, 기타2 …)마다 마디·열·(줄, 프렛)으로 낸다."""
    from bandscribe import atomic
    from bandscribe.tabvid import fetch as F
    from bandscribe.tabvid import sheet as SH
    from bandscribe.tabvid import video as V

    name, frames = _frames(source, fps, set_)
    out = Console(highlight=False)
    dest = F.root() / "read"
    for n in strings or [6]:
        doc = V.read_video(frames, n)
        doc["source"] = source
        atomic.write_json(dest / f"{name}_n{n}.json", doc)
        t = Table(title=f"{name}: {n}줄 탭 {len(doc['tracks'])}트랙 ({len(frames)}프레임)", title_justify="left")
        for col in ("트랙", "마디", "읽은 마디", "음", "프레임 일치", "의심 마디(<0.8)", "파트 파일"):
            t.add_column(col, justify="left" if col in ("파트 파일",) else "right")
        for k, tr in enumerate(doc["tracks"], 1):
            bars = tr["bars"]
            part = dest / f"{name}_n{n}_t{k}.json"
            if bars:
                atomic.write_json(part, SH.to_sheet_part(bars, n_strings=n, source=f"tabvid:{name}:n{n}:t{k}"))
            low = sum(1 for b in bars if b["agree"] < 0.8)
            t.add_row(str(k), f"{bars[0]['number']}–{bars[-1]['number']}" if bars else "–",
                      str(len(bars)), str(tr["notes"]), f"{tr['agree']:.3f}" if tr["agree"] is not None else "–",
                      f"{low} ({100 * low / max(1, len(bars)):.0f} %)", escape(part.name) if bars else "–")
        out.print(t)
    out.print(f"결과: {escape(str(dest))}", soft_wrap=True)


def compare_cmd(
    part: Annotated[Path, typer.Argument(help="tabvid read 가 낸 파트 JSON")],
    sheet: Annotated[Path, typer.Argument(help="비교할 악보 JSON(같은 형식)")],
    tuning: Annotated[str, typer.Option("--tuning", help="guitar | bass | bass-dropd | bass5, 또는 낮은 줄부터 MIDI 번호 "
                                                       "(예 26,33,38,43). 기본: 줄 수로 고른다")] = "",
    agree: Annotated[float, typer.Option("--agree", help="프레임 일치가 이보다 낮은 마디는 빼고 잰다(의심 마디)")] = 0.0,
) -> None:
    """파트 JSON 을 악보 JSON 과 마디별 음높이 묶음으로 비교한다(가장 잘 맞는 마디 오프셋)."""
    from bandscribe.tabvid import sheet as SH

    try:
        a = json.loads(part.read_text(encoding="utf-8"))
        b = json.loads(sheet.read_text(encoding="utf-8"))
        tun = SH.tuning(tuning, int(a.get("n_strings", 6)))
    except (OSError, ValueError) as e:
        _fail(str(e), 2)
    kept = [x for x in a["bars"] if x.get("read") and (x.get("agree") or 0.0) >= agree]
    res = SH.compare({"bars": kept}, b, tuning=tun)
    Console(highlight=False).print(
        f"마디 {res['bars']}개(일치 {agree} 이상, 읽은 마디만) · 오프셋 {res['offset']:+d} · F1 [bold]{res['F1']:.3f}[/] "
        f"(맞음 {res['tp']} · 더 읽음 {res['fp']} · 놓침 {res['fn']}) · 같은 마디 {res['bars_same']}")


def score_cmd(
    read: Annotated[Path, typer.Argument(help="tabvid read 가 낸 <id>_n<줄>.json")],
    job: Annotated[str, typer.Option("--job", help="채점할 작업 키(data/jobs/<키>)")],
    track: Annotated[int, typer.Option("--track", help="트랙 번호(1부터)")] = 1,
    agree: Annotated[float, typer.Option("--agree", help="프레임 일치가 이보다 낮은 마디는 뺀다(의심 마디)")] = 0.8,
    tuning: Annotated[str, typer.Option("--tuning", help="탭의 튜닝(compare 와 같음). 기본: 줄 수로 고른다")] = "",
) -> None:
    """작업의 기타(합친 것)를 영상 탭 파트와 마디별 음높이 묶음으로 채점한다(eval sheets 와 같은 방식, 실버 정답).
    영상 탭은 한 연주자의 판인 경우가 많아 재현율(탭 음을 우리가 잡은 몫)이 더 중요하다."""
    from bandscribe import paths
    from bandscribe.eval import sheets as E
    from bandscribe.tabvid import sheet as SH

    try:
        doc = json.loads(read.read_text(encoding="utf-8"))
        tr = doc["tracks"][track - 1]
        n = int(doc.get("strings", 6))
        tun = SH.tuning(tuning, n)
        o = E.ours(Path(paths.JOBS) / job)
        keep = {b["number"] for b in tr["bars"] if b["agree"] >= agree}
        est = o.bass if n in (4, 5) else o.guitar
        res = SH.score(SH.to_sheet_part(tr["bars"], n_strings=n), o.tempo, est, tuning=tun, bars=keep)
    except (OSError, ValueError, KeyError, IndexError, E.SheetEvalError) as e:
        _fail(str(e), 2)
    out = Console(highlight=False)
    d = res["drifting"]
    out.print(f"{escape(read.name)} 트랙 {track} ↔ {escape(job)} {'베이스' if n in (4, 5) else '기타(합친 것)'}: "
              f"마디 {res['bars']}개(일치 {agree} 이상) · 오프셋 {res['offset']:+d}박(떠도는 오프셋 {len(res['changes'])}번 바뀜)")
    out.print(f"재현율 [bold]{d['R']:.1f}[/] · 정밀도 {d['P']:.1f} · F1 {d['F1']:.1f} "
              f"(탭 음 {d['tp'] + d['fn']} 중 {d['tp']} 잡음, 우리만 낸 음 {d['fp']})")
    t = Table(title="구간별", title_justify="left")
    for col in ("마디", "탭 음", "재현율", "정밀도"):
        t.add_column(col, justify="right")
    for b in res["blocks"]:
        t.add_row(f"{b['bar']}–{b['last']}", str(b["tab_notes"]), f"{b['R']:.1f}", f"{b['P']:.1f}")
    out.print(t)


def register(app: typer.Typer) -> None:
    tv = typer.Typer(help="탭 영상에서 실버 정답 탭 뽑기(docs/TAB_VIDEO.md). 영상·프레임·결과는 data/eval/tabvid/ 에만 둔다.",
                     no_args_is_help=True)
    app.add_typer(tv, name="tabvid")
    tv.command("read")(read_cmd)
    tv.command("compare")(compare_cmd)
    tv.command("score")(score_cmd)
