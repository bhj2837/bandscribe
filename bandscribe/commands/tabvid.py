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
        doc = V.read_video(frames, n, fps=fps)
        doc["source"] = source
        atomic.write_json(dest / f"{name}_n{n}.json", doc)
        t = Table(title=f"{name}: {n}줄 탭 {len(doc['tracks'])}트랙 ({len(frames)}프레임)", title_justify="left")
        for col in ("트랙", "마디", "읽은 마디", "음", "프레임 일치", "의심 마디(<0.8)", "판정", "파트 파일"):
            t.add_column(col, justify="left" if col in ("판정", "파트 파일") else "right")
        for k, tr in enumerate(doc["tracks"], 1):
            bars = tr["bars"]
            part = dest / f"{name}_n{n}_t{k}.json"
            if bars:
                atomic.write_json(part, SH.to_sheet_part(bars, n_strings=n, source=f"tabvid:{name}:n{n}:t{k}"))
            low = sum(1 for b in bars if b["agree"] < 0.8)
            t.add_row(str(k), f"{bars[0]['number']}–{bars[-1]['number']}" if bars else "–",
                      str(len(bars)), str(tr["notes"]), f"{tr['agree']:.3f}" if tr["agree"] is not None else "–",
                      f"{low} ({100 * low / max(1, len(bars)):.0f} %)", _quality_text(tr.get("quality")),
                      escape(part.name) if bars else "–")
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


QUALITY = {"usable": "[green]쓸 수 있음[/]", "doubtful": "[yellow]의심[/]", "unusable": "[bold red]못 씀[/]"}


def _quality_text(q: dict | None) -> str:
    if not q:
        return "–"
    return QUALITY[q["verdict"]] + (f" ({escape('; '.join(q['reasons']))})" if q["reasons"] else "")


def _vid(read: Path) -> str:
    return read.name.rsplit("_n", 1)[0]


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _alignment(read: Path, doc: dict, track: int, job: str, overrides: list[str] | None) -> dict:
    """The track's alignment with the job's song: from ``<read>_t<k>.align.json`` when made for this job, else made
    now (the video's sound downloaded once into audio/) and saved there."""
    from bandscribe import atomic, config, paths
    from bandscribe.tabvid import align as A
    from bandscribe.tabvid import fetch as F

    dest = read.with_name(f"{read.stem}_t{track}.align.json")
    if dest.is_file():
        have = json.loads(dest.read_text(encoding="utf-8"))
        if have.get("job") == job:
            return have
    vid = _vid(read)
    audio = F.find_audio(vid)
    if audio is None:
        audio = F.download_audio(vid, config.load_config(overrides=overrides or ()))
    span = doc["tracks"][track - 1].get("span_s")
    res = A.align(A.load(audio), A.load(Path(paths.JOBS) / job / "input" / "mix_44k_f32.wav"),
                  (span[0] - 5, span[1] + 5) if span else None)
    kind, why = A.verdict(res)
    res.update(job=job, verdict=kind, reasons=why, title=F.title(vid), title_flags=A.title_flags(F.title(vid)))
    atomic.write_json(dest, res)
    return res


def _print_alignment(out: Console, res: dict) -> None:
    name = {"same": "같은 판", "cut": "축약판(빠진 구간 있음)", "other": "다른 곡이거나 다른 판"}[res["verdict"]]
    out.print(f"소리 맞추기: [bold]{name}[/] · 유사도 {res['similarity']} · 빠르기 비 {res['rate']} · "
              f"영상 {_mmss(res['span'][0])}–{_mmss(res['span'][1])} ↔ 원곡 {_mmss(res['song_s'])}")
    for a, b in res["cuts"]:
        out.print(f"  원곡 {_mmss(a)}–{_mmss(b)} 가 영상에 없다")
    for a, b in res["extra"]:
        out.print(f"  영상 {_mmss(a)}–{_mmss(b)} 는 원곡과 맞지 않는다")
    for r in res["reasons"] if res["verdict"] == "other" else []:
        out.print(f"  [bold red]{escape(r)}[/]")
    if res.get("title_flags"):
        out.print(f"  [bold yellow]제목에 다른 판 표시: {', '.join(res['title_flags'])}[/] — 소리로는 못 가리니 사람이 확인"
                  f" ({escape(res.get('title') or '')})")


def align_cmd(
    read: Annotated[Path, typer.Argument(help="tabvid read 가 낸 <id>_n<줄>.json")],
    job: Annotated[str, typer.Option("--job", help="맞출 곡의 작업 키(data/jobs/<키>)")],
    track: Annotated[int, typer.Option("--track", help="트랙 번호(1부터): 그 트랙이 화면에 나오는 구간만 맞춘다")] = 1,
    set_: SetOpt = None,
) -> None:
    """영상의 소리를 작업의 원곡과 맞춰 같은 판 · 축약판 · 다른 곡을 가리고, 영상 시간 → 원곡 시간을 남긴다."""
    try:
        doc = json.loads(read.read_text(encoding="utf-8"))
        res = _alignment(read, doc, track, job, set_)
    except Exception as e:  # noqa: BLE001 - download, decode or alignment: one message
        _fail(f"소리를 맞추지 못했습니다: {e}")
    _print_alignment(Console(highlight=False), res)


def score_cmd(
    read: Annotated[Path, typer.Argument(help="tabvid read 가 낸 <id>_n<줄>.json")],
    job: Annotated[str, typer.Option("--job", help="채점할 작업 키(data/jobs/<키>)")],
    track: Annotated[int, typer.Option("--track", help="트랙 번호(1부터)")] = 1,
    agree: Annotated[float, typer.Option("--agree", help="프레임 일치가 이보다 낮은 마디는 뺀다(의심 마디)")] = 0.8,
    tuning: Annotated[str, typer.Option("--tuning", help="탭의 튜닝(compare 와 같음). 기본: 줄 수로 고른다")] = "",
    how: Annotated[str, typer.Option("--how", help="auto(기본: 둘 다 해 보고 더 잘 맞는 쪽) | time(소리 맞추기로 마디를 원곡 "
                                                    "시간에 놓기) | beats(박 오프셋 찾기)")] = "auto",
    scale: Annotated[float, typer.Option("--scale", help="beats: 탭의 박 / 우리 박")] = 1.0,
    search: Annotated[int, typer.Option("--search", help="beats: 전역 오프셋을 찾는 폭(박)")] = 16,
    set_: SetOpt = None,
) -> None:
    """작업의 기타(합친 것, 4 · 5줄 탭이면 베이스)를 영상 탭 파트와 마디별 음높이 묶음으로 채점한다(eval sheets 와 같은
    묶음, 실버 정답). time 은 영상 소리를 원곡과 맞춰 마디를 원곡 시간에 놓는다 — 축약판 · 두 번 나오는 곡 · 다른 박 표기 ·
    번호가 불확실한 판독에 강하다. beats 는 박 오프셋을 찾는다 — 같은 판을 번호대로 잘 읽은 영상에서 조금 더 정확하다.
    영상 탭은 한 연주자의 판인 경우가 많아 재현율(탭 음을 우리가 잡은 몫)이 더 중요하다."""
    from bandscribe import paths
    from bandscribe.eval import sheets as E
    from bandscribe.tabvid import sheet as SH

    out = Console(highlight=False)
    al = None
    try:
        doc = json.loads(read.read_text(encoding="utf-8"))
        tr = doc["tracks"][track - 1]
        n = int(doc.get("strings", 6))
        tun = SH.tuning(tuning, n)
        o = E.ours(Path(paths.JOBS) / job)
        keep = {b["number"] for b in tr["bars"] if b["agree"] >= agree}
        est = o.bass if n in (4, 5) else o.guitar
        part = SH.to_sheet_part(tr["bars"], n_strings=n)
        if how not in ("auto", "time", "beats"):
            _fail(f"--how 는 auto | time | beats 입니다 (받은 값: {how})", 2)
        tried = {}
        if how in ("auto", "beats"):
            tried["beats"] = SH.score(part, o.tempo, est, tuning=tun, bars=keep, scale=scale, search=search)
        times = {b["number"]: b.get("t") for b in tr["bars"]}
        if how in ("auto", "time") and any(t is not None for t in times.values()):
            al = _alignment(read, doc, track, job, set_)
            tried["time"] = SH.score_time(part, times, al["path"], o.tempo, est, tuning=tun, bars=keep)
        if not tried:
            _fail("이 판독에는 마디 시간이 없습니다 — tabvid read 를 다시 돌리거나 --how beats 를 쓰세요", 2)
        used = max(tried, key=lambda k: tried[k]["drifting"]["F1"])
        res = tried[used]
    except (OSError, ValueError, KeyError, IndexError, E.SheetEvalError) as e:
        _fail(str(e), 2)
    if al is not None:
        _print_alignment(out, al)
    q = tr.get("quality")
    if q and q["verdict"] != "usable":
        out.print(f"판독 판정: {_quality_text(q)} — 이 점수는 참고로만 보세요")
    d = res["drifting"]
    other = "".join(f"; {k} 는 F1 {v['drifting']['F1']:.1f}" for k, v in tried.items() if k != used)
    way = (f"시간으로 놓음(탭 1마디 = 우리 {res['beats_per_bar']}박, 쪽이 소리보다 {res['lead_s']:+.2f} s{other})"
           if used == "time" else f"박 오프셋 {res['offset']:+d}박(떠도는 오프셋 {len(res['changes'])}번 바뀜{other})")
    out.print(f"{escape(read.name)} 트랙 {track} ↔ {escape(job)} {'베이스' if n in (4, 5) else '기타(합친 것)'}: "
              f"마디 {res['bars']}개(일치 {agree} 이상) · {way}")
    out.print(f"재현율 [bold]{d['R']:.1f}[/] · 정밀도 {d['P']:.1f} · F1 {d['F1']:.1f} "
              f"(탭 음 {d['tp'] + d['fn']} 중 {d['tp']} 잡음, 우리만 낸 음 {d['fp']})")
    t = Table(title="구간별", title_justify="left")
    for col in ("마디", "탭 음", "재현율", "정밀도"):
        t.add_column(col, justify="right")
    for b in res["blocks"]:
        t.add_row(f"{b['bar']}–{b['last']}", str(b["tab_notes"]), f"{b['R']:.1f}", f"{b['P']:.1f}")
    out.print(t)


def search_cmd(
    job: Annotated[str, typer.Option("--job", help="찾을 곡의 작업 키(data/jobs/<키>): 제목과 길이를 여기서 읽는다")],
    part: Annotated[str, typer.Option("--part", help="guitar | bass")] = "guitar",
    query: Annotated[str, typer.Option("--query", help="검색어(기본: 곡 제목 + TAB + 파트)")] = "",
    count: Annotated[int, typer.Option("--count", help="검색 결과 수")] = 20,
) -> None:
    """YouTube 에서 곡의 탭 영상을 찾아, 원곡과 길이가 맞는 순으로 다른 판 표기와 함께 보여 준다(받지는 않는다)."""
    from bandscribe import paths
    from bandscribe.jobs import hints as H
    from bandscribe.tabvid import align as A
    from bandscribe.tabvid import fetch as F

    job_dir = Path(paths.JOBS) / job
    try:
        title = str((H.read_hints(job_dir).get("hints") or {}).get("title") or "")
        meta = json.loads((job_dir / "input" / "meta.json").read_text(encoding="utf-8"))
        dur = float(meta.get("duration_s") or 0.0)
    except (OSError, ValueError) as e:
        _fail(f"작업을 읽지 못했습니다: {e}", 2)
    if not title and not query:
        _fail("작업에 곡 제목(hints.toml [hints] title)이 없습니다 — --query 를 주세요", 2)
    q = query or f"{title} {'ベース' if part == 'bass' else 'ギター'} TAB"
    try:
        found = F.search(q, count)
    except Exception as e:  # noqa: BLE001 - yt-dlp: one message
        _fail(f"검색하지 못했습니다: {e}")
    rows = sorted(found, key=lambda v: abs((v.get("duration") or 0) - dur))
    t = Table(title=f"'{escape(q)}' — 원곡 {_mmss(dur)}", title_justify="left")
    for col in ("id", "길이", "차이", "채널", "제목", "확인할 것"):
        t.add_column(col, justify="right" if col in ("길이", "차이") else "left")
    for v in rows:
        d = float(v.get("duration") or 0.0)
        notes = A.title_flags(v.get("title") or "")
        if "tab" not in (v.get("title") or "").lower():
            notes.append("제목에 TAB 없음")
        if dur and abs(d - dur) > 0.05 * dur:
            notes.append("길이가 다름(축약판·다른 판?)")
        t.add_row(escape(v["id"]), _mmss(d), f"{d - dur:+.0f} s", escape(str(v.get("channel") or ""))[:24],
                  escape(str(v.get("title") or ""))[:60], escape(", ".join(notes)) or "[green]–[/]")
    Console(highlight=False).print(t)


def review_cmd(
    read: Annotated[Path, typer.Argument(help="tabvid read 가 낸 <id>_n<줄>.json")],
    track: Annotated[int, typer.Option("--track", help="트랙 번호(1부터)")] = 1,
    agree: Annotated[float, typer.Option("--agree", help="프레임 일치가 이보다 낮은 마디만(의심 마디). 1 이면 모든 마디")] = 0.8,
    bars: Annotated[str, typer.Option("--bars", help="마디 범위만(예 100-120)")] = "",
) -> None:
    """사람 확인용 쪽(관문 3-5): 마디마다 영상에서 그 마디를 잘라낸 그림과 읽어 낸 탭을 나란히 놓은 HTML 을
    data/eval/tabvid/review/ 에 만든다(영상 그림은 로컬에만)."""
    import html

    from PIL import Image

    from bandscribe.tabvid import fetch as F
    from bandscribe.tabvid import sheet as SH

    try:
        doc = json.loads(read.read_text(encoding="utf-8"))
        tr = doc["tracks"][track - 1]
        frames = sorted(Path(doc["frames_dir"]).glob("f*.png")) if doc.get("frames_dir") else []
    except (OSError, ValueError, KeyError, IndexError) as e:
        _fail(str(e), 2)
    if not frames:
        _fail("프레임 폴더를 찾지 못했습니다 — tabvid read 를 다시 돌려 주세요(frames_dir)", 2)
    lo, hi = (int(x) for x in bars.split("-")) if bars else (-10 ** 9, 10 ** 9)
    pick = [b for b in tr["bars"] if lo <= b["number"] <= hi and (agree >= 1 or b["agree"] < agree) and b.get("view")]
    name = f"{read.stem}_t{track}"
    dest = F.root() / "review" / name
    dest.mkdir(parents=True, exist_ok=True)
    n = int(doc.get("strings", 6))
    rows = []
    for b in pick:
        fi, x0, x1, ytop, ybot = b["view"]
        sp = (ybot - ytop) / max(1, n - 1)
        im = Image.open(frames[min(fi, len(frames) - 1)])
        box = (max(0, int(x0 - 10)), max(0, int(ytop - 4 * sp)), min(im.width, int(x1 + 10)),
               min(im.height, int(ybot + 2.5 * sp)))
        im.crop(box).save(dest / f"bar{b['number']:04d}.png")
        flags = []
        if any(b.get("bend") or []):
            flags.append(f"벤딩 {sum(b['bend'])}")
        if any(b.get("slide") or []):
            flags.append(f"슬라이드 {sum(b['slide'])}")
        rows.append(f"<tr><td class=n>{b['number']}</td><td>{b['agree']:.2f}<br>{b['frames']}프레임"
                    f"<br>{html.escape(' · '.join(flags))}</td><td><img src='{name}/bar{b['number']:04d}.png'></td>"
                    f"<td><pre>{html.escape(chr(10).join(SH.ascii_bar(b, n)))}</pre></td></tr>")
    q = tr.get("quality") or {}
    page = f"""<!doctype html><html lang=ko><meta charset=utf-8><title>{html.escape(name)} 확인</title>
<style>
:root {{ color-scheme: light dark; --bg: #fafafa; --fg: #222; --line: #ccc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #1b1b1b; --fg: #ddd; --line: #444; }} }}
body {{ background: var(--bg); color: var(--fg); font: 14px system-ui, sans-serif; margin: 16px; }}
table {{ border-collapse: collapse; }}
td {{ border-bottom: 1px solid var(--line); padding: 6px 10px; vertical-align: top; }}
td.n {{ font-weight: 600; font-size: 18px; }} img {{ max-width: 900px; }} pre {{ margin: 0; font-size: 13px; }}
</style>
<h1>{html.escape(name)}</h1>
<p>{html.escape(doc.get('source') or '')} · 판정 {html.escape(q.get('verdict', '-'))}
{html.escape('; '.join(q.get('reasons') or []))}
· 보인 마디 {len(pick)}개 (프레임 일치 {agree} 미만{', 마디 ' + bars if bars else ''})</p>
<p>왼쪽 그림은 그 마디가 가장 오래 보인 프레임, 오른쪽은 읽어 낸 탭(b 벤딩, / 슬라이드). 다르면 영상 쪽이 맞다.</p>
<table>{''.join(rows)}</table></html>"""
    out = F.root() / "review" / f"{name}.html"
    out.write_text(page, encoding="utf-8")
    Console(highlight=False).print(f"확인할 마디 {len(pick)}개: {escape(str(out))}", soft_wrap=True)


def register(app: typer.Typer) -> None:
    tv = typer.Typer(help="탭 영상에서 실버 정답 탭 뽑기(docs/TAB_VIDEO.md). 영상·프레임·결과는 data/eval/tabvid/ 에만 둔다.",
                     no_args_is_help=True)
    app.add_typer(tv, name="tabvid")
    tv.command("read")(read_cmd)
    tv.command("compare")(compare_cmd)
    tv.command("search")(search_cmd)
    tv.command("align")(align_cmd)
    tv.command("score")(score_cmd)
    tv.command("review")(review_cmd)
