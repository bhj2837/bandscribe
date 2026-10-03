"""`gtab data ...`: datasets (Tier B/C) and the Tier A candidate registry (DATA owner, M1_M2_SPEC §5, §9.5).

Module level imports only typer/rich/stdlib (CLI start-up stays fast, test_import_weight); the backends
(gtab.datasets.*) are imported inside the commands.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

log = logging.getLogger(__name__)

STATUS_KO = {"absent": "없음", "partial": "일부만 받음", "fetched": "받음", "verified": "검증됨", "manual": "받음(직접)"}


def _out() -> Console:
    return Console(highlight=False)


def _err() -> Console:
    return Console(stderr=True, highlight=False)


def _fail(msg: str, code: int = 1) -> NoReturn:
    _err().print(f"[bold red]오류:[/] {escape(msg)}", soft_wrap=True)
    raise typer.Exit(code)


def _size(n: int | float | None) -> str:
    if not n:
        return "-"
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:,.1f} MB"


def _status_ko(row: dict[str, Any]) -> str:
    st = row["status"]
    if st == "absent" and row.get("access") == "manual":
        return "수동 필요"
    return STATUS_KO.get(st, st)


def _registry_error(e: Exception) -> NoReturn:
    _fail(str(e), code=2)


# ------------------------------------------------------------------------------------------ commands


def cmd_list() -> None:
    """데이터셋 목록: 계층, 라이선스, 크기, 상태, 위치"""
    from gtab import datasets

    t = Table(title="데이터셋")
    for col in ("이름", "계층", "라이선스", "크기", "상태", "위치"):
        t.add_column(col, overflow="fold")
    for r in datasets.list_datasets():
        size = r["local_bytes"] or r["approx_bytes"]
        t.add_row(r["name"], r["tier"], escape(r["license"]), _size(size), _status_ko(r), escape(str(r["path"])))
    _out().print(t)


def cmd_fetch(name: str, yes: bool, part: list[str] | None) -> None:
    """URL·크기·라이선스·저장 위치를 보여 주고 y/N 을 받은 뒤 받는다."""
    from gtab.datasets import fetch as F
    from gtab.datasets import registry

    out = _out()

    def confirm(text: str) -> bool:
        out.print(escape(text), soft_wrap=True)
        return typer.confirm("받을까요?", default=False)

    try:
        plan = F.plan_fetch(name, parts=part or None)
        if not F.free_space_ok(plan.root, max(plan.known_bytes, plan.spec.approx_bytes)):
            _fail(f"디스크 여유 공간이 모자랍니다 (필요 약 {_size(2.2 * max(plan.known_bytes, plan.spec.approx_bytes))}).")
        if not yes and not confirm(plan.describe_ko()):
            _fail("취소했습니다.")
        res = F.execute(plan, progress=True, notify=lambda s: out.print(escape(s)))
    except registry.RegistryError as e:
        _registry_error(e)
    except F.ManualDownloadRequired as e:
        _fail(f"{e}\n직접 받은 뒤: gtab data import-local {name} <폴더>")
    except F.FetchError as e:
        _fail(str(e))
    out.print(f"완료: 새로 받음 {res['downloaded']}개, 이미 있음 {res['skipped']}개, 상태 {STATUS_KO.get(res['status'], res['status'])}")
    for p in res["extracted"]:
        out.print(f"  압축 해제: {escape(p)}")
    if name == "cambridge_mt":
        out.print("곡마다 lines.yaml 초안을 만들었습니다. 귀로 확인해 kind·line·families_present 를 고친 뒤 reviewed: true 로 바꾸세요.")


def cmd_verify(name: str | None) -> None:
    """받은 파일의 SHA256 을 다시 계산해 기록과 비교한다."""
    from gtab.datasets import registry, store

    names = [name] if name else sorted(store.read_lock()["datasets"])
    if name:
        try:
            registry.get(name)
        except registry.RegistryError as e:
            _registry_error(e)
    if not names:
        _out().print("기록된 데이터셋이 없습니다.")
        return
    bad = 0
    for n in names:
        r = store.verify(n)
        if r["ok"]:
            _out().print(f"[green]OK[/] {n}: 파일 {r['checked']}개 일치 ({STATUS_KO.get(r['status'], r['status'])})")
            if r["status"] == "partial":
                _out().print(f"    필수 파일이 아직 다 있지 않습니다. `gtab data fetch {n}` 로 나머지를 이어 받으세요.")
        else:
            bad += 1
            ext = r.get("extracted") or []
            _out().print(f"[red]실패[/] {n}: 없음 {len(r['missing'])}개, 불일치 {len(r['mismatch'])}개"
                         + (f", 압축 해제본 문제 {len(ext)}개" if ext else ""))
            for rel in (r["missing"] + r["mismatch"] + ext)[:20]:
                _out().print(f"    {escape(rel)}")
            if ext and not r["missing"] and not r["mismatch"]:
                _out().print(f"    받은 파일은 정상입니다. `gtab data fetch {n}` 를 다시 실행하면 압축을 다시 풉니다.")
    if bad:
        raise typer.Exit(1)


def cmd_import_local(name: str, folder: Path, song: str | None, copy: bool) -> None:
    """직접 받은 데이터(수동 다운로드)를 등록한다."""
    from gtab.datasets import registry, store

    try:
        res = store.import_local(name, folder, song=song, link=not copy)
    except registry.RegistryError as e:
        _registry_error(e)
    except (store.StoreError, ValueError) as e:
        _fail(str(e))
    out = _out()
    out.print(f"등록: {res['files']}개 ({_size(res['bytes'])}; 하드링크 {res['linked']}, 복사 {res['copied']}) -> "
              f"{escape(str(res['dest']))}")
    out.print(f"라이선스: {escape(res['license'])} - 이 PC 안에서만 씁니다.")
    if res["lines_yaml"]:
        out.print(f"lines.yaml 초안: {escape(str(res['lines_yaml']))} - 귀로 확인해 고친 뒤 reviewed: true 로 바꾸세요.")


# ---------------------------------------------------------------------------------------------- Tier A


def cmd_tiera_list() -> None:
    """Tier A 후보곡 목록 (data/eval/tierA/songs.yaml)"""
    from gtab.datasets import tiera

    try:
        songs = tiera.load_songs()
    except tiera.TierAError as e:
        _fail(str(e))
    if not songs:
        _out().print(f"등록된 곡이 없습니다 ({escape(str(tiera.songs_path()))}).")
        return
    t = Table(title=f"Tier A 후보곡 ({tiera.songs_path()})")
    for col in ("ID", "제목", "아티스트", "길이", "무손실", "반주", "반주 점검", "정답"):
        t.add_column(col, overflow="fold")
    for e in songs.values():
        dur = f"{int(e.duration_s // 60)}:{e.duration_s % 60:04.1f}" if e.duration_s else "?"
        chk = e.extra.get("backing_check") or {}
        ex = chk.get("excess_over_codec_floor_db")
        verdict = "-" if not chk else ("빠진 파트 있음" if chk.get("removed_part_likely") else "빠진 것 거의 없음")
        if ex is not None:
            verdict += f" ({ex:+.1f} dB)"
        t.add_row(e.id, escape(e.title), escape(e.artist), dur, "예" if e.lossless else "아니오",
                  "있음" if e.backing else "-", verdict, escape(e.gt) if e.gt else "없음")
    _out().print(t)


def cmd_tiera_add(song_id: str, path: Path, title: str, artist: str, backing: Path | None, backing_note: str,
                  notes: str) -> None:
    """Tier A 후보곡을 등록한다(파일은 복사하지 않고 경로만 적는다)."""
    from gtab.datasets import tiera

    try:
        e = tiera.add_song(song_id, path, title=title, artist=artist, backing=backing, backing_note=backing_note,
                           notes=notes)
    except (tiera.TierAError, OSError) as ex:
        _fail(str(ex))
    except Exception as ex:  # ffprobe failures carry Korean messages (gtab.ingest.decode.DecodeError)
        _fail(str(ex))
    _out().print(f"등록: {e.id} ({escape(e.title)} - {escape(e.artist)}), {e.duration_s} s, {escape(e.codec)} -> "
                 f"{escape(str(tiera.songs_path()))}")


def cmd_tiera_align(song_ids: list[str] | None, max_offset_s: float) -> None:
    """반주를 원곡에 맞추고(오프셋·게인) 진단용 잔차(원곡 - 반주)를 쓴다. 잔차는 정답이 아니다."""
    from gtab.datasets import tiera

    try:
        songs = tiera.load_songs()
    except tiera.TierAError as e:
        _fail(str(e))
    ids = song_ids or [s for s, e in songs.items() if e.backing]
    if not ids:
        _fail("반주가 등록된 곡이 없습니다.")
    out = _out()
    t = Table(title="반주 정렬 (잔차는 정답이 아님: 진단용)")
    for col in ("곡", "오프셋 s", "게인 dB", "잔차 dB", "코덱 바닥 대비 dB", "드리프트 ppm", "상관", "잔차 파일"):
        t.add_column(col, overflow="fold")
    failed = 0
    for sid in ids:
        out.print(f"[bold]{escape(sid)}[/]")
        try:
            rep = tiera.align_backing(sid, max_offset_s=max_offset_s, progress=lambda s: out.print(f"  {escape(s)}"))
        except Exception as e:  # one broken song must not stop the others; message is Korean
            failed += 1
            out.print(f"  [red]실패:[/] {escape(str(e))}")
            continue
        r = tiera.summary_row(rep)
        t.add_row(sid, f"{r['offset_s']:+.4f}", f"{r['gain_db']:+.2f}" if r["gain_db"] is not None else "?",
                  f"{r['residual_db']:.1f}" if r["residual_db"] is not None else "?",
                  f"{r['excess_db']:+.1f}" if r.get("excess_db") is not None else "-", f"{r['drift_ppm']:+.2f}",
                  f"{r['corr']:.2f}", escape(str(tiera.tier_a_dir() / sid / "backing" / rep["residual_wav"])))
        out.print(f"  {escape(rep['hint_ko'])}")
    out.print(t)
    out.print(escape(tiera.NOT_GT_KO))
    if failed:
        raise typer.Exit(1)


# ------------------------------------------------------------------------------------------ register


def register(app: typer.Typer) -> None:
    data_app = typer.Typer(help="데이터셋(Tier B/C) 관리: 목록, 받기, 검증, 직접 받은 데이터 등록. Tier A 후보곡(tier-a).",
                           no_args_is_help=True)
    tiera_app = typer.Typer(help="Tier A 후보곡(실제로 커버하는 곡) 목록과 반주 정렬 진단.", no_args_is_help=True)

    @data_app.command("list")
    def _list() -> None:
        """데이터셋 목록: 계층, 라이선스, 크기, 상태, 위치"""
        cmd_list()

    @data_app.command("fetch")
    def _fetch(
        name: Annotated[str, typer.Argument(metavar="이름", help="데이터셋 이름 (gtab data list)")],
        yes: Annotated[bool, typer.Option("--yes", "-y", help="확인 없이 받는다")] = False,
        part: Annotated[list[str] | None, typer.Option("--part", metavar="파일", help="받을 부분(여러 번 가능, all = 선택 파일까지 전부)")] = None,
    ) -> None:
        """URL·크기·라이선스·저장 위치를 보여 주고 y/N 을 받은 뒤 받는다."""
        cmd_fetch(name, yes, part)

    @data_app.command("verify")
    def _verify(name: Annotated[str | None, typer.Argument(metavar="[이름]", help="생략하면 전부")] = None) -> None:
        """받은 파일의 SHA256 을 다시 계산해 기록과 비교한다."""
        cmd_verify(name)

    @data_app.command("import-local")
    def _import_local(
        name: Annotated[str, typer.Argument(metavar="이름")],
        folder: Annotated[Path, typer.Argument(metavar="폴더", help="직접 받아 푼 폴더")],
        song: Annotated[str | None, typer.Option("--song", metavar="이름", help="Cambridge-MT: 곡 이름(곡마다 한 번)")] = None,
        copy: Annotated[bool, typer.Option("--copy", help="하드링크 대신 복사")] = False,
    ) -> None:
        """직접 받은 데이터(수동 다운로드)를 등록한다."""
        cmd_import_local(name, folder, song, copy)

    @tiera_app.command("list")
    def _tiera_list() -> None:
        """Tier A 후보곡 목록"""
        cmd_tiera_list()

    @tiera_app.command("add")
    def _tiera_add(
        song_id: Annotated[str, typer.Argument(metavar="곡ID", help="ASCII 소문자 ID (예: sc)")],
        path: Annotated[Path, typer.Argument(metavar="원곡파일")],
        title: Annotated[str, typer.Option("--title", help="곡 제목")],
        artist: Annotated[str, typer.Option("--artist", help="아티스트")],
        backing: Annotated[Path | None, typer.Option("--backing", metavar="파일", help="연습용 반주 파일")] = None,
        backing_note: Annotated[str, typer.Option("--backing-note", help="반주 설명")] = "",
        notes: Annotated[str, typer.Option("--notes", help="메모(편성 등)")] = "",
    ) -> None:
        """Tier A 후보곡을 등록한다(파일은 복사하지 않고 경로만 적는다)."""
        cmd_tiera_add(song_id, path, title, artist, backing, backing_note, notes)

    @tiera_app.command("align")
    def _tiera_align(
        song_ids: Annotated[list[str] | None, typer.Argument(metavar="[곡ID...]", help="생략하면 반주가 있는 곡 전부")] = None,
        max_offset_s: Annotated[float, typer.Option("--max-offset-s", help="찾을 최대 시작 위치 차이(초)")] = 30.0,
    ) -> None:
        """반주를 원곡에 맞추고(오프셋·게인) 진단용 잔차(원곡 - 반주)를 쓴다. 잔차는 정답이 아니다."""
        cmd_tiera_align(song_ids, max_offset_s)

    data_app.add_typer(tiera_app, name="tier-a")
    app.add_typer(data_app, name="data")
