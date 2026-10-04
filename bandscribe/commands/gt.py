"""``bandscribe gt``: Tier A ground truth — import a reference tab, render it, align it to the recording, taps, approval.

Module level imports only typer / rich / stdlib (spec §0).
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape

log = logging.getLogger(__name__)


def _out() -> Console:
    return Console(highlight=False)


def _fail(msg: str, code: int = 1) -> None:
    Console(stderr=True, highlight=False).print(f"[bold red]오류:[/] {escape(msg)}", soft_wrap=True)
    raise typer.Exit(code)


def _cfg() -> Any:
    from bandscribe import config

    try:
        return config.load_config()
    except config.ConfigError as e:
        _fail(str(e), 2)


def _song_dir(song_id: str, must_exist: bool = True) -> Path:
    from bandscribe.eval import gtstore

    try:
        d = gtstore.song_dir(song_id)
    except gtstore.GtError as e:
        _fail(str(e), 2)
    if must_exist and not (d / "gt.yaml").is_file():
        _fail(f"곡 '{song_id}' 의 gt.yaml 이 없습니다. 먼저 bandscribe gt import {song_id} <참조탭> --audio <파일> 을 실행하세요.")
    return d


def _resolve_song_key(audio: str) -> str:
    """An existing job key (prefix ok) or a file/URL to ingest."""
    from bandscribe import paths

    jobs = paths.JOBS
    if jobs.is_dir() and not Path(audio).exists():
        cands = [p.name for p in jobs.iterdir() if p.is_dir() and p.name.startswith(audio)]
        if len(cands) == 1:
            return cands[0]
    from bandscribe.ingest import ingest
    from bandscribe.jobs.store import JobStore

    res = ingest(audio, store=JobStore(), cfg=_cfg())
    return res.song_key


def _mix_path(song_key: str) -> Path:
    from bandscribe.jobs.store import JobStore

    p = JobStore().input_dir(song_key) / "mix_44k_f32.wav"
    if not p.is_file():
        _fail(f"작업 {song_key} 의 믹스(mix_44k_f32.wav)가 없습니다. bandscribe ingest 를 다시 실행하세요.")
    return p


def register(app: typer.Typer) -> None:
    gt = typer.Typer(help="정답(Tier A) 준비: 참조 탭 가져오기, 렌더, DTW 정렬, 다운비트 탭, 승인.", no_args_is_help=True)
    app.add_typer(gt, name="gt")

    @gt.command("import")
    def import_cmd(
        song_id: Annotated[str, typer.Argument(help="곡 ID (영문 소문자·숫자·_·-)")],
        ref: Annotated[Path, typer.Argument(help="참조 탭 (.gp3/.gp4/.gp5/.gp/.gpx)", exists=True, dir_okay=False)],
        audio: Annotated[str, typer.Option("--audio", help="같은 곡의 오디오 파일 또는 작업 키")],
        encoding: Annotated[str | None, typer.Option("--encoding", help="GP3-5 문자 인코딩 (기본: cp949 -> cp1252)")] = None,
        engine: Annotated[str, typer.Option("--engine", help="auto | pyguitarpro | alphatab")] = "auto",
    ) -> None:
        """참조 탭을 가져와 반복·다른 엔딩을 펼치고 gt.yaml 초안을 만든다(구간·Line 지도는 직접 채운다)."""
        from bandscribe import atomic
        from bandscribe.eval import gp_import, gtstore
        from bandscribe.eval.node import NodeError
        from bandscribe.eval.refscore import detect_pickup, play_order

        d = _song_dir(song_id, must_exist=False)
        d.mkdir(parents=True, exist_ok=True)
        spec_path = d / "gt.yaml"
        spec = gtstore.load_spec(spec_path) if spec_path.is_file() else None
        enc = encoding or ((spec.reference or {}).get("encoding") if spec else None)
        try:
            rs = gp_import.import_reference(ref, encoding=enc, engine=engine)
        except (gp_import.ReferenceImportError, NodeError) as e:
            _fail(str(e))
        ext = ref.suffix.lower()
        atomic.write_bytes(d / f"ref{ext}", ref.read_bytes())
        rs.save(d / "refscore.json")
        try:
            song_key = _resolve_song_key(audio)
        except Exception as e:  # ingest errors carry Korean messages
            _fail(f"오디오를 가져오지 못했습니다: {e}")
        con = _out()
        order = play_order(rs.masterbars, (spec.reference or {}).get("bar_order") if spec else None)
        if spec is None:
            text = gtstore.skeleton_yaml(song_id, rs, song_key=song_key, ref_file=f"ref{ext}", expanded_bars=len(order),
                                         encoding=rs.source.get("encoding"), pickup=detect_pickup(rs.masterbars, order))
            atomic.write_text(spec_path, text)
            con.print(f"gt.yaml 초안을 만들었습니다: {escape(str(spec_path))}")
        else:
            con.print("기존 gt.yaml 을 유지합니다 (덮어쓰지 않음).")
            if (spec.audio or {}).get("song_key") != song_key:
                con.print(f"[yellow]주의:[/] gt.yaml 의 audio.song_key 가 이번 오디오({song_key})와 다릅니다.")
        song = gtstore.load_song(d)
        gtn = gtstore.rebuild_gt_notes(song)
        con.print(f"트랙 {len(rs.tracks)}개: " + ", ".join(f"{t.index}.{escape(t.name)}({gtstore.guess_kind(t)})" for t in rs.tracks))
        con.print(f"펼친 마디 {len(order)}개, 정답 음 {len(gtn.notes)}개, 문자 인코딩 {rs.source.get('encoding')}, 작업 키 {song_key}")
        con.print("다음: gt.yaml 의 sections(반열림 마디 범위)·families_present·lines 를 채우고 "
                  f"bandscribe gt align {song_id} 로 정렬하세요.")

    @gt.command("render")
    def render_cmd(
        song_id: Annotated[str, typer.Argument(help="곡 ID")],
        engine: Annotated[str, typer.Option("--engine", help="internal | alphatab")] = "internal",
    ) -> None:
        """참조 탭을 오디오로 렌더한다(청취·A/B용)."""
        from bandscribe.eval import gtstore, render_ref
        from bandscribe.eval.node import NodeError

        d = _song_dir(song_id)
        try:
            out = render_ref.render_song(d, engine=engine)
        except (gtstore.GtError, NodeError, ValueError) as e:
            _fail(str(e))
        _out().print(f"렌더: {escape(str(out))}")

    def _align(song_id: str, taps: list[tuple[float, int]] | None) -> None:
        from bandscribe.eval import align, gtstore

        d = _song_dir(song_id)
        spec = gtstore.load_spec(d / "gt.yaml")
        song_key = (spec.audio or {}).get("song_key")
        if not song_key:
            _fail("gt.yaml 의 audio.song_key 가 비어 있습니다")
        try:
            res = align.align_song(d, _mix_path(song_key), taps=taps)
        except gtstore.GtError as e:
            _fail(str(e))
        rep = res.report
        con = _out()
        con.print(f"정렬 완료 ({rep['method']}): 박 {rep['n_beats']}개, 펼친 마디 {rep['expanded_bars']}개"
                  + ("" if rep["bars_match"] in (None, True) else f" [yellow](gt.yaml 기록 {rep['reference_expanded_bars']}개와 다름)[/]"))
        if rep["taps"]:
            con.print(f"탭 잔차 최대 {rep['max_tap_residual_ms']} ms")
        con.print(f"국소 템포 비(명목 대비) {rep['local_tempo_ratio']['min']}–{rep['local_tempo_ratio']['max']}, "
                  f"명목 격자 대비 누적 이탈 최대 {rep['drift_s']['max_abs']} s "
                  f"(박 사이 최대 변화 {rep['drift_s']['max_beat_to_beat']} s)")
        con.print(f"들어 보기: {escape(str(d / 'align_check.wav'))}  → 맞으면 bandscribe gt approve {song_id}")

    @gt.command("align")
    def align_cmd(
        song_id: Annotated[str, typer.Argument(help="곡 ID")],
        taps: Annotated[Path | None, typer.Option("--taps", help="Audacity 라벨 파일 (다운비트 탭)", exists=True, dir_okay=False)] = None,
    ) -> None:
        """DTW로 정답 템포맵을 만들고 align_check.wav 를 쓴다(Beat This! 안 씀)."""
        from bandscribe.eval import taps as taps_mod

        d = _song_dir(song_id)
        tp: list[tuple[float, int]] | None = None
        if taps is not None:
            tp = _import_taps(song_id, d, taps)
        elif (d / "taps.json").is_file():
            tp = taps_mod.load_taps(d / "taps.json")
        _align(song_id, tp)

    def _import_taps(song_id: str, d: Path, label_file: Path) -> list[tuple[float, int]]:
        from bandscribe import atomic
        from bandscribe.eval import gtstore
        from bandscribe.eval import taps as taps_mod
        from bandscribe.eval.refscore import nominal_tempo_map

        song = gtstore.load_song(d)
        if song.gt_notes is None:
            _fail("gt_notes.json 이 없습니다 (bandscribe gt import 먼저)")
        parsed = taps_mod.parse_audacity_labels(taps_mod.read_label_text(label_file))
        if not parsed:
            _fail(f"라벨을 하나도 읽지 못했습니다: {label_file}")
        tm = song.tempo_map or nominal_tempo_map(song.gt_notes)
        assigned = taps_mod.assign_bars(parsed, tm, [b["bar"] for b in song.gt_notes.bars])
        atomic.write_json(d / "taps.json", taps_mod.taps_json(label_file.name, assigned))
        shutil.copyfile(label_file, d / "taps_labels.txt")
        _out().print(f"탭 {len(assigned)}개를 가져왔습니다 → taps.json")
        return [(t["t_s"], t["bar"]) for t in assigned if t["bar"] is not None]

    @gt.command("taps")
    def taps_cmd(
        song_id: Annotated[str, typer.Argument(help="곡 ID")],
        label_file: Annotated[Path, typer.Argument(help="Audacity 라벨 파일", exists=True, dir_okay=False)],
    ) -> None:
        """Audacity 라벨 트랙의 다운비트 탭을 가져와 정렬을 보정한다."""
        d = _song_dir(song_id)
        _align(song_id, _import_taps(song_id, d, label_file))

    @gt.command("approve")
    def approve_cmd(
        song_id: Annotated[str, typer.Argument(help="곡 ID")],
        minutes: Annotated[float | None, typer.Option("--minutes", help="정답 준비에 든 시간(분), 예산 보정용")] = None,
    ) -> None:
        """align_check.wav 를 듣고 승인한 것으로 기록한다."""
        from bandscribe.eval import gtstore

        d = _song_dir(song_id)
        if not (d / "tempo_map.json").is_file():
            _fail("아직 정렬되지 않았습니다 (bandscribe gt align 먼저)")
        spec = gtstore.load_spec(d / "gt.yaml")
        al = dict(spec.alignment or {})
        al.update(approved=True, approved_utc=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
        if minutes is not None:
            al["minutes_spent"] = minutes
        gtstore.save_spec_fields(d / "gt.yaml", {"alignment": al})
        _out().print(f"{song_id}: 정렬 승인을 기록했습니다.")
