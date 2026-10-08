"""``bandscribe eval``: evaluation runs, comparisons, external results, pre-registered experiments, reference transcription.

Module level imports only typer / rich / stdlib (CLI start-up stays fast; spec §0). Backends load inside commands.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

log = logging.getLogger(__name__)

SetOpt = Annotated[list[str] | None, typer.Option("--set", "-s", metavar="KEY=VALUE", help="설정 덮어쓰기(여러 번 가능)")]
EXP_IDS = ("E1", "E2", "E3", "E3b", "E3c", "E12", "E12b", "E13", "E23", "E24-sep", "E24-amt", "latency", "gate-m2",
           "stereo-preservation")


def _out() -> Console:
    return Console(highlight=False)


def _fail(msg: str, code: int = 1) -> None:
    Console(stderr=True, highlight=False).print(f"[bold red]오류:[/] {escape(msg)}", soft_wrap=True)
    raise typer.Exit(code)


def _cfg(overrides: list[str] | None) -> Any:
    from bandscribe import config

    try:
        return config.load_config(overrides=overrides or ())
    except config.ConfigError as e:
        _fail(str(e), 2)


def _num(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if f != f else f"{f:.3f}"


def _parse_args(items: list[str] | None) -> dict[str, Any]:
    """``--arg key=value``: JSON values when they parse (numbers, lists, true/false), else the raw string."""
    import json

    out: dict[str, Any] = {}
    for it in items or []:
        key, sep, val = it.partition("=")
        if not sep or not key.strip():
            _fail(f"--arg 형식이 잘못되었습니다: {it!r} (예: --arg max_tracks=4)", 2)
        try:
            out[key.strip()] = json.loads(val)
        except ValueError:
            out[key.strip()] = val
    return out


def _resolve_run(ref: str) -> Path:
    from bandscribe import paths

    p = Path(ref)
    if p.is_dir():
        return p
    cands = sorted(d for d in paths.RUNS.glob(f"*{ref}*") if d.is_dir()) if paths.RUNS.is_dir() else []
    if len(cands) == 1:
        return cands[0]
    if not cands:
        _fail(f"평가 실행을 찾지 못했습니다: {ref}")
    _fail(f"'{ref}' 에 맞는 실행이 여러 개입니다: {', '.join(c.name for c in cands[:5])}")
    raise AssertionError


def _ci_txt(d: Any) -> str:
    if not d:
        return "–"
    pt = d.get("delta", d.get("value"))
    lo, hi = (d.get("ci") or [None, None])[:2]
    if lo is None or hi is None or lo != lo or hi != hi:
        return f"{_num(pt)}"
    return f"{_num(pt)} [{_num(lo)}, {_num(hi)}]"


def _print_special(con: Console, suite: str, s: dict[str, Any]) -> None:
    """Short console view of the distractors / backing suites (the full report is report.md / report.html)."""
    if s.get("dry_run"):
        con.print(f"계획만 세웠습니다(dry_run): 구간 {len(s.get('windows') or [])}개, 예상 GPU 약 {s.get('gpu_estimate_min')}분")
        for w in s.get("windows") or []:
            con.print(f"  {escape(w['item'])} {w['start_s']:.0f}–{w['end_s']:.0f} s: {escape(', '.join(w['conditions']))}")
        return
    if suite == "backing":
        t = Table("쌍", "원곡 기타 음", "반주에도 있음", "비율")
        for p in s.get("pairs") or []:
            t.add_row(escape(p["label"]), str(p.get("orig_n", "–")), str(p.get("on_backing", "–")),
                      _num(p.get("share")))
        con.print(t)
        return
    from bandscribe.eval import distractor_report as R

    t = Table("조건", "구간", "ΔFP/분", "ΔFN/분", "Δ재현율(점)", "ΔF1(점)")
    for cond, d in ((s.get("deltas") or {}).get("production") or {}).items():
        t.add_row(escape(R.cond_ko(cond)), str(d["n_items"]), _ci_txt(d["fp_per_min"]), _ci_txt(d["fn_per_min"]),
                  _ci_txt(d["recall"]), _ci_txt(d["f1"]))
    con.print(t)
    gpu = s.get("gpu") or {}
    con.print(f"GPU: 분리 {(gpu.get('sep_s') or 0) / 60:.1f}분, MuScriptor {(gpu.get('ms_s') or 0) / 60:.1f}분")


def register(app: typer.Typer) -> None:
    ev = typer.Typer(help="평가: 지표 계산, 실행 비교, 외부 결과 가져오기, 사전 등록 실험, 참조 전사.", no_args_is_help=True)
    app.add_typer(ev, name="eval")

    @ev.command("run")
    def run_cmd(
        suite: Annotated[str, typer.Option("--suite", help="quick | bp-smoke | synth | tierA | tierB | components | distractors | backing")] = "quick",
        system: Annotated[str | None, typer.Option("--system", help="oracle, oracle-noisy, majority, b0, no_split, job:<단계>, external:<이름>")] = None,
        out: Annotated[Path | None, typer.Option("--out", help="실행 폴더를 만들 상위 폴더 (기본 data/runs)")] = None,
        split: Annotated[str, typer.Option("--split", help="tierA 분할: dev | test | all")] = "dev",
        data: Annotated[str | None, typer.Option("--data", help="tierB 데이터셋(쉼표로 여러 개, 기본 cambridge_mt,medleydb)")] = None,
        arg: Annotated[list[str] | None, typer.Option("--arg", metavar="KEY=VALUE",
                                                      help="세트 인자(여러 번 가능): 예 distractors 의 songs=a,b · max_windows=1 · "
                                                           "gpu_budget_min=60 · dry_run=true, backing 의 pairs=<toml>, "
                                                           "job:<단계>·b0·no_split 의 allow_stale=true(현재 설정으로 만든 "
                                                           "단계 결과가 없을 때 예전 설정의 결과로 평가, 기본은 그 곡을 뺌)")] = None,
        set_: SetOpt = None,
    ) -> None:
        """평가 세트를 돌려 metrics.csv, summary.json, report.html 을 만든다."""
        from bandscribe.eval import suites

        cfg = _cfg(set_)
        con = _out()
        if suite not in suites.SUITES:
            _fail(f"알 수 없는 평가 세트 '{suite}' ({', '.join(suites.SUITES)})", 2)
        if split not in ("dev", "test", "all"):
            _fail(f"--split 은 dev | test | all 중 하나입니다 (받은 값: {split})", 2)
        args: dict[str, Any] = {"split": split}
        if data:
            args["datasets"] = [d.strip() for d in data.split(",") if d.strip()]
        args.update(_parse_args(arg))

        def progress(ev_: str, d: dict) -> None:
            if ev_ == "message":
                con.print(escape(str(d.get("message", ""))))
            else:
                log.debug("%s %s", ev_, d)

        try:
            res = suites.run_suite(suite, system, cfg, out=out, args=args, progress=progress)
        except suites.SuiteError as e:
            _fail(str(e))
        s = res.summary
        if suite in ("distractors", "backing"):
            _print_special(con, suite, s)
            con.print(f"결과 폴더: {escape(str(res.run_dir))}")
            return
        con.print(f"평가 세트 [bold]{suite}[/] · 시스템 [bold]{escape(str(s.get('system', '')))}[/] · 항목 {s.get('n_items', '-')}개")
        if s.get("note"):
            con.print(f"[yellow]{escape(s['note'])}[/]")
        t = Table("지표", "값", "95% CI")
        for k, v in (s.get("aggregates") or {}).items():
            t.add_row(escape(k), _num(v["value"]), f"{_num(v['ci'][0])} – {_num(v['ci'][1])}")
        for k, v in (s.get("baselines") or {}).items():
            t.add_row(escape(f"기준선 {k}"), _num(v["value"]), f"{_num(v['ci'][0])} – {_num(v['ci'][1])}")
        if s.get("experiments"):
            for k, v in s["experiments"].items():
                t.add_row(escape(k), escape(str(v.get("decision") or v.get("error"))), escape(str(v.get("run_dir", ""))))
        con.print(t)
        con.print(f"결과 폴더: {escape(str(res.run_dir))}")

    @ev.command("compare")
    def compare_cmd(
        run_a: Annotated[str, typer.Argument(help="기준 실행 (폴더 또는 이름 일부)")],
        run_b: Annotated[str, typer.Argument(help="비교할 실행")],
        set_: SetOpt = None,
    ) -> None:
        """두 평가 실행을 곡 블록 부트스트랩으로 비교한다."""
        from bandscribe.eval import compare

        cfg = _cfg(set_)
        a, b = _resolve_run(run_a), _resolve_run(run_b)
        try:
            res = compare.compare_runs(a, b, cfg)
        except compare.CompareError as e:
            _fail(str(e))
        con = _out()
        con.print(f"B − A: [bold]{escape(b.name)}[/] − [bold]{escape(a.name)}[/] (공통 항목 {res['n_common_items']}개, 점 = ×100)")
        t = Table("지표", "A", "B", "Δ(점)", "95% CI", "판정(MDE)")
        for m, v in res["metrics"].items():
            t.add_row(escape(m), _num(v["a"]), _num(v["b"]), _num(v["delta"]), f"{_num(v['ci'][0])} – {_num(v['ci'][1])}",
                      escape(v["decision_ko"]))
        con.print(t)
        for w in res["warnings"]:
            con.print(f"[yellow]주의:[/] {escape(w)}")

    @ev.command("import-external")
    def import_external_cmd(
        item: Annotated[str, typer.Argument(help="곡 ID (Tier A 곡 ID 또는 평가 항목 ID)")],
        file: Annotated[Path, typer.Argument(help=".mid / .gp5 / .gp 파일", exists=True, dir_okay=False)],
        system: Annotated[str, typer.Option("--system", help="songsterr_ai | klangio | mvsep101 | other")] = "other",
        map_: Annotated[str | None, typer.Option("--map", help='원본 트랙 -> Line, 예: "1=L1,2=L2"')] = None,
    ) -> None:
        """외부 도구의 결과(MIDI/GP)를 평가용 예측으로 가져온다."""
        from bandscribe import paths
        from bandscribe.eval import external, gtstore

        try:
            tm = None
            song_id = item.split(":", 1)[0]
            if gtstore.SONG_ID_RE.fullmatch(song_id) and (paths.EVAL / "tierA" / song_id / "tempo_map.json").is_file():
                tm = gtstore.load_tempo_map(paths.EVAL / "tierA" / song_id)
            out = external.import_external(item, file, system, mapping=external.parse_map(map_), tempo_map=tm)
        except (external.ExternalError, gtstore.GtError, ValueError) as e:
            _fail(str(e))
        _out().print(f"가져왔습니다: {escape(str(out))}  (평가: bandscribe eval run --suite tierA --system external:{system})")

    @ev.command("exp")
    def exp_cmd(
        exp_id: Annotated[str, typer.Argument(help="실험 id: " + ", ".join(EXP_IDS))],
        dry_run: Annotated[bool, typer.Option("--dry-run", help="실행하지 않고 사전 등록 내용만 확인한다")] = False,
        data: Annotated[str | None, typer.Option("--data", help="데이터 세트 이름(실험이 지원할 때, 쉼표로 여러 개)")] = None,
        arg: Annotated[list[str] | None, typer.Option("--arg", metavar="KEY=VALUE",
                                                      help="실험 인자(여러 번 가능, 값은 JSON 이면 JSON 으로): 예 max_tracks=4")] = None,
        record: Annotated[str | None, typer.Option("--record", help='사람이 확인한 결정을 기록: 예 "채택 (-14)" / "판단 보류 (데이터 대기)"')] = None,
        set_: SetOpt = None,
    ) -> None:
        """사전 등록된 실험을 돌린다."""
        from bandscribe.eval import experiments

        if record is not None:
            try:
                experiments.record_decision(exp_id, record)
            except experiments.ExperimentStateError as e:
                _fail(str(e))
            _out().print(f"{exp_id}: 상태를 '결정: {escape(record)}' 로 기록했습니다 (docs/decisions.md).")
            return
        cfg = _cfg(set_)
        args = _parse_args(arg)
        if data:
            args["data"] = data
        try:
            res = experiments.run_experiment(exp_id, cfg, args=args, dry_run=dry_run)
        except experiments.ExperimentStateError as e:
            _fail(str(e))
        except experiments.ExperimentRunError as e:
            msg = f"{exp_id} 실행 실패: {e.cause}"
            if e.run_dir is not None:
                msg += f"\n(작업 폴더: {e.run_dir})"
            if e.data_missing:
                msg += (f"\n데이터가 준비되기 전에는 결정할 수 없습니다. 지금 기록하려면: "
                        f"bandscribe eval exp {exp_id} --record \"판단 보류 (데이터 대기)\"")
            _fail(msg)
        con = _out()
        if dry_run:
            con.print(f"[bold]{exp_id}[/] ({escape(res.status)}): {escape(res.fields.get('제목', ''))}")
            for k in ("주 지표", "MDE", "데이터(세트, 분할, 창 정의)", "분석", "결정 규칙", "판정 설정", "실행 명령"):
                if k in res.fields:
                    con.print(f"  {k}: {escape(res.full.get(k, res.fields[k]))}", soft_wrap=True)
            return
        con.print(f"{exp_id}: 계산된 판정 = [bold]{escape(str(res.summary.get('decision_ko')))}[/]")
        for w in res.summary.get("warnings", []):
            con.print(f"[yellow]주의:[/] {escape(w)}")
        con.print(f"결과 폴더: {escape(str(res.run_dir))}")
        con.print("확인 후 결정 기록: bandscribe eval exp " + exp_id + ' --record "채택 (…)" | "기각" | "판단 보류 (…)"')

    @ev.command("reftx")
    def reftx_cmd(
        dataset: Annotated[str, typer.Argument(help="데이터셋 이름 (예: cambridge_mt)")],
        track: Annotated[str | None, typer.Option("--track", help="트랙 ID 하나만")] = None,
        set_: SetOpt = None,
    ) -> None:
        """Tier B 참 단독 트랙을 MuScriptor로 참조 전사한다."""
        cfg = _cfg(set_)
        try:
            from bandscribe import datasets, paths
            from bandscribe.amt.reftx import reference_transcribe
        except ImportError as e:
            _fail(f"참조 전사 모듈을 불러오지 못했습니다: {e}")
        try:
            available = datasets.is_available(dataset)
        except Exception as e:  # unknown name -> the registry's Korean message
            _fail(str(e), 2)
        if not available:
            _fail(f"데이터셋 '{dataset}' 이 없습니다 (bandscribe data list / fetch / import-local)")
        con = _out()
        try:
            tracks = [datasets.load_track(dataset, track)] if track else list(datasets.iter_tracks(dataset))
        except Exception as e:  # loader errors (lines.yaml problems, unknown track) carry Korean messages
            _fail(f"트랙을 읽지 못했습니다: {e}")
        # reftx writes into the item's own folder data/eval/tierB/<dataset>/<track>/ (bandscribe.amt.reftx contract);
        # the tierB suite reads the references back from there (bandscribe.eval.suites.reftx_dir)
        from bandscribe.eval.suites import reftx_dir

        out_root = paths.EVAL / "tierB"
        failed = 0
        for k, tr in enumerate(tracks, 1):
            con.print(f"[{k}/{len(tracks)}] {escape(tr.dataset)}:{escape(tr.track_id)} 참조 전사 중…")
            try:
                res = reference_transcribe(tr, reftx_dir(tr), cfg)
            except Exception as e:  # ModelMissing / VramInsufficient / worker failures: report, go on
                failed += 1
                con.print(f"  [red]실패:[/] {escape(str(e))}")
                continue
            if not res:
                con.print("  [yellow]기타·베이스 Line 의 단독 트랙이 없어 건너뜀[/] (lines.yaml 의 parts.line 확인)")
                continue
            con.print("  " + ", ".join(f"{escape(k_)}={len((v or {}).get('notes') or [])}음" for k_, v in res.items()))
        con.print(f"저장 위치: {escape(str(out_root))}")
        if failed:
            raise typer.Exit(1)
