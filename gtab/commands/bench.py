"""``gtab bench gpu`` (M1_M2_SPEC 5; SEP). Imports only typer/rich/stdlib at module level (CLI start-up stays fast);
the bench itself (``gtab.bench``) is imported inside the command.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

bench_app = typer.Typer(help="GPU 벤치마크.", no_args_is_help=True)

GPU_HELP = ("GPU 벤치마크: 백엔드와 사다리 칸마다 VRAM(reserved·NVML·Shared Usage)과 실시간 배수를 잰다. "
            "결과는 data/bench/ 아래에 저장되고 VRAM 사전 점검 표(vram_table.json)를 갱신한다. "
            "--fallback 은 NVIDIA 'Sysmem Fallback' 설정 상태를 기록용으로 적는다(프로그램이 바꾸지 않는다).")


def _num(v: Any, fmt: str = ".0f") -> str:
    return "-" if v is None or v == "" else format(float(v), fmt)


_STATUS_KO = {"ok": "성공", "oom": "OOM", "shared_growth": "공유 메모리 증가", "unavailable": "없음",
              "skipped_budget": "예산 초과(건너뜀)", "error": "오류"}


@bench_app.command("gpu", help=GPU_HELP)
def gpu_cmd(
    clip: Annotated[str | None, typer.Option("--clip", help="측정할 곡: 파일 경로나 작업 키(없으면 합성 클립)")] = None,
    seconds: Annotated[float, typer.Option("--seconds", help="클립 길이(초). 0 이면 전체")] = 240.0,
    backend: Annotated[str, typer.Option("--backend", help="sep | amt | beats | all (쉼표로 여러 개)")] = "sep",
    rungs: Annotated[str, typer.Option("--rungs", help="all = 칸마다 따로(force_rung), auto = 0번 칸부터 보통 사다리")] = "all",
    cap_mb: Annotated[float | None, typer.Option(
        "--cap-mb", help="PyTorch 할당기 한도(OOM 사다리 검증용). 드라이버 폴백은 일어나지 않는다.")] = None,
    ballast_mb: Annotated[str | None, typer.Option(
        "--ballast-mb", help="별도 프로세스가 실제 VRAM을 N MB 붙잡아 여유를 줄인다(Sysmem Fallback 켬/끔 비교용). "
                             "auto = 첫 칸이 들어가지 않을 만큼.")] = None,
    fallback: Annotated[str, typer.Option("--fallback", help="NVIDIA Sysmem Fallback 설정 상태(기록용): on | off | unknown")] = "unknown",
    repeat: Annotated[int, typer.Option("--repeat", help="같은 칸을 몇 번 잴지(표에는 중앙값)")] = 1,
    set_: Annotated[list[str] | None, typer.Option("--set", "-s", metavar="KEY=VALUE", help="설정 덮어쓰기")] = None,
) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from gtab import bench, config

    out, err = Console(highlight=False), Console(stderr=True, highlight=False)
    try:
        cfg = config.load_config(overrides=set_ or [])
    except config.ConfigError as e:
        err.print(f"[bold red]설정 오류:[/] {escape(str(e))}")
        raise typer.Exit(2) from e
    names = ["sep", "amt", "beats"] if backend.strip() == "all" else [b.strip() for b in backend.split(",") if b.strip()]
    bal: int | str | None = None
    if ballast_mb is not None:
        bal = "auto" if ballast_mb.strip().lower() == "auto" else ballast_mb.strip()
    opts = bench.BenchOptions(clip=clip, seconds=seconds or None, backends=names, rungs=rungs, cap_mb=cap_mb,
                              ballast_mb=bal, fallback=fallback, repeat=repeat)
    try:
        opts.validate()
        if isinstance(opts.ballast_mb, str) and opts.ballast_mb != "auto":
            opts.ballast_mb = int(opts.ballast_mb)
        doc = bench.run_bench(opts, cfg, progress=lambda m: err.print(escape(m)))
    except bench.BenchError as e:
        err.print(f"[bold red]오류:[/] {escape(str(e))}")
        raise typer.Exit(1) from e

    procs = doc.get("processes_at_start") or []
    if procs:
        names_ = ", ".join(str(p.get("name") or p.get("pid")).replace("\\", "/").rsplit("/", 1)[-1] for p in procs)
        out.print(f"시작 때 GPU 를 쓰던 프로그램: {escape(names_)}")
    t = Table(title=f"GPU 벤치 ({escape(str(doc.get('gpu')))}, 드라이버 {escape(str(doc.get('driver')))})")
    for col in ("실행", "칸", "상태", "reserved MB", "NVML Δ MB", "공유 증가 MB", "실시간 배수", "로드 s", "컨텍스트 MB"):
        t.add_column(col)
    for r in doc.get("rows", []):
        t.add_row(escape(str(r.get("run"))), escape(str(r.get("rung_label") or "-")),
                  _STATUS_KO.get(str(r.get("status")), str(r.get("status"))) + (" (느림)" if r.get("slow") else ""),
                  _num(r.get("max_reserved_mb")), _num(r.get("nvml_delta_peak_mb")),
                  _num(r.get("pdh_self_shared_growth_mb"), ".1f"), _num(r.get("rtf"), ".2f"),
                  _num(r.get("load_s"), ".1f"), _num(r.get("ctx_mb")))
    out.print(t)
    out.print(f"결과: {escape(str(doc.get('run_dir')))}  (bench.json, bench.csv)")
    if doc.get("vram_table"):
        out.print(f"VRAM 표 갱신: {escape(str(doc['vram_table']))} (상한·밸러스트 없이 성공한 칸만 반영)")
    cal = doc.get("calibration") or {}
    if cal.get("shared_growth_noise_mb") is not None:
        rec = cal.get("recommended_shared_growth_fail_mb")
        cur = float(cfg.get("gpu.shared_growth_fail_mb", 64.0) or 64.0)
        line = (f"공유 메모리 잡음(폴백 끔 측정의 최대 증가): {cal['shared_growth_noise_mb']} MB "
                f"-> 권장 gpu.shared_growth_fail_mb = {rec} (현재 {cur:g})")
        out.print(line if rec == cur else line + "  [yellow]값이 다릅니다: 설정 변경이 필요합니다[/]")


def register(app: typer.Typer) -> None:
    app.add_typer(bench_app, name="bench")
