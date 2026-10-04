"""bandscribe.bench: row extraction, vram_table merge rules, calibration, the ballast launcher, and a faked full run."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bandscribe import bench, paths, vram
from bandscribe.vram import GpuSnapshot


@pytest.fixture(autouse=True)
def _tmp_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    for name, value in (("DATA", data), ("BENCH", data / "bench"), ("VRAM_TABLE", data / "bench" / "vram_table.json"),
                        ("MODELS", data / "models"), ("MODELS_LOCK", data / "models" / "models.lock.json")):
        monkeypatch.setattr(paths, name, value)
    return data


def _row(label: str = "chunk=588800", status: str = "ok", reserved: float = 1800.0, **kw: Any) -> dict[str, Any]:
    r = {"backend": "sep_msst", "rung_label": label, "status": status, "device": "cuda", "max_reserved_mb": reserved,
         "ctx_mb": 400.0, "nvml_delta_peak_mb": 2100.0, "rtf": 6.0, "fallback": "off", "clip_s": 240.0,
         "cap_mb": None, "ballast_mb": None, "pdh_self_shared_growth_mb": 3.0}
    r.update(kw)
    return r


def test_merge_only_clean_successful_rows_median_over_repeats() -> None:
    rows = [_row(reserved=1790), _row(reserved=1800, rtf=5.0), _row(reserved=1830, rtf=7.0),
            _row(reserved=900, cap_mb=2500.0),  # capped: evidence for b4 only
            _row(reserved=950, ballast_mb=3000.0),  # ballasted: evidence for b3 only
            _row(reserved=100, status="oom"),  # failed attempt
            _row(label="chunk=262144@cpu", device="cpu", reserved=None),
            _row(label="chunk=352256", reserved=1354)]
    t = bench.merge_table(None, rows, gpu="RTX 2060", driver="610.47", measured_utc="2026-09-30T12:00:00+00:00")
    e = t["entries"]["sep_msst"]
    assert set(e) == {"chunk=588800", "chunk=352256"}
    assert e["chunk=588800"]["reserved_peak_mb"] == 1800.0 and e["chunk=588800"]["rtf"] == 6.0
    assert e["chunk=588800"]["n"] == 3 and e["chunk=588800"]["ctx_mb"] == 400.0
    assert e["chunk=588800"]["fallback"] == "off" and e["chunk=352256"]["reserved_peak_mb"] == 1354.0
    # ctx: median over every clean row of the backend (rung-independent; a single reading is noisy)
    t2 = bench.merge_table(None, [_row(ctx_mb=6.0), _row(label="chunk=352256", ctx_mb=89.0),
                                  _row(label="chunk=262144", ctx_mb=97.0)], gpu=None, driver=None, measured_utc="x")
    assert {v["ctx_mb"] for v in t2["entries"]["sep_msst"].values()} == {89.0}
    assert t["format"] == "bandscribe.vram_table/1" and t["gpu"] == "RTX 2060"
    # the merged table drives the VRAM pre-check
    p = vram.vram_table_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(t), encoding="utf-8")
    needs = vram.load_table(ctx_default_mb=500)["sep_msst"]
    assert needs["chunk=588800"] == vram.RungNeed(1800.0, 400.0)
    assert needs["chunk=262144"] == vram.RungNeed(1200.0, 500.0)  # not benched: default


def test_merge_keeps_other_entries_and_replaces_benched_ones() -> None:
    old = {"format": "bandscribe.vram_table/1", "entries": {"amt_muscriptor": {"medium": {"reserved_peak_mb": 2360}},
                                                        "sep_msst": {"chunk=588800": {"reserved_peak_mb": 2000}}}}
    t = bench.merge_table(old, [_row(reserved=1796)], gpu=None, driver=None, measured_utc="x")
    assert t["entries"]["amt_muscriptor"]["medium"]["reserved_peak_mb"] == 2360
    assert t["entries"]["sep_msst"]["chunk=588800"]["reserved_peak_mb"] == 1796.0


def test_calibration_from_fallback_off_runs() -> None:
    rows = [_row(pdh_self_shared_growth_mb=12.0), _row(pdh_self_shared_growth_mb=41.0, status="oom"),
            _row(pdh_self_shared_growth_mb=900.0, fallback="on"),  # a real spill: not noise
            _row(pdh_self_shared_growth_mb=None)]
    t = bench.merge_table(None, rows, gpu=None, driver=None, measured_utc="t1")
    cal = t["calibration"]
    assert cal["shared_growth_noise_mb"] == 41.0 and cal["runs"] == 2
    assert cal["recommended_shared_growth_fail_mb"] == 82.0
    t2 = bench.merge_table(t, [_row(pdh_self_shared_growth_mb=5.0)], gpu=None, driver=None, measured_utc="t2")
    assert t2["calibration"]["shared_growth_noise_mb"] == 41.0 and t2["calibration"]["runs"] == 3
    assert bench.recommended_fail_mb(10.0) == 64.0 and bench.recommended_fail_mb(None) == 64.0


def test_rows_from_result_and_csv() -> None:
    out = {"status": "ok", "vram": {"ctx_mb": 330.0, "start_free_mb": 5100.0, "start_nvml_used_mb": 1500.0,
                                    "attempts": [
                                        {"rung_index": 0, "rung_label": "chunk=588800", "device": "cuda",
                                         "status": "oom", "max_reserved_mb": 2900.0, "nvml_used_peak_mb": 4500.0},
                                        {"rung_index": 1, "rung_label": "chunk=352256", "device": "cuda",
                                         "status": "ok", "max_reserved_mb": 1354.0, "max_allocated_mb": 1100.0,
                                         "nvml_used_peak_mb": 3300.0, "pdh_self_shared_growth_mb": 2.5,
                                         "rtf": 5.8, "slow": False}]},
           "timing": {"load_s": 4.2}}
    res = SimpleNamespace(ok=True, output=out, result={"output": out})
    rows = bench.rows_from_result("sep_msst", "sep-auto-r1", 1, "auto", res, cap_mb=2500.0, ballast_mb=None,
                                  fallback="off", clip_s=240.0)
    assert [r["status"] for r in rows] == ["oom", "ok"]
    assert rows[1]["nvml_delta_peak_mb"] == 1800.0 and rows[1]["load_s"] == 4.2 and rows[1]["ctx_mb"] == 330.0
    assert rows[0]["cap_mb"] == 2500.0
    failed = bench.rows_from_result("sep_msst", "x", 1, "auto", SimpleNamespace(ok=False, output={}, result={
        "error": "RuntimeError: boom"}), cap_mb=None, ballast_mb=None, fallback="unknown", clip_s=1.0)
    assert failed[0]["status"] == "error" and "boom" in failed[0]["error"]
    text = bench.csv_text(rows + failed)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert list(parsed[0]) == list(bench.CSV_COLUMNS) and len(parsed) == 3
    assert parsed[1]["max_reserved_mb"] == "1354.0" and parsed[1]["pdh_self_shared_growth_mb"] == "2.5"


def test_auto_ballast_size() -> None:
    snap = GpuSnapshot(total_mb=6144, used_mb=2000, free_mb=4144, processes=[])
    # rung 0 needs 1800 + 150 (default ctx) = 1950 -> leave 1950 - 400 = 1550 free
    assert bench.auto_ballast_mb("sep", {}, snap) == 4144 - 1550
    with pytest.raises(bench.BenchError):
        bench.auto_ballast_mb("sep", {}, None)


def test_options_validation() -> None:
    bench.BenchOptions().validate()
    for bad in (dict(backends=["gpu"]), dict(rungs="some"), dict(fallback="maybe"), dict(repeat=0),
                dict(cap_mb=-1), dict(ballast_mb="lots")):
        with pytest.raises(bench.BenchError):
            bench.BenchOptions(**bad).validate()


# ------------------------------------------------------------------------------------------- ballast


def test_ballast_launcher_with_dry_worker(tmp_path: Path) -> None:
    """The real launcher (own Job Object, ready.json handshake, stop file) with the torch-free dry mode."""
    b = bench.Ballast(700, tmp_path / "bal", python=paths.CORE_PYTHON, mode="dry", verify=False,
                      ready_timeout_s=60)
    with b:
        assert b.held_mb == 700
        assert json.loads((tmp_path / "bal" / "ready.json").read_text(encoding="utf-8"))["held_mb"] == 700
        assert b.proc is not None and b.proc.poll() is None  # still holding
    assert b.proc is None and b.job is None
    result = json.loads((tmp_path / "bal" / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] and result["output"]["released_because"] == "stop_file"


def test_ballast_nvml_check_aborts(tmp_path: Path) -> None:
    snaps = iter([GpuSnapshot(6144, 1000, 5144), GpuSnapshot(6144, 1100, 5044)])  # rose 100 MB, not 700
    b = bench.Ballast(700, tmp_path / "bal", python=paths.CORE_PYTHON, mode="dry", verify=True,
                      snap_fn=lambda: next(snaps), ready_timeout_s=60)
    with pytest.raises(bench.BenchError, match="밸러스트 확인 실패"):
        b.start()
    assert b.job is None


# --------------------------------------------------------------------------------------- faked full run


class FakeBallast:
    instances: list[FakeBallast] = []

    def __init__(self, mb: int, work_dir: Path) -> None:
        self.mb, self.held_mb, self.stopped = mb, mb, False
        FakeBallast.instances.append(self)

    def start(self) -> int:
        return self.mb

    def stop(self) -> None:
        self.stopped = True


def test_run_bench_faked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("bandscribe.audio")
    mdir = tmp_path / "models"
    mdir.mkdir()
    (mdir / "sw.ckpt").write_bytes(b"w")
    (mdir / "sw.yaml").write_text("model: {}\n", encoding="utf-8")
    cfg = {"sep": {"model_dir": str(mdir), "ckpt": "sw.ckpt", "config": "sw.yaml"}}
    monkeypatch.setattr(bench, "git_sha7", lambda: "abc1234")
    monkeypatch.setattr(bench, "_gpu_identity", lambda: ("RTX 2060", "610.47", [{"pid": 7, "name": "claude.exe",
                                                                               "used_mb": None}]))
    calls: list[dict] = []

    def fake_worker(name: str, req: dict, work_dir: Path, **kw: Any) -> Any:
        calls.append({"name": name, "req": req, **kw})
        label = req["vram"]["force_rung"]
        out = {"status": "ok", "vram": {"ctx_mb": 400.0, "start_nvml_used_mb": 1000.0, "attempts": [
            {"rung_index": 0, "rung_label": label, "device": "cuda", "status": "ok",
             "max_reserved_mb": {"chunk=588800": 1790, "chunk=352256": 1350, "chunk=262144": 1190}[label],
             "nvml_used_peak_mb": 3000.0, "pdh_self_shared_growth_mb": 4.0, "rtf": 6.0}]},
               "timing": {"load_s": 5.0}}
        return SimpleNamespace(ok=True, output=out, result={"ok": True, "output": out}, stderr_path=work_dir / "e",
                               lock_released_dirty=False)

    opts = bench.BenchOptions(clip=None, seconds=3.0, backends=["sep"], rungs="all", fallback="off", repeat=2)
    doc = bench.run_bench(opts, cfg, run_worker=fake_worker)
    run_dir = Path(doc["run_dir"])
    assert run_dir.parent == paths.DATA / "bench" and run_dir.name.endswith("_abc1234")
    assert len(calls) == 6 and {c["req"]["vram"]["precheck"] for c in calls} == {False}
    assert all(c["req"]["vram"]["cpu_allowed"] is False for c in calls)
    assert doc["processes_at_start"][0]["name"] == "claude.exe"
    assert (run_dir / "bench.json").is_file() and (run_dir / "bench.csv").is_file() and (run_dir / "clip.wav").is_file()
    table = json.loads(vram.vram_table_path().read_text(encoding="utf-8"))
    assert table["entries"]["sep_msst"]["chunk=352256"]["reserved_peak_mb"] == 1350.0
    assert table["calibration"]["shared_growth_noise_mb"] == 4.0

    # ballasted run: recorded, never merged
    FakeBallast.instances.clear()
    vram.vram_table_path().unlink()
    opts = bench.BenchOptions(clip=None, seconds=2.0, backends=["sep"], rungs="auto", ballast_mb=1500,
                              fallback="on")

    def auto_worker(name: str, req: dict, work_dir: Path, **kw: Any) -> Any:
        assert req["vram"]["force_rung"] is None and kw["use_lock"] is False
        out = {"status": "ok", "vram": {"attempts": [
            {"rung_index": 0, "rung_label": "chunk=588800", "device": "cuda", "status": "shared_growth",
             "pdh_self_shared_growth_mb": 800.0, "rtf": 0.6, "slow": True},
            {"rung_index": 1, "rung_label": "chunk=352256", "device": "cuda", "status": "ok",
             "max_reserved_mb": 1350, "rtf": 5.9}]}}
        return SimpleNamespace(ok=True, output=out, result={"output": out}, stderr_path=work_dir / "e",
                               lock_released_dirty=False)

    import contextlib

    locks: list[float] = []

    def fake_lock(timeout: float) -> Any:
        locks.append(timeout)
        return contextlib.nullcontext()

    doc = bench.run_bench(opts, cfg, run_worker=auto_worker, ballast_factory=FakeBallast, lock_factory=fake_lock)
    assert len(locks) == 1  # the bench holds the GPU lock around ballast + measured worker
    assert [r["status"] for r in doc["rows"]] == ["shared_growth", "ok"]
    assert all(r["ballast_mb"] == 1500.0 for r in doc["rows"]) and FakeBallast.instances[0].stopped
    table = json.loads(vram.vram_table_path().read_text(encoding="utf-8"))
    assert table["entries"] == {} and "shared_growth_noise_mb" not in table["calibration"]  # fallback on


# ------------------------------------------------------------------------------------------------ CLI


def _cli_app() -> Any:
    import typer

    from bandscribe.commands import bench as bench_cmd

    app = typer.Typer()

    @app.callback()
    def _root() -> None:
        pass

    bench_cmd.register(app)
    return app


def test_cli_bench_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    seen: dict[str, Any] = {}

    def fake_run(opts: Any, cfg: Any, **kw: Any) -> dict:
        seen["opts"] = opts
        return {"gpu": "RTX 2060", "driver": "610.47", "run_dir": "D:/gtab/data/bench/x",
                "processes_at_start": [{"pid": 1, "name": r"C:\x\claude.exe"}],
                "rows": [{"run": "sep-chunk588800-r1", "rung_label": "chunk=588800", "status": "ok",
                          "max_reserved_mb": 1688.0, "nvml_delta_peak_mb": 1777.0, "pdh_self_shared_growth_mb": 2.0,
                          "rtf": 5.48, "load_s": 4.4, "ctx_mb": 89.0}],
                "vram_table": "D:/gtab/data/bench/vram_table.json",
                "calibration": {"shared_growth_noise_mb": 20.0, "recommended_shared_growth_fail_mb": 64.0}}

    monkeypatch.setattr(bench, "run_bench", fake_run)
    monkeypatch.setenv("COLUMNS", "200")  # rich wraps long lines at 80 columns otherwise
    r = CliRunner().invoke(_cli_app(), ["bench", "gpu", "--backend", "sep,amt", "--rungs", "auto", "--ballast-mb",
                                        "auto", "--fallback", "off", "--repeat", "2", "--seconds", "0"])
    assert r.exit_code == 0, r.output
    o = seen["opts"]
    assert o.backends == ["sep", "amt"] and o.rungs == "auto" and o.ballast_mb == "auto" and o.fallback == "off"
    assert o.repeat == 2 and o.seconds is None
    assert "claude.exe" in r.output and "성공" in r.output and "권장 gpu.shared_growth_fail_mb = 64.0" in r.output
    r = CliRunner().invoke(_cli_app(), ["bench", "gpu", "--ballast-mb", "1500"])
    assert r.exit_code == 0 and seen["opts"].ballast_mb == 1500


def test_cli_bench_gpu_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    r = CliRunner().invoke(_cli_app(), ["bench", "gpu", "--backend", "gpu"])
    assert r.exit_code == 1 and "알 수 없는 백엔드" in r.output
    r = CliRunner().invoke(_cli_app(), ["bench", "gpu", "--set", "nosuch.key=1"])
    assert r.exit_code == 2 and "설정 오류" in r.output


def test_amt_fallback_rung_without_cached_weights_is_skipped_before_any_worker(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Only muscriptor-medium is accepted on this PC: the bench must not take the GPU lock for a rung whose
    weights are not in the HF cache (the worker never downloads); run_bench records it as skipped."""
    from bandscribe import paths

    rev = "f32236969308476e01fd3aae67357de5feb05a2d"
    w = tmp_path / "hf" / "hub" / "models--MuScriptor--muscriptor-medium" / "snapshots" / rev / "model.safetensors"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"x")
    monkeypatch.setattr(paths, "HF_HOME", tmp_path / "hf")
    assert bench.amt_labels({}) == ["medium-lean", "small"]
    with pytest.raises(bench.BenchError, match="HF 캐시"):
        bench.amt_request(tmp_path / "missing.wav", tmp_path / "w", {}, force_rung="small", cap_mb=None)
    assert not bench._hf_cached("small") and bench._hf_cached("medium")
