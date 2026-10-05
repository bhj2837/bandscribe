"""bandscribe.vram: needs, budget, wait policy, GPU-stage retries, gpu_run.json; plus the doctor's M2 checks (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bandscribe import paths, vram
from bandscribe.vram import GpuSnapshot, RungNeed

SEP_LABELS = ["chunk=588800", "chunk=352256", "chunk=262144"]


@pytest.fixture(autouse=True)
def _tmp_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Every test gets its own DATA (VRAM table, models lock): never read or write the real ones."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(paths, "DATA", data)
    monkeypatch.setattr(paths, "BENCH", data / "bench")
    monkeypatch.setattr(paths, "VRAM_TABLE", data / "bench" / "vram_table.json")
    monkeypatch.setattr(paths, "MODELS", data / "models")
    monkeypatch.setattr(paths, "MODELS_LOCK", data / "models" / "models.lock.json")
    return data


def _cfg(**gpu: Any) -> dict[str, Any]:
    base = {"insufficient_vram_policy": "wait", "vram_margin_gb": 0.5, "wait_poll_s": 10, "wait_max_s": 60,
            "max_worker_retries": 2, "worker_headroom_mb": 256, "ctx_mb_default": 500,
            "cpu_backends": ["beats_beatthis"], "worker_timeout_s": 100, "lock_timeout_s": 100}
    base.update(gpu)
    return {"gpu": base}


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _snaps(*free_mb: float) -> Any:
    seq = list(free_mb)

    def fn() -> GpuSnapshot:
        f = seq.pop(0) if len(seq) > 1 else seq[0]
        return GpuSnapshot(total_mb=6144, used_mb=6144 - f, free_mb=f,
                           processes=[{"pid": 1, "name": r"C:\Games\League of Legends.exe", "used_mb": None}])
    return fn


# ----------------------------------------------------------------------------------------- needs


def test_orchestrator_and_worker_needs_use_their_own_terms() -> None:
    n = RungNeed(reserved_peak_mb=1800, ctx_mb=500)
    assert vram.orchestrator_need_mb(n) == 2300  # reserved + context (vs NVML free - margin)
    assert vram.worker_need_mb(n, 256) == 2056  # reserved + headroom (vs mem_get_info free); ctx already exists


def test_choose_rung_uses_orchestrator_need_only() -> None:
    needs = {"a": RungNeed(1800, 500), "b": RungNeed(1400, 500), "c": RungNeed(1200, 500)}
    assert vram.choose_rung(["a", "b", "c"], 2300, needs) == 0
    assert vram.choose_rung(["a", "b", "c"], 2299, needs) == 1
    # 2100 would fit "a" by the worker's measure (1800 + 256) but not by the orchestrator's (+ ctx 500).
    assert vram.choose_rung(["a", "b", "c"], 2100, needs) == 1
    assert vram.choose_rung(["a", "b", "c"], 1000, needs) is None
    assert vram.choose_rung(["x", "a"], 5000, needs) == 1  # unknown labels never fit


def test_load_table_defaults_and_measured_entries(tmp_path: Path) -> None:
    t = vram.load_table(ctx_default_mb=500)
    assert t["sep_msst"]["chunk=588800"] == RungNeed(1800, 500)
    assert t["amt_muscriptor"]["medium"] == RungNeed(2400, 500)
    assert t["beats_beatthis"]["final0"] == RungNeed(500, 500)
    p = vram.vram_table_path()
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"format": "bandscribe.vram_table/1", "entries": {
        "sep_msst": {"chunk=588800": {"reserved_peak_mb": 1796.0, "ctx_mb": 330.0, "rtf": 6.1},
                     "chunk=352256": {"reserved_peak_mb": 1354.0}},  # no ctx -> default
        "new_backend": {"x": {"reserved_peak_mb": 10.0, "ctx_mb": 5}}}}), encoding="utf-8")
    t = vram.load_table(ctx_default_mb=450)
    assert t["sep_msst"]["chunk=588800"] == RungNeed(1796.0, 330.0)
    assert t["sep_msst"]["chunk=352256"] == RungNeed(1354.0, 450.0)
    assert t["sep_msst"]["chunk=262144"] == RungNeed(1200.0, 450.0)  # per-entry fallback
    assert t["new_backend"]["x"] == RungNeed(10.0, 5.0)
    assert vram.rtf_refs("sep_msst", SEP_LABELS) == {"chunk=588800": 6.1, "chunk=352256": None,
                                                     "chunk=262144": None}


def test_corrupt_table_is_ignored() -> None:
    p = vram.vram_table_path()
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    assert vram.load_table(ctx_default_mb=500)["sep_msst"]["chunk=588800"].reserved_peak_mb == 1800


def test_cfg_get_with_real_config_and_dicts() -> None:
    from bandscribe.config import Config

    cfg = Config()
    assert vram.cfg_get(cfg, "gpu.vram_margin_gb") == pytest.approx(0.7)
    assert vram.cfg_get(cfg, "gpu.wait_poll_s") == 30.0 or vram.cfg_get(cfg, "gpu.wait_poll_s") > 0
    assert vram.cfg_get({"gpu": {"wait_poll_s": 5}}, "gpu.wait_poll_s") == 5
    assert vram.cfg_get({}, "gpu.max_worker_retries") == 3
    assert vram.cfg_get(None, "gpu.cpu_backends") == ["beats_beatthis"]
    assert vram.cfg_get({"gpu": {}}, "nope.x", 7) == 7
    # A real Config is read strictly: a misspelled key is a bug, never a silent default.
    with pytest.raises(KeyError):
        vram.cfg_get(cfg, "gpu.wait_pol_s")
    with pytest.raises(KeyError):
        vram.cfg_get(cfg, "gpu.wait_pol_s", 30.0)
    assert vram.cfg_get(Config.model_validate({"gpu": {"wait_poll_s": 5}}), "gpu.wait_poll_s") == 5


def test_default_mirrors_equal_the_config_defaults() -> None:
    """GPU_DEFAULTS / SEP_DEFAULTS stand in for a missing Config (None, partial dicts): they must not drift."""
    from bandscribe.config import Config
    from bandscribe.sep.run import SEP_DEFAULTS

    cfg = Config()
    for mirror in (vram.GPU_DEFAULTS, SEP_DEFAULTS):
        for key, value in mirror.items():
            assert cfg.get(key) == value, key


# ------------------------------------------------------------------------------------ wait policy


def test_wait_returns_first_fitting_rung_after_apps_close() -> None:
    clk = Clock()
    msgs: list[str] = []
    # margin 0.5 GB = 512 MB; rung 0 needs 2300 -> NVML free 2812 fits.
    idx, waited = vram.wait_for_budget("sep_msst", SEP_LABELS, _cfg(), clock=clk, sleep=clk.sleep,
                                       notify=msgs.append, snap_fn=_snaps(1000, 1000, 2900))
    assert idx == 0 and waited == 20.0 and clk.sleeps == [10, 10]
    assert "VRAM 부족" in msgs[0] and "League of Legends.exe" in msgs[0] and "대기 중" in msgs[0]
    assert "이어서" in msgs[-1]


def test_wait_picks_lower_rung_that_fits() -> None:
    clk = Clock()
    idx, waited = vram.wait_for_budget("sep_msst", SEP_LABELS, _cfg(), clock=clk, sleep=clk.sleep,
                                       snap_fn=_snaps(2300))  # budget 1788 -> 1200 + 500 fits
    assert (idx, waited) == (2, 0.0)


def test_wait_times_out_with_korean_message() -> None:
    clk = Clock()
    with pytest.raises(vram.VramInsufficient) as ei:
        vram.wait_for_budget("sep_msst", SEP_LABELS, _cfg(wait_max_s=35), clock=clk, sleep=clk.sleep,
                             snap_fn=_snaps(800))
    assert clk.t == pytest.approx(35.0)
    msg = str(ei.value)
    assert "VRAM 부족" in msg and "필요 약 1.7 GB" in msg and "League of Legends.exe" in msg


def test_error_policy_raises_at_once() -> None:
    clk = Clock()
    with pytest.raises(vram.VramInsufficient):
        vram.wait_for_budget("sep_msst", SEP_LABELS, _cfg(insufficient_vram_policy="error"), clock=clk,
                             sleep=clk.sleep, snap_fn=_snaps(800))
    assert clk.sleeps == []


def test_cpu_policy_honours_cpu_backends() -> None:
    clk = Clock()
    cfg = _cfg(insufficient_vram_policy="cpu", wait_max_s=20)
    assert vram.wait_for_budget("beats_beatthis", ["final0"], cfg, clock=clk, sleep=clk.sleep,
                                snap_fn=_snaps(100)) == (None, 0.0)
    assert vram.cpu_allowed("beats_beatthis", cfg) and not vram.cpu_allowed("sep_msst", cfg)
    # sep_msst is not in cpu_backends: behaves like "wait" (never a silent 23-minute CPU separation)
    with pytest.raises(vram.VramInsufficient):
        vram.wait_for_budget("sep_msst", SEP_LABELS, cfg, clock=clk, sleep=clk.sleep, snap_fn=_snaps(100))
    assert clk.sleeps  # it waited first
    assert not vram.cpu_allowed("beats_beatthis", _cfg())  # policy wait: no CPU rung at all


def test_no_nvml_defers_to_worker() -> None:
    assert vram.wait_for_budget("sep_msst", SEP_LABELS, _cfg(), snap_fn=lambda: None) == (0, 0.0)


# ------------------------------------------------------------------------------------ run_gpu_stage


class FakeRunner:
    def __init__(self, outputs: list[dict], ok: bool = True) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []
        self.ok = ok

    def __call__(self, name: str, request: dict, work_dir: Path, **kw: Any) -> Any:
        self.calls.append({"name": name, "request": request, "work_dir": work_dir, **kw})
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        return SimpleNamespace(ok=self.ok, output=out, result={"output": out, "error": None if self.ok else "boom"},
                               lock_released_dirty=False, stderr_path=work_dir / "stderr.log", timed_out=False)


INSUFF = {"status": "insufficient_vram", "vram": {"start_free_mb": 900.0}}
OK = {"status": "ok", "vram": {"rung_index": 0, "rung_label": "chunk=588800"}, "backend": {"device_used": "cuda"}}


def test_run_gpu_stage_retries_insufficient_then_succeeds(tmp_path: Path) -> None:
    clk = Clock()
    runner = FakeRunner([INSUFF, INSUFF, OK])
    msgs: list[str] = []
    res = vram.run_gpu_stage("sep_msst", SEP_LABELS, {"x": 1}, _cfg(), tmp_path, notify=msgs.append,
                             run_worker=runner, clock=clk, sleep=clk.sleep, snap_fn=_snaps(5000))
    assert len(runner.calls) == 3 and res.output["status"] == "ok"
    assert res.waited_s == 20.0 and clk.sleeps == [10, 10]
    v = runner.calls[0]["request"]["vram"]
    assert v["policy"] == "wait" and v["cpu_allowed"] is False and v["worker_headroom_mb"] == 256
    assert v["need_table"]["chunk=588800"] == {"reserved_peak_mb": 1800.0, "ctx_mb": 500.0}
    assert runner.calls[0]["timeout_s"] == 100 and runner.calls[0]["lock_timeout_s"] == 100
    assert sum("다시 시도" in m for m in msgs) == 2


def test_run_gpu_stage_retry_bound(tmp_path: Path) -> None:
    clk = Clock()
    runner = FakeRunner([INSUFF])
    with pytest.raises(vram.VramInsufficient) as ei:
        vram.run_gpu_stage("sep_msst", SEP_LABELS, {}, _cfg(max_worker_retries=2), tmp_path, run_worker=runner,
                           clock=clk, sleep=clk.sleep, snap_fn=_snaps(5000))
    assert len(runner.calls) == 3  # first try + 2 retries
    assert "3번" in str(ei.value)


def test_run_gpu_stage_error_policy_no_retry(tmp_path: Path) -> None:
    runner = FakeRunner([INSUFF])
    with pytest.raises(vram.VramInsufficient):
        vram.run_gpu_stage("sep_msst", SEP_LABELS, {}, _cfg(insufficient_vram_policy="error"), tmp_path,
                           run_worker=runner, snap_fn=_snaps(5000))
    assert len(runner.calls) == 1


def test_run_gpu_stage_failed_worker_raises(tmp_path: Path) -> None:
    runner = FakeRunner([{}], ok=False)
    with pytest.raises(vram.GpuWorkerFailed) as ei:
        vram.run_gpu_stage("sep_msst", SEP_LABELS, {}, _cfg(), tmp_path, run_worker=runner, snap_fn=_snaps(5000))
    assert "stderr.log" in str(ei.value) and "boom" in str(ei.value)
    res = vram.run_gpu_stage("sep_msst", SEP_LABELS, {}, _cfg(), tmp_path, run_worker=runner,
                             snap_fn=_snaps(5000), check=False)
    assert res.ok is False


def test_run_gpu_stage_cpu_policy_runs_on_cpu(tmp_path: Path) -> None:
    runner = FakeRunner([OK])
    vram.run_gpu_stage("beats_beatthis", ["final0"], {}, _cfg(insufficient_vram_policy="cpu"), tmp_path,
                       run_worker=runner, snap_fn=_snaps(100))
    req = runner.calls[0]["request"]
    assert req["device"] == "cpu" and req["vram"]["cpu_allowed"] is True


def test_run_gpu_stage_passes_the_chosen_lower_rung_to_the_worker(tmp_path: Path) -> None:
    runner = FakeRunner([OK])
    # NVML free 2300 -> budget 1788: rung 0 (2300) and rung 1 (1900) do not fit, rung 2 (1700) does (b2)
    vram.run_gpu_stage("sep_msst", SEP_LABELS, {}, _cfg(), tmp_path, run_worker=runner, snap_fn=_snaps(2300))
    assert runner.calls[0]["request"]["vram"]["start_rung"] == "chunk=262144"
    runner = FakeRunner([OK])
    vram.run_gpu_stage("sep_msst", SEP_LABELS, {}, _cfg(), tmp_path, run_worker=runner, snap_fn=_snaps(5000))
    assert "start_rung" not in runner.calls[0]["request"]["vram"]  # top rung fits: the worker starts at rung 0


def test_run_gpu_stage_force_rung_checks_only_that_rung(tmp_path: Path) -> None:
    clk = Clock()
    runner = FakeRunner([OK])
    req = {"vram": vram.build_vram_request("sep_msst", SEP_LABELS, _cfg(), force_rung="chunk=262144")}
    # 2300 free -> budget 1788: rung 0 would not fit, the forced rung 2 does -> no wait
    vram.run_gpu_stage("sep_msst", SEP_LABELS, req, _cfg(), tmp_path, run_worker=runner, clock=clk,
                       sleep=clk.sleep, snap_fn=_snaps(2300))
    assert clk.sleeps == [] and runner.calls[0]["request"]["vram"]["force_rung"] == "chunk=262144"


# ------------------------------------------------------------------------------------- gpu_run.json


@pytest.mark.parametrize("label,device,index,degraded", [
    ("chunk=588800", "cuda", 0, False),
    ("chunk=352256", "cuda", 1, True),
    ("chunk=262144@cpu", "cpu", 3, True),
])
def test_gpu_run_round_trip(tmp_path: Path, label: str, device: str, index: int, degraded: bool) -> None:
    out = {"vram": {"rung_label": label, "rung_index": index, "attempts": []}, "backend": {"device_used": device}}
    doc = vram.write_gpu_run(tmp_path, "sep_msst", SEP_LABELS, out, 12.5)
    back = vram.read_gpu_run(tmp_path)
    assert back == doc
    assert back["format"] == "bandscribe.gpu_run/1" and back["ladder"] == SEP_LABELS
    assert back["rung"] == {"index": index, "label": label, "device": device}
    assert back["degraded"] is degraded and back["waited_s"] == 12.5


def test_read_gpu_run_missing_or_foreign(tmp_path: Path) -> None:
    assert vram.read_gpu_run(tmp_path) is None
    (tmp_path / "gpu_run.json").write_text('{"format": "other"}', encoding="utf-8")
    assert vram.read_gpu_run(tmp_path) is None


# ------------------------------------------------------------------------------- doctor M2 checks


def _doctor_checks() -> dict[str, Any]:
    from bandscribe import doctor

    return {c.name: c for c in doctor._check_models(None)}


def test_doctor_vram_table_absent_is_informational() -> None:
    c = _doctor_checks()["gpu.vram_table"]
    assert c.ok and not c.required and "bandscribe bench gpu" in c.detail


def test_doctor_vram_table_present_and_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date

    from bandscribe import doctor

    p = vram.vram_table_path()
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"format": "bandscribe.vram_table/1", "entries": {"sep_msst": {
        "chunk=588800": {"reserved_peak_mb": 1796, "measured_utc": "2026-09-30T10:00:00+00:00"}}},
        "calibration": {"shared_growth_noise_mb": 12.0}}), encoding="utf-8")
    monkeypatch.setattr(doctor, "_today", lambda: date(2026, 10, 2))
    c = _doctor_checks()["gpu.vram_table"]
    assert c.ok and "1칸" in c.detail and "2일 전" in c.detail and "12.0 MB" in c.detail
    monkeypatch.setattr(doctor, "_today", lambda: date(2027, 3, 1))
    c = _doctor_checks()["gpu.vram_table"]
    assert not c.ok and c.status == "warn" and "다시 재세요" in c.detail


def test_doctor_models_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    c = _doctor_checks()["models.present"]
    assert not c.ok and c.status == "warn"  # no lock file
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "sw.ckpt").write_bytes(b"x")
    lock = paths.DATA / "models" / "models.lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"format": "bandscribe.models/1", "models": {
        "bs_roformer_sw.ckpt": {"path": "m/sw.ckpt"}, "gone": {"path": "m/gone.bin"}}}), encoding="utf-8")
    c = _doctor_checks()["models.present"]
    assert not c.ok and "gone" in c.detail and "bandscribe models fetch beat_this.final0" in c.detail
    (tmp_path / "m" / "final0.ckpt").write_bytes(b"x")
    lock.write_text(json.dumps({"format": "bandscribe.models/1", "models": {
        "bs_roformer_sw.ckpt": {"path": "m/sw.ckpt"}, "beat_this.final0": {"path": "m/final0.ckpt"}}}),
        encoding="utf-8")
    c = _doctor_checks()["models.present"]
    assert c.ok and "모두 있음" in c.detail


# ------------------------------------------------------------- overall wait deadline, periodic notices


def test_long_wait_says_so_every_few_minutes() -> None:
    """Review 2026-10-05: the reason was announced once, then a wait of up to an hour said nothing."""
    clk = Clock()
    msgs: list[str] = []
    with pytest.raises(vram.VramInsufficient):
        vram.wait_for_budget("sep_msst", SEP_LABELS, _cfg(wait_poll_s=30, wait_max_s=1000), clock=clk,
                             sleep=clk.sleep, notify=msgs.append, snap_fn=_snaps(800))
    assert "대기 중" in msgs[0]
    still = [m for m in msgs if m.startswith("아직 VRAM 여유를 기다리는 중입니다(sep_msst)")]
    assert len(still) == 3  # at 5, 10 and 15 minutes of a 16.7-minute wait
    assert "5분째" in still[0] and "10분째" in still[1] and "League of Legends.exe" in still[0]


def test_all_waiting_of_a_stage_is_bounded_by_wait_total_max_s(tmp_path: Path) -> None:
    """Each pre-check used to get the full wait_max_s again after every insufficient_vram retry (up to ~4 h)."""
    clk = Clock()
    # pre-check 1 waits 40 s, the worker answers insufficient_vram, 10 s pause, pre-check 2 gets only the
    # remaining 50 s of the 100 s total (not wait_max_s = 80)
    snaps = _snaps(1000, 1000, 1000, 1000, 5000, *([1000] * 50))
    runner = FakeRunner([INSUFF])
    with pytest.raises(vram.VramInsufficient) as ei:
        vram.run_gpu_stage("sep_msst", SEP_LABELS, {}, _cfg(wait_max_s=80, wait_total_max_s=100, max_worker_retries=5),
                           tmp_path, run_worker=runner, clock=clk, sleep=clk.sleep, snap_fn=snaps)
    assert len(runner.calls) == 1 and clk.t == pytest.approx(100.0)
    assert "gpu.wait_total_max_s" in str(ei.value) and "2분" in str(ei.value)
    # retries alone: the pauses stop before they pass the total either
    clk2 = Clock()
    runner2 = FakeRunner([INSUFF])
    with pytest.raises(vram.VramInsufficient) as ei:
        vram.run_gpu_stage("sep_msst", SEP_LABELS, {}, _cfg(wait_total_max_s=25, max_worker_retries=10), tmp_path,
                           run_worker=runner2, clock=clk2, sleep=clk2.sleep, snap_fn=_snaps(5000))
    assert len(runner2.calls) == 3 and clk2.sleeps == [10, 10]  # a third pause would end at 30 s > 25 s
    assert "wait_total_max_s" in str(ei.value)
