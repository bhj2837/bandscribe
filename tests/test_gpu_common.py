"""bandscribe.workers.gpu_common (ladder, watchdog, bf16 refusal) and the additive ResourceSampler marks.

Most tests run in the core env with fakes: the ladder's torch calls are behind seams (``mem_info``,
``cuda_available``) and an OOM is simulated with the RuntimeError text CUDA uses. The ``gpuenv`` tests run the
same code in the gpu venv (real ``torch.OutOfMemoryError``); ``gpu`` tests use the real device (b4 cap).
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

from bandscribe import paths
from bandscribe.workers import base, gpu_common
from bandscribe.workers.gpu_common import SharedGrowth, Watchdog, run_ladder

BANDSCRIBE_ROOT = Path(__file__).resolve().parents[1]


class FakeSampler:
    """snapshot() returns the next scripted reading (the last one repeats); mark() just records."""

    def __init__(self, readings: list[dict[str, Any]]) -> None:
        self.readings = list(readings)
        self.marks: list[str] = []

    def snapshot(self) -> dict[str, Any]:
        return dict(self.readings.pop(0) if len(self.readings) > 1 else self.readings[0])

    def mark(self, label: str) -> None:
        self.marks.append(label)


def _r(self_shared: float | None, adapter: float | None = 100.0, nvml: float = 3000.0) -> dict[str, Any]:
    return {"pdh_self_shared_mb": self_shared, "pdh_adapter_shared_mb": adapter, "nvml_used_mb": nvml}


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# -------------------------------------------------------------------------------------------- watchdog


def test_watchdog_growth_is_measured_from_arm() -> None:
    # 300 MB of shared usage before arm() (weights staged through shared memory) must not count.
    s = FakeSampler([_r(300.0), _r(310.0), _r(330.0), _r(380.0)])
    wd = Watchdog(s, shared_growth_fail_mb=64, rtf_ref=None, rtf_fail_ratio=0.5, audio_s_total=100)
    wd.arm()
    wd.check(10)
    wd.check(20)
    assert wd.max_growth_mb == 30.0
    with pytest.raises(SharedGrowth):
        wd.check(30)  # 380 - 300 = 80 > 64


def test_watchdog_ignores_adapter_growth() -> None:
    s = FakeSampler([_r(50.0, adapter=100.0), _r(52.0, adapter=2100.0)])  # Chrome moved 2 GB, we did not
    wd = Watchdog(s, shared_growth_fail_mb=64, rtf_ref=None, rtf_fail_ratio=0.5, audio_s_total=100)
    wd.arm()
    wd.check(50)
    assert wd.adapter_delta_mb == 2000.0 and wd.max_growth_mb == 2.0


def test_watchdog_slow_is_only_a_warning_with_pdh() -> None:
    clk = Clock()
    s = FakeSampler([_r(50.0)])
    wd = Watchdog(s, shared_growth_fail_mb=64, rtf_ref=6.0, rtf_fail_ratio=0.5, audio_s_total=100, clock=clk)
    wd.arm()
    clk.t = 20.0
    wd.check(5)  # first chunk: warm-up, only the rate baseline
    assert not wd.slow and wd.rtf is None
    clk.t = 25.0
    wd.check(8)  # 8 % done: too early to judge (rtf 0.6)
    assert not wd.slow and wd.rtf == 0.6
    clk.t = 40.0
    wd.check(20)  # rtf (20 - 5) / (40 - 20) = 0.75 < 3.0, 20 % done
    assert wd.slow and wd.rtf == 0.75 and not wd.degraded_detection


def test_watchdog_rtf_excludes_the_first_chunk() -> None:
    """A slow first chunk (kernel selection) must not make a normal rung look slow."""
    clk = Clock()
    wd = Watchdog(FakeSampler([_r(50.0)]), shared_growth_fail_mb=64, rtf_ref=6.0, rtf_fail_ratio=0.5,
                  audio_s_total=100, clock=clk)
    wd.arm()
    clk.t = 8.0
    wd.check(6.7)  # warm-up: 0.8x
    for k in range(2, 6):
        clk.t = 8.0 + (k - 1) * 1.1  # then 6.7 s of audio per 1.1 s = 6.1x
        wd.check(6.7 * k)
    assert not wd.slow and wd.rtf == pytest.approx(6.091, abs=0.01)


def test_watchdog_slow_without_pdh_counts_as_spill() -> None:
    clk = Clock()
    s = FakeSampler([_r(None, adapter=None)])
    wd = Watchdog(s, shared_growth_fail_mb=64, rtf_ref=6.0, rtf_fail_ratio=0.5, audio_s_total=100, clock=clk)
    wd.arm()
    clk.t = 2.0
    wd.check(1)  # rate baseline
    clk.t = 40.0
    with pytest.raises(SharedGrowth, match="degraded detection"):
        wd.check(20)
    assert wd.degraded_detection


def test_watchdog_without_reference_never_slow() -> None:
    clk = Clock()
    wd = Watchdog(FakeSampler([_r(None)]), shared_growth_fail_mb=64, rtf_ref=None, rtf_fail_ratio=0.5,
                  audio_s_total=100, clock=clk)
    wd.arm()
    clk.t = 1000.0
    wd.check(50)
    assert not wd.slow


# ---------------------------------------------------------------------------------------------- ladder


RUNGS = [{"label": "chunk=588800", "device": "cuda", "params": {}},
         {"label": "chunk=352256", "device": "cuda", "params": {}},
         {"label": "chunk=262144", "device": "cuda", "params": {}},
         {"label": "chunk=262144@cpu", "device": "cpu", "params": {}}]


def _req(**vram: Any) -> dict[str, Any]:
    v = {"precheck": True, "worker_headroom_mb": 256, "cpu_allowed": False, "shared_growth_fail_mb": 64,
         "rtf_fail_ratio": 0.5, "rtf_ref": {},
         "need_table": {"chunk=588800": {"reserved_peak_mb": 1800}, "chunk=352256": {"reserved_peak_mb": 1400},
                        "chunk=262144": {"reserved_peak_mb": 1200}}}
    v.update(vram)
    return {"device": "auto", "vram": v}


def _ladder(req: dict, fail: dict[str, BaseException], free_mb: float = 5000.0) -> tuple[Any, list]:
    seen: list[str] = []

    def attempt(rung: dict, wd: Watchdog) -> str:
        seen.append(rung["label"])
        wd.arm()
        if rung["label"] in fail:
            raise fail[rung["label"]]
        wd.check(10.0)
        return rung["label"]

    res, attempts = run_ladder(RUNGS, attempt, sampler=FakeSampler([_r(10.0)]), req=req, audio_s_total=10.0,
                               mem_info=lambda: (free_mb, 6144.0), cuda_available=lambda: True)
    return res, attempts


def test_ladder_steps_down_on_oom() -> None:
    oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    res, attempts = _ladder(_req(), {"chunk=588800": oom})
    assert res == "chunk=352256"
    assert [(a.rung_label, a.status) for a in attempts] == [("chunk=588800", "oom"), ("chunk=352256", "ok")]
    assert attempts[0].rung_index == 0 and attempts[1].rung_index == 1 and attempts[1].rtf is not None


def test_ladder_steps_down_on_shared_growth() -> None:
    res, attempts = _ladder(_req(), {"chunk=588800": SharedGrowth("spill")})
    assert res == "chunk=352256" and attempts[0].status == "shared_growth"


def test_all_gpu_rungs_fail_and_no_cpu_means_insufficient() -> None:
    oom = RuntimeError("CUDA out of memory")
    res, attempts = _ladder(_req(), {r["label"]: oom for r in RUNGS[:3]})
    assert res is None and [a.status for a in attempts] == ["oom", "oom", "oom"]  # CPU rung never tried


def test_cpu_rung_only_when_allowed() -> None:
    oom = RuntimeError("CUDA out of memory")
    res, attempts = _ladder(_req(cpu_allowed=True), {r["label"]: oom for r in RUNGS[:3]})
    assert res == "chunk=262144@cpu" and attempts[-1].device == "cpu" and attempts[-1].rung_index == 3


def test_precheck_skips_rungs_over_worker_need() -> None:
    # mem_get_info free 1600: 1800+256 and 1400+256 do not fit, 1200+256 does
    res, attempts = _ladder(_req(), {}, free_mb=1600.0)
    assert res == "chunk=262144"
    assert [a.status for a in attempts] == ["skipped_budget", "skipped_budget", "ok"]
    assert attempts[0].need_mb == 2056.0 and attempts[0].free_mb == 1600.0


def test_precheck_off_attempts_anyway() -> None:
    res, attempts = _ladder(_req(precheck=False), {}, free_mb=100.0)
    assert res == "chunk=588800" and attempts[0].status == "ok"


def test_start_rung_from_orchestrator_skips_larger_rungs() -> None:
    # mem_get_info says everything fits (WDDM), but the NVML pre-check chose rung 1: rung 0 is not attempted.
    oom = RuntimeError("CUDA out of memory")
    res, attempts = _ladder(_req(start_rung="chunk=352256"), {"chunk=352256": oom})
    assert res == "chunk=262144"
    assert [(a.rung_label, a.status) for a in attempts] == [
        ("chunk=588800", "skipped_budget"), ("chunk=352256", "oom"), ("chunk=262144", "ok")]
    assert "orchestrator" in (attempts[0].error or "")
    # unknown label (another worker's naming): ignored, the ladder starts at rung 0
    res, attempts = _ladder(_req(start_rung="medium"), {})
    assert res == "chunk=588800" and len(attempts) == 1
    # the bench (precheck off) and force_rung never skip
    res, _ = _ladder(_req(start_rung="chunk=262144", precheck=False), {})
    assert res == "chunk=588800"
    res, _ = _ladder(_req(start_rung="chunk=262144", force_rung="chunk=352256"), {})
    assert res == "chunk=352256"


def test_hide_cuda_only_for_cpu_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # recorded, so teardown restores the original state
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert gpu_common.hide_cuda_for_cpu_request({"device": "auto"}) is False
    assert "CUDA_VISIBLE_DEVICES" not in __import__("os").environ
    assert gpu_common.hide_cuda_for_cpu_request({"device": "cpu"}) is True  # torch not imported in core
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == ""


def test_force_rung_does_not_step_down() -> None:
    oom = RuntimeError("CUDA out of memory")
    res, attempts = _ladder(_req(force_rung="chunk=352256"), {"chunk=352256": oom})
    assert res is None and [(a.rung_label, a.status) for a in attempts] == [("chunk=352256", "oom")]
    with pytest.raises(ValueError):
        _ladder(_req(force_rung="chunk=1"), {})
    with pytest.raises(ValueError):
        _ladder(_req(force_rung="chunk=262144@cpu"), {})  # CPU rung needs cpu_allowed


def test_other_exceptions_propagate() -> None:
    with pytest.raises(KeyError):
        _ladder(_req(), {"chunk=588800": KeyError("bug")})


def test_no_cuda_and_no_cpu_is_an_error() -> None:
    with pytest.raises(gpu_common.CudaUnavailable):
        run_ladder(RUNGS, lambda r, wd: None, sampler=None, req=_req(), cuda_available=lambda: False,
                   mem_info=lambda: (0.0, 0.0))


def test_device_cpu_request_uses_cpu_rungs_only() -> None:
    req = _req()
    req["device"] = "cpu"
    res, attempts = _ladder(req, {})
    assert res == "chunk=262144@cpu" and len(attempts) == 1


# ---------------------------------------------------------------------------------------------- bf16


@pytest.mark.parametrize("req", [{"dtype": "bf16"}, {"dtype": "bfloat16"}, {"model": {"dtype": "torch.bfloat16"}},
                                 {"views": [{"autocast_dtype": "BF16"}]}])
def test_refuse_bf16(req: dict) -> None:
    with pytest.raises(ValueError, match="bfloat16"):
        gpu_common.refuse_bf16(req)


def test_refuse_bf16_env(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu_common.refuse_bf16({"dtype": "upstream", "x": "bf16 is only a word here"})
    monkeypatch.setenv("BANDSCRIBE_AUTOCAST_DTYPE", "bfloat16")
    with pytest.raises(ValueError):
        gpu_common.refuse_bf16({"dtype": "fp16"})


# ------------------------------------------------------------------------------ sampler marks (base.py)


class FakePdh:
    def __init__(self, seq: list[dict]) -> None:
        self.seq = iter(seq)

    def sample(self) -> dict:
        return next(self.seq)


def _pdh(me: int, shared_mb: int, adapter_mb: int) -> dict:
    return {"process": {me: {"dedicated": 500 << 20, "shared": shared_mb << 20}},
            "adapter": {"dedicated": 0, "shared": adapter_mb << 20}}


def test_sampler_mark_and_growth_after_model_loaded() -> None:
    me = 4242
    s = base.ResourceSampler(pid=me)
    pdh = FakePdh([_pdh(me, 10, 100), _pdh(me, 400, 150),  # staging during upload
                   _pdh(me, 380, 900),  # model resident: mark here
                   _pdh(me, 420, 2000), _pdh(me, 390, 300)])
    s._sample(None, pdh)
    s._sample(None, pdh)
    s._sample(None, pdh)
    snap = s.snapshot()
    assert snap["pdh_self_shared_mb"] == 380.0 and snap["pdh_self_shared_peak_mb"] == 400.0
    m = s.mark(base.MODEL_LOADED)
    assert m["pdh_self_shared_mb"] == 380.0 and m["pdh_adapter_shared_mb"] == 900.0
    s._sample(None, pdh)
    s._sample(None, pdh)
    st = s.stats()
    assert st["pdh_self_shared_growth_mb"] == 40.0  # 420 peak after mark - 380 at mark (not 400 - 10)
    assert st["pdh_self_shared_delta_mb"] == 410.0  # info: peak - start, includes staging
    assert st["pdh_adapter_shared_delta_mb"] == 1900.0  # info only
    assert st["shared_growth"] is False and st["marks"][base.MODEL_LOADED]["pdh_self_shared_mb"] == 380.0
    s.shared_growth_fail_mb = 30.0
    assert s.stats()["shared_growth"] is True


def test_sampler_without_pdh_has_null_growth() -> None:
    s = base.ResourceSampler(pid=1)
    s.mark(base.MODEL_LOADED)
    st = s.stats()
    assert st["pdh_self_shared_growth_mb"] is None and st["shared_growth"] is False


def test_sampler_threshold_from_request() -> None:
    s = base.ResourceSampler(pid=1)
    base._configure_sampler(s, {"vram": {"shared_growth_fail_mb": 128}})
    assert s.shared_growth_fail_mb == 128.0
    base._configure_sampler(s, {"vram": {}})
    assert s.shared_growth_fail_mb == 128.0


def test_worker_result_carries_new_sysmon_fields(tmp_path: Path) -> None:
    from bandscribe.gpu import run_worker

    r = run_worker("selftest", {"mode": "echo", "vram": {"shared_growth_fail_mb": 99}}, tmp_path / "w",
                   python=paths.CORE_PYTHON, use_lock=False, timeout_s=60)
    assert r.ok, r.stderr_path.read_text(encoding="utf-8")
    sm = r.result["sysmon"]
    assert {"marks", "pdh_self_shared_growth_mb", "pdh_self_shared_delta_mb", "pdh_adapter_shared_delta_mb",
            "shared_growth"} <= set(sm)
    assert sm["shared_growth_fail_mb"] == 99.0 and sm["shared_growth"] is False


# --------------------------------------------------------------------------------- gpu env (real torch)


def _gpu_env_run(code: str, timeout: float = 300) -> dict:
    env = paths.child_env({"PYTHONPATH": str(BANDSCRIBE_ROOT)})
    proc = subprocess.run([str(paths.GPU_PYTHON), "-c", textwrap.dedent(code)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env, cwd=str(BANDSCRIBE_ROOT))
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.gpuenv
def test_ladder_with_real_torch_oom_class() -> None:
    out = _gpu_env_run("""
        import json, torch
        from bandscribe.workers import gpu_common as g
        rungs = [{"label": "a", "device": "cuda", "params": {}}, {"label": "b", "device": "cuda", "params": {}}]
        def attempt(r, wd):
            wd.arm()
            if r["label"] == "a":
                raise torch.OutOfMemoryError("fake OOM")
            return "b"
        res, att = g.run_ladder(rungs, attempt, sampler=None, req={"vram": {}}, mem_info=lambda: (5000.0, 6144.0),
                                cuda_available=lambda: True)
        print(json.dumps({"res": res, "status": [a.status for a in att]}))
    """)
    assert out == {"res": "b", "status": ["oom", "ok"]}


@pytest.mark.gpu
@pytest.mark.slow
def test_cap_makes_large_allocation_oom_then_smaller_fits() -> None:
    """b4 in miniature: under a 700 MB allocator cap a 1 GB rung OOMs and a 256 MB rung succeeds."""
    out = _gpu_env_run("""
        import json, torch
        from filelock import FileLock
        from bandscribe import paths
        from bandscribe.workers import gpu_common as g
        with FileLock(str(paths.GPU_LOCK), timeout=600):
            g.apply_cap({"vram": {"cap_mb": 700}})
            sizes = {"big": 1024, "small": 256}
            rungs = [{"label": k, "device": "cuda", "params": {}} for k in sizes]
            def attempt(r, wd):
                wd.arm()
                x = torch.empty(sizes[r["label"]] << 20, dtype=torch.uint8, device="cuda").fill_(1)
                torch.cuda.synchronize()
                return r["label"]
            res, att = g.run_ladder(rungs, attempt, sampler=None, req={"vram": {"precheck": False}})
            print(json.dumps({"res": res, "status": [a.status for a in att],
                              "reserved": [a.max_reserved_mb for a in att]}))
    """)
    assert out["res"] == "small" and out["status"] == ["oom", "ok"]
    assert out["reserved"][1] is not None and out["reserved"][1] < 700


# ------------------------------------------------------------------- load-time spill, slow after a spill


class LevelSampler:
    """A sampler whose shared usage the test sets (``level``); marks are stored like ResourceSampler's."""

    def __init__(self, level: float | None = 66.0) -> None:
        self.level = level
        self.marks: dict[str, dict[str, Any]] = {}

    def snapshot(self) -> dict[str, Any]:
        return {"pdh_self_shared_mb": self.level, "pdh_adapter_shared_mb": 100.0, "nvml_used_mb": 3000.0}

    def mark(self, label: str) -> dict[str, Any]:
        e = {"t_s": 0.0, **self.snapshot()}
        self.marks[label] = e
        return dict(e)


def _armed(s: LevelSampler, loaded: float, **kw: Any) -> Watchdog:
    s.mark(gpu_common.PRE_LOAD)
    s.level = loaded
    s.mark(base.MODEL_LOADED)
    wd = Watchdog(s, shared_growth_fail_mb=64, rtf_ref=None, rtf_fail_ratio=0.5, audio_s_total=100, **kw)
    wd.arm()
    return wd


def test_load_time_spill_fails_the_rung() -> None:
    """Bench 2026-09-30 17:58/18:01: 130-136 MB shared at model_loaded vs 66 MB after the CUDA context; the growth
    check (from model_loaded on) saw nothing, so a spilled rung was accepted at a fraction of its speed."""
    with pytest.raises(SharedGrowth, match="load time"):
        _armed(LevelSampler(66.0), 136.0, load_spill_fail_mb=48)
    # normal loads stage a few MB through shared memory; MuScriptor's upstream loader +32 MB (bench 2026-10-03)
    assert _armed(LevelSampler(66.0), 68.0, load_spill_fail_mb=48).load_growth_mb == 2.0
    assert _armed(LevelSampler(66.0), 98.0, load_spill_fail_mb=48).load_growth_mb == 32.0
    # off (None / 0), no PDH, a fake sampler without dict marks, or no reference mark: never fails
    assert _armed(LevelSampler(66.0), 200.0).load_growth_mb is None
    assert _armed(LevelSampler(66.0), 200.0, load_spill_fail_mb=0).load_growth_mb is None
    assert _armed(LevelSampler(None), None, load_spill_fail_mb=48).load_growth_mb is None
    wd = Watchdog(FakeSampler([_r(500.0)]), shared_growth_fail_mb=64, rtf_ref=None, rtf_fail_ratio=0.5,
                  audio_s_total=10, load_spill_fail_mb=48)
    wd.arm()
    assert wd.load_growth_mb is None
    s = LevelSampler(66.0)
    s.mark("cuda_ctx")  # without pre_load the CUDA-context mark is the reference
    s.level = 140.0
    with pytest.raises(SharedGrowth):
        Watchdog(s, shared_growth_fail_mb=64, rtf_ref=None, rtf_fail_ratio=0.5, audio_s_total=10,
                 load_spill_fail_mb=48).arm()
    # a stale 0 MB reading (bench 2026-09-30 17:49: cuda_ctx 0.0, model_loaded 66 with nothing spilled) is never
    # the reference: the other mark is used, and with no non-zero reference there is no check at all
    s = LevelSampler(0.0)
    s.mark(gpu_common.PRE_LOAD)
    s.marks["cuda_ctx"] = {"pdh_self_shared_mb": 64.0}
    s.level = 66.0
    wd = Watchdog(s, shared_growth_fail_mb=64, rtf_ref=None, rtf_fail_ratio=0.5, audio_s_total=10,
                  load_spill_fail_mb=48)
    wd.arm()
    assert wd.load_growth_mb == 2.0
    s = LevelSampler(0.0)
    s.mark("cuda_ctx")
    s.level = 66.0
    wd = Watchdog(s, shared_growth_fail_mb=64, rtf_ref=None, rtf_fail_ratio=0.5, audio_s_total=10,
                  load_spill_fail_mb=48)
    wd.arm()
    assert wd.load_growth_mb is None


def test_spill_during_load_steps_down_and_a_spilled_gpu_answers_insufficient() -> None:
    """Through the ladder: every GPU rung whose load leaves the shared usage up fails; with no CPU rung the worker
    answers insufficient_vram (the stage waits and retries) instead of accepting a spilled, 4-6x slower rung."""
    s = LevelSampler(66.0)

    def attempt(rung: dict, wd: Watchdog) -> str:
        s.level = 136.0  # the upload spilled (and stays spilled: the other app still holds the VRAM)
        s.mark(base.MODEL_LOADED)
        wd.arm()
        wd.check(10.0)
        return rung["label"]

    res, attempts = run_ladder(RUNGS, attempt, sampler=s, req=_req(load_spill_fail_mb=48), audio_s_total=10.0,
                               mem_info=lambda: (5000.0, 6144.0), cuda_available=lambda: True)
    assert res is None and [a.status for a in attempts] == ["shared_growth"] * 3
    assert [a.load_growth_mb for a in attempts] == [70.0] * 3 and "load time" in (attempts[0].error or "")
    assert s.marks[gpu_common.PRE_LOAD]["pdh_self_shared_mb"] == 66.0  # taken once, before the first upload
    # the same ladder with a clean load succeeds at once
    s2 = LevelSampler(66.0)

    def clean(rung: dict, wd: Watchdog) -> str:
        s2.level = 68.0
        s2.mark(base.MODEL_LOADED)
        wd.arm()
        return rung["label"]

    res, attempts = run_ladder(RUNGS, clean, sampler=s2, req=_req(load_spill_fail_mb=48), audio_s_total=10.0,
                               mem_info=lambda: (5000.0, 6144.0), cuda_available=lambda: True)
    assert res == "chunk=588800" and attempts[0].load_growth_mb == 2.0


def test_slow_after_a_spill_is_a_failure_unless_the_backend_says_otherwise() -> None:
    clk = Clock()

    def slow_run(**kw: Any) -> Watchdog:
        wd = Watchdog(FakeSampler([_r(50.0)]), shared_growth_fail_mb=64, rtf_ref=6.0, rtf_fail_ratio=0.5,
                      audio_s_total=100, clock=clk, **kw)
        wd.arm()
        clk.t = 0.0
        wd.check(5)
        clk.t = 40.0
        wd.check(20)  # 0.375x vs 6x reference
        return wd

    assert slow_run().slow  # PDH readable, no spill above: contention, a warning only
    with pytest.raises(SharedGrowth, match="spilled"):
        slow_run(slow_is_failure=True)
    assert slow_run(slow_is_failure=True, slow_can_fail=False).slow  # MuScriptor: never fatal
    # MuScriptor without PDH: slow is not "degraded detection" failure either (only recorded)
    clk.t = 0.0
    wd = Watchdog(FakeSampler([_r(None, adapter=None)]), shared_growth_fail_mb=64, rtf_ref=6.0, rtf_fail_ratio=0.5,
                  audio_s_total=100, clock=clk, slow_can_fail=False)
    wd.arm()
    wd.check(1)
    clk.t = 40.0
    wd.check(20)
    assert wd.slow and wd.degraded_detection


def test_ladder_passes_the_spill_and_backend_flags_to_each_rung(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[dict[str, Any]] = []

    class Rec(Watchdog):
        def __init__(self, *a: Any, **kw: Any) -> None:
            made.append(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(gpu_common, "Watchdog", Rec)
    res, _attempts = _ladder(_req(load_spill_fail_mb=48), {"chunk=588800": SharedGrowth("spill")})
    assert res == "chunk=352256"
    assert [m["slow_is_failure"] for m in made] == [False, True]  # the rung below a spill may not be slow
    assert all(m["load_spill_fail_mb"] == 48.0 and m["slow_can_fail"] is True for m in made)
    made.clear()
    _ladder(_req(), {"chunk=588800": RuntimeError("CUDA out of memory")})
    assert [m["slow_is_failure"] for m in made] == [False, False]  # an OOM is no spill
    assert all(m["load_spill_fail_mb"] is None for m in made)  # an old request without the key: no load check
    made.clear()
    run_ladder(RUNGS, lambda r, wd: r["label"], sampler=FakeSampler([_r(1.0)]), req=_req(), audio_s_total=1.0,
               mem_info=lambda: (5000.0, 6144.0), cuda_available=lambda: True, slow_can_fail=False)
    assert made[0]["slow_can_fail"] is False


def test_vram_request_carries_the_load_spill_threshold() -> None:
    from bandscribe import vram

    assert vram.build_vram_request("sep_msst", ["chunk=588800"], None)["load_spill_fail_mb"] == 48.0
    assert vram.build_vram_request("sep_msst", ["chunk=588800"], {"gpu": {"load_spill_fail_mb": 0}})[
        "load_spill_fail_mb"] == 0.0


def test_sampler_wait_for_sample() -> None:
    s = base.ResourceSampler(pid=1, interval_s=0.01)
    assert s.wait_for_sample() is False  # not running: never blocks
    s.start()
    try:
        assert s.wait_for_sample(n=2, timeout_s=10.0) is True
    finally:
        s.stop()


def test_muscriptor_eos_warning_is_one_korean_line_per_view() -> None:
    from bandscribe.workers import amt_muscriptor as W

    msg = W.eos_warning("bass_mono", [12.04, 3.5, None, 40.0, 41.0, 42.0, 43.0])
    assert msg.startswith("bass_mono: MuScriptor 가 7개 구간에서 끝 토큰(EOS)") and "(4, 12, 40, 41, 42 ... s 부근)" in msg
    assert "부근" not in W.eos_warning("g", [None])
