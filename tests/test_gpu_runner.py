"""run_worker: echo round trip, errors, timeout kill of the whole tree, and the GPU lock guarantee.

Non-GPU modes run with the core venv python (a launcher too), so these tests need no torch.
"""
from __future__ import annotations

import _thread
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects")

from gtab import gpu, paths  # noqa: E402
from gtab.gpu import GpuLockTimeout, run_worker  # noqa: E402
from gtab.winjob import pid_alive  # noqa: E402

CORE = paths.CORE_PYTHON


@pytest.fixture
def lock_path(tmp_path, monkeypatch) -> Path:
    p = tmp_path / "gpu.lock"
    monkeypatch.setattr(paths, "GPU_LOCK", p)
    return p


@pytest.fixture
def lock_release_probe(monkeypatch):
    """Replace gpu.FileLock with one that records, at the moment of release, which watched pids are alive."""
    state: dict = {"watch": None, "alive_at_release": None, "releases": 0}

    class ProbeLock(FileLock):
        def release(self, force: bool = False) -> None:
            files = state["watch"]
            if files is not None and state["alive_at_release"] is None:
                pids = _pids_from(files) if files.exists() else []
                state["alive_at_release"] = {p: pid_alive(p) for p in pids}
            state["releases"] += 1
            super().release(force=force)

    monkeypatch.setattr(gpu, "FileLock", ProbeLock)
    return state


def _pids_from(pid_file: Path) -> list[int]:
    info = json.loads(pid_file.read_text(encoding="utf-8"))
    return [int(info[k]) for k in ("worker_pid", "worker_ppid", "grandchild_pid", "grandchild_self_pid")
            if info.get(k) is not None]


def _lock_is_free(path: Path) -> bool:
    probe = FileLock(str(path))
    try:
        probe.acquire(timeout=0)
    except Timeout:
        return False
    probe.release()
    return True


def _wait_for(pred, timeout: float = 20.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


def test_echo_roundtrip_and_launcher_report(tmp_path, lock_path):
    req = {"mode": "echo", "text": "기타 탭", "n": [1, 2, 3]}
    r = run_worker("selftest", req, tmp_path / "w", python=CORE, lock_timeout_s=30)
    assert r.ok, r.stderr_path.read_text(encoding="utf-8")
    assert r.returncode == 0 and not r.timed_out and not r.killed
    assert r.output == req
    res = r.result
    assert res["worker"] == "selftest" and res["error"] is None and res["traceback"] is None
    assert res["torch_imported"] is False and res["cuda"] is None
    # The worker runs in the base interpreter started by the venv launcher we spawned.
    assert Path(res["python"]) == CORE
    assert Path(res["base_executable"]) != CORE
    assert res["ppid"] == r.pid
    assert res["pid"] in r.job_pids and r.pid in r.job_pids
    assert set(res["sysmon"]) >= {"nvml_used_peak_mb", "pdh_self_dedicated_peak_mb",
                                  "pdh_self_shared_peak_mb", "pdh_self_shared_start_mb"}
    assert json.loads((tmp_path / "w" / "request.json").read_text(encoding="utf-8")) == req
    assert r.stdout_path.exists() and r.stderr_path.exists()
    assert _lock_is_free(lock_path)


def test_worker_exception_reports_error(tmp_path, lock_path):
    r = run_worker("selftest", {"mode": "fail", "message": "boom"}, tmp_path, python=CORE)
    assert not r.ok
    assert r.returncode == 1 and not r.killed
    assert r.result["ok"] is False
    assert "RuntimeError: boom" in r.error
    assert "Traceback" in r.result["traceback"]
    assert r.output == {"reached": "fail"}  # ctx.partial survives the failure


def test_missing_python_raises_and_releases_lock(tmp_path, lock_path):
    with pytest.raises(OSError):
        run_worker("selftest", {"mode": "echo"}, tmp_path, python=tmp_path / "no-python.exe")
    assert _lock_is_free(lock_path)


def test_unknown_mode_fails(tmp_path, lock_path):
    r = run_worker("selftest", {"mode": "nope"}, tmp_path, python=CORE, use_lock=False)
    assert not r.ok and "unknown selftest mode" in r.error


def test_timeout_kills_launcher_worker_and_grandchild_before_lock_release(tmp_path, lock_path, lock_release_probe):
    pid_file = tmp_path / "pids.json"
    lock_release_probe["watch"] = pid_file
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "result.json").write_text('{"ok": true}', encoding="utf-8")  # stale, must be ignored
    t0 = time.monotonic()
    r = run_worker("selftest", {"mode": "spawn_grandchild", "seconds": 600, "pid_file": str(pid_file)},
                   tmp_path / "w", python=CORE, timeout_s=6)
    elapsed = time.monotonic() - t0
    assert r.timed_out and r.killed and not r.ok
    assert r.result == {}
    assert elapsed < 6 + gpu.DRAIN_TIMEOUT_S + 5

    pids = _pids_from(pid_file)
    assert len(pids) == 4  # launcher, worker interpreter, grandchild launcher, grandchild interpreter
    assert len(set(pids)) == 4
    assert set(pids) <= set(r.job_pids)
    assert not any(pid_alive(p) for p in pids)
    # The guarantee: when the lock was released, nothing from the job was alive any more.
    assert lock_release_probe["alive_at_release"] == {p: False for p in pids}
    assert _lock_is_free(lock_path)


def test_straggler_after_normal_exit_is_killed_before_lock_release(tmp_path, lock_path, lock_release_probe,
                                                                   monkeypatch):
    """The worker exits fine but leaves a grandchild behind: the lock must wait for (and kill) it."""
    monkeypatch.setattr(gpu, "STRAGGLER_GRACE_S", 1.0)
    pid_file = tmp_path / "pids.json"
    lock_release_probe["watch"] = pid_file
    r = run_worker("selftest", {"mode": "spawn_grandchild", "seconds": 0.2, "pid_file": str(pid_file)},
                   tmp_path / "w", python=CORE, timeout_s=60)
    assert r.ok and r.returncode == 0 and not r.timed_out and not r.killed
    pids = _pids_from(pid_file)
    assert len(pids) == 4
    assert not any(pid_alive(p) for p in pids)
    assert lock_release_probe["alive_at_release"] == {p: False for p in pids}


def test_second_runner_cannot_get_the_lock_while_first_runs(tmp_path, lock_path):
    first: dict = {}

    def run_first() -> None:
        first["r"] = run_worker("selftest", {"mode": "hold", "seconds": 5}, tmp_path / "a", python=CORE)

    th = threading.Thread(target=run_first)
    th.start()
    try:
        _wait_for(lambda: not _lock_is_free(lock_path), timeout=10, interval=0.02)
        t0 = time.monotonic()
        with pytest.raises(GpuLockTimeout):
            run_worker("selftest", {"mode": "echo"}, tmp_path / "b", python=CORE, lock_timeout_s=0.5)
        assert time.monotonic() - t0 < 5
        assert issubclass(GpuLockTimeout, TimeoutError)
        assert not (tmp_path / "b" / "result.json").exists()
    finally:
        th.join(60)
    assert first["r"].ok
    r = run_worker("selftest", {"mode": "echo"}, tmp_path / "c", python=CORE, lock_timeout_s=0.5)
    assert r.ok


def test_lock_waits_then_proceeds(tmp_path, lock_path):
    first: dict = {}
    th = threading.Thread(target=lambda: first.setdefault(
        "r", run_worker("selftest", {"mode": "hold", "seconds": 2}, tmp_path / "a", python=CORE)))
    th.start()
    try:
        _wait_for(lambda: not _lock_is_free(lock_path), timeout=10, interval=0.02)
        r = run_worker("selftest", {"mode": "echo"}, tmp_path / "b", python=CORE, lock_timeout_s=60)
    finally:
        th.join(60)
    assert r.ok and r.lock_wait_s > 0.5
    assert first["r"].ok


def test_keyboard_interrupt_terminates_tree_then_releases_lock(tmp_path, lock_path, lock_release_probe):
    pid_file = tmp_path / "pids.json"
    lock_release_probe["watch"] = pid_file

    def interrupt_when_started() -> None:
        deadline = time.monotonic() + 30
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.3)
        _thread.interrupt_main()

    threading.Thread(target=interrupt_when_started, daemon=True).start()
    with pytest.raises(KeyboardInterrupt):
        run_worker("selftest", {"mode": "spawn_grandchild", "seconds": 600, "pid_file": str(pid_file)},
                   tmp_path / "w", python=CORE)
    pids = _pids_from(pid_file)
    assert len(pids) == 4
    assert not any(pid_alive(p) for p in pids)
    assert lock_release_probe["alive_at_release"] == {p: False for p in pids}
    assert _lock_is_free(lock_path)


@pytest.mark.gpu
def test_selftest_cuda_in_gpu_env(tmp_path, lock_path):
    r = run_worker("selftest", {"mode": "cuda"}, tmp_path, timeout_s=600, lock_timeout_s=600)
    assert r.ok, (r.error, r.stderr_path.read_text(encoding="utf-8")[-3000:])
    out = r.output
    assert out["cuda_available"] is True and out["passed"] is True
    assert out["torch_version"].startswith("2.11.0") and out["cuda_version"] == "12.8"
    assert out["fp16_matmul"]["ok"] and out["autocast_conv"]["ok"]
    assert len(out["capability"]) == 2
    if tuple(out["capability"]) == (7, 5):  # RTX 2060 (Turing): no native bf16
        assert out["bf16_supported_native"] is False
    res = r.result
    assert res["torch_imported"] is True
    assert res["cuda"]["max_reserved_mb"] >= res["cuda"]["max_allocated_mb"] > 0
    # GPU venv python.exe is a launcher too; the NVIDIA per-program profile belongs on base_executable.
    assert Path(res["python"]) == paths.GPU_PYTHON
    assert Path(res["base_executable"]) != paths.GPU_PYTHON
    assert res["ppid"] == r.pid and res["pid"] in r.job_pids
    # PDH saw the CUDA context of the worker's own pid (wildcard counters pick up new instances).
    assert res["sysmon"]["pdh_self_dedicated_peak_mb"] and res["sysmon"]["pdh_self_dedicated_peak_mb"] > 0
    assert _lock_is_free(lock_path)


def test_unkillable_leftover_is_reported_as_a_dirty_lock_release(tmp_path, lock_path, monkeypatch, caplog):
    """A process that survives terminate (e.g. stuck in the driver) must not be released silently."""
    monkeypatch.setattr(gpu, "STRAGGLER_GRACE_S", 0.0)
    monkeypatch.setattr(gpu, "DRAIN_TIMEOUT_S", 0.0)
    monkeypatch.setattr(gpu, "DRAIN_EXTRA_S", 0.0)
    # Simulate "never empties": the real processes still die when the job closes (kill-on-close).
    monkeypatch.setattr(gpu.JobObject, "wait_empty", lambda self, timeout_s: False)
    r = run_worker("selftest", {"mode": "echo"}, tmp_path, python=CORE, lock_timeout_s=30)
    assert r.lock_released_dirty is True
    assert r.ok is False  # a worker whose tree could not be cleaned up is never a success
    assert r.returncode == 0 and r.result.get("ok") is True
    assert "lock is released anyway" in caplog.text
    assert _lock_is_free(lock_path)


def test_clean_run_is_not_dirty(tmp_path, lock_path):
    r = run_worker("selftest", {"mode": "echo"}, tmp_path, python=CORE, lock_timeout_s=30)
    assert r.ok and r.lock_released_dirty is False


def test_pid_sidecar_is_removed_after_a_clean_run(tmp_path, lock_path):
    r = run_worker("selftest", {"mode": "echo"}, tmp_path / "w", python=CORE, lock_timeout_s=30)
    assert r.ok
    assert not gpu._pids_sidecar().exists()


def test_next_holder_waits_for_a_hard_killed_holders_workers(tmp_path, lock_path):
    """A leftover sidecar (previous orchestrator hard-killed) makes the next run wait for those pids."""
    import subprocess

    straggler = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])
    try:
        from gtab import atomic
        atomic.write_json(gpu._pids_sidecar(), {"owner_pid": 0, "pids": [straggler.pid]})
        t0 = time.monotonic()
        r = run_worker("selftest", {"mode": "echo"}, tmp_path / "w", python=CORE, lock_timeout_s=30)
        assert r.ok
        # run_worker could only start once the straggler had fully exited
        assert straggler.poll() is not None
        assert time.monotonic() - t0 >= 1.0
        assert not gpu._pids_sidecar().exists()
    finally:
        straggler.kill()
