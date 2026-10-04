"""Job Object wrapper: struct layouts, launcher capture, terminate / kill-on-close, cleanup on spawn failure."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects")

from bandscribe import paths  # noqa: E402
from bandscribe import winjob  # noqa: E402
from bandscribe.winjob import JobObject, pid_alive, spawn_in_job  # noqa: E402

PY = str(paths.CORE_PYTHON)
REPORT_AND_SLEEP = "import os, time; print(os.getpid(), os.getppid(), flush=True); time.sleep(120)"


def _wait_for(pred, timeout: float = 20.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


def _read_pids(path: Path) -> tuple[int, int] | None:
    try:
        parts = path.read_text(encoding="utf-8").split()
    except OSError:
        return None
    return (int(parts[0]), int(parts[1])) if len(parts) >= 2 else None


def _spawn_reporter(job: JobObject, tmp_path: Path, name: str = "out.txt") -> tuple[subprocess.Popen, Path]:
    out = tmp_path / name
    with open(out, "wb") as f:
        proc = spawn_in_job([PY, "-c", REPORT_AND_SLEEP], job, env=paths.child_env(), cwd=tmp_path,
                            stdout=f, stderr=subprocess.STDOUT)
    return proc, out


def test_struct_layouts_x64():
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    B = winjob.JOBOBJECT_BASIC_LIMIT_INFORMATION
    E = winjob.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    A = winjob.JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
    assert ctypes.sizeof(B) == 64
    assert (B.LimitFlags.offset, B.MinimumWorkingSetSize.offset, B.ActiveProcessLimit.offset,
            B.Affinity.offset, B.PriorityClass.offset, B.SchedulingClass.offset) == (16, 24, 40, 48, 56, 60)
    assert ctypes.sizeof(winjob.IO_COUNTERS) == 48
    assert ctypes.sizeof(E) == 144
    assert (E.IoInfo.offset, E.ProcessMemoryLimit.offset, E.PeakJobMemoryUsed.offset) == (64, 112, 136)
    assert ctypes.sizeof(A) == 48
    assert (A.TotalProcesses.offset, A.ActiveProcesses.offset) == (36, 40)
    L1, L4 = winjob._pid_list_type(1), winjob._pid_list_type(4)
    assert ctypes.sizeof(L1) == 16 and L1.ProcessIdList.offset == 8
    assert ctypes.sizeof(L4) == 8 + 4 * 8


def test_empty_job_and_close_idempotent():
    job = JobObject()
    assert job.active_processes() == 0
    assert job.total_processes() == 0
    assert job.process_ids() == []
    assert job.wait_empty(0)
    job.close()
    job.close()
    assert job.closed
    with pytest.raises(ValueError):
        job.active_processes()


def test_venv_launcher_child_is_born_inside_the_job(tmp_path):
    """The venv python.exe is a launcher; its child (the real interpreter) must be in the job too."""
    with JobObject() as job:
        proc, out = _spawn_reporter(job, tmp_path)
        child_pid, child_ppid = _wait_for(lambda: _read_pids(out))
        # This PC: .venv\Scripts\python.exe starts the base interpreter as a child process (DESIGN 9.5).
        assert child_pid != proc.pid, "venv python.exe is no longer a launcher - revisit DESIGN 9.5"
        assert child_ppid == proc.pid
        ids = set(job.process_ids())
        assert {proc.pid, child_pid} <= ids
        assert job.active_processes() == len(ids) == 2

        job.terminate(7)
        assert job.wait_empty(10)
        assert not pid_alive(proc.pid)
        assert not pid_alive(child_pid)
        assert proc.wait(5) == 7
        assert job.total_processes() == 2


def test_close_kills_remaining_processes(tmp_path):
    job = JobObject(kill_on_close=True)
    proc, out = _spawn_reporter(job, tmp_path)
    child_pid, _ = _wait_for(lambda: _read_pids(out))
    assert pid_alive(child_pid)
    job.close()
    _wait_for(lambda: not pid_alive(child_pid) and not pid_alive(proc.pid), timeout=10)
    proc.wait(5)


def test_process_ids_grows_buffer(tmp_path):
    """More pids than the initial 16-slot buffer -> ERROR_MORE_DATA path."""
    code = ("import subprocess, sys, time\n"
            "ps = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']) for _ in range(8)]\n"
            "time.sleep(120)\n")
    with JobObject() as job:
        proc = spawn_in_job([PY, "-c", code], job, env=paths.child_env(), cwd=tmp_path,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # launcher + interpreter + 8 x (launcher + interpreter)
        _wait_for(lambda: job.active_processes() >= 18, timeout=60, interval=0.2)
        ids = job.process_ids()
        assert len(ids) == len(set(ids)) >= 18
        assert proc.pid in ids
        job.terminate()
        assert job.wait_empty(15)
        assert not any(pid_alive(p) for p in ids)
        proc.wait(5)


def test_spawn_failure_raises_without_leak(tmp_path):
    with JobObject() as job:
        with pytest.raises(OSError):
            spawn_in_job([str(tmp_path / "does-not-exist.exe")], job, env=None, cwd=tmp_path,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert job.total_processes() == 0


def test_assign_failure_kills_the_suspended_process(tmp_path, monkeypatch):
    started: list[subprocess.Popen] = []
    real_popen = winjob.subprocess.Popen

    def recording_popen(*a, **kw):
        p = real_popen(*a, **kw)
        started.append(p)
        return p

    def failing_assign(self, handle):
        raise OSError("simulated AssignProcessToJobObject failure")

    monkeypatch.setattr(winjob.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(JobObject, "assign", failing_assign)
    with JobObject() as job:
        with pytest.raises(OSError, match="simulated"):
            spawn_in_job([PY, "-c", "import time; time.sleep(120)"], job, env=paths.child_env(), cwd=tmp_path,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert len(started) == 1
    assert started[0].returncode is not None  # reaped
    assert not pid_alive(started[0].pid)


def test_pid_alive():
    import os

    assert pid_alive(os.getpid())
    p = subprocess.Popen([PY, "-c", "pass"])
    p.wait(30)
    assert not pid_alive(p.pid)


# ------------------------------------------------------------------------------- run_captured
# A launcher (venv python.exe) whose real interpreter starts a grandchild that inherits stdout/stderr:
# the same shape as yt-dlp.exe (PyInstaller bootloader -> yt-dlp -> node). subprocess.run(timeout=)
# would kill only the launcher and then wait for the pipes, i.e. for the grandchild's 600 s sleep.
PIPE_HOLDER = (
    "import os, subprocess, sys, time\n"
    "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
    "open(sys.argv[1], 'w', encoding='utf-8').write(f'{os.getpid()} {g.pid}')\n"
    "print('started', flush=True)\n"
    "time.sleep(600)\n"
)


def test_run_captured_output_and_exit_code(tmp_path):
    code = "import sys; print('out 한글'); print('err ✓', file=sys.stderr); sys.exit(3)"
    r = winjob.run_captured([PY, "-c", code], timeout_s=60, env=paths.child_env(), cwd=tmp_path)
    assert r.returncode == 3 and not r.timed_out and not r.killed
    assert r.stdout.strip() == "out 한글" and r.stderr.strip() == "err ✓"
    assert len(r.pids) >= 1


def test_run_captured_timeout_kills_the_whole_tree(tmp_path):
    pid_file = tmp_path / "pids.txt"
    timeout = 3.0
    t0 = time.monotonic()
    r = winjob.run_captured([PY, "-c", PIPE_HOLDER, str(pid_file)], timeout_s=timeout,
                            env=paths.child_env(), cwd=tmp_path)
    elapsed = time.monotonic() - t0
    assert r.timed_out and r.killed
    assert elapsed < timeout + 8, f"returned after {elapsed:.1f} s"
    assert "started" in r.stdout  # output produced before the kill is kept
    child, grandchild = (int(p) for p in pid_file.read_text(encoding="utf-8").split())
    assert {child, grandchild} <= set(r.pids)
    _wait_for(lambda: not pid_alive(child) and not pid_alive(grandchild), timeout=5)


def test_run_captured_kills_stragglers_after_normal_exit(tmp_path):
    """The main process exits but leaves a pipe-holding child behind: the call still returns promptly."""
    code = ("import subprocess, sys\n"
            "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
            "open(sys.argv[1], 'w', encoding='utf-8').write(str(g.pid))\n")
    pid_file = tmp_path / "pid.txt"
    t0 = time.monotonic()
    r = winjob.run_captured([PY, "-c", code, str(pid_file)], timeout_s=60, env=paths.child_env(), cwd=tmp_path)
    assert time.monotonic() - t0 < 20
    assert r.returncode == 0 and not r.timed_out and r.killed
    _wait_for(lambda: not pid_alive(int(pid_file.read_text(encoding="utf-8"))), timeout=5)


def test_run_captured_missing_exe_raises(tmp_path):
    with pytest.raises(OSError):
        winjob.run_captured([str(tmp_path / "nope.exe")], timeout_s=5)
