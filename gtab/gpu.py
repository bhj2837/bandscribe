"""GPU worker runner: one worker process tree at a time, inside a Job Object, under data/gpu.lock.

The guarantee (DESIGN 9.5): at most one model on the GPU. The lock is taken *before* spawning and released
only once every process in the job is gone - including the base interpreter that the venv launcher starts
as a child. Releasing when only the launcher has exited would let the next worker load while the previous
one still holds VRAM, pushing it into Sysmem Fallback.

One deliberate exception: a process that survives TerminateJobObject for DRAIN_TIMEOUT_S + DRAIN_EXTRA_S
(stuck in the driver, e.g. during a TDR) is given up on and the lock is released anyway, so one hung
kernel call cannot block every later GPU run until a reboot. That release is never silent: the result
carries ``lock_released_dirty=True`` and an error is logged, so callers (and the doctor) can refuse to go
on or warn that the next worker may share the GPU with the leftover (M0_SPEC 10).
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

import gtab
from gtab import atomic, paths
from gtab.winjob import JobObject, pid_alive, spawn_in_job

log = logging.getLogger(__name__)

POLL_S = 0.2
# After the worker's main process exits, stragglers (processes it spawned and did not reap) get this long
# before the job is terminated. The lock stays held meanwhile.
STRAGGLER_GRACE_S = 3.0
# After TerminateJobObject, how long to wait for the kernel to finish tearing processes down.
DRAIN_TIMEOUT_S = 10.0
# If a terminated job still is not empty (e.g. a process stuck in the driver), keep the lock this much longer.
DRAIN_EXTRA_S = 60.0
# After acquiring the lock: how long to wait for a hard-killed previous holder's workers to finish exiting.
PREVIOUS_HOLDER_WAIT_S = 60.0


class GpuLockTimeout(TimeoutError):
    """Another worker (CLI or UI) held the GPU lock for longer than lock_timeout_s."""


@dataclass
class WorkerResult:
    ok: bool
    returncode: int | None
    result: dict
    work_dir: Path
    stdout_path: Path
    stderr_path: Path
    duration_s: float
    timed_out: bool
    killed: bool
    lock_wait_s: float
    pid: int | None = None  # pid of the process we spawned (the venv launcher)
    job_pids: list[int] = field(default_factory=list)  # pids seen in the job (sampled every POLL_S)
    # True if the GPU lock had to be released while processes of this worker were still in the job
    # (they survived terminate for DRAIN_TIMEOUT_S + DRAIN_EXTRA_S). Such a run is never ok.
    lock_released_dirty: bool = False

    @property
    def output(self) -> dict:
        out = self.result.get("output")
        return out if isinstance(out, dict) else {}

    @property
    def error(self) -> str | None:
        return self.result.get("error")


def _worker_env() -> dict[str, str]:
    # Run the same gtab source tree as the orchestrator even if the target env's editable install drifts.
    src_root = str(Path(gtab.__file__).resolve().parents[1])
    env = paths.child_env()
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_root, env.get("PYTHONPATH", "")) if p)
    return env


def _drain(job: JobObject, *, terminate: bool) -> bool:
    """Make the job empty; True once it is.

    terminate=False is the normal-exit path: the launcher only exits after its child, so the job is usually
    empty already. Anything still running after STRAGGLER_GRACE_S is an orphan that may hold VRAM -> kill it.
    """
    if not terminate:
        if job.wait_empty(STRAGGLER_GRACE_S):
            return True
        log.warning("worker exited but left %d process(es) in its job; terminating them", job.active_processes())
    try:
        job.terminate(1)
    except OSError as e:  # the job may have emptied between the check and the call
        log.debug("TerminateJobObject: %s", e)
    if job.wait_empty(DRAIN_TIMEOUT_S):
        return True
    log.error("job still has %d process(es) %.0fs after terminate; waiting up to %.0fs more before "
              "releasing the GPU lock", job.active_processes(), DRAIN_TIMEOUT_S, DRAIN_EXTRA_S)
    if job.wait_empty(DRAIN_EXTRA_S):
        return True
    log.error("giving up on %d unkillable process(es) (pids %s); the GPU lock is released anyway and the "
              "next GPU worker may share VRAM with them", job.active_processes(), _pids(job))
    return False


def _pids(job: JobObject) -> list[int]:
    try:
        return job.process_ids()
    except OSError:
        return []


def _pids_sidecar() -> Path:
    return paths.GPU_LOCK.with_name(paths.GPU_LOCK.name + ".pids")


def _write_sidecar(pids: set[int]) -> None:
    try:
        atomic.write_json(_pids_sidecar(), {"owner_pid": os.getpid(), "pids": sorted(pids)})
    except OSError as e:  # best effort: the sidecar only narrows a crash-only race
        log.debug("could not write GPU pid sidecar: %s", e)


def _wait_for_previous_holder() -> None:
    """Called right after acquiring the GPU lock.

    If the previous holder was hard-killed, Windows released its lock file immediately, but kill-on-close
    takes a few more ms to tear down its worker (measured 9-38 ms), which may still hold VRAM. A leftover
    sidecar means exactly that situation (a clean release deletes it): wait for those pids to be gone.
    """
    sidecar = _pids_sidecar()
    if not sidecar.exists():
        return
    try:
        pids = [int(p) for p in atomic.read_json(sidecar).get("pids", [])]
    except (OSError, ValueError, AttributeError) as e:
        log.debug("unreadable GPU pid sidecar (%s); ignoring", e)
        pids = []
    deadline = time.monotonic() + PREVIOUS_HOLDER_WAIT_S
    alive = [p for p in pids if pid_alive(p)]
    if alive:
        log.warning("previous GPU worker processes %s are still exiting; waiting for them", alive)
    while alive and time.monotonic() < deadline:
        time.sleep(0.02)
        alive = [p for p in alive if pid_alive(p)]
    if alive:
        log.error("previous GPU worker processes %s did not exit within %.0fs; continuing anyway",
                  alive, PREVIOUS_HOLDER_WAIT_S)
    sidecar.unlink(missing_ok=True)


def _run_in_job(args: list[str], work_dir: Path, stdout_path: Path, stderr_path: Path,
                timeout_s: float | None, record_pids: bool = False,
                ) -> tuple[int | None, bool, bool, int | None, list[int], bool]:
    """Spawn, wait, clean up. Returns (returncode, timed_out, killed, pid, job_pids, drained).

    Returns (or raises) only after the job is empty, so the caller may release the GPU lock right after -
    except when ``drained`` is False: then processes survived every attempt to kill them (see module doc).
    record_pids: keep the GPU pid sidecar up to date (only while holding the GPU lock).
    """
    timed_out = killed = False
    seen: set[int] = set()
    with JobObject(kill_on_close=True) as job:  # closing the job is the last line of defence
        with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
            proc = spawn_in_job(args, job, env=_worker_env(), cwd=work_dir, stdout=out, stderr=err)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        try:
            while True:
                try:
                    new = set(job.track()) - seen
                    if new:
                        seen.update(new)
                        if record_pids:
                            _write_sidecar(seen)
                except OSError:
                    pass
                try:
                    proc.wait(timeout=POLL_S)  # short waits keep Ctrl+C responsive
                    break
                except subprocess.TimeoutExpired:
                    pass
                if deadline is not None and time.monotonic() >= deadline:
                    log.warning("worker timed out after %.1fs; terminating its job", timeout_s)
                    timed_out = killed = True
                    break
            drained = _drain(job, terminate=timed_out)
        except BaseException:
            # KeyboardInterrupt or anything else: kill the whole tree before the lock can be released.
            log.warning("interrupted; terminating worker job")
            _drain(job, terminate=True)
            raise
        try:
            proc.wait(timeout=DRAIN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            pass
    return proc.returncode, timed_out, killed, proc.pid, sorted(seen), drained


def run_worker(name: str, request: dict, work_dir: Path, *, timeout_s: float | None = None,
               python: Path = paths.GPU_PYTHON, use_lock: bool = True,
               lock_timeout_s: float | None = None) -> WorkerResult:
    """Run `python -m gtab.workers.<name> request.json result.json` in a Job Object.

    Raises GpuLockTimeout if the lock could not be acquired within lock_timeout_s (None = wait forever).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    req_path = work_dir / "request.json"
    res_path = work_dir / "result.json"
    stdout_path = work_dir / "stdout.log"
    stderr_path = work_dir / "stderr.log"
    # A result left over from an earlier run must never be mistaken for this run's.
    res_path.unlink(missing_ok=True)
    atomic.write_json(req_path, request)
    args = [str(python), "-m", f"gtab.workers.{name}", str(req_path), str(res_path)]

    lock_wait_s = 0.0
    lock: FileLock | None = None
    t0 = time.monotonic()
    if use_lock:
        lock_path = paths.GPU_LOCK  # read at call time so tests / GTAB_ROOT overrides apply
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path))
        try:
            lock.acquire(timeout=-1 if lock_timeout_s is None else lock_timeout_s)
        except Timeout as e:
            raise GpuLockTimeout(f"GPU lock {lock_path} busy for more than {lock_timeout_s}s") from e
    drained = False
    try:  # from here on the lock (if any) is held and must be released on every path
        if lock is not None:
            _wait_for_previous_holder()
            lock_wait_s = time.monotonic() - t0
            if lock_wait_s > 1.0:
                log.info("waited %.1fs for the GPU lock", lock_wait_s)
        t_start = time.monotonic()
        returncode, timed_out, killed, pid, job_pids, drained = _run_in_job(
            args, work_dir, stdout_path, stderr_path, timeout_s, record_pids=lock is not None)
        duration_s = time.monotonic() - t_start
    finally:
        if lock is not None:
            if drained:  # every worker process is gone: nothing for the next holder to wait for
                _pids_sidecar().unlink(missing_ok=True)
            lock.release()

    result: dict[str, Any] = {}
    if res_path.exists():
        try:
            loaded = atomic.read_json(res_path)
            result = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError) as e:
            log.warning("unreadable result.json from worker %s: %s", name, e)
    ok = returncode == 0 and result.get("ok") is True and drained
    if not ok:
        log.warning("worker %s failed (rc=%s, timed_out=%s): %s; see %s", name, returncode, timed_out,
                    result.get("error"), stderr_path)
    return WorkerResult(ok=ok, returncode=returncode, result=result, work_dir=work_dir,
                        stdout_path=stdout_path, stderr_path=stderr_path, duration_s=duration_s,
                        timed_out=timed_out, killed=killed, lock_wait_s=lock_wait_s,
                        pid=pid, job_pids=job_pids, lock_released_dirty=not drained)
