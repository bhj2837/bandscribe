"""Windows Job Objects via ctypes (no pywin32).

Why: the venv `Scripts\\python.exe` is a *launcher* that starts the base interpreter as a child process
(verified on this PC for both uv venvs, DESIGN 9.5). Killing only the process we spawned would leave the
real interpreter - the one holding VRAM - running. Every worker therefore lives in a Job Object with
KILL_ON_JOB_CLOSE, and it is started suspended so the launcher cannot spawn its child before the
assignment happens.

Struct layouts below are for x64 (natural alignment, which ctypes reproduces); sizes are asserted in tests.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

log = logging.getLogger(__name__)

# --- constants (winnt.h) ---------------------------------------------------------------------------
JobObjectBasicAccountingInformation = 1
JobObjectBasicProcessIdList = 3
JobObjectExtendedLimitInformation = 9

JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
STILL_ACTIVE = 259
WAIT_OBJECT_0 = 0x0
WAIT_TIMEOUT = 0x102
ERROR_MORE_DATA = 234
ERROR_ACCESS_DENIED = 5


# --- structures ------------------------------------------------------------------------------------
class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),  # ULONG_PTR
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]  # 64 bytes on x64


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]  # 48 bytes


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]  # 144 bytes on x64


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", wintypes.LARGE_INTEGER),
        ("TotalKernelTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]  # 48 bytes


def _pid_list_type(n: int) -> type[ctypes.Structure]:
    """JOBOBJECT_BASIC_PROCESS_ID_LIST with room for n ids (the C struct declares ProcessIdList[1])."""

    class JOBOBJECT_BASIC_PROCESS_ID_LIST(ctypes.Structure):
        _fields_ = [
            ("NumberOfAssignedProcesses", wintypes.DWORD),
            ("NumberOfProcessIdsInList", wintypes.DWORD),
            ("ProcessIdList", ctypes.c_size_t * max(1, n)),  # ULONG_PTR
        ]

    return JOBOBJECT_BASIC_PROCESS_ID_LIST


# --- DLL bindings ----------------------------------------------------------------------------------
_k32: Any = None
_ntdll: Any = None


def _kernel32() -> Any:
    global _k32
    if _k32 is None:
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        H, BOOL, DWORD = wintypes.HANDLE, wintypes.BOOL, wintypes.DWORD
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.CreateJobObjectW.restype = H
        k.SetInformationJobObject.argtypes = [H, ctypes.c_int, ctypes.c_void_p, DWORD]
        k.SetInformationJobObject.restype = BOOL
        k.QueryInformationJobObject.argtypes = [H, ctypes.c_int, ctypes.c_void_p, DWORD, ctypes.POINTER(DWORD)]
        k.QueryInformationJobObject.restype = BOOL
        k.AssignProcessToJobObject.argtypes = [H, H]
        k.AssignProcessToJobObject.restype = BOOL
        k.TerminateJobObject.argtypes = [H, wintypes.UINT]
        k.TerminateJobObject.restype = BOOL
        k.TerminateProcess.argtypes = [H, wintypes.UINT]
        k.TerminateProcess.restype = BOOL
        k.CloseHandle.argtypes = [H]
        k.CloseHandle.restype = BOOL
        k.OpenProcess.argtypes = [DWORD, BOOL, DWORD]
        k.OpenProcess.restype = H
        k.GetExitCodeProcess.argtypes = [H, ctypes.POINTER(DWORD)]
        k.GetExitCodeProcess.restype = BOOL
        k.WaitForSingleObject.argtypes = [H, DWORD]
        k.WaitForSingleObject.restype = DWORD
        _k32 = k
    return _k32


def _nt() -> Any:
    global _ntdll
    if _ntdll is None:
        n = ctypes.WinDLL("ntdll")
        n.NtResumeProcess.argtypes = [wintypes.HANDLE]
        n.NtResumeProcess.restype = ctypes.c_long  # NTSTATUS
        _ntdll = n
    return _ntdll


def _check(ok: Any, what: str) -> None:
    if not ok:
        err = ctypes.get_last_error()
        raise ctypes.WinError(err, f"{what} failed: {ctypes.FormatError(err).strip()}")


# --- API -------------------------------------------------------------------------------------------
class JobObject:
    """An anonymous Job Object; every process assigned to it (and all their descendants) dies with it."""

    def __init__(self, kill_on_close: bool = True) -> None:
        k = _kernel32()
        h = k.CreateJobObjectW(None, None)
        _check(h, "CreateJobObjectW")
        self._h: int | None = h
        # pid -> SYNCHRONIZE handle for every process seen in the job (see track()). Holding the handle
        # also keeps the pid from being reused while we still care about it.
        self._proc_handles: dict[int, int] = {}
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        # DIE_ON_UNHANDLED_EXCEPTION: a crashing worker must not sit in a WER dialog while we hold the GPU lock.
        flags = JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        if kill_on_close:
            flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        info.BasicLimitInformation.LimitFlags = flags
        try:
            _check(k.SetInformationJobObject(h, JobObjectExtendedLimitInformation,
                                             ctypes.byref(info), ctypes.sizeof(info)),
                   "SetInformationJobObject")
        except BaseException:
            self.close()
            raise

    @property
    def handle(self) -> int:
        if self._h is None:
            raise ValueError("JobObject is closed")
        return self._h

    def assign(self, process_handle: int) -> None:
        _check(_kernel32().AssignProcessToJobObject(self.handle, int(process_handle)), "AssignProcessToJobObject")

    def _accounting(self) -> JOBOBJECT_BASIC_ACCOUNTING_INFORMATION:
        info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        _check(_kernel32().QueryInformationJobObject(
            self.handle, JobObjectBasicAccountingInformation, ctypes.byref(info), ctypes.sizeof(info), None),
            "QueryInformationJobObject(BasicAccounting)")
        return info

    def active_processes(self) -> int:
        return int(self._accounting().ActiveProcesses)

    def total_processes(self) -> int:
        """Processes ever assigned (including exited ones) - useful to tell 'empty' from 'never started'."""
        return int(self._accounting().TotalProcesses)

    def process_ids(self) -> list[int]:
        k = _kernel32()
        n = 16
        while True:
            T = _pid_list_type(n)
            buf = T()
            ok = k.QueryInformationJobObject(self.handle, JobObjectBasicProcessIdList,
                                             ctypes.byref(buf), ctypes.sizeof(buf), None)
            if not ok and ctypes.get_last_error() == ERROR_MORE_DATA:
                n = max(n * 2, int(buf.NumberOfAssignedProcesses) + 8)
                continue
            _check(ok, "QueryInformationJobObject(BasicProcessIdList)")
            if buf.NumberOfProcessIdsInList < buf.NumberOfAssignedProcesses:  # raced with a new process
                n = int(buf.NumberOfAssignedProcesses) + 8
                continue
            return [int(buf.ProcessIdList[i]) for i in range(buf.NumberOfProcessIdsInList)]

    def track(self) -> list[int]:
        """Sample the job's pids and open a SYNCHRONIZE handle to each new one; returns the current pids.

        Why: ActiveProcesses drops to 0 as soon as a process *starts* exiting, but its teardown (closing
        the CUDA context, freeing VRAM) finishes later - measured up to ~130 ms after TerminateJobObject.
        Only the process object being signaled means it is really gone, and waiting on it needs a handle
        opened while the process still existed. Call this periodically and right before terminating.
        """
        pids = self.process_ids()
        k = _kernel32()
        for pid in pids:
            if pid in self._proc_handles:
                continue
            h = k.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if h:
                self._proc_handles[pid] = h
            # else: it already exited (and was fully torn down) between the listing and the open.
        return pids

    def terminate(self, exit_code: int = 1) -> None:
        try:
            self.track()  # catch processes started since the last sample before they are killed
        except OSError:
            pass
        _check(_kernel32().TerminateJobObject(self.handle, exit_code), "TerminateJobObject")

    def _wait_tracked(self, deadline: float) -> bool:
        """Wait until every tracked process object is signaled (fully exited)."""
        k = _kernel32()
        for pid, h in list(self._proc_handles.items()):
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            rc = k.WaitForSingleObject(h, remaining_ms)
            if rc == WAIT_TIMEOUT:
                return False
            if rc != WAIT_OBJECT_0:
                raise ctypes.WinError(ctypes.get_last_error(), f"WaitForSingleObject(pid {pid}) failed")
        return True

    def wait_empty(self, timeout_s: float) -> bool:
        """True once no process in the job is alive *and* every tracked process has fully exited.

        Jobs have no waitable 'empty' state, so the active count is polled; tracked processes are then
        waited on individually (see track()).
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            if self.active_processes() == 0:
                return self._wait_tracked(deadline)
            if time.monotonic() >= deadline:
                return False
            try:
                self.track()
            except OSError:
                pass
            time.sleep(0.05)

    def close(self) -> None:
        """Close the handle; with kill_on_close this kills whatever is still running in the job."""
        h, self._h = self._h, None
        k = _kernel32()
        if h:
            k.CloseHandle(h)
        handles, self._proc_handles = self._proc_handles, {}
        for ph in handles.values():
            k.CloseHandle(ph)

    @property
    def closed(self) -> bool:
        return self._h is None

    def __enter__(self) -> JobObject:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def spawn_in_job(args: list[str], job: JobObject, env: dict[str, str] | None, cwd: Path | str | None,
                 stdout: IO[Any] | int | None, stderr: IO[Any] | int | None,
                 stdin: IO[Any] | int | None = subprocess.DEVNULL) -> subprocess.Popen:
    """Start `args` suspended, put it in `job`, then resume it.

    CREATE_NEW_PROCESS_GROUP keeps console Ctrl+C away from the worker: cancellation goes through the job.
    Popen closes the primary thread handle, so the whole process is resumed with ntdll.NtResumeProcess.
    """
    proc = subprocess.Popen(
        [str(a) for a in args], env=env, cwd=None if cwd is None else str(cwd),
        stdin=stdin, stdout=stdout, stderr=stderr,
        creationflags=CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP,
    )
    handle = int(proc._handle)  # type: ignore[attr-defined]  # full-access handle from CreateProcess
    try:
        job.assign(handle)
        status = _nt().NtResumeProcess(handle)
        if status < 0:
            raise OSError(f"NtResumeProcess failed: NTSTATUS 0x{status & 0xFFFFFFFF:08X}")
    except BaseException:
        # Never leave a suspended orphan behind.
        _kernel32().TerminateProcess(handle, 1)
        proc.wait()
        raise
    log.debug("spawned pid %d in job: %s", proc.pid, args)
    return proc


# --- bounded, tree-killing replacement for subprocess.run(capture_output=True, timeout=...) --------
RUN_POLL_S = 0.2
# After the main process exits, processes it left behind get this long before the job is terminated.
RUN_STRAGGLER_GRACE_S = 2.0
# After TerminateJobObject: how long to wait for the kernel to tear the processes down.
RUN_DRAIN_S = 10.0
# Pipe readers get this long to hit EOF once the job is empty (a process outside the job could hold them).
RUN_READER_JOIN_S = 5.0


@dataclass
class CapturedRun:
    args: list[str]
    returncode: int | None  # exit code of the process we started (None if it could not be reaped)
    stdout: str
    stderr: str
    timed_out: bool
    killed: bool  # processes in the job had to be terminated (timeout, leftovers or interrupt)
    duration_s: float
    pids: list[int] = field(default_factory=list)  # every pid seen in the job (sampled)


class _PipeReader(threading.Thread):
    """Drains one pipe to EOF in the background, so no child ever blocks on a full pipe and we never
    block on a pipe that a straggler still holds open."""

    def __init__(self, stream: IO[bytes]) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._chunks: list[bytes] = []

    def run(self) -> None:
        try:
            while chunk := self._stream.read1(65536):  # type: ignore[attr-defined]
                self._chunks.append(chunk)
        except (OSError, ValueError):
            pass
        finally:
            try:
                self._stream.close()
            except OSError:
                pass

    def text(self, encoding: str, errors: str) -> str:
        return b"".join(self._chunks).decode(encoding, errors)


def _terminate_job(job: JobObject) -> bool:
    try:
        job.terminate(1)
    except OSError as e:  # the job may have emptied between the check and the call
        log.debug("TerminateJobObject: %s", e)
    if job.wait_empty(RUN_DRAIN_S):
        return True
    log.error("job still has %d process(es) %.0f s after terminate", job.active_processes(), RUN_DRAIN_S)
    return False


def run_captured(args: list[str] | list[str | Path], *, timeout_s: float | None,
                 env: dict[str, str] | None = None, cwd: Path | str | None = None,
                 encoding: str = "utf-8", errors: str = "replace") -> CapturedRun:
    """Run ``args`` to completion inside a Job Object and return its decoded stdout/stderr.

    Why not ``subprocess.run(timeout=)``: it kills only the process it started, then waits for the
    pipes to close. PyInstaller one-file exes (yt-dlp.exe), the venv python.exe and ``py.exe`` are all
    launchers whose real work happens in a child that inherits the pipes, so the "timeout" returns
    only when that orphaned child finishes by itself. Here the whole tree lives in a kill-on-close
    job: on expiry (or Ctrl+C) every process is terminated, and the call returns within about
    ``timeout_s + RUN_DRAIN_S`` whatever the children do. Raises OSError if ``args[0]`` cannot start.
    """
    argv = [str(a) for a in args]
    t0 = time.monotonic()
    deadline = None if timeout_s is None else t0 + max(0.0, float(timeout_s))
    timed_out = killed = False
    seen: set[int] = set()
    with JobObject(kill_on_close=True) as job:
        proc = spawn_in_job(argv, job, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        readers = [_PipeReader(proc.stdout), _PipeReader(proc.stderr)]  # type: ignore[arg-type]
        for r in readers:
            r.start()
        try:
            while True:
                try:
                    seen.update(job.track())
                except OSError:
                    pass
                wait = RUN_POLL_S if deadline is None else max(0.0, min(RUN_POLL_S, deadline - time.monotonic()))
                try:
                    proc.wait(timeout=wait)  # short waits keep Ctrl+C responsive
                    break
                except subprocess.TimeoutExpired:
                    pass
                if deadline is not None and time.monotonic() >= deadline:
                    log.warning("%s timed out after %.1f s; terminating its process tree", Path(argv[0]).name, timeout_s)
                    timed_out = True
                    break
            if timed_out or not job.wait_empty(RUN_STRAGGLER_GRACE_S):
                if not timed_out:
                    log.debug("%s left %d process(es) behind; terminating them", Path(argv[0]).name, job.active_processes())
                killed = True
                _terminate_job(job)
        except BaseException:
            # KeyboardInterrupt or anything else: never leave the tree running behind us.
            _terminate_job(job)
            raise
        try:
            proc.wait(timeout=RUN_DRAIN_S)
        except subprocess.TimeoutExpired:
            pass
    # Every process in the job is gone (or the job handle was closed, which kills them), so the pipes
    # reach EOF now; the bounded join only guards against a process that escaped the job.
    for r in readers:
        r.join(RUN_READER_JOIN_S)
        if r.is_alive():
            log.warning("output pipe of %s still open after the job ended; output may be cut short", Path(argv[0]).name)
    return CapturedRun(argv, proc.returncode, readers[0].text(encoding, errors), readers[1].text(encoding, errors),
                       timed_out, killed, time.monotonic() - t0, sorted(seen))


def pid_alive(pid: int) -> bool:
    """True if a process with this pid has not fully exited yet (pid reuse aside).

    Uses the process object's signaled state, not GetExitCodeProcess: the exit code is set when
    termination starts, while the object is signaled only once teardown (e.g. freeing VRAM) is done.
    """
    k = _kernel32()
    h = k.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        # Gone (ERROR_INVALID_PARAMETER) - unless we are merely not allowed to look at it.
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        return k.WaitForSingleObject(h, 0) == WAIT_TIMEOUT
    finally:
        k.CloseHandle(h)
