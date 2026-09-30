"""Common worker scaffolding: request/result files, resource sampling, exit code.

A worker module does:

    def handle(request: dict, ctx: WorkerContext) -> dict: ...
    if __name__ == "__main__":
        worker_main(handle)

and is started by `gtab.gpu.run_worker` as `python -m gtab.workers.<name> request.json result.json`.
This module must stay importable without torch (stdlib + gtab.atomic/sysmon only).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NoReturn

import gtab
from gtab import atomic, sysmon

log = logging.getLogger(__name__)

SAMPLE_INTERVAL_S = 0.25
_MB = 1024 * 1024


def _mb(v: float | None) -> float | None:
    return None if v is None else round(v / _MB, 1)


class ResourceSampler(threading.Thread):
    """Background peak tracker: NVML device memory and PDH GPU memory of *this* pid.

    Runs in the worker (the real interpreter, not the venv launcher), so os.getpid() is the process that
    owns the CUDA context. The PDH query is opened inside the thread because opening costs ~0.3 s.
    """

    def __init__(self, pid: int | None = None, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        super().__init__(name="gtab-sampler", daemon=True)
        self.pid = os.getpid() if pid is None else pid
        self.interval_s = interval_s
        self._stop_evt = threading.Event()
        self.samples = 0
        self.errors: list[str] = []
        self.nvml_used_start: float | None = None  # MB
        self.nvml_used_peak: float | None = None  # MB
        self.self_dedicated_peak: int | None = None  # bytes
        self.self_shared_peak: int | None = None
        self.self_shared_start: int | None = None  # first sample in which this pid had a GPU instance
        self.self_shared_end: int | None = None
        self.adapter_shared_start: int | None = None
        self.adapter_shared_peak: int | None = None
        self.adapter_shared_end: int | None = None

    def _note_error(self, what: str, e: BaseException) -> None:
        msg = f"{what}: {type(e).__name__}: {e}"
        if msg not in self.errors and len(self.errors) < 10:
            self.errors.append(msg)
            log.debug("sampler %s", msg)

    def _sample(self, nvml: sysmon.NvmlSampler | None, pdh: sysmon.PdhGpuMemory | None) -> None:
        self.samples += 1
        if nvml is not None:
            used = nvml.used_mb()
            if used is not None:
                if self.nvml_used_start is None:
                    self.nvml_used_start = used
                self.nvml_used_peak = used if self.nvml_used_peak is None else max(self.nvml_used_peak, used)
        if pdh is not None:
            try:
                s = pdh.sample()
            except Exception as e:
                self._note_error("pdh sample", e)
                return
            ad = s["adapter"]["shared"]
            if self.adapter_shared_start is None:
                self.adapter_shared_start = ad
            self.adapter_shared_peak = ad if self.adapter_shared_peak is None else max(self.adapter_shared_peak, ad)
            self.adapter_shared_end = ad
            mine = s["process"].get(self.pid)
            if mine is not None:
                ded, sh = mine["dedicated"], mine["shared"]
                if self.self_shared_start is None:
                    self.self_shared_start = sh
                self.self_dedicated_peak = ded if self.self_dedicated_peak is None else max(self.self_dedicated_peak, ded)
                self.self_shared_peak = sh if self.self_shared_peak is None else max(self.self_shared_peak, sh)
                self.self_shared_end = sh

    def run(self) -> None:
        nvml: sysmon.NvmlSampler | None = None
        pdh: sysmon.PdhGpuMemory | None = None
        try:
            try:
                nvml = sysmon.NvmlSampler()
                if not nvml.available:
                    nvml = None
            except Exception as e:
                self._note_error("nvml init", e)
            try:
                pdh = sysmon.PdhGpuMemory()
            except Exception as e:
                self._note_error("pdh init", e)
            while True:
                self._sample(nvml, pdh)
                if self._stop_evt.wait(self.interval_s):
                    self._sample(nvml, pdh)  # final reading after the handler finished
                    break
        except Exception as e:  # a monitoring failure must never fail the worker
            self._note_error("sampler", e)
        finally:
            if pdh is not None:
                pdh.close()
            if nvml is not None:
                nvml.close()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout_s)

    def stats(self) -> dict[str, Any]:
        return {
            "nvml_used_peak_mb": self.nvml_used_peak,
            "nvml_used_start_mb": self.nvml_used_start,
            "pdh_self_dedicated_peak_mb": _mb(self.self_dedicated_peak),
            "pdh_self_shared_peak_mb": _mb(self.self_shared_peak),
            "pdh_self_shared_start_mb": _mb(self.self_shared_start),
            "pdh_self_shared_end_mb": _mb(self.self_shared_end),
            "pdh_adapter_shared_start_mb": _mb(self.adapter_shared_start),
            "pdh_adapter_shared_peak_mb": _mb(self.adapter_shared_peak),
            "pdh_adapter_shared_end_mb": _mb(self.adapter_shared_end),
            "samples": self.samples,
            "errors": list(self.errors),
        }


@dataclass
class WorkerContext:
    name: str
    request_path: Path
    result_path: Path
    work_dir: Path
    # Handlers may fill this progressively; it becomes result["output"] if the handler raises,
    # so a failing self-test still reports what it found (versions, device, ...).
    partial: dict[str, Any] = field(default_factory=dict)
    sampler: ResourceSampler | None = None


def _cuda_stats() -> dict[str, float] | None:
    """torch allocator peaks - only if the handler imported torch and actually initialised CUDA."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_initialized():
            return None
        return {
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / _MB, 1),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved() / _MB, 1),
        }
    except Exception as e:
        log.warning("could not read torch CUDA memory stats: %s", e)
        return None


def _worker_name() -> str:
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name:
        return spec.name.rsplit(".", 1)[-1]
    return Path(sys.argv[0]).stem


def _exit(code: int) -> NoReturn:
    # os._exit skips interpreter finalisation: with a CUDA context on Windows that teardown can hang or
    # crash *after* result.json is written, turning a good run into a timeout. The result file is the contract.
    logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)


def worker_main(handler: Callable[[dict, WorkerContext], dict], *, name: str | None = None,
                version: str | None = None) -> NoReturn:
    logging.basicConfig(level=os.environ.get("GTAB_WORKER_LOG_LEVEL", "INFO"), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 3:
        log.error("usage: python -m gtab.workers.<name> REQUEST.json RESULT.json")
        _exit(2)
    request_path, result_path = Path(sys.argv[1]), Path(sys.argv[2])
    ctx = WorkerContext(name=name or _worker_name(), request_path=request_path, result_path=result_path,
                        work_dir=request_path.parent)

    sampler = ResourceSampler()
    sampler.start()
    ctx.sampler = sampler
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    output: dict[str, Any] | None = None
    error: str | None = None
    tb: str | None = None
    try:
        request = atomic.read_json(request_path)
        if not isinstance(request, dict):
            raise TypeError(f"request must be a JSON object, got {type(request).__name__}")
        output = handler(request, ctx)
        if output is None:
            output = {}
        if not isinstance(output, dict):
            raise TypeError(f"handler must return a dict, got {type(output).__name__}")
    except BaseException as e:  # KeyboardInterrupt/SystemExit from a handler are failures too
        error = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        output = ctx.partial or None
        log.error("worker %s failed: %s\n%s", ctx.name, error, tb)
    duration_s = time.perf_counter() - t0
    sampler.stop()
    ok = error is None

    result = {
        "ok": ok,
        "worker": ctx.name,
        "version": version or gtab.__version__,
        "started_utc": started_utc,
        "duration_s": round(duration_s, 3),
        "output": output,
        "error": error,
        "traceback": tb,
        "python": sys.executable,
        "base_executable": getattr(sys, "_base_executable", sys.executable),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "torch_imported": "torch" in sys.modules,
        "cuda": _cuda_stats(),
        "sysmon": sampler.stats(),
    }
    try:
        atomic.write_json(result_path, result)
    except Exception as e:
        log.error("could not write %s: %s", result_path, e)
        _exit(1)
    _exit(0 if ok else 1)
