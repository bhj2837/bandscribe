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


# Default threshold for `shared_growth` in stats(); the orchestrator sends the configured value in
# request["vram"]["shared_growth_fail_mb"] (gpu.shared_growth_fail_mb, calibrated by `gtab bench gpu`).
DEFAULT_SHARED_GROWTH_FAIL_MB = 64.0
# The mark that separates legitimate staging (weights uploaded through shared memory) from a spill.
MODEL_LOADED = "model_loaded"


class ResourceSampler(threading.Thread):
    """Background peak tracker: NVML device memory and PDH GPU memory of *this* pid.

    Runs in the worker (the real interpreter, not the venv launcher), so os.getpid() is the process that
    owns the CUDA context. The PDH query is opened inside the thread because opening costs ~0.3 s.

    M2 additions (M1_M2_SPEC 7.1, 8.2): ``snapshot()`` returns the latest and peak readings thread-safely,
    ``mark(label)`` stores a snapshot under ``marks[label]`` and starts tracking this pid's shared-usage peak
    *after* that point. Sysmem Fallback shows up as this process's Shared Usage growing after its model is
    resident (``mark("model_loaded")``); shared usage during the upload itself is legitimate staging, and the
    adapter-wide value moves with every other app (Chrome, Discord, the Claude app), so neither decides.
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
        # Latest readings (for snapshot()); _sample runs on the sampler thread, snapshot()/mark() elsewhere.
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self.nvml_used_last: float | None = None  # MB
        self.self_dedicated_last: int | None = None  # bytes
        self.adapter_dedicated_last: int | None = None
        self.marks: dict[str, dict[str, Any]] = {}
        # label -> this pid's shared-usage peak (bytes) seen at or after the mark
        self._self_shared_peak_after: dict[str, int | None] = {}
        self._adapter_shared_peak_after: dict[str, int | None] = {}
        self.shared_growth_fail_mb: float = DEFAULT_SHARED_GROWTH_FAIL_MB

    def _note_error(self, what: str, e: BaseException) -> None:
        msg = f"{what}: {type(e).__name__}: {e}"
        if msg not in self.errors and len(self.errors) < 10:
            self.errors.append(msg)
            log.debug("sampler %s", msg)

    def _sample(self, nvml: sysmon.NvmlSampler | None, pdh: sysmon.PdhGpuMemory | None) -> None:
        self.samples += 1
        used = nvml.used_mb() if nvml is not None else None
        s: dict[str, Any] | None = None
        if pdh is not None:
            try:
                s = pdh.sample()
            except Exception as e:
                self._note_error("pdh sample", e)
        with self._lock:
            if used is not None:
                if self.nvml_used_start is None:
                    self.nvml_used_start = used
                self.nvml_used_peak = used if self.nvml_used_peak is None else max(self.nvml_used_peak, used)
                self.nvml_used_last = used
            if s is None:
                return
            ad = s["adapter"]["shared"]
            if self.adapter_shared_start is None:
                self.adapter_shared_start = ad
            self.adapter_shared_peak = ad if self.adapter_shared_peak is None else max(self.adapter_shared_peak, ad)
            self.adapter_shared_end = ad
            self.adapter_dedicated_last = s["adapter"].get("dedicated")
            for label, peak in self._adapter_shared_peak_after.items():
                self._adapter_shared_peak_after[label] = ad if peak is None else max(peak, ad)
            mine = s["process"].get(self.pid)
            if mine is not None:
                ded, sh = mine["dedicated"], mine["shared"]
                if self.self_shared_start is None:
                    self.self_shared_start = sh
                self.self_dedicated_peak = ded if self.self_dedicated_peak is None else max(self.self_dedicated_peak, ded)
                self.self_shared_peak = sh if self.self_shared_peak is None else max(self.self_shared_peak, sh)
                self.self_shared_end = sh
                self.self_dedicated_last = ded
                for label, peak in self._self_shared_peak_after.items():
                    self._self_shared_peak_after[label] = sh if peak is None else max(peak, sh)

    def snapshot(self) -> dict[str, Any]:
        """Latest and peak readings (MB), thread-safe. Values are None where the source is unavailable.

        The sampler thread refreshes them every ``interval_s``, so a snapshot may be that much old.
        """
        with self._lock:
            return {
                "t_s": round(time.perf_counter() - self._t0, 3),
                "nvml_used_mb": self.nvml_used_last,
                "nvml_used_peak_mb": self.nvml_used_peak,
                "pdh_self_shared_mb": _mb(self.self_shared_end),
                "pdh_self_shared_peak_mb": _mb(self.self_shared_peak),
                "pdh_self_dedicated_mb": _mb(self.self_dedicated_last),
                "pdh_adapter_shared_mb": _mb(self.adapter_shared_end),
                "pdh_adapter_shared_peak_mb": _mb(self.adapter_shared_peak),
                "pdh_adapter_dedicated_mb": _mb(self.adapter_dedicated_last),
                "samples": self.samples,
            }

    def mark(self, label: str) -> dict[str, Any]:
        """Store the current snapshot under ``marks[label]`` and track peaks from here on.

        Marking the same label again replaces it (a worker that steps down its ladder re-marks
        ``model_loaded`` for the rung it retries, so the growth fields describe the rung that finished).
        """
        snap = self.snapshot()
        entry = {"t_s": snap["t_s"], "pdh_self_shared_mb": snap["pdh_self_shared_mb"],
                 "pdh_adapter_shared_mb": snap["pdh_adapter_shared_mb"], "nvml_used_mb": snap["nvml_used_mb"]}
        with self._lock:
            self.marks[label] = entry
            self._self_shared_peak_after[label] = self.self_shared_end
            self._adapter_shared_peak_after[label] = self.adapter_shared_end
        return dict(entry)

    def growth_since(self, label: str) -> float | None:
        """This pid's shared-usage peak after ``mark(label)`` minus its value at the mark (MB); None if unknown."""
        with self._lock:
            mark = self.marks.get(label)
            peak = self._self_shared_peak_after.get(label)
        if mark is None or peak is None or mark.get("pdh_self_shared_mb") is None:
            return None
        return round(_mb(peak) - mark["pdh_self_shared_mb"], 1)

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
        growth = self.growth_since(MODEL_LOADED)
        with self._lock:
            self_start, self_peak = self.self_shared_start, self.self_shared_peak
            ad_start, ad_peak = self.adapter_shared_start, self.adapter_shared_peak
            marks = {k: dict(v) for k, v in self.marks.items()}
        self_delta = None if self_start is None or self_peak is None else _mb(self_peak - self_start)
        ad_delta = None if ad_start is None or ad_peak is None else _mb(ad_peak - ad_start)
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
            # M2 (M1_M2_SPEC 7.1): the only hard Sysmem-Fallback signal is growth after model_loaded.
            "marks": marks,
            "pdh_self_shared_growth_mb": growth,
            "pdh_self_shared_delta_mb": self_delta,  # info: includes staging during the upload
            "pdh_adapter_shared_delta_mb": ad_delta,  # info only: other apps move it
            "shared_growth_fail_mb": self.shared_growth_fail_mb,
            "shared_growth": bool(growth is not None and growth > self.shared_growth_fail_mb),
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


def _configure_sampler(sampler: ResourceSampler, request: dict) -> None:
    """Take the shared-growth threshold from the request envelope (M1_M2_SPEC 7.1), if present."""
    vram = request.get("vram")
    if isinstance(vram, dict) and vram.get("shared_growth_fail_mb") is not None:
        try:
            sampler.shared_growth_fail_mb = float(vram["shared_growth_fail_mb"])
        except (TypeError, ValueError):
            log.warning("ignoring invalid vram.shared_growth_fail_mb=%r", vram.get("shared_growth_fail_mb"))


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
        _configure_sampler(sampler, request)
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
