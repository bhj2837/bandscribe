"""Shared GPU-worker machinery: device, determinism, VRAM cap, context measurement, OOM ladder, watchdog.

Used by every GPU worker (sep_msst, amt_muscriptor, beats_beatthis). torch is imported lazily inside
functions, so importing this module is cheap and torch-free (M1_M2_SPEC 0 torch boundary).

Ladder semantics (M1_M2_SPEC 7.1, 8.2):
- Each rung is ``{"label", "device": "cuda"|"cpu", "params"}``. GPU rungs whose worker-side need
  (``reserved_peak_mb + worker_headroom_mb``) exceeds ``torch.cuda.mem_get_info`` free are recorded as
  ``skipped_budget`` unless ``vram.precheck`` is false (bench). The orchestrator's NVML budget is never
  used here: the two sources disagree by ~350 MB on this PC. The orchestrator's *decision* is honoured,
  though: ``vram.start_rung`` (set by ``bandscribe.vram.run_gpu_stage`` when its NVML pre-check picked a lower rung)
  makes the rungs above it ``skipped_budget`` - on WDDM ``mem_get_info`` barely sees other processes' usage,
  so the worker alone would attempt rung 0 while a game holds the VRAM.
- ``torch.OutOfMemoryError`` and ``SharedGrowth`` step down to the next GPU rung. SharedGrowth means this
  process's own Shared Usage grew after its model was resident (Sysmem Fallback spilled VRAM into RAM) -
  the only hard trigger. Adapter-wide shared usage is recorded, never decisive (other apps move it).
- **Load-time spill** (review 2026-10-05): the growth check starts at ``model_loaded``, so a model that already
  spilled while it was uploaded looked clean (bench 2026-09-30 17:58/18:01: shared usage 130-136 MB at
  ``model_loaded`` vs 66 MB after the CUDA context, and chunk=352256 was accepted at 1.4x / 0.87x realtime
  against 5.3x benched). So when the model is resident, this pid's shared usage minus its value before the first
  upload (the higher non-zero reading of the ``pre_load`` and ``cuda_ctx`` marks; a 0 is a stale sample) above
  ``vram.load_spill_fail_mb`` (``gpu.load_spill_fail_mb``,
  48 MB, provisional: normal loads add 0-4 MB, MuScriptor's upstream loader 32 MB, the spills 64-70 MB) is a
  SharedGrowth failure of that rung too.
- Slowness is a warning only (usually contention from a game rendering, which a smaller rung does not fix),
  except (a) when this worker cannot read its PDH counters: then slow alone counts as a spill ("degraded
  detection"), because the spill cannot be seen directly; (b) on a rung below one that failed with
  SharedGrowth: the memory is already over-committed, so a slow lower rung is the spill continuing, not
  contention - it fails too and the stage waits and retries instead of accepting a 4-6x slower result.
  A backend whose speed depends on its output (MuScriptor: decode time grows with note density) passes
  ``slow_can_fail=False``: for it slowness is only ever a warning, and only shared-usage growth fails a rung.
- CPU rungs are used only if ``vram.cpu_allowed`` (policy ``cpu`` and a backend in ``gpu.cpu_backends``,
  decided by the orchestrator) - never because of slowness.
- If nothing succeeded the ladder returns ``(None, attempts)`` and the worker answers
  ``status: "insufficient_vram"``; the stage waits and retries. Other exceptions propagate (a real bug).
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")
_MB = 1024 * 1024
# Rung-level slowness is judged only after this share of the audio (the first chunks include warm-up).
SLOW_MIN_DONE_RATIO = 0.10
CTX_SETTLE_S = 0.25

_BF16_NAMES = ("bf16", "bfloat16")


class SharedGrowth(RuntimeError):
    """This process's Shared Usage grew after its model became resident (Sysmem Fallback spill)."""


class CudaUnavailable(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------ device


def pick_device(req: dict) -> Any:
    """``req["device"]``: ``auto`` (CUDA if available, else CPU) | ``cuda`` (must exist) | ``cpu``."""
    import torch

    want = str(req.get("device", "auto") or "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if want == "cuda":
        raise CudaUnavailable("CUDA requested but torch.cuda.is_available() is False")
    return torch.device("cpu")


def hide_cuda_for_cpu_request(req: dict) -> bool:
    """For ``device: "cpu"`` (the orchestrator's ``cpu`` policy branch) hide the GPU before CUDA initialises.

    Building some models touches CUDA even when they stay on the CPU (MSST's ``Attend`` reads the device
    properties whenever CUDA is available), which creates a ~100-300 MB CUDA context on a GPU that was just
    judged too full. CUDA reads ``CUDA_VISIBLE_DEVICES`` when it initialises (first CUDA call), so this works as
    long as nothing has used CUDA yet. Returns True if the GPU was hidden.
    """
    if str(req.get("device", "auto") or "auto").lower() != "cpu":
        return False
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        log.warning("CUDA already initialised: cannot hide the GPU for a CPU-only request")
        return False
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return True


def apply_determinism(on: bool) -> None:
    """Deterministic kernels where torch has them. Must run before CUDA / cuBLAS initialisation."""
    if not on:
        return
    # cuBLAS reads this when its handle is created; setting it later has no effect.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def apply_cap(req: dict) -> float | None:
    """``vram.cap_mb`` -> ``set_per_process_memory_fraction`` (b4 OOM-ladder test; returns the fraction).

    The PyTorch allocator then raises OOM before real memory runs out, so the driver's Sysmem Fallback never
    engages under a cap (b3 uses a ballast process instead).
    """
    cap = (req.get("vram") or {}).get("cap_mb")
    if cap is None:
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory / _MB
    frac = max(0.01, min(1.0, float(cap) / total))
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    log.info("VRAM cap %.0f MB of %.0f MB (fraction %.3f)", float(cap), total, frac)
    return frac


def _nvml_used_now() -> float | None:
    from bandscribe import sysmon

    s = sysmon.NvmlSampler()
    try:
        return s.used_mb() if s.available else None
    finally:
        s.close()


def init_cuda_measure_ctx(sampler: Any = None) -> float | None:
    """Create the CUDA context and return its size in MB (NVML used after - before), or None.

    NVML device used includes other processes, so this is noisy by whatever they allocate meanwhile; the
    bench takes the median over repeats. The sampler (if given) gets a ``cuda_ctx`` mark for provenance.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    before = _nvml_used_now()
    torch.cuda.init()
    torch.empty(1, device="cuda").fill_(0)  # the context (and allocator) really exist only after a first alloc
    torch.cuda.synchronize()
    time.sleep(CTX_SETTLE_S)  # NVML's device total can lag the allocation by a sample
    after = _nvml_used_now()
    if sampler is not None and hasattr(sampler, "mark"):
        sampler.mark("cuda_ctx")
    if before is None or after is None:
        return None
    return round(max(0.0, float(after) - float(before)), 1)


def mem_get_info_mb() -> tuple[float, float]:
    """(free_mb, total_mb) from the CUDA driver, as seen by this process (its context exists)."""
    import torch

    free, total = torch.cuda.mem_get_info()
    return round(free / _MB, 1), round(total / _MB, 1)


def _has_bf16(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in _BF16_NAMES or "bfloat16" in v.lower()
    if isinstance(v, dict):
        return any(_has_bf16(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_has_bf16(x) for x in v)
    return False


def _dtype_values(obj: Any) -> list[Any]:
    """Every value stored under a key containing 'dtype', anywhere in the request."""
    out: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "dtype" in str(k).lower():
                out.append(v)
            out.extend(_dtype_values(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_dtype_values(v))
    return out


def refuse_bf16(req: dict) -> None:
    """Raise on any bfloat16 request or environment (RTX 2060 = sm_75: no native bf16, DESIGN 9.5)."""
    bad = [v for v in _dtype_values(req) if _has_bf16(v)]
    env_bad = [k for k, v in os.environ.items() if "DTYPE" in k.upper() and _has_bf16(v)]
    if bad or env_bad:
        raise ValueError(f"bfloat16 is refused on this GPU (sm_75 has no native bf16): request={bad} env={env_bad}")


# ----------------------------------------------------------------------------------------- ladder


@dataclass
class Attempt:
    rung_index: int
    rung_label: str
    device: str
    status: str  # ok | oom | shared_growth | unavailable | skipped_budget
    max_allocated_mb: float | None = None
    max_reserved_mb: float | None = None
    nvml_used_peak_mb: float | None = None
    pdh_self_shared_growth_mb: float | None = None
    pdh_adapter_shared_delta_mb: float | None = None
    rtf: float | None = None
    slow: bool = False
    error: str | None = None
    # extras (not in the spec list, informational)
    free_mb: float | None = None
    need_mb: float | None = None
    process_s: float | None = None
    degraded_detection: bool = False
    load_growth_mb: float | None = None  # this pid's shared usage at model_loaded - before the first upload

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRE_LOAD = "pre_load"  # = bandscribe.workers.base.PRE_LOAD (kept here so tests' fake samplers need no base import)
CUDA_CTX = "cuda_ctx"
MODEL_LOADED = "model_loaded"


def _marks(sampler: Any) -> dict[str, Any] | None:
    marks = getattr(sampler, "marks", None)
    return marks if isinstance(marks, dict) else None


def fresh_mark(sampler: Any, label: str) -> dict[str, Any] | None:
    """``sampler.mark(label)`` after waiting for a reading taken now (real ResourceSampler); fakes just mark."""
    if sampler is None or not hasattr(sampler, "mark"):
        return None
    wait = getattr(sampler, "wait_for_sample", None)
    if callable(wait):
        try:
            wait()
        except Exception as e:  # monitoring must never break a run
            log.debug("wait_for_sample failed: %s", e)
    try:
        return sampler.mark(label)
    except Exception as e:
        log.debug("sampler mark %s failed: %s", label, e)
        return None


class Watchdog:
    """Called by the worker between chunks / on progress events while one rung runs.

    ``arm()`` once the rung's model is resident: the baseline for this pid's shared-usage growth, and (GPU rungs,
    ``load_spill_fail_mb`` set) the load-time spill check: shared usage now minus before the first upload above
    the threshold raises SharedGrowth. ``check(audio_s_done)`` raises SharedGrowth iff the growth since arm()
    exceeds ``shared_growth_fail_mb``; slowness (rtf below ``rtf_fail_ratio * rtf_ref`` after >= 10 % of the
    audio) only sets ``slow`` - unless ``slow_can_fail`` and either the PDH self counters are unavailable in this
    worker (degraded detection) or ``slow_is_failure`` (a higher rung already failed with SharedGrowth): then
    slow raises SharedGrowth.

    The running rtf is measured from the *first* check on, so the first chunk (cuDNN/cuBLAS kernel selection,
    first-touch allocations) does not count: on 2026-10-03 a real 205 s song ran at 3.5x overall but was
    flagged slow against a 5.4x reference because of that warm-up alone. ``rtf`` is None until two checks.
    """

    def __init__(self, sampler: Any, *, shared_growth_fail_mb: float, rtf_ref: float | None, rtf_fail_ratio: float,
                 audio_s_total: float, clock: Callable[[], float] = time.perf_counter,
                 load_spill_fail_mb: float | None = None, slow_is_failure: bool = False,
                 slow_can_fail: bool = True) -> None:
        self.sampler = sampler
        self.shared_growth_fail_mb = float(shared_growth_fail_mb)
        self.rtf_ref = None if rtf_ref in (None, 0) else float(rtf_ref)
        self.rtf_fail_ratio = float(rtf_fail_ratio)
        self.audio_s_total = float(audio_s_total)
        self.clock = clock
        self.load_spill_fail_mb = None if load_spill_fail_mb in (None, 0) else float(load_spill_fail_mb)
        self.slow_is_failure = bool(slow_is_failure)
        self.slow_can_fail = bool(slow_can_fail)
        self.armed = False
        self.baseline: dict[str, Any] | None = None
        self.t0: float | None = None
        self.max_growth_mb: float | None = None
        self.adapter_delta_mb: float | None = None
        self.nvml_used_peak_mb: float | None = None
        self.load_growth_mb: float | None = None
        self.slow = False
        self.rtf: float | None = None
        self.degraded_detection = False
        self.checks = 0
        self._rate_base: tuple[float, float] | None = None  # (clock, audio_s_done) at the first check

    def _snap(self) -> dict[str, Any]:
        if self.sampler is None or not hasattr(self.sampler, "snapshot"):
            return {}
        try:
            return dict(self.sampler.snapshot())
        except Exception as e:  # monitoring must never break a run
            log.debug("sampler snapshot failed: %s", e)
            return {}

    @property
    def pdh_available(self) -> bool:
        return self.baseline is not None and self.baseline.get("pdh_self_shared_mb") is not None

    def _load_growth(self) -> float | None:
        """This pid's shared usage with the model resident minus before the first upload (MB), or None when it
        cannot be measured (no PDH, fake sampler, no reference mark). Re-marks ``model_loaded`` with a reading
        taken now: the worker's own mark may be up to one sampling interval old."""
        if self.load_spill_fail_mb is None:
            return None
        marks = _marks(self.sampler)
        if marks is None:
            return None
        # Both marks precede any upload; a reading of exactly 0 is a stale sample from before the CUDA context
        # showed up in the counter (bench 2026-09-30 17:49: cuda_ctx 0.0, then 66 MB at model_loaded with nothing
        # spilled), so zeros are ignored and the higher reading is the reference. None of them: no check.
        refs = [float(m["pdh_self_shared_mb"]) for m in (marks.get(PRE_LOAD), marks.get(CUDA_CTX))
                if isinstance(m, dict) and m.get("pdh_self_shared_mb") is not None]
        refs = [r for r in refs if r > 0.0]
        if not refs:
            return None
        loaded = fresh_mark(self.sampler, MODEL_LOADED) or {}
        cur = loaded.get("pdh_self_shared_mb")
        return None if cur is None else round(float(cur) - max(refs), 1)

    def arm(self) -> None:
        self.load_growth_mb = self._load_growth()
        self.baseline = self._snap()
        self.t0 = self.clock()
        self.armed = True
        self.max_growth_mb = 0.0 if self.pdh_available else None
        self.adapter_delta_mb = 0.0 if self.baseline.get("pdh_adapter_shared_mb") is not None else None
        self.slow = False
        self.rtf = None
        self._rate_base = None
        if self.load_growth_mb is not None and self.load_growth_mb > (self.load_spill_fail_mb or 0.0):
            raise SharedGrowth(f"this process's shared GPU memory grew {self.load_growth_mb:.0f} MB while the model was "
                               f"loaded (> {self.load_spill_fail_mb:.0f} MB): Sysmem Fallback spill at load time")

    def check(self, audio_s_done: float) -> None:
        if not self.armed:
            self.arm()
        self.checks += 1
        snap = self._snap()
        base = self.baseline or {}
        cur_nvml = snap.get("nvml_used_mb")
        if cur_nvml is not None:
            self.nvml_used_peak_mb = cur_nvml if self.nvml_used_peak_mb is None else max(self.nvml_used_peak_mb, cur_nvml)
        cur_ad, base_ad = snap.get("pdh_adapter_shared_mb"), base.get("pdh_adapter_shared_mb")
        if cur_ad is not None and base_ad is not None:
            d = round(cur_ad - base_ad, 1)
            self.adapter_delta_mb = d if self.adapter_delta_mb is None else max(self.adapter_delta_mb, d)
        cur, b = snap.get("pdh_self_shared_mb"), base.get("pdh_self_shared_mb")
        growth = None
        if cur is not None and b is not None:
            growth = round(cur - b, 1)
            self.max_growth_mb = growth if self.max_growth_mb is None else max(self.max_growth_mb, growth)
        now = self.clock()
        if self._rate_base is None:
            self._rate_base = (now, float(audio_s_done))  # end of the first chunk: warm-up excluded from rtf
        else:
            t_base, done_base = self._rate_base
            if now > t_base and audio_s_done > done_base:
                self.rtf = round((audio_s_done - done_base) / (now - t_base), 3)
        if growth is not None and growth > self.shared_growth_fail_mb:
            raise SharedGrowth(f"this process's shared GPU memory grew {growth:.0f} MB after the model was loaded "
                               f"(> {self.shared_growth_fail_mb:.0f} MB): Sysmem Fallback spill")
        done_ratio = audio_s_done / self.audio_s_total if self.audio_s_total > 0 else 0.0
        if (self.rtf_ref is not None and self.rtf is not None and done_ratio >= SLOW_MIN_DONE_RATIO
                and self.rtf < self.rtf_fail_ratio * self.rtf_ref):
            if not self.slow:
                log.warning("rung is slow: %.2fx realtime vs reference %.2fx", self.rtf, self.rtf_ref)
            self.slow = True
            if not self.pdh_available:
                self.degraded_detection = True
            if not self.slow_can_fail:
                return  # e.g. MuScriptor: its speed depends on how many notes it decodes, never a spill signal
            if self.slow_is_failure:
                raise SharedGrowth(f"slow ({self.rtf:.2f}x vs reference {self.rtf_ref:.2f}x) on a rung below one that "
                                   "spilled into shared memory: treated as the spill continuing")
            if not self.pdh_available:
                raise SharedGrowth(f"slow ({self.rtf:.2f}x vs reference {self.rtf_ref:.2f}x) and this worker "
                                   "cannot read its PDH shared-memory counter: treated as a spill (degraded detection)")


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as e:  # after an OOM the allocator may complain; freeing is best effort
        log.debug("empty_cache failed: %s", e)


def _cuda_peaks() -> tuple[float | None, float | None]:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return (round(torch.cuda.max_memory_allocated() / _MB, 1),
                    round(torch.cuda.max_memory_reserved() / _MB, 1))
    except Exception:
        pass
    return None, None


def _reset_peaks() -> None:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        log.debug("reset_peak_memory_stats failed: %s", e)


def _is_oom(e: BaseException) -> bool:
    try:
        import torch

        if isinstance(e, torch.OutOfMemoryError):
            return True
    except Exception:
        pass
    # Some CUDA paths (cuFFT, cuBLAS workspace) raise a plain RuntimeError on allocation failure.
    msg = str(e).lower()
    return isinstance(e, RuntimeError) and ("out of memory" in msg or "cuda_error_out_of_memory" in msg)


def select_rungs(rungs: list[dict], req: dict) -> list[tuple[int, dict]]:
    """(index in the full ladder, rung) pairs to try, applying force_rung and cpu_allowed."""
    vram = req.get("vram") or {}
    force = vram.get("force_rung")
    # A request with device "cpu" comes from the orchestrator's "cpu" policy branch (run_gpu_stage), which
    # only exists for backends in gpu.cpu_backends.
    cpu_ok = bool(vram.get("cpu_allowed")) or str(req.get("device", "auto")).lower() == "cpu"
    indexed = list(enumerate(rungs))
    if force:
        chosen = [(i, r) for i, r in indexed if r["label"] == force]
        if not chosen:
            raise ValueError(f"force_rung {force!r} is not one of {[r['label'] for r in rungs]}")
        if chosen[0][1]["device"] == "cpu" and not cpu_ok:
            raise ValueError(f"force_rung {force!r} is a CPU rung but vram.cpu_allowed is false")
        return chosen
    if str(req.get("device", "auto")).lower() == "cpu":
        return [(i, r) for i, r in indexed if r["device"] == "cpu"]
    return [(i, r) for i, r in indexed if r["device"] != "cpu" or cpu_ok]


def run_ladder(rungs: list[dict], attempt: Callable[[dict, Watchdog], T], *, sampler: Any, req: dict,
               audio_s_total: float = 0.0,
               mem_info: Callable[[], tuple[float, float]] | None = None,
               cuda_available: Callable[[], bool] | None = None,
               slow_can_fail: bool = True) -> tuple[T | None, list[Attempt]]:
    """Try rungs in order; returns (result of the first rung that succeeded | None, attempts).

    ``attempt(rung, watchdog)`` loads the model on ``rung["device"]``, calls ``watchdog.arm()`` (and
    ``sampler.mark("model_loaded")``) once it is resident, calls ``watchdog.check(audio_s_done)`` as it goes,
    and must move its model off the GPU before re-raising (the ladder then frees the cache).
    ``mem_info`` / ``cuda_available`` are seams for tests (default: torch). ``slow_can_fail=False`` (MuScriptor)
    keeps slowness a warning in every case (see the module docstring).
    """
    vram = req.get("vram") or {}
    precheck = bool(vram.get("precheck", True))
    headroom = float(vram.get("worker_headroom_mb", 256.0))
    need_table = vram.get("need_table") or {}
    rtf_ref = vram.get("rtf_ref") or {}
    fail_mb = float(vram.get("shared_growth_fail_mb", 64.0))
    ratio = float(vram.get("rtf_fail_ratio", 0.5))
    load_fail = vram.get("load_spill_fail_mb")  # absent (old requests, hand-built ones): no load-time check
    load_fail = None if load_fail is None else float(load_fail)
    spilled = False  # a rung above failed with SharedGrowth
    pre_marked = False
    if mem_info is None:
        mem_info = mem_get_info_mb
    if cuda_available is None:
        def cuda_available() -> bool:
            import torch

            return bool(torch.cuda.is_available())

    selected = select_rungs(rungs, req)
    # The orchestrator's NVML pre-check may have chosen a lower rung (vram.start_rung, set by
    # bandscribe.vram.run_gpu_stage): GPU rungs above it are recorded as skipped_budget and not attempted. Ignored
    # under force_rung, without the pre-check (bench) and for a label this worker does not know.
    start = vram.get("start_rung")
    skip_before = 0
    if start and precheck and not vram.get("force_rung"):
        pos = next((k for k, (_i, r) in enumerate(selected) if r["label"] == start), None)
        if pos is None:
            log.warning("vram.start_rung %r is not one of this worker's rungs; ignored", start)
        else:
            skip_before = pos

    attempts: list[Attempt] = []
    for k, (index, rung) in enumerate(selected):
        label, device = rung["label"], rung["device"]
        rec = Attempt(rung_index=index, rung_label=label, device=device, status="ok")
        if k < skip_before and device != "cpu":
            rec.status = "skipped_budget"
            rec.error = f"orchestrator NVML budget chose {start}"
            log.info("rung %s skipped: %s", label, rec.error)
            attempts.append(rec)
            continue
        if device != "cpu":
            if not cuda_available():
                rec.status, rec.error = "unavailable", "CUDA not available"
                attempts.append(rec)
                continue
            need = need_table.get(label) or {}
            reserved = need.get("reserved_peak_mb")
            free, _total = mem_info()
            rec.free_mb = free
            if reserved is not None:
                rec.need_mb = round(float(reserved) + headroom, 1)
            if precheck and rec.need_mb is not None and rec.need_mb > free:
                rec.status = "skipped_budget"
                rec.error = f"need {rec.need_mb:.0f} MB (reserved + headroom) > mem_get_info free {free:.0f} MB"
                log.info("rung %s skipped: %s", label, rec.error)
                attempts.append(rec)
                continue
            _free_cuda()
            _reset_peaks()
            if not pre_marked:  # the load-spill reference: context exists, no model uploaded yet
                fresh_mark(sampler, PRE_LOAD)
                pre_marked = True
        gpu = device != "cpu"
        wd = Watchdog(sampler, shared_growth_fail_mb=fail_mb, rtf_ref=rtf_ref.get(label), rtf_fail_ratio=ratio,
                      audio_s_total=audio_s_total, load_spill_fail_mb=load_fail if gpu else None,
                      slow_is_failure=spilled and gpu, slow_can_fail=slow_can_fail)
        t0 = time.perf_counter()
        failure: BaseException | None = None
        try:
            result = attempt(rung, wd)
        except SharedGrowth as e:
            rec.status, rec.error = "shared_growth", str(e)
            failure = e
        except BaseException as e:
            if device != "cpu" and _is_oom(e):
                rec.status, rec.error = "oom", f"{type(e).__name__}: {str(e)[:300]}"
                failure = e
            else:
                raise
        dt = time.perf_counter() - t0
        rec.process_s = round(dt, 3)
        if device != "cpu":
            rec.max_allocated_mb, rec.max_reserved_mb = _cuda_peaks()
        rec.nvml_used_peak_mb = wd.nvml_used_peak_mb
        rec.pdh_self_shared_growth_mb = wd.max_growth_mb
        rec.pdh_adapter_shared_delta_mb = wd.adapter_delta_mb
        rec.load_growth_mb = wd.load_growth_mb
        rec.slow = wd.slow
        rec.degraded_detection = wd.degraded_detection
        if rec.status == "shared_growth":
            spilled = True
        # A finished rung: whole-run rtf (incl. model load to GPU and warm-up, like the user experiences it);
        # a failed one: the watchdog's running rtf up to the failure.
        if failure is None and audio_s_total > 0 and dt > 0:
            rec.rtf = round(audio_s_total / dt, 3)
        else:
            rec.rtf = wd.rtf
        attempts.append(rec)
        if failure is None:
            return result, attempts
        del failure  # drop the traceback (it references the attempt's GPU tensors) before freeing
        log.warning("rung %s failed (%s); stepping down", label, rec.status)
        _free_cuda()
    if attempts and all(a.status == "unavailable" for a in attempts):
        raise CudaUnavailable("no rung could run: CUDA is not available and no CPU rung is allowed")
    return None, attempts


def attempts_summary(attempts: list[Attempt]) -> list[dict[str, Any]]:
    return [a.to_dict() for a in attempts]
