"""GPU memory monitoring: NVML (via pynvml) and Windows PDH performance counters.

Why both (DESIGN 9.5):
- NVML gives the device-wide picture (driver, total/used/free, which processes touch the GPU). Under WDDM
  it usually cannot attribute memory per process (`usedGpuMemory` comes back as None).
- The Windows "GPU Process Memory" / "GPU Adapter Memory" counters do attribute per pid, and their
  *Shared Usage* is how we notice NVIDIA's Sysmem Fallback silently spilling VRAM into system RAM
  (no OOM, just a ~10x slowdown).

Importable from both the core and the GPU worker env: stdlib + ctypes only; pynvml is imported lazily.
Every public function returns None instead of raising when its backend is unavailable.
"""

from __future__ import annotations

import ctypes
import logging
import re
import sys
from ctypes import wintypes
from typing import Any, Iterable

log = logging.getLogger(__name__)

MB = 1024 * 1024

# --------------------------------------------------------------------------------------------------
# NVML
# --------------------------------------------------------------------------------------------------


def _s(v: Any) -> str:
    """pynvml returned bytes in old releases and str in new ones."""
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def _process_image_name(pid: int) -> str | None:
    """Full image path via QueryFullProcessImageNameW (NVML's own lookup often says 'Insufficient Permissions')."""
    if sys.platform != "win32":
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    k32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buf))
        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value
        return None
    finally:
        k32.CloseHandle(h)


def _import_pynvml() -> Any | None:
    try:
        import pynvml  # package nvidia-ml-py
    except Exception as e:  # ImportError, or a broken install
        log.debug("pynvml unavailable: %s", e)
        return None
    return pynvml


def nvml_info() -> dict | None:
    """Driver, per-device memory and GPU processes; None if NVML is unavailable.

    Shape: {driver, devices: [{index, name, total_mb, used_mb, free_mb, capability: (maj, min)}],
            processes: [{pid, name, used_mb | None, device, kinds: ["compute"|"graphics", ...]}]}
    """
    nv = _import_pynvml()
    if nv is None:
        return None
    try:
        nv.nvmlInit()
    except Exception as e:
        log.debug("nvmlInit failed: %s", e)
        return None
    try:
        info: dict[str, Any] = {"driver": _s(nv.nvmlSystemGetDriverVersion()), "devices": [], "processes": []}
        procs: dict[tuple[int, int], dict[str, Any]] = {}
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            try:
                cap = tuple(nv.nvmlDeviceGetCudaComputeCapability(h))
            except Exception:
                cap = None
            info["devices"].append({
                "index": i, "name": _s(nv.nvmlDeviceGetName(h)),
                "total_mb": mem.total // MB, "used_mb": mem.used // MB, "free_mb": mem.free // MB,
                "capability": cap,
            })
            for kind, fn in (("compute", nv.nvmlDeviceGetComputeRunningProcesses),
                             ("graphics", nv.nvmlDeviceGetGraphicsRunningProcesses)):
                try:
                    plist = fn(h)
                except Exception as e:
                    log.debug("NVML %s process list failed: %s", kind, e)
                    continue
                for p in plist:
                    entry = procs.setdefault((i, p.pid), {
                        "pid": int(p.pid), "name": None, "used_mb": None, "device": i, "kinds": []})
                    entry["kinds"].append(kind)
                    used = getattr(p, "usedGpuMemory", None)  # None under WDDM (NVML_VALUE_NOT_AVAILABLE)
                    if used is not None:
                        entry["used_mb"] = max(entry["used_mb"] or 0, int(used) // MB)
        for entry in procs.values():
            try:
                entry["name"] = _s(nv.nvmlSystemGetProcessName(entry["pid"]))
            except Exception:
                entry["name"] = _process_image_name(entry["pid"])
        info["processes"] = sorted(procs.values(), key=lambda e: (e["device"], e["pid"]))
        return info
    except Exception as e:
        log.warning("NVML query failed: %s", e)
        return None
    finally:
        try:
            nv.nvmlShutdown()
        except Exception:
            pass


class NvmlSampler:
    """Keeps NVML initialised for repeated cheap reads (worker sampler thread polls every 0.25 s)."""

    def __init__(self) -> None:
        self._nv = _import_pynvml()
        self._handles: list[Any] = []
        if self._nv is None:
            return
        try:
            self._nv.nvmlInit()
            self._handles = [self._nv.nvmlDeviceGetHandleByIndex(i) for i in range(self._nv.nvmlDeviceGetCount())]
        except Exception as e:
            log.debug("NVML sampler unavailable: %s", e)
            self._nv = None

    @property
    def available(self) -> bool:
        return self._nv is not None and bool(self._handles)

    def used_mb(self) -> int | None:
        """Device-wide used memory summed over NVIDIA devices (includes other processes)."""
        if not self.available:
            return None
        try:
            return sum(self._nv.nvmlDeviceGetMemoryInfo(h).used for h in self._handles) // MB
        except Exception:
            return None

    def close(self) -> None:
        if self._nv is not None:
            try:
                self._nv.nvmlShutdown()
            except Exception:
                pass
            self._nv = None


# --------------------------------------------------------------------------------------------------
# PDH (Windows performance counters)
# --------------------------------------------------------------------------------------------------

PDH_FMT_LARGE = 0x00000400
PDH_MORE_DATA = 0x800007D2
PDH_CSTATUS_VALID_DATA = 0x00000000
PDH_CSTATUS_NEW_DATA = 0x00000001

# English names work on the Korean-localised Windows because we add them with PdhAddEnglishCounterW.
COUNTERS: dict[tuple[str, str], str] = {
    ("process", "dedicated"): r"\GPU Process Memory(*)\Dedicated Usage",
    ("process", "shared"): r"\GPU Process Memory(*)\Shared Usage",
    ("adapter", "dedicated"): r"\GPU Adapter Memory(*)\Dedicated Usage",
    ("adapter", "shared"): r"\GPU Adapter Memory(*)\Shared Usage",
}


class _PDH_FMT_COUNTERVALUE_U(ctypes.Union):
    _fields_ = [
        ("longValue", wintypes.LONG),
        ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong),
        ("AnsiStringValue", ctypes.c_char_p),
        ("WideStringValue", ctypes.c_wchar_p),
    ]


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    # DWORD CStatus; <4 bytes padding on x64>; union{...}  -> 16 bytes on x64
    _fields_ = [("CStatus", wintypes.DWORD), ("u", _PDH_FMT_COUNTERVALUE_U)]


class PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    # LPWSTR szName; PDH_FMT_COUNTERVALUE FmtValue  -> 24 bytes on x64
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", PDH_FMT_COUNTERVALUE)]


_PID_RE = re.compile(r"^pid_(\d+)_", re.IGNORECASE)


def parse_instance_pid(instance: str) -> int | None:
    """'pid_1234_luid_0x00000000_0x0000D1B5_phys_0' -> 1234; adapter instances ('luid_...') -> None."""
    m = _PID_RE.match(instance or "")
    return int(m.group(1)) if m else None


def sum_by_pid(items: Iterable[tuple[str, int]]) -> dict[int, int]:
    """Sum counter values per pid. A process has one instance per adapter/physical engine set."""
    out: dict[int, int] = {}
    for name, value in items:
        pid = parse_instance_pid(name)
        if pid is not None:
            out[pid] = out.get(pid, 0) + int(value)
    return out


_pdh_dll: Any = None


def _pdh() -> Any:
    global _pdh_dll
    if _pdh_dll is None:
        dll = ctypes.WinDLL("pdh")
        H = wintypes.HANDLE
        dll.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(H)]
        dll.PdhAddEnglishCounterW.argtypes = [H, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(H)]
        dll.PdhCollectQueryData.argtypes = [H]
        dll.PdhGetFormattedCounterArrayW.argtypes = [
            H, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        dll.PdhCloseQuery.argtypes = [H]
        # PDH_STATUS is a LONG, but compare against the documented unsigned hex codes.
        for fn in (dll.PdhOpenQueryW, dll.PdhAddEnglishCounterW, dll.PdhCollectQueryData,
                   dll.PdhGetFormattedCounterArrayW, dll.PdhCloseQuery):
            fn.restype = wintypes.DWORD
        _pdh_dll = dll
    return _pdh_dll


class PdhError(OSError):
    def __init__(self, fn: str, status: int) -> None:
        super().__init__(f"{fn} failed: PDH status 0x{status:08X}")
        self.status = status


class PdhGpuMemory:
    """One open PDH query with the four GPU memory counters.

    Opening a query is the expensive part; re-collecting is cheap, so samplers keep one instance.
    Wildcard counters re-enumerate instances on each collect, so processes that start using the GPU
    after the query was opened (e.g. the worker itself after CUDA init) still show up.
    """

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("PDH is Windows-only")
        self._dll = _pdh()
        self._query = wintypes.HANDLE()
        st = self._dll.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
        if st != 0:
            raise PdhError("PdhOpenQueryW", st)
        self._counters: dict[tuple[str, str], wintypes.HANDLE] = {}
        try:
            for key, path in COUNTERS.items():
                h = wintypes.HANDLE()
                st = self._dll.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(h))
                if st != 0:
                    raise PdhError(f"PdhAddEnglishCounterW({path})", st)
                self._counters[key] = h
        except BaseException:
            self.close()
            raise

    def _array(self, counter: wintypes.HANDLE) -> list[tuple[str, int]]:
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        st = self._dll.PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
        if st == 0 and size.value == 0:
            return []  # no instances (e.g. no GPU process yet)
        if st != PDH_MORE_DATA:
            raise PdhError("PdhGetFormattedCounterArrayW(size)", st)
        # Instances can appear between the two calls; retry with the new size a few times.
        for _ in range(4):
            buf = (ctypes.c_byte * size.value)()
            st = self._dll.PdhGetFormattedCounterArrayW(
                counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buf)
            if st == PDH_MORE_DATA:
                continue
            if st != 0:
                raise PdhError("PdhGetFormattedCounterArrayW", st)
            items = ctypes.cast(buf, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM_W))
            out: list[tuple[str, int]] = []
            for i in range(count.value):  # names point into `buf`, so copy them out before it is freed
                it = items[i]
                if it.FmtValue.CStatus in (PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA):
                    out.append((it.szName or "", int(it.FmtValue.u.largeValue)))
            return out
        raise PdhError("PdhGetFormattedCounterArrayW(retry)", PDH_MORE_DATA)

    def sample(self) -> dict[str, Any]:
        """{"process": {pid: {"dedicated": bytes, "shared": bytes}}, "adapter": {"dedicated": bytes, "shared": bytes}}"""
        st = self._dll.PdhCollectQueryData(self._query)
        if st != 0:
            raise PdhError("PdhCollectQueryData", st)
        process: dict[int, dict[str, int]] = {}
        adapter = {"dedicated": 0, "shared": 0}
        for (scope, kind), counter in self._counters.items():
            items = self._array(counter)
            if scope == "process":
                for pid, value in sum_by_pid(items).items():
                    process.setdefault(pid, {"dedicated": 0, "shared": 0})[kind] = value
            else:
                adapter[kind] = sum(v for _, v in items)
        return {"process": process, "adapter": adapter}

    def close(self) -> None:
        if getattr(self, "_query", None) and self._query.value:
            self._dll.PdhCloseQuery(self._query)
            self._query = wintypes.HANDLE()

    def __enter__(self) -> PdhGpuMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _pdh_sample() -> dict[str, Any] | None:
    try:
        with PdhGpuMemory() as q:
            return q.sample()
    except Exception as e:
        log.debug("PDH GPU counters unavailable: %s", e)
        return None


def pdh_gpu_process_memory() -> dict[int, dict[str, int]] | None:
    """pid -> {"dedicated": bytes, "shared": bytes}; None if the counters are unreadable."""
    s = _pdh_sample()
    return None if s is None else s["process"]


def pdh_gpu_adapter_memory() -> dict[str, int] | None:
    """{"dedicated": bytes, "shared": bytes} summed over adapters; None if the counters are unreadable."""
    s = _pdh_sample()
    return None if s is None else s["adapter"]
