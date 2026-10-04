"""sysmon: PDH instance parsing, struct layouts, graceful None, live reads, and the worker's sampler thread."""
from __future__ import annotations

import ctypes
import os
import sys
import time
import types

import pytest

from bandscribe import sysmon
from bandscribe.workers import base


@pytest.mark.parametrize("name, pid", [
    ("pid_1234_luid_0x00000000_0x0000D1B5_phys_0", 1234),
    ("pid_4_luid_0x00000000_0x0000EAF1_phys_0", 4),
    ("PID_99_luid_0x0_0x1_phys_0", 99),
    ("luid_0x00000000_0x0000D1B5_phys_0", None),  # adapter instance
    ("pid__luid_0x0", None),
    ("pid_12x_luid", None),
    ("", None),
    ("_Total", None),
])
def test_parse_instance_pid(name, pid):
    assert sysmon.parse_instance_pid(name) == pid


def test_sum_by_pid_sums_instances_of_one_process():
    items = [
        ("pid_10_luid_0x00000000_0x0000D1B5_phys_0", 100),
        ("pid_10_luid_0x00000000_0x0000EAF1_phys_0", 5),  # same process on a second adapter
        ("pid_20_luid_0x00000000_0x0000D1B5_phys_0", 7),
        ("luid_0x00000000_0x0000D1B5_phys_0", 1000),  # ignored
    ]
    assert sysmon.sum_by_pid(items) == {10: 105, 20: 7}


def test_pdh_struct_layout_x64():
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(sysmon.PDH_FMT_COUNTERVALUE) == 16
    assert sysmon.PDH_FMT_COUNTERVALUE.u.offset == 8
    assert ctypes.sizeof(sysmon.PDH_FMT_COUNTERVALUE_ITEM_W) == 24
    assert sysmon.PDH_FMT_COUNTERVALUE_ITEM_W.FmtValue.offset == 8


def test_nvml_info_none_without_pynvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # makes `import pynvml` raise ImportError
    assert sysmon.nvml_info() is None
    s = sysmon.NvmlSampler()
    assert not s.available and s.used_mb() is None
    s.close()


def test_nvml_info_none_when_init_fails(monkeypatch):
    fake = types.ModuleType("pynvml")

    def boom():
        raise RuntimeError("NVML Shared Library Not Found")

    fake.nvmlInit = boom
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert sysmon.nvml_info() is None


def test_nvml_info_shape_with_fake_wddm(monkeypatch):
    """usedGpuMemory=None (WDDM) must map to used_mb=None; name falls back when NVML refuses."""
    fake = types.ModuleType("pynvml")
    mem = types.SimpleNamespace(total=6 << 30, used=2 << 30, free=4 << 30)
    procs = [types.SimpleNamespace(pid=111, usedGpuMemory=None), types.SimpleNamespace(pid=222, usedGpuMemory=512 << 20)]
    fake.nvmlInit = lambda: None
    fake.nvmlShutdown = lambda: None
    fake.nvmlSystemGetDriverVersion = lambda: b"610.47"
    fake.nvmlDeviceGetCount = lambda: 1
    fake.nvmlDeviceGetHandleByIndex = lambda i: object()
    fake.nvmlDeviceGetMemoryInfo = lambda h: mem
    fake.nvmlDeviceGetCudaComputeCapability = lambda h: (7, 5)
    fake.nvmlDeviceGetName = lambda h: "NVIDIA GeForce RTX 2060"
    fake.nvmlDeviceGetComputeRunningProcesses = lambda h: procs
    fake.nvmlDeviceGetGraphicsRunningProcesses = lambda h: procs[:1]

    def name(pid):
        if pid == 111:
            raise RuntimeError("Insufficient Permissions")
        return "C:\\x\\worker.exe"

    fake.nvmlSystemGetProcessName = name
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(sysmon, "_process_image_name", lambda pid: None)
    info = sysmon.nvml_info()
    assert info["driver"] == "610.47"
    assert info["devices"] == [{"index": 0, "name": "NVIDIA GeForce RTX 2060", "total_mb": 6144,
                                "used_mb": 2048, "free_mb": 4096, "capability": (7, 5)}]
    by_pid = {p["pid"]: p for p in info["processes"]}
    assert by_pid[111]["used_mb"] is None and by_pid[111]["name"] is None
    assert sorted(by_pid[111]["kinds"]) == ["compute", "graphics"]
    assert by_pid[222]["used_mb"] == 512 and by_pid[222]["name"] == "C:\\x\\worker.exe"


def test_nvml_info_live_does_not_raise():
    info = sysmon.nvml_info()
    assert info is None or isinstance(info, dict)
    if info:
        assert info["driver"]
        for d in info["devices"]:
            assert d["total_mb"] > 0 and d["used_mb"] + d["free_mb"] <= d["total_mb"] + 1
        for p in info["processes"]:
            assert set(p) >= {"pid", "name", "used_mb"}


def test_pdh_none_when_query_cannot_open(monkeypatch):
    def fail(self):
        raise sysmon.PdhError("PdhOpenQueryW", 0xC0000BB8)

    monkeypatch.setattr(sysmon.PdhGpuMemory, "__init__", fail)
    assert sysmon.pdh_gpu_process_memory() is None
    assert sysmon.pdh_gpu_adapter_memory() is None


@pytest.mark.skipif(sys.platform != "win32", reason="PDH is Windows-only")
def test_pdh_live_reads():
    adapter = sysmon.pdh_gpu_adapter_memory()
    procs = sysmon.pdh_gpu_process_memory()
    # The counters exist on this PC (DESIGN 9.5); on a machine without them both are None, not an exception.
    assert (adapter is None) == (procs is None)
    if adapter is None:
        pytest.skip("GPU PDH counters not available")
    assert set(adapter) == {"dedicated", "shared"}
    assert all(isinstance(v, int) and v >= 0 for v in adapter.values())
    for pid, v in procs.items():
        assert isinstance(pid, int) and pid >= 0
        assert set(v) == {"dedicated", "shared"} and v["dedicated"] >= 0 and v["shared"] >= 0
    with sysmon.PdhGpuMemory() as q:  # repeated samples on one open query
        a, b = q.sample(), q.sample()
    assert set(a) == set(b) == {"process", "adapter"}


def test_resource_sampler_collects_and_stops():
    s = base.ResourceSampler(interval_s=0.05)
    s.start()
    time.sleep(0.8)  # PDH open alone takes ~0.3 s
    s.stop()
    assert not s.is_alive()
    st = s.stats()
    assert st["samples"] >= 2
    assert set(st) >= {"nvml_used_peak_mb", "pdh_self_dedicated_peak_mb", "pdh_self_shared_peak_mb",
                       "pdh_self_shared_start_mb"}
    if sys.platform == "win32" and not st["errors"]:
        assert st["pdh_adapter_shared_peak_mb"] is not None
    assert s.pid == os.getpid()


def test_resource_sampler_tracks_peaks_from_fake_samples():
    class FakePdh:
        def __init__(self, seq):
            self.seq = iter(seq)

        def sample(self):
            return next(self.seq)

    me = 4242
    seq = [
        {"process": {}, "adapter": {"dedicated": 0, "shared": 100 << 20}},
        {"process": {me: {"dedicated": 50 << 20, "shared": 10 << 20}}, "adapter": {"dedicated": 0, "shared": 120 << 20}},
        {"process": {me: {"dedicated": 900 << 20, "shared": 30 << 20}}, "adapter": {"dedicated": 0, "shared": 140 << 20}},
        {"process": {me: {"dedicated": 20 << 20, "shared": 12 << 20}}, "adapter": {"dedicated": 0, "shared": 110 << 20}},
    ]
    s = base.ResourceSampler(pid=me)
    pdh = FakePdh(seq)
    for _ in seq:
        s._sample(None, pdh)  # type: ignore[arg-type]
    st = s.stats()
    assert st["pdh_self_shared_start_mb"] == 10.0  # first sample in which the pid appeared
    assert st["pdh_self_shared_peak_mb"] == 30.0
    assert st["pdh_self_shared_end_mb"] == 12.0
    assert st["pdh_self_dedicated_peak_mb"] == 900.0
    assert (st["pdh_adapter_shared_start_mb"], st["pdh_adapter_shared_peak_mb"]) == (100.0, 140.0)


def test_cuda_stats_none_without_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert base._cuda_stats() is None
