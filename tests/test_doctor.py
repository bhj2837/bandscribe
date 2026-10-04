"""Tests for bandscribe.doctor: parsers, check isolation, required-vs-warning logic, plus a real smoke run."""

from __future__ import annotations

import json
import sys
import time
from collections import namedtuple
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bandscribe import doctor, paths
from bandscribe.doctor import Check, DoctorReport, _Proc

MB = 1024 * 1024
DiskUsage = namedtuple("DiskUsage", "total used free")

FFMPEG_VERSION = "ffmpeg version 8.0.1-full_build-www.gyan.dev Copyright (c) 2000-2025 the FFmpeg developers\n"
BUILDCONF = """\
  configuration:
    --enable-gpl
    --enable-version3
    --enable-libsoxr
    --enable-chromaprint
    --enable-libmp3lame
"""
FILTERS = """\
Filters:
  T.. = Timeline support
  .S. = Slice threading
  A = Audio input/output
  | = Source or sink filter
 TS aap               AA->A      Apply Affine Projection algorithm to first audio stream.
 .. abuffer           |->A       Buffer audio frames, and make them accessible to the filterchain.
 .. ebur128           A->N       EBU R128 scanner.
 T. loudnorm          A->A       EBU R128 loudness normalization
"""
GLOBAL_INFO = {"version": "3.10.8", "executable": r"C:\Python310\python.exe", "dists": 58,
               "dists_sha256": "ab" * 32, "torch": "2.10.0+cpu"}

GOOD_SELFTEST = {
    "ok": True,
    "worker": "selftest",
    "output": {
        "torch_version": "2.11.0+cu128", "cuda_version": "12.8", "cuda_available": True,
        "device_count": 1, "device_name": "NVIDIA GeForce RTX 2060", "capability": [7, 5],
        "total_vram_mb": 6143.7, "bf16_supported_native": False, "bf16_supported_emulated": True,
        "fp16_matmul": {"n": 1024, "finite": True, "max_rel_err": 4e-4, "dtype": "torch.float16", "ok": True, "ms": 160},
        "autocast_conv": {"finite": True, "max_rel_err": 5e-4, "dtype": "torch.float16", "ok": True, "ms": 80},
        "passed": True,
    },
    "error": None,
    "python": r"D:\gtab\envs\gpu\.venv\Scripts\python.exe",
    "base_executable": r"D:\gtab\tools\python\cpython-3.12\python.exe",
    "pid": 222, "ppid": 111,
    "cuda": {"max_allocated_mb": 20.0, "max_reserved_mb": 44.0},
    "sysmon": {"nvml_used_peak_mb": 900, "pdh_self_dedicated_peak_mb": 300,
               "pdh_self_shared_start_mb": 48, "pdh_self_shared_peak_mb": 68},
}


def _worker_result(result: dict[str, Any] | None = None, *, ok: bool | None = None, work_dir: Path | None = None,
                   **kw: Any) -> SimpleNamespace:
    result = GOOD_SELFTEST if result is None else result
    base = dict(ok=result.get("ok") is True if ok is None else ok, returncode=0 if result.get("ok") else 1,
                result=result, work_dir=work_dir or Path("."), stdout_path=None, stderr_path=None,
                duration_s=3.0, timed_out=False, killed=False, lock_wait_s=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


class FakeGpu:
    class GpuLockTimeout(TimeoutError):
        pass

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.next: Any = None  # WorkerResult-like or an exception to raise

    def run_worker(self, name: str, request: dict, work_dir: Path, **kw: Any) -> Any:
        self.calls.append((name, request, Path(work_dir), kw))
        if isinstance(self.next, BaseException):
            raise self.next
        return self.next if self.next is not None else _worker_result(work_dir=work_dir)


class FakeSysmon:
    def __init__(self) -> None:
        self.nvml: Any = {
            "driver": "610.47",
            "devices": [{"index": 0, "name": "NVIDIA GeForce RTX 2060", "total_mb": 6144, "used_mb": 400,
                         "free_mb": 5744, "capability": (7, 5)}],
            "processes": [{"pid": 100, "name": r"C:\Program Files\Google\chrome.exe", "used_mb": None}],
        }
        self.procs: Any = {100: {"dedicated": 300 * MB, "shared": 10 * MB}, 200: {"dedicated": 0, "shared": 0}}
        self.adapter: Any = {"dedicated": 400 * MB, "shared": 50 * MB}

    def nvml_info(self) -> Any:
        return self.nvml() if callable(self.nvml) else self.nvml

    def pdh_gpu_process_memory(self) -> Any:
        return self.procs() if callable(self.procs) else self.procs

    def pdh_gpu_adapter_memory(self) -> Any:
        return self.adapter


class FakeRunner:
    """Stands in for doctor._run; responses keyed by executable stem (ffmpeg, node, yt-dlp, py)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.overrides: dict[str, _Proc] = {}

    def __call__(self, args: list[Any], *, timeout: float = 0, env: Any = None) -> _Proc:
        argv = [str(a) for a in args]
        self.calls.append(argv)
        stem = Path(argv[0]).stem.lower()
        if stem == "ffmpeg":
            key = next(a for a in argv[1:] if a != "-hide_banner")
            if f"ffmpeg{key}" in self.overrides:
                return self.overrides[f"ffmpeg{key}"]
            return _Proc(0, {"-version": FFMPEG_VERSION, "-buildconf": BUILDCONF, "-filters": FILTERS}[key], "")
        if stem in self.overrides:
            return self.overrides[stem]
        if stem == "node":
            return _Proc(0, "v24.10.0\n")
        if stem == "yt-dlp":
            return _Proc(0, "2026.09.27.232945\n")
        if stem == "py":
            return _Proc(0, json.dumps(GLOBAL_INFO) + "\n")
        return _Proc(None, error=f"unexpected command {argv}")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Every probe faked: a healthy machine. Tests break one thing at a time."""
    data = tmp_path / "data"
    tools = tmp_path / "tools"
    tools.mkdir()
    gpu_py = tmp_path / "gpu" / "python.exe"
    gpu_py.parent.mkdir()
    gpu_py.write_bytes(b"")
    exe = tools / "yt-dlp.exe"
    exe.write_bytes(b"")
    conf = tools / "yt-dlp.conf"
    conf.write_text("# bandscribe\n--js-runtimes node\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA", data)
    # M2 "models" group (SEP): a healthy machine has its models recorded and present (absolute paths
    # override BANDSCRIBE_ROOT below). No vram_table.json yet is fine (defaults are used, reported as ok).
    monkeypatch.setattr(paths, "MODELS", data / "models")
    monkeypatch.setattr(paths, "MODELS_LOCK", data / "models" / "models.lock.json")
    monkeypatch.setattr(paths, "VRAM_TABLE", data / "bench" / "vram_table.json")
    ckpt = tmp_path / "final0.ckpt"
    ckpt.write_bytes(b"x")
    (data / "models").mkdir(parents=True)
    (data / "models" / "models.lock.json").write_text(json.dumps(
        {"format": "bandscribe.models/1", "models": {"beat_this.final0": {"path": str(ckpt)}}}), encoding="utf-8")
    monkeypatch.setattr(paths, "GPU_PYTHON", gpu_py)
    monkeypatch.setattr(paths, "YTDLP_EXE", exe)
    monkeypatch.setattr(paths, "YTDLP_CONF", conf)
    monkeypatch.setattr(paths, "ROOT", Path(r"D:\gtab"))
    monkeypatch.setattr(paths, "child_env", lambda extra=None: {
        "HF_HOME": r"D:\gtab\data\hf", "TORCH_HOME": r"D:\gtab\data\torch", "UV_CACHE_DIR": r"D:\gtab\data\uv-cache"})

    from bandscribe.eval import node as gnode

    monkeypatch.setattr(gnode, "alphatab_available", lambda: True)  # node.alphatab check (non-required)

    gpu, sysmon, runner = FakeGpu(), FakeSysmon(), FakeRunner()
    modules: dict[str, Any] = {"bandscribe.gpu": gpu, "bandscribe.sysmon": sysmon}

    def fake_import(name: str) -> Any:
        mod = modules[name]
        if isinstance(mod, BaseException):
            raise mod
        return mod

    which = {"ffmpeg": r"C:\ffmpeg\bin\ffmpeg.exe", "ffprobe": r"C:\ffmpeg\bin\ffprobe.exe",
             "node": r"C:\Program Files\nodejs\node.exe", "py": r"C:\Windows\py.exe"}
    disk = {"usage": DiskUsage(1000 * 2**30, 700 * 2**30, 300 * 2**30)}
    monkeypatch.setattr(doctor, "_import", fake_import)
    monkeypatch.setattr(doctor, "_run", runner)
    monkeypatch.setattr(doctor, "_which", lambda name: which.get(name))
    monkeypatch.setattr(doctor, "_disk_usage", lambda p: disk["usage"])
    monkeypatch.setattr(doctor, "_today", lambda: date(2026, 9, 30))
    cfg = {"youtube": {"enabled": True, "cookies": False}, "doctor": {"min_free_disk_gb": 60, "min_node_major": 22},
           "gpu": {"vram_margin_gb": 0.7, "target_reserved_gb": 4.5, "insufficient_vram_policy": "wait"},
           "consent": {"youtube": "2026-09-30 사용자 동의"}}
    return SimpleNamespace(gpu=gpu, sysmon=sysmon, runner=runner, modules=modules, which=which, disk=disk,
                           cfg=cfg, data=data, exe=exe, conf=conf, gpu_py=gpu_py)


def _statuses(report: DoctorReport) -> dict[str, str]:
    return {c.name: c.status for c in report.checks}


# ------------------------------------------------------------------------------------------ parsers


def test_parse_ffmpeg_buildconf() -> None:
    flags = doctor.parse_ffmpeg_buildconf(BUILDCONF)
    assert {"--enable-libsoxr", "--enable-chromaprint", "--enable-gpl"} <= flags
    assert "configuration:" not in flags
    assert doctor.parse_ffmpeg_buildconf("") == set()


def test_parse_ffmpeg_filters_skips_legend_and_accepts_old_layout() -> None:
    names = doctor.parse_ffmpeg_filters(FILTERS)
    assert names == {"aap", "abuffer", "ebur128", "loudnorm"}
    # ffmpeg <= 6 had a third "C" (command support) flag column.
    assert doctor.parse_ffmpeg_filters(" TSC ebur128           A->N       EBU R128 scanner.") == {"ebur128"}
    assert doctor.parse_ffmpeg_filters("  T.. = Timeline support\n") == set()


def test_parse_ffmpeg_version() -> None:
    assert doctor.parse_ffmpeg_version(FFMPEG_VERSION) == "8.0.1-full_build-www.gyan.dev"
    assert doctor.parse_ffmpeg_version("garbage") is None


@pytest.mark.parametrize(("text", "expected"), [
    ("v24.10.0\n", (24, 10, 0)),
    ("v22.0.0-nightly20240101abc", (22, 0, 0)),
    ("18.19.1", (18, 19, 1)),
    ("", None),
    ("node: not found", None),
])
def test_parse_node_version(text: str, expected: tuple[int, int, int] | None) -> None:
    assert doctor.parse_node_version(text) == expected


@pytest.mark.parametrize(("conf", "runtimes", "cookies"), [
    ("--js-runtimes node\n", ["node"], []),
    ("# comment --js-runtimes deno\n--js-runtimes=node:C:/nodejs/node.exe\n", ["node"], []),
    ('--js-runtimes deno --js-runtimes "node"\n', ["deno", "node"], []),
    ("--js-runtimes node\n--no-js-runtimes\n", [], []),
    ("--js-runtimes node\n--cookies-from-browser firefox\n--cookies=c.txt\n", ["node"],
     ["--cookies-from-browser", "--cookies"]),
    ("--js-runtimes\n", [], []),  # dangling option: nothing enabled
    ('--js-runtimes node\n-o "unterminated\n', ["node"], []),  # bad quoting falls back to split
])
def test_ytdlp_conf_parsing(conf: str, runtimes: list[str], cookies: list[str]) -> None:
    tokens = doctor.parse_ytdlp_conf(conf)
    assert doctor.ytdlp_js_runtimes(tokens) == runtimes
    assert doctor.ytdlp_cookie_options(tokens) == cookies


@pytest.mark.parametrize(("version", "expected"), [
    ("2026.09.27.232945", date(2026, 9, 27)),
    ("2026.08.19", date(2026, 8, 19)),
    ("2026.13.40", None),
    ("nightly", None),
])
def test_parse_ytdlp_version_date(version: str, expected: date | None) -> None:
    assert doctor.parse_ytdlp_version_date(version) == expected


@pytest.mark.parametrize(("value", "expected"), [
    ([7, 5], (7, 5)), ((8, 6), (8, 6)), ("7.5", (7, 5)), ("(7, 5)", (7, 5)), ("sm_75", (7, 5)),
    ({"major": 7, "minor": 5}, (7, 5)), (None, None), ("?", None), ([7], None),
])
def test_parse_capability(value: Any, expected: tuple[int, int] | None) -> None:
    assert doctor._parse_capability(value) == expected


def test_cfg_value_sources() -> None:
    assert doctor._cfg_value({"doctor": {"min_node_major": 20}}, "doctor.min_node_major") == 20
    assert doctor._cfg_value(SimpleNamespace(doctor=SimpleNamespace(min_node_major=18)), "doctor.min_node_major") == 18
    # Missing -> built-in default; YouTube input defaults to ON (user request 2026-09-30).
    assert doctor._cfg_value(None, "youtube.enabled") is True
    assert doctor._cfg_value({}, "doctor.min_free_disk_gb") == 60
    # Unknown sections live in Config.extra.
    assert doctor._cfg_value(SimpleNamespace(extra={"consent": {"youtube": "ok"}}), "consent.youtube") == "ok"

    class RaisingGet:
        def get(self, dotted: str) -> Any:
            raise KeyError(dotted)

    assert doctor._cfg_value(RaisingGet(), "doctor.min_node_major") == 22


def test_cfg_value_with_real_config() -> None:
    config = pytest.importorskip("bandscribe.config")
    cfg = config.Config()
    assert doctor._cfg_value(cfg, "youtube.enabled") is True
    assert doctor._cfg_value(cfg, "doctor.min_node_major") == 22
    assert doctor._cfg_value(cfg, "consent.youtube", "") == ""


# --------------------------------------------------------------------------------- run_doctor (faked)


def test_all_ok(env: SimpleNamespace) -> None:
    report = doctor.run_doctor(env.cfg)
    st = _statuses(report)
    assert report.ok, [c.to_dict() for c in report.checks if not c.ok]
    assert all(s == "ok" for s in st.values()), st
    # Spec order: gpu, ffmpeg, node, disk, nvml/pdh, yt-dlp, global python, then the M2 models group (SEP).
    names = [c.name for c in report.checks]
    assert names == ["gpu.venv", "gpu.cuda", "gpu.capability", "gpu.fp16_matmul", "gpu.autocast_conv", "gpu.launcher",
                     "ffmpeg", "ffmpeg.features", "node", "disk.free", "disk.cache_paths",
                     "nvml", "gpu.budget", "pdh", "youtube", "ytdlp", "ytdlp.freshness", "ytdlp.conf", "global_python",
                     "gpu.vram_table", "models.present", "node.alphatab"]
    json.dumps(report.to_dict(), ensure_ascii=False)  # --json must be serialisable

    (name, request, work_dir, kw) = env.gpu.calls[0]
    assert (name, request) == ("selftest", {"mode": "cuda"})
    assert work_dir.parent == env.data / "doctor"
    assert kw["timeout_s"] > 0 and kw["lock_timeout_s"] > 0

    launcher = report.get("gpu.launcher")
    assert launcher is not None and launcher.data["launcher"] is True
    assert r"cpython-3.12\python.exe" in launcher.detail
    assert "max_reserved 44 MB" in report.get("gpu.cuda").detail
    assert "chrome.exe(100) 300 MB" in report.get("pdh").detail
    assert "chrome.exe" in report.get("nvml").detail


def test_required_failure_fails_report(env: SimpleNamespace) -> None:
    env.runner.overrides["node"] = _Proc(0, "v20.11.1\n")
    report = doctor.run_doctor(env.cfg)
    assert not report.ok
    assert report.get("node").status == "fail"
    assert "22" in report.get("node").detail
    assert [c.name for c in report.failures] == ["node"]


def test_warnings_do_not_fail_report(env: SimpleNamespace) -> None:
    out = dict(GOOD_SELFTEST["output"], capability=[8, 6])
    env.gpu.next = _worker_result(dict(GOOD_SELFTEST, output=out))
    env.sysmon.nvml["devices"][0]["free_mb"] = 2000  # other apps hog VRAM -> budget below target
    env.runner.overrides["yt-dlp"] = _Proc(0, "2026.06.01\n")  # stale build
    report = doctor.run_doctor(env.cfg)
    assert report.ok
    assert {c.name for c in report.warnings} == {"gpu.capability", "gpu.budget", "ytdlp.freshness"}
    assert "--update-to nightly" in report.get("ytdlp.freshness").detail


def test_exception_in_group_is_isolated(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(cfg: Any) -> list[Check]:
        raise RuntimeError("probe exploded")

    monkeypatch.setitem(doctor._GROUP_FUNCS, "node", boom)
    report = doctor.run_doctor(env.cfg)
    node = report.get("node")
    assert node is not None and node.status == "fail"
    assert "RuntimeError" in node.detail and "probe exploded" in node.detail
    assert report.get("ffmpeg").ok and report.get("gpu.cuda").ok  # everything else still ran


def test_exception_in_single_probe_is_isolated(env: SimpleNamespace) -> None:
    def pdh_broken() -> Any:
        raise OSError("PdhCollectQueryData failed")

    env.sysmon.procs = pdh_broken
    report = doctor.run_doctor(env.cfg)
    assert report.get("pdh").status == "fail" and "OSError" in report.get("pdh").detail
    assert report.get("nvml").ok and report.get("gpu.budget").ok


def test_missing_tools_are_reported(env: SimpleNamespace) -> None:
    env.which["node"] = None
    env.modules["bandscribe.sysmon"] = ImportError("no pynvml")
    report = doctor.run_doctor(env.cfg)
    assert report.get("node").status == "fail" and "PATH" in report.get("node").detail
    assert report.get("nvml").status == "fail" and "bandscribe.sysmon" in report.get("nvml").detail
    assert report.get("pdh").status == "fail"


def test_gpu_venv_missing(env: SimpleNamespace) -> None:
    env.gpu_py.unlink()
    report = doctor.run_doctor(env.cfg)
    venv = report.get("gpu.venv")
    assert venv.status == "fail" and "GPU venv" in venv.detail
    assert env.gpu.calls == []
    assert not report.ok


def test_gpu_module_missing(env: SimpleNamespace) -> None:
    env.modules["bandscribe.gpu"] = ModuleNotFoundError("No module named 'bandscribe.gpu'")
    report = doctor.run_doctor(env.cfg)
    assert report.get("gpu.venv").ok
    sel = report.get("gpu.selftest")
    assert sel.status == "fail" and "bandscribe.gpu" in sel.detail


def test_gpu_lock_busy(env: SimpleNamespace) -> None:
    env.gpu.next = FakeGpu.GpuLockTimeout("busy")
    report = doctor.run_doctor(env.cfg)
    sel = report.get("gpu.selftest")
    assert sel.status == "fail" and "gpu.lock" in sel.detail


def test_selftest_failure_keeps_partial_info(env: SimpleNamespace) -> None:
    partial = {"torch_version": "2.11.0+cpu", "cuda_version": None, "cuda_available": False}
    res = dict(GOOD_SELFTEST, ok=False, output=partial,
               error="SelftestFailed: torch.cuda.is_available() is False")
    env.gpu.next = _worker_result(res)
    report = doctor.run_doctor(env.cfg)
    assert report.get("gpu.cuda").status == "fail"
    assert "2.11.0+cpu" in report.get("gpu.cuda").detail
    assert "is_available() is False" in report.get("gpu.selftest").detail
    assert report.get("gpu.fp16_matmul") is None  # never measured: no misleading "not reported" rows
    assert report.get("gpu.launcher").ok
    assert not report.ok


def test_selftest_fp16_failure(env: SimpleNamespace) -> None:
    out = dict(GOOD_SELFTEST["output"],
               fp16_matmul={"finite": False, "max_rel_err": float("nan"), "ok": False, "ms": 5})
    env.gpu.next = _worker_result(dict(GOOD_SELFTEST, ok=False, output=out, error="SelftestFailed: fp16 matmul"))
    report = doctor.run_doctor(env.cfg)
    assert report.get("gpu.fp16_matmul").status == "fail"
    assert "inf/nan" in report.get("gpu.fp16_matmul").detail
    assert report.get("gpu.autocast_conv").ok


def test_dirty_gpu_lock_release_fails_the_doctor(env: SimpleNamespace) -> None:
    """gpu.run_worker had to release the lock with leftovers in the job: never a silent pass."""
    env.gpu.next = _worker_result(lock_released_dirty=True, job_pids=[111, 222])
    report = doctor.run_doctor(env.cfg)
    c = report.get("gpu.cleanup")
    assert c is not None and c.status == "fail" and "111, 222" in c.detail
    assert not report.ok
    env.gpu.next = _worker_result()
    assert doctor.run_doctor(env.cfg).get("gpu.cleanup") is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects")
def test_run_probe_timeout_is_bounded_for_launchers(tmp_path: Path) -> None:
    """doctor._run on a launcher whose child keeps the pipes open returns at the timeout, not when
    the child exits (the old subprocess.run(timeout=) behaviour with yt-dlp.exe / py.exe)."""
    code = ("import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
            "time.sleep(600)\n")
    t0 = time.monotonic()
    p = doctor._run([paths.CORE_PYTHON, "-c", code], timeout=2)
    assert time.monotonic() - t0 < 2 + 8
    assert p.returncode is None and p.error and "끝나지 않음" in p.error
    missing = doctor._run([tmp_path / "nope.exe", "--version"])
    assert missing.returncode is None and "찾지 못함" in (missing.error or "")


def test_selftest_timeout_without_result(env: SimpleNamespace, tmp_path: Path) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text("loading...\n", encoding="utf-8")
    env.gpu.next = _worker_result({}, ok=False, timed_out=True, killed=True, returncode=None, stderr_path=stderr)
    report = doctor.run_doctor(env.cfg)
    sel = report.get("gpu.selftest")
    assert sel.status == "fail" and "시간 초과" in sel.detail
    assert report.get("gpu.cuda") is None


def test_selftest_error_from_stderr_tail(tmp_path: Path) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text("Traceback...\nImportError: DLL load failed while importing _C\n", encoding="utf-8")
    checks = doctor._interpret_selftest(_worker_result({}, ok=False, returncode=1, stderr_path=stderr))
    assert [c.name for c in checks] == ["gpu.selftest"]
    assert "DLL load failed" in checks[0].detail


def test_not_a_launcher(env: SimpleNamespace) -> None:
    same = r"D:\gtab\envs\gpu\.venv\Scripts\python.exe"
    env.gpu.next = _worker_result(dict(GOOD_SELFTEST, python=same, base_executable=same.lower()))
    report = doctor.run_doctor(env.cfg)
    launcher = report.get("gpu.launcher")
    assert launcher.ok and launcher.data["launcher"] is False and "런처가 아닙니다" in launcher.detail


def test_ffmpeg_missing_features(env: SimpleNamespace) -> None:
    env.runner.overrides["ffmpeg-buildconf"] = _Proc(0, "  configuration:\n    --enable-libsoxr\n")
    env.runner.overrides["ffmpeg-filters"] = _Proc(0, " TS aap   AA->A   x\n")
    report = doctor.run_doctor(env.cfg)
    feat = report.get("ffmpeg.features")
    assert feat.status == "fail"
    assert feat.data["missing"] == ["chromaprint", "ebur128"]
    assert report.get("ffmpeg").ok


def test_ffmpeg_not_on_path(env: SimpleNamespace) -> None:
    env.which["ffmpeg"] = None
    report = doctor.run_doctor(env.cfg)
    assert report.get("ffmpeg").status == "fail" and report.get("ffmpeg.features") is None


def test_ffprobe_missing(env: SimpleNamespace) -> None:
    env.which["ffprobe"] = None
    report = doctor.run_doctor(env.cfg)
    assert report.get("ffmpeg").status == "fail" and "ffprobe" in report.get("ffmpeg").detail


def test_disk_low_and_cache_on_other_drive(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    env.disk["usage"] = DiskUsage(1000 * 2**30, 990 * 2**30, 10 * 2**30)
    monkeypatch.setattr(paths, "child_env", lambda extra=None: {
        "HF_HOME": r"C:\Users\someone\.cache\huggingface", "TORCH_HOME": r"D:\gtab\data\torch",
        "UV_CACHE_DIR": r"D:\elsewhere\uv"})
    report = doctor.run_doctor(env.cfg)
    assert report.get("disk.free").status == "fail"
    cache = report.get("disk.cache_paths")
    assert cache.status == "fail" and "HF_HOME" in cache.detail and "다른 드라이브" in cache.detail
    assert "BANDSCRIBE_ROOT 밖" in cache.detail  # D:\elsewhere is allowed but flagged


def test_youtube_disabled_makes_ytdlp_optional(env: SimpleNamespace) -> None:
    env.exe.unlink()
    env.cfg["youtube"]["enabled"] = False
    report = doctor.run_doctor(env.cfg)
    assert report.ok
    assert report.get("youtube").ok and report.get("ytdlp") is None


def test_youtube_enabled_requires_ytdlp(env: SimpleNamespace) -> None:
    env.exe.unlink()
    report = doctor.run_doctor(env.cfg)
    yt = report.get("ytdlp")
    assert yt.status == "fail" and yt.required
    assert not report.ok


def test_youtube_default_is_on_and_required(env: SimpleNamespace) -> None:
    env.exe.unlink()
    cfg = {k: v for k, v in env.cfg.items() if k != "youtube"}  # no [youtube] section -> default ON
    report = doctor.run_doctor(cfg)
    assert report.get("ytdlp").required and not report.ok


def test_ytdlp_conf_problems(env: SimpleNamespace) -> None:
    env.conf.write_text("--cookies-from-browser firefox\n", encoding="utf-8")
    report = doctor.run_doctor(env.cfg)
    conf = report.get("ytdlp.conf")
    assert conf.status == "fail"
    assert "--js-runtimes node" in conf.detail and "쿠키" in conf.detail


def test_youtube_consent_missing_is_warning(env: SimpleNamespace) -> None:
    del env.cfg["consent"]
    report = doctor.run_doctor(env.cfg)
    assert report.get("youtube").status == "warn"
    assert report.ok


def test_global_python_baseline(env: SimpleNamespace) -> None:
    first = doctor.run_doctor(env.cfg, skip={"gpu"}).get("global_python")
    assert first.ok and "기준 기록" in first.detail
    baseline = env.data / "doctor" / "global_python_baseline.json"
    assert json.loads(baseline.read_text(encoding="utf-8"))["torch"] == "2.10.0+cpu"

    second = doctor.run_doctor(env.cfg, skip={"gpu"}).get("global_python")
    assert second.ok and "변경 없음" in second.detail

    env.runner.overrides["py"] = _Proc(0, json.dumps(dict(GLOBAL_INFO, torch="2.11.0+cpu")) + "\n")
    third = doctor.run_doctor(env.cfg, skip={"gpu"}).get("global_python")
    assert third.status == "warn" and "2.10.0+cpu → 2.11.0+cpu" in third.detail
    assert not third.required

    py_call = next(c for c in env.runner.calls if Path(c[0]).stem == "py")
    assert py_call[1:3] == ["-3.10", "-B"]  # read-only: no bytecode written into the global install


def test_global_python_absent_is_informational(env: SimpleNamespace) -> None:
    env.which["py"] = None
    assert doctor.run_doctor(env.cfg, skip={"gpu"}).get("global_python").ok
    env.which["py"] = r"C:\Windows\py.exe"
    env.runner.overrides["py"] = _Proc(103, "", "No suitable Python runtime found")
    chk = doctor.run_doctor(env.cfg, skip={"gpu"}).get("global_python")
    assert chk.ok and not chk.required


def test_skip_groups(env: SimpleNamespace) -> None:
    report = doctor.run_doctor(env.cfg, skip={"gpu", "global_python"})
    gpu = report.get("gpu")
    assert gpu is not None and gpu.status == "warn" and gpu.data["skipped"]
    assert env.gpu.calls == []
    assert report.ok
    with pytest.raises(ValueError):
        doctor.run_doctor(env.cfg, skip={"gpus"})


def test_cfg_none_falls_back_to_defaults(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "bandscribe.config", None)  # import now raises ImportError
    report = doctor.run_doctor(None, skip={"gpu"})
    assert report.checks[0].name == "config" and report.checks[0].status == "fail"
    assert report.get("node").ok  # defaults (min_node_major=22) still applied


def test_selftest_dirs_are_fresh_and_pruned(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "KEEP_SELFTEST_RUNS", 2)
    dirs = [doctor._selftest_dir() for _ in range(4)]
    assert len(set(dirs)) == 4
    left = sorted(p.name for p in (env.data / "doctor").glob("selftest-*"))
    assert left == sorted(d.name for d in dirs[-2:])


# ------------------------------------------------------------------------------------------- real runs


def test_smoke_real_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Non-GPU checks against the real machine: must return a report, never crash."""
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")  # baseline file goes to tmp, not D:\gtab\data
    report = doctor.run_doctor(None, skip={"gpu"})
    assert isinstance(report, DoctorReport) and report.checks
    assert all(isinstance(c, Check) and isinstance(c.detail, str) and c.detail for c in report.checks)
    names = {c.name for c in report.checks}
    assert {"ffmpeg", "node", "disk.free", "disk.cache_paths", "youtube", "global_python"} <= names
    crashed = [c.to_dict() for c in report.checks if c.data.get("exception")]
    assert not crashed, crashed
    json.dumps(report.to_dict(), ensure_ascii=False)


@pytest.mark.gpu
@pytest.mark.slow
def test_real_cuda_selftest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("bandscribe.gpu")
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    skip = set(doctor.GROUPS) - {"gpu"}
    report = doctor.run_doctor(None, skip=skip)
    gpu_checks = [c for c in report.checks if c.name.startswith("gpu.")]
    assert gpu_checks
    failed = [c.to_dict() for c in gpu_checks if c.required and not c.ok]
    assert not failed, failed
    assert report.get("gpu.launcher").data["base_executable"]
