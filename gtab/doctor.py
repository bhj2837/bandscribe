"""`gtab doctor`: environment checks for M0 (docs/M0_SPEC.md 13, docs/DESIGN.md 10 "M0 기반과 입력").

Every probe is isolated: an exception inside one check group becomes a failed `Check` and the
remaining groups still run. `DoctorReport.ok` is True iff every *required* check passed;
non-required checks are warnings (shown, but they never change the exit code).

Execution order matters in one place: NVML/PDH are read first, before the CUDA selftest worker
starts, so the "other GPU processes" list and the VRAM budget are not polluted by our own worker.
The selftest runs on the calling thread (so Ctrl+C reaches `gpu.run_worker`, which kills the
worker's Job Object before releasing the GPU lock); the remaining subprocess-bound probes
(ffmpeg, node, yt-dlp, `py -3.10`) run in a small thread pool meanwhile.

User-facing text (`Check.detail`) is Korean; check names are stable English ids.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shlex
import shutil
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from gtab import atomic, paths

log = logging.getLogger(__name__)

__all__ = [
    "Check",
    "DoctorReport",
    "GROUPS",
    "run_doctor",
    "parse_ffmpeg_buildconf",
    "parse_ffmpeg_filters",
    "parse_ffmpeg_version",
    "parse_node_version",
    "parse_ytdlp_conf",
    "ytdlp_js_runtimes",
    "ytdlp_cookie_options",
    "parse_ytdlp_version_date",
]

# ---------------------------------------------------------------------------------------------
# Tunables (deliberately not config keys: config sections are extra="forbid")

SELFTEST_TIMEOUT_S = 300.0
# Doctor should not sit behind a multi-hour separation job; report "busy" instead.
SELFTEST_LOCK_TIMEOUT_S = 60.0
SUBPROCESS_TIMEOUT_S = 30.0
# yt-dlp.exe is a PyInstaller onefile (unpacks on every start, Defender scans it) and the global
# Python imports CPU torch: both can take tens of seconds on a cold cache.
SLOW_SUBPROCESS_TIMEOUT_S = 120.0
EXPECTED_CAPABILITY = (7, 5)  # RTX 2060 (Turing); other GPUs are a warning, not a failure
YTDLP_STALE_DAYS = 30  # YouTube breaks extractors often; nightly is the recommended channel
KEEP_SELFTEST_RUNS = 10
CACHE_ENV_VARS = ("HF_HOME", "TORCH_HOME", "UV_CACHE_DIR")
REQUIRED_FFMPEG_FLAGS = {"--enable-libsoxr": "soxr", "--enable-chromaprint": "chromaprint"}
REQUIRED_FFMPEG_FILTERS = ("ebur128",)

# Fallbacks used when a key is missing from the config object (mirror gtab/defaults.toml).
# youtube.enabled defaults to True: user request 2026-09-30 ("YouTube 링크 입력은 기본 켜기").
_DEFAULTS: dict[str, Any] = {
    "youtube.enabled": True,
    "youtube.cookies": False,
    "doctor.min_free_disk_gb": 60,
    "doctor.min_node_major": 22,
    "gpu.vram_margin_gb": 0.7,
    "gpu.target_reserved_gb": 4.5,
    "gpu.insufficient_vram_policy": "wait",
}


# ---------------------------------------------------------------------------------------------
# Report types


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True
    data: dict[str, Any] = field(default_factory=dict)  # structured values for `gtab doctor --json`

    @property
    def status(self) -> str:
        """"ok" | "warn" (failed, not required) | "fail" (failed, required)."""
        if self.ok:
            return "ok"
        return "fail" if self.required else "warn"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status
        return d


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.required)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]

    def get(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "duration_s": round(self.duration_s, 3),
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------------------------
# Probe seams (tests monkeypatch these)


@dataclass
class _Proc:
    returncode: int | None  # None: could not start, or timed out
    stdout: str = ""
    stderr: str = ""
    error: str | None = None  # Korean reason when the process did not run to completion

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"


def _run(args: list[str | Path], *, timeout: float = SUBPROCESS_TIMEOUT_S,
         env: Mapping[str, str] | None = None) -> _Proc:
    """Run a probe with a deadline that really holds.

    Several probes are launchers (yt-dlp.exe is a PyInstaller one-file exe, `py.exe` starts
    python.exe), so plain subprocess.run(timeout=) would kill only the launcher and then keep waiting
    for the child. winjob.run_captured kills the whole process tree when the time is up.
    """
    from gtab import winjob  # ctypes only; imported here so `import gtab.doctor` stays cheap

    argv = [str(a) for a in args]
    try:
        run = winjob.run_captured(argv, timeout_s=timeout,
                                  env=dict(env) if env is not None else paths.child_env())
    except FileNotFoundError:
        return _Proc(None, error=f"실행 파일을 찾지 못함: {argv[0]}")
    except OSError as e:
        return _Proc(None, error=f"실행 실패: {e}")
    if run.timed_out:
        return _Proc(None, run.stdout, run.stderr, error=f"{timeout:.0f}초 안에 끝나지 않음")
    return _Proc(run.returncode, run.stdout, run.stderr)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _import(module: str) -> Any:
    return importlib.import_module(module)


def _disk_usage(path: Path) -> Any:
    return shutil.disk_usage(path)


def _today() -> date:
    return date.today()


# ---------------------------------------------------------------------------------------------
# Small helpers


def _cfg_value(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Read `a.b` from a pydantic Config, a nested dict, or None; fall back to `_DEFAULTS`.

    Tries `cfg.get(dotted)` (Config helper), attribute/key traversal, then `cfg.extra`
    (unknown top-level sections such as [consent] live there).
    """
    if default is None:
        default = _DEFAULTS.get(dotted)
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter) and not isinstance(cfg, Mapping):
        try:
            v = getter(dotted)
            if v is not None:
                return v
        except Exception:  # noqa: BLE001 - Config.get may raise KeyError for missing keys
            pass
    for root in (cfg, getattr(cfg, "extra", None)):
        cur = root
        for part in dotted.split("."):
            if cur is None:
                break
            cur = cur.get(part) if isinstance(cur, Mapping) else getattr(cur, part, None)
        if cur is not None:
            return cur
    return default


def _tail(text: str, lines: int = 3, width: int = 300) -> str:
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    s = " | ".join(rows[-lines:])
    return s if len(s) <= width else "…" + s[-width:]


def _proc_problem(p: _Proc) -> str:
    if p.error:
        return p.error + (f" ({_tail(p.stderr)})" if p.stderr.strip() else "")
    return f"종료 코드 {p.returncode}" + (f": {_tail(p.stderr or p.stdout)}" if (p.stderr or p.stdout).strip() else "")


def _norm_path(p: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _mb(n_bytes: float | int | None) -> int | None:
    return None if n_bytes is None else int(round(n_bytes / 2**20))


def _names_summary(names: list[str], limit: int = 8) -> str:
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" 외 {len(names) - limit}개"


def _exception_check(name: str, required: bool, e: BaseException) -> Check:
    tb = traceback.extract_tb(e.__traceback__)
    where = f" ({Path(tb[-1].filename).name}:{tb[-1].lineno})" if tb else ""
    log.debug("doctor check %s raised", name, exc_info=e)
    return Check(name, False, f"검사 중 예외: {type(e).__name__}: {e}{where}", required,
                 {"exception": type(e).__name__})


def _one(name: str, required: bool, fn: Callable[[], Check]) -> Check:
    """Run a single probe; any exception becomes a failed Check with the same name."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - isolation is the point
        return _exception_check(name, required, e)


# ---------------------------------------------------------------------------------------------
# Parsers (pure, unit-tested)

_FFMPEG_VERSION_RE = re.compile(r"^\s*ffmpeg version (\S+)", re.MULTILINE)
# ffmpeg 8: " TS aap  AA->A  desc" (2 flag chars); older builds have a 3rd "C" column.
_FFMPEG_FILTER_RE = re.compile(r"^\s*[T.][S.][C.]?\s+(\S+)\s+\S*->\S*(?:\s|$)")
_NODE_VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")
_YTDLP_DATE_RE = re.compile(r"^\s*(\d{4})\.(\d{2})\.(\d{2})")


def parse_ffmpeg_version(text: str) -> str | None:
    m = _FFMPEG_VERSION_RE.search(text)
    return m.group(1) if m else None


def parse_ffmpeg_buildconf(text: str) -> set[str]:
    """`ffmpeg -buildconf` → set of configure flags (`--enable-libsoxr`, ...)."""
    flags: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("--"):
            flags.add(s.split()[0])
    return flags


def parse_ffmpeg_filters(text: str) -> set[str]:
    """`ffmpeg -filters` → set of filter names. Legend/header lines have no `->` and are skipped."""
    names: set[str] = set()
    for line in text.splitlines():
        m = _FFMPEG_FILTER_RE.match(line)
        if m:
            names.add(m.group(1))
    return names


def parse_node_version(text: str) -> tuple[int, int, int] | None:
    """`node --version` ("v24.10.0") → (24, 10, 0)."""
    m = _NODE_VERSION_RE.search(text.strip())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def parse_ytdlp_conf(text: str) -> list[str]:
    """Tokenise a yt-dlp config file the way yt-dlp does (shell-like, `#` comments)."""
    try:
        return shlex.split(text, comments=True, posix=True)
    except ValueError:  # unbalanced quote: fall back to whitespace split, comment lines dropped
        return [tok for line in text.splitlines() if not line.lstrip().startswith("#")
                for tok in line.split()]


def ytdlp_js_runtimes(tokens: list[str]) -> list[str]:
    """Runtimes enabled by `--js-runtimes NAME[:PATH]` (repeatable); `--no-js-runtimes` resets."""
    runtimes: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        value: str | None = None
        if tok == "--no-js-runtimes":
            runtimes = []
        elif tok == "--js-runtimes" and i + 1 < len(tokens):
            value = tokens[i + 1]
            i += 1  # consume the argument
        elif tok.startswith("--js-runtimes="):
            value = tok.split("=", 1)[1]
        i += 1
        if value:
            for part in value.split(","):
                name = part.split(":", 1)[0].strip().lower()
                if name and name not in runtimes:
                    runtimes.append(name)
    return runtimes


def ytdlp_cookie_options(tokens: list[str]) -> list[str]:
    """Cookie options present in the config (M0: none allowed unless youtube.cookies)."""
    found = []
    for tok in tokens:
        opt = tok.split("=", 1)[0]
        if opt in ("--cookies", "--cookies-from-browser"):
            found.append(opt)
    return found


def parse_ytdlp_version_date(version: str) -> date | None:
    """yt-dlp versions are dates: "2026.09.27.232945" (nightly) or "2026.08.19" (stable)."""
    m = _YTDLP_DATE_RE.match(version or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _parse_capability(v: Any) -> tuple[int, int] | None:
    """Accept (7, 5) / [7, 5] / "7.5" / "(7, 5)" / "sm_75" / {"major": 7, "minor": 5}."""
    if isinstance(v, Mapping):
        v = (v.get("major"), v.get("minor"))
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return int(v[0]), int(v[1])
        except (TypeError, ValueError):
            return None
    if isinstance(v, str):
        m = re.search(r"(\d+)\D+(\d+)", v) or re.search(r"sm_?(\d)(\d)", v)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


# ---------------------------------------------------------------------------------------------
# 1-2. GPU env + CUDA selftest (+ launcher / base interpreter)


def _pick(d: Mapping[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _as_ok(v: Any) -> bool | None:
    """A selftest sub-result may be a bool or a dict with an "ok" field."""
    if isinstance(v, bool):
        return v
    if isinstance(v, Mapping) and "ok" in v:
        return bool(v["ok"])
    return None


def _total_vram_mb(out: Mapping[str, Any]) -> int | None:
    v = _pick(out, "total_vram_mb", "vram_total_mb", "total_mb", "total_memory_mb")
    if isinstance(v, (int, float)):
        return int(v)
    v = _pick(out, "total_vram_gb", "vram_total_gb", "total_gb")
    if isinstance(v, (int, float)):
        return int(round(v * 1024))
    v = _pick(out, "total_memory", "total_memory_bytes", "total_vram_bytes")
    if isinstance(v, (int, float)):
        return _mb(v)
    return None


def _selftest_dir() -> Path:
    """Fresh dir per run: a reused dir could hold a stale result.json that looks like success."""
    root = paths.DATA / "doctor"
    root.mkdir(parents=True, exist_ok=True)
    # Names sort chronologically (for pruning); the counter keeps them unique because the Windows
    # clock can return the same value for back-to-back calls.
    prefix = f"selftest-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-"
    used = [int(p.name[len(prefix):]) for p in root.glob(prefix + "*") if p.name[len(prefix):].isdigit()]
    n = max(used, default=-1) + 1  # never reuse a lower (possibly pruned) number: it would sort as older
    while True:
        d = root / f"{prefix}{n:03d}"
        try:
            d.mkdir()
            break
        except FileExistsError:
            n += 1
    old = sorted(p for p in root.glob("selftest-*") if p.is_dir())
    for p in old[:-KEEP_SELFTEST_RUNS]:
        shutil.rmtree(p, ignore_errors=True)
    return d


def _worker_failure_reason(res: Any, result: Mapping[str, Any]) -> str:
    if getattr(res, "timed_out", False):
        return f"시간 초과({SELFTEST_TIMEOUT_S:.0f}초) — 워커를 강제 종료함"
    err = result.get("error")
    if err:
        return str(err).strip().splitlines()[0][:300]
    stderr_path = getattr(res, "stderr_path", None)
    if stderr_path:
        try:
            t = _tail(Path(stderr_path).read_text(encoding="utf-8", errors="replace"))
            if t:
                return t
        except OSError:
            pass
    return f"returncode={getattr(res, 'returncode', None)}, result.json 없음 또는 ok=false"


def _interpret_selftest(res: Any) -> list[Check]:
    """Turn a `gpu.WorkerResult` of selftest(mode=cuda) into checks (pure: no I/O but stderr tail)."""
    result = res.result if isinstance(getattr(res, "result", None), Mapping) else {}
    raw_out = result.get("output")
    out: Mapping[str, Any] = raw_out if isinstance(raw_out, Mapping) else {}
    work_dir = str(getattr(res, "work_dir", "") or "")
    if not out:
        reason = _worker_failure_reason(res, result)
        return [Check("gpu.selftest", False, f"CUDA 셀프테스트 실패: {reason} (로그: {work_dir})", True,
                      {"work_dir": work_dir, "returncode": getattr(res, "returncode", None)})]

    checks: list[Check] = []
    torch_v = _pick(out, "torch_version", "torch")
    cuda_v = _pick(out, "cuda_version", "torch_cuda", "cuda")
    cuda_v = cuda_v if isinstance(cuda_v, str) else None
    available = _pick(out, "cuda_available", "cuda_is_available", "available")
    device = _pick(out, "device_name", "gpu_name", "device", "name")
    device = device if isinstance(device, str) else None
    total_mb = _total_vram_mb(out)
    cuda_stats = result.get("cuda") if isinstance(result.get("cuda"), Mapping) else {}
    max_reserved = cuda_stats.get("max_reserved_mb")
    sysmon = result.get("sysmon") if isinstance(result.get("sysmon"), Mapping) else {}
    shared_start = sysmon.get("pdh_self_shared_start_mb")
    shared_peak = sysmon.get("pdh_self_shared_peak_mb")
    data = {
        "torch": torch_v, "cuda": cuda_v, "cuda_available": available, "device": device,
        "total_vram_mb": total_mb, "max_reserved_mb": max_reserved,
        "max_allocated_mb": cuda_stats.get("max_allocated_mb"),
        "nvml_used_peak_mb": sysmon.get("nvml_used_peak_mb"),
        "pdh_self_dedicated_peak_mb": sysmon.get("pdh_self_dedicated_peak_mb"),
        "pdh_self_shared_start_mb": shared_start, "pdh_self_shared_peak_mb": shared_peak,
        "duration_s": getattr(res, "duration_s", None), "lock_wait_s": getattr(res, "lock_wait_s", None),
        "work_dir": work_dir,
    }
    parts = [f"torch {torch_v or '?'}", f"CUDA {cuda_v or '?'}", device or "장치 이름 없음"]
    if total_mb is not None:
        parts.append(f"VRAM {total_mb} MB")
    if isinstance(max_reserved, (int, float)):
        parts.append(f"max_reserved {max_reserved:.0f} MB")
    if isinstance(shared_start, (int, float)) and isinstance(shared_peak, (int, float)):
        parts.append(f"공유 메모리 {shared_start:.0f}→{shared_peak:.0f} MB")
    if available is True:
        checks.append(Check("gpu.cuda", True, " · ".join(parts), True, data))
    else:
        checks.append(Check("gpu.cuda", False,
                            f"torch.cuda.is_available() = {available} — gpu venv에 CUDA 빌드 torch(cu128)가 "
                            f"깔렸는지, 드라이버가 살아 있는지 확인하세요. ({' · '.join(parts)})", True, data))

    # Without CUDA the capability / fp16 / autocast items were never measured; "not reported" rows
    # would only bury the real cause, so go straight to the launcher info and the failure summary.
    if available is True:
        checks.extend(_selftest_compute_checks(out))
    checks.extend(_selftest_launcher_and_status(res, result, work_dir))
    return checks


def _selftest_compute_checks(out: Mapping[str, Any]) -> list[Check]:
    checks: list[Check] = []
    cap = _parse_capability(_pick(out, "capability", "device_capability", "compute_capability"))
    bf16_ok = _as_ok(_pick(out, "bf16_supported_native", "bf16_supported", "bf16"))
    bf16_txt = "" if bf16_ok is None else f" · bf16 네이티브 {'지원' if bf16_ok else '미지원'}"
    if cap is None:
        checks.append(Check("gpu.capability", False, "셀프테스트가 compute capability를 보고하지 않았습니다.",
                            False, {"capability": None}))
    elif cap == EXPECTED_CAPABILITY:
        checks.append(Check("gpu.capability", True,
                            f"compute capability {cap[0]}.{cap[1]} (Turing, RTX 2060){bf16_txt} — bf16은 쓰지 않습니다.",
                            False, {"capability": list(cap), "bf16": bf16_ok}))
    else:
        checks.append(Check("gpu.capability", False,
                            f"compute capability {cap[0]}.{cap[1]} — 설계 기준(7.5, RTX 2060)과 다른 GPU입니다. "
                            f"VRAM 예산과 dtype 정책을 다시 확인하세요.{bf16_txt}",
                            False, {"capability": list(cap), "bf16": bf16_ok}))

    for name, keys, label in (
        ("gpu.fp16_matmul", ("fp16_matmul_ok", "fp16_matmul", "matmul_fp16", "fp16"), "fp16 matmul(1024²)"),
        ("gpu.autocast_conv", ("autocast_conv_ok", "autocast_conv", "conv_autocast", "autocast"),
         "autocast(fp16) conv2d"),
    ):
        raw = _pick(out, *keys)
        ok = _as_ok(raw)
        extra = {k: v for k, v in raw.items() if k != "ok"} if isinstance(raw, Mapping) else {}
        if ok is None:
            checks.append(Check(name, False, f"셀프테스트가 {label} 결과를 보고하지 않았습니다.", True))
            continue
        info = []
        if isinstance(extra.get("max_rel_err"), (int, float)):
            info.append(f"상대 오차 {extra['max_rel_err']:.1e}")
        if extra.get("finite") is False:
            info.append("inf/nan 발생")
        if isinstance(extra.get("ms"), (int, float)):
            info.append(f"{extra['ms']:.0f} ms")
        checks.append(Check(name, ok, f"{label} {'성공' if ok else '실패'}" + (f" ({', '.join(info)})" if info else ""),
                            True, dict(extra)))
    return checks


def _selftest_launcher_and_status(res: Any, result: Mapping[str, Any], work_dir: str) -> list[Check]:
    checks: list[Check] = []
    python = result.get("python")
    base = result.get("base_executable")
    if not python or not base:
        checks.append(Check("gpu.launcher", False,
                            "셀프테스트 결과에 python/base_executable이 없어 런처 여부를 알 수 없습니다.", True))
    else:
        launcher = _norm_path(python) != _norm_path(base)
        if launcher:
            detail = (f"venv python.exe는 런처입니다. 실제 CUDA 프로세스는 기본 인터프리터 {base} 입니다 — "
                      f"NVIDIA 제어판 프로그램별 설정(CUDA 시스템 메모리 폴백 정책)은 이 경로에 거세요.")
        else:
            detail = (f"런처가 아닙니다(직접 실행): {python} — NVIDIA 프로그램별 설정은 이 경로에 거세요.")
        checks.append(Check("gpu.launcher", True, detail, True,
                            {"python": python, "base_executable": base, "launcher": launcher,
                             "pid": result.get("pid"), "ppid": result.get("ppid")}))

    # Items above can all look fine while the worker still failed (e.g. a later assertion).
    if not getattr(res, "ok", False):
        checks.append(Check("gpu.selftest", False,
                            f"셀프테스트 워커가 실패로 끝났습니다: {_worker_failure_reason(res, result)} (로그: {work_dir})",
                            True, {"work_dir": work_dir}))
    return checks


def _is_lock_timeout(gpu: Any, e: BaseException) -> bool:
    lock_errors: tuple[type[BaseException], ...] = tuple(
        t for t in (getattr(gpu, "GpuLockTimeout", None),) if isinstance(t, type))
    try:
        import filelock

        lock_errors += (filelock.Timeout,)
    except ImportError:
        pass
    return bool(lock_errors) and isinstance(e, lock_errors)


def _check_gpu(cfg: Any) -> list[Check]:
    py = paths.GPU_PYTHON
    if not Path(py).exists():
        return [Check("gpu.venv", False,
                      f"GPU venv 파이썬이 없습니다: {py} — envs\\gpu 동기화(torch cu128)가 끝났는지 확인하세요. "
                      f"CUDA 셀프테스트를 할 수 없습니다.", True)]
    checks = [Check("gpu.venv", True, str(py), True)]
    try:
        gpu = _import("gtab.gpu")
    except Exception as e:  # noqa: BLE001
        checks.append(Check("gpu.selftest", False,
                            f"gtab.gpu 모듈을 불러오지 못했습니다({type(e).__name__}: {e}) — CUDA 셀프테스트를 실행할 수 없습니다.",
                            True))
        return checks
    work_dir = _selftest_dir()
    try:
        res = gpu.run_worker("selftest", {"mode": "cuda"}, work_dir,
                             timeout_s=SELFTEST_TIMEOUT_S, lock_timeout_s=SELFTEST_LOCK_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        if _is_lock_timeout(gpu, e):
            detail = (f"GPU 락(data\\gpu.lock)을 {SELFTEST_LOCK_TIMEOUT_S:.0f}초 안에 얻지 못했습니다 — "
                      f"다른 GPU 작업이 끝난 뒤 다시 실행하세요.")
        else:
            detail = f"셀프테스트 워커를 실행하지 못했습니다: {type(e).__name__}: {e}"
        checks.append(Check("gpu.selftest", False, detail, True, {"work_dir": str(work_dir)}))
        return checks
    checks.extend(_interpret_selftest(res))
    if getattr(res, "lock_released_dirty", False) is True:
        # gpu.run_worker gave up on processes that survived TerminateJobObject (stuck in the driver)
        # and released the GPU lock anyway; the next worker could share VRAM with them.
        pids = getattr(res, "job_pids", None) or []
        checks.append(Check("gpu.cleanup", False,
                            "셀프테스트 워커 프로세스가 강제 종료 뒤에도 남아 GPU 락을 억지로 풀었습니다 "
                            f"(pid {', '.join(map(str, pids)) or '?'}) — 작업 관리자에서 확인하고, 안 끝나면 재부팅하세요.",
                            True, {"job_pids": list(pids)}))
    return checks


# ---------------------------------------------------------------------------------------------
# 3. ffmpeg


def _check_ffmpeg(cfg: Any) -> list[Check]:
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        return [Check("ffmpeg", False, "PATH에서 ffmpeg를 찾지 못했습니다 (gyan.dev full 빌드 8.x 필요).", True)]
    ffprobe = _which("ffprobe")
    checks: list[Check] = []

    def version_check() -> Check:
        p = _run([ffmpeg, "-hide_banner", "-version"])
        ver = parse_ffmpeg_version(p.output)
        if not p.ok or not ver:
            return Check("ffmpeg", False, f"ffmpeg -version 실패: {_proc_problem(p)}", True)
        detail = f"ffmpeg {ver} · {ffmpeg}"
        if not ffprobe:
            return Check("ffmpeg", False, detail + " · ffprobe가 PATH에 없습니다", True, {"version": ver})
        return Check("ffmpeg", True, detail, True, {"version": ver, "ffmpeg": ffmpeg, "ffprobe": ffprobe})

    def features_check() -> Check:
        bc = _run([ffmpeg, "-hide_banner", "-buildconf"])
        flags = parse_ffmpeg_buildconf(bc.output)
        if not flags:
            return Check("ffmpeg.features", False, f"ffmpeg -buildconf를 읽지 못했습니다: {_proc_problem(bc)}", True)
        fl = _run([ffmpeg, "-hide_banner", "-filters"])
        filters = parse_ffmpeg_filters(fl.output)
        if not filters:
            return Check("ffmpeg.features", False, f"ffmpeg -filters를 읽지 못했습니다: {_proc_problem(fl)}", True)
        missing = [label for flag, label in REQUIRED_FFMPEG_FLAGS.items() if flag not in flags]
        missing += [f for f in REQUIRED_FFMPEG_FILTERS if f not in filters]
        if missing:
            return Check("ffmpeg.features", False,
                         f"없음: {', '.join(missing)} — soxr·chromaprint·ebur128이 켜진 gyan.dev full 빌드로 바꾸세요.",
                         True, {"missing": missing})
        return Check("ffmpeg.features", True, "libsoxr · chromaprint · ebur128 필터 모두 있음", True, {"missing": []})

    checks.append(_one("ffmpeg", True, version_check))
    checks.append(_one("ffmpeg.features", True, features_check))
    return checks


# ---------------------------------------------------------------------------------------------
# 4. Node


def _check_node(cfg: Any) -> list[Check]:
    min_major = int(_cfg_value(cfg, "doctor.min_node_major"))
    node = _which("node")
    if not node:
        return [Check("node", False,
                      f"node가 PATH에 없습니다 — Node {min_major} 이상이 필요합니다 (yt-dlp JS 챌린지, alphaTab 내보내기).",
                      True)]
    p = _run([node, "--version"])
    v = parse_node_version(p.stdout) if p.ok else None
    if v is None:
        return [Check("node", False, f"node --version 실패: {_proc_problem(p)}", True)]
    vs = ".".join(map(str, v))
    if v[0] < min_major:
        return [Check("node", False, f"Node v{vs} — {min_major} 이상이 필요합니다 ({node}).", True,
                      {"version": vs, "path": node})]
    return [Check("node", True, f"Node v{vs} (최소 {min_major}) · {node}", True, {"version": vs, "path": node})]


# ---------------------------------------------------------------------------------------------
# 5. Disk + cache locations


def _check_disk(cfg: Any) -> list[Check]:
    root = Path(paths.GTAB_ROOT)
    drive = root.drive or root.anchor or str(root)
    min_gb = float(_cfg_value(cfg, "doctor.min_free_disk_gb"))

    def free_check() -> Check:
        target = root if root.exists() else Path(root.anchor or root)
        du = _disk_usage(target)
        free_gb = du.free / 2**30
        total_gb = du.total / 2**30
        data = {"drive": drive, "free_gb": round(free_gb, 1), "total_gb": round(total_gb, 1), "min_gb": min_gb}
        detail = f"{drive} 여유 {free_gb:.0f} GB / 전체 {total_gb:.0f} GB (최소 {min_gb:.0f} GB)"
        if free_gb < min_gb:
            detail += " — 공간이 부족합니다. `gtab gc` / `gtab clean --archive`로 정리하세요."
        return Check("disk.free", free_gb >= min_gb, detail, True, data)

    def cache_check() -> Check:
        env = paths.child_env()
        rows, bad = [], []
        root_norm = _norm_path(root)
        for k in CACHE_ENV_VARS:
            v = env.get(k)
            if not v:
                bad.append(k)
                rows.append(f"{k}=(없음)")
                continue
            p = Path(v)
            same_drive = bool(p.drive) and p.drive.upper() == root.drive.upper()
            inside = _norm_path(p).startswith(root_norm + os.sep)
            suffix = ""
            if not same_drive:
                bad.append(k)
                suffix = " ← 다른 드라이브"
            elif not inside:
                suffix = " (GTAB_ROOT 밖)"  # allowed by the spec, but worth seeing
            rows.append(f"{k}={v}{suffix}")
        detail = " · ".join(rows)
        if bad:
            detail = f"{drive}가 아닌 곳을 가리킴: {', '.join(bad)} — " + detail
        return Check("disk.cache_paths", not bad, detail, True, {k: env.get(k) for k in CACHE_ENV_VARS})

    return [_one("disk.free", True, free_check), _one("disk.cache_paths", True, cache_check)]


# ---------------------------------------------------------------------------------------------
# 6. NVML + PDH (gtab.sysmon)


def _check_sysmon(cfg: Any) -> list[Check]:
    try:
        sysmon = _import("gtab.sysmon")
    except Exception as e:  # noqa: BLE001
        msg = f"gtab.sysmon 모듈을 불러오지 못했습니다: {type(e).__name__}: {e}"
        return [Check("nvml", False, msg, True), Check("pdh", False, msg, True)]

    nvml_names: dict[int, str] = {}
    nvml_holder: dict[str, Any] = {}

    def nvml_check() -> Check:
        info = sysmon.nvml_info()
        if not info:
            return Check("nvml", False, "NVML을 읽지 못했습니다 — NVIDIA 드라이버와 nvidia-ml-py 설치를 확인하세요.", True)
        nvml_holder["info"] = info
        devices = list(info.get("devices") or [])
        procs = list(info.get("processes") or [])
        for p in procs:
            if p.get("pid") is not None and p.get("name"):
                nvml_names[int(p["pid"])] = Path(str(p["name"])).name
        if not devices:
            return Check("nvml", False, f"NVML은 열렸지만 GPU가 보이지 않습니다 (드라이버 {info.get('driver')}).", True,
                         {"driver": info.get("driver")})
        dev_txt = [
            f"GPU{d.get('index', i)} {d.get('name', '?')}: 전체 {d.get('total_mb', '?')} MB, "
            f"사용 {d.get('used_mb', '?')} MB, 여유 {d.get('free_mb', '?')} MB"
            for i, d in enumerate(devices)
        ]
        names = [Path(str(p.get("name") or f"pid {p.get('pid')}")).name for p in procs]
        proc_txt = f"GPU 사용 프로세스 {len(procs)}개" + (f": {_names_summary(names)}" if names else "")
        return Check("nvml", True, f"드라이버 {info.get('driver', '?')} · " + " · ".join(dev_txt) + " · " + proc_txt,
                     True, {"driver": info.get("driver"), "devices": devices, "processes": procs})

    def budget_check() -> Check:
        info = nvml_holder.get("info")
        devices = list((info or {}).get("devices") or [])
        if not devices or devices[0].get("free_mb") is None:
            return Check("gpu.budget", False, "NVML 값이 없어 VRAM 예산을 계산하지 못했습니다.", False)
        free_mb = float(devices[0]["free_mb"])
        margin = float(_cfg_value(cfg, "gpu.vram_margin_gb"))
        target = float(_cfg_value(cfg, "gpu.target_reserved_gb"))
        policy = _cfg_value(cfg, "gpu.insufficient_vram_policy")
        budget_gb = (free_mb / 1024) - margin
        ok = budget_gb >= target
        detail = (f"GPU 예산 {budget_gb:.1f} GB (여유 {free_mb / 1024:.1f} GB − 여분 {margin:.1f} GB) / "
                  f"목표 피크 {target:.1f} GB")
        if not ok:
            detail += (f" — 다른 앱이 VRAM을 쓰고 있어 무거운 단계는 정책({policy})대로 대기하거나 작은 설정으로 내려갑니다. "
                       f"게임·브라우저·메신저를 닫으면 늘어납니다.")
        return Check("gpu.budget", ok, detail, False,
                     {"budget_gb": round(budget_gb, 2), "target_gb": target, "margin_gb": margin, "policy": policy})

    def pdh_check() -> Check:
        adapter = sysmon.pdh_gpu_adapter_memory()
        procs = sysmon.pdh_gpu_process_memory()
        problems = []
        if adapter is None or "shared" not in adapter:
            problems.append("GPU Adapter Memory(*)\\Shared Usage")
        if procs is None:
            problems.append("GPU Process Memory(*)\\Shared Usage")
        if problems:
            return Check("pdh", False, f"성능 카운터를 읽지 못했습니다: {', '.join(problems)} — Sysmem Fallback 감시가 불가합니다.",
                         True)
        top = sorted(procs.items(), key=lambda kv: kv[1].get("dedicated", 0), reverse=True)
        top = [(pid, m) for pid, m in top if m.get("dedicated", 0) > 0][:5]
        # PDH only knows pids; names come from NVML's process list (which already resolves image names).
        top_txt = ", ".join(
            (f"{nvml_names[int(pid)]}({pid})" if int(pid) in nvml_names else f"pid {pid}")
            + f" {_mb(m.get('dedicated', 0))} MB"
            for pid, m in top
        )
        detail = (f"어댑터 전용 {_mb(adapter.get('dedicated', 0))} MB · 공유 {_mb(adapter.get('shared', 0))} MB · "
                  f"프로세스별 카운터 {len(procs)}개" + (f" (전용 상위: {top_txt})" if top_txt else ""))
        return Check("pdh", True, detail, True,
                     {"adapter_mb": {k: _mb(v) for k, v in adapter.items()},
                      "processes_mb": {str(pid): {k: _mb(v) for k, v in m.items()} for pid, m in procs.items()}})

    return [_one("nvml", True, nvml_check), _one("gpu.budget", False, budget_check), _one("pdh", True, pdh_check)]


# ---------------------------------------------------------------------------------------------
# 7. YouTube / yt-dlp (required only when youtube.enabled)


def _check_ytdlp(cfg: Any) -> list[Check]:
    enabled = bool(_cfg_value(cfg, "youtube.enabled"))
    cookies_allowed = bool(_cfg_value(cfg, "youtube.cookies"))
    exe = Path(paths.YTDLP_EXE)
    conf = Path(paths.YTDLP_CONF)
    checks: list[Check] = []

    consent = _cfg_value(cfg, "consent.youtube", "")
    if enabled:
        checks.append(Check(
            "youtube", bool(consent),
            "YouTube URL 입력 켜짐 · 동의 기록 있음 (data\\config.toml [consent])" if consent else
            "YouTube URL 입력이 켜져 있지만 data\\config.toml에 [consent] youtube 동의 기록이 없습니다.",
            False, {"enabled": True, "consent": bool(consent)}))
    else:
        checks.append(Check("youtube", True, "YouTube URL 입력 꺼짐 (youtube.enabled=false) — yt-dlp는 선택 사항입니다.",
                            False, {"enabled": False}))
        if not exe.exists():
            return checks

    def exe_check() -> list[Check]:
        if not exe.exists():
            return [Check("ytdlp", False, f"{exe}가 없습니다 — yt-dlp nightly exe를 받아 두세요 (YouTube 입력이 켜져 있음).",
                          enabled)]
        p = _run([exe, "--version"], timeout=SLOW_SUBPROCESS_TIMEOUT_S)
        lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
        if not p.ok or not lines:
            return [Check("ytdlp", False, f"yt-dlp --version 실패: {_proc_problem(p)}", enabled)]
        version = lines[-1]
        out = [Check("ytdlp", True, f"yt-dlp {version} · {exe}", enabled, {"version": version})]
        built = parse_ytdlp_version_date(version)
        if built is not None:
            age = (_today() - built).days
            fresh = age <= YTDLP_STALE_DAYS
            out.append(Check("ytdlp.freshness", fresh,
                             f"{age}일 전 빌드" + ("" if fresh else
                                                  f" — YouTube 변경에 막히기 쉽습니다. `{exe} --update-to nightly`를 권합니다."),
                             False, {"built": built.isoformat(), "age_days": age}))
        return out

    def conf_check() -> Check:
        if not conf.exists():
            return Check("ytdlp.conf", False, f"{conf}가 없습니다 — `--js-runtimes node` 한 줄을 넣어 두세요.", enabled)
        tokens = parse_ytdlp_conf(conf.read_text(encoding="utf-8", errors="replace"))
        runtimes = ytdlp_js_runtimes(tokens)
        cookie_opts = ytdlp_cookie_options(tokens)
        problems = []
        if "node" not in runtimes:
            problems.append("`--js-runtimes node`가 없음 (JS 챌린지를 못 풂)")
        if cookie_opts and not cookies_allowed:
            problems.append(f"쿠키 옵션({', '.join(cookie_opts)})이 있음 — youtube.cookies=false인데 계정 정지 위험")
        data = {"js_runtimes": runtimes, "cookie_options": cookie_opts}
        if problems:
            return Check("ytdlp.conf", False, f"{conf}: " + "; ".join(problems), enabled, data)
        return Check("ytdlp.conf", True, f"--js-runtimes {','.join(runtimes)} · 쿠키 없음" if not cookie_opts
                     else f"--js-runtimes {','.join(runtimes)} · 쿠키 옵션 있음(설정에서 허용)", enabled, data)

    try:
        checks.extend(exe_check())
    except Exception as e:  # noqa: BLE001
        checks.append(_exception_check("ytdlp", enabled, e))
    checks.append(_one("ytdlp.conf", enabled, conf_check))
    return checks


# ---------------------------------------------------------------------------------------------
# 8. Global Python 3.10 (informational, read-only)

# Runs inside the *global* interpreter. Read-only: `-B` + PYTHONDONTWRITEBYTECODE so not even
# __pycache__ is written into its site-packages. The package-list hash lets later doctor runs
# prove the interpreter was left untouched (DESIGN 10 M0: "전역 Python 3.10은 바뀌지 않는다").
_GLOBAL_PY_SCRIPT = """\
import hashlib, json, sys
info = {"version": sys.version.split()[0], "executable": sys.executable}
try:
    import importlib.metadata as md
    names = sorted({(d.metadata["Name"] or "").lower() + "==" + str(d.version) for d in md.distributions()})
    info["dists"] = len(names)
    info["dists_sha256"] = hashlib.sha256("\\n".join(names).encode("utf-8")).hexdigest()
except Exception as e:
    info["dists_error"] = repr(e)
try:
    import torch
    info["torch"] = torch.__version__
except Exception as e:
    info["torch"] = None
    info["torch_error"] = type(e).__name__
print(json.dumps(info))
"""

_BASELINE_KEYS = ("version", "executable", "torch", "dists_sha256")


def _global_python_env() -> dict[str, str]:
    """The user's own environment, minus anything that would make the global Python look different
    from how the user runs it (our PYTHONNOUSERSITE would hide user-site torch)."""
    env = dict(os.environ)
    for k in ("PYTHONNOUSERSITE", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP"):
        env.pop(k, None)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def _check_global_python(cfg: Any) -> list[Check]:
    baseline_path = paths.DATA / "doctor" / "global_python_baseline.json"
    baseline: dict[str, Any] | None = None
    if baseline_path.exists():
        try:
            baseline = atomic.read_json(baseline_path)
        except (OSError, ValueError):
            baseline = None

    py = _which("py")
    if not py:
        return [Check("global_python", True, "py 런처가 없어 전역 Python 3.10 확인을 건너뜁니다.", False)]
    p = _run([py, "-3.10", "-B", "-c", _GLOBAL_PY_SCRIPT], timeout=SLOW_SUBPROCESS_TIMEOUT_S,
             env=_global_python_env())
    info: dict[str, Any] | None = None
    if p.ok:
        for line in reversed(p.stdout.splitlines()):
            if line.strip().startswith("{"):
                try:
                    info = json.loads(line)
                except ValueError:
                    info = None
                break
    if info is None:
        if baseline:
            return [Check("global_python", False,
                          f"전역 Python 3.10을 실행하지 못했습니다(기준에는 있었음): {_proc_problem(p)}", False)]
        return [Check("global_python", True, f"전역 Python 3.10 없음 또는 실행 불가 — 확인만 하고 넘어갑니다 "
                                             f"({_proc_problem(p)})", False)]

    torch_txt = f"torch {info['torch']}" if info.get("torch") else "torch 없음"
    summary = f"Python {info.get('version')} · {torch_txt} · 패키지 {info.get('dists', '?')}개 · {info.get('executable')}"
    current = {k: info.get(k) for k in _BASELINE_KEYS}
    if baseline is None:
        atomic.write_json(baseline_path, {**current, "recorded": datetime.now().isoformat(timespec="seconds")})
        return [Check("global_python", True, f"{summary} — 기준 기록함(이후 실행에서 변경 여부 비교)", False, info)]
    changed = [k for k in _BASELINE_KEYS if baseline.get(k) != current[k]]
    if changed:
        diffs = ", ".join(
            f"{k}: {baseline.get(k)} → {current[k]}" if k != "dists_sha256" else "설치 패키지 목록 변경"
            for k in changed
        )
        return [Check("global_python", False,
                      f"전역 Python 3.10이 기준({baseline.get('recorded', '?')})과 다릅니다: {diffs} — gtab은 이 인터프리터를 "
                      f"건드리지 않습니다. 의도한 변경이면 {baseline_path}를 지우면 새 기준이 잡힙니다.",
                      False, {**info, "changed": changed})]
    return [Check("global_python", True, f"{summary} — 기준({baseline.get('recorded', '?')})과 같음(변경 없음)", False,
                  info)]


# ---------------------------------------------------------------------------------------------
# 9. M2 state (SEP, M1_M2_SPEC 9.1 item 11): VRAM table and recorded models. Never required.

VRAM_TABLE_STALE_DAYS = 90
BEAT_THIS_MODEL = "beat_this.final0"


def _check_models(cfg: Any) -> list[Check]:
    def vram_table_check() -> Check:
        p = Path(paths.VRAM_TABLE)  # read at call time (tests repoint it)
        if not p.is_file():
            return Check("gpu.vram_table", True, "VRAM 측정표가 아직 없습니다: 기본 필요량으로 사전 점검합니다 "
                         "(`gtab bench gpu --backend all` 로 이 PC 값을 재면 정확해집니다).", False, {"present": False})
        try:
            doc = atomic.read_json(p)
        except (OSError, ValueError) as e:
            return Check("gpu.vram_table", False, f"VRAM 측정표를 읽지 못했습니다: {p} ({e}) — 기본값을 씁니다.", False,
                         {"present": True, "readable": False})
        entries = doc.get("entries") if isinstance(doc, dict) else None
        if not isinstance(entries, dict) or doc.get("format") != "gtab.vram_table/1":
            return Check("gpu.vram_table", False, f"VRAM 측정표 형식이 올바르지 않습니다: {p} — 기본값을 씁니다.", False,
                         {"present": True, "readable": False})
        stamps = [str(e.get("measured_utc", "")) for rungs in entries.values() if isinstance(rungs, dict)
                  for e in rungs.values() if isinstance(e, dict)]
        n = len(stamps)
        newest = max((t for t in stamps if t), default="")
        age_days = None
        if newest:
            try:
                age_days = (_today() - datetime.fromisoformat(newest).date()).days
            except ValueError:
                age_days = None
        summary = ", ".join(f"{b} {len(r)}칸" for b, r in sorted(entries.items()) if isinstance(r, dict)) or "항목 없음"
        cal = doc.get("calibration") or {}
        detail = f"VRAM 측정표 {n}칸 ({summary})"
        if age_days is not None:
            detail += f", 최근 측정 {age_days}일 전"
        if cal.get("shared_growth_noise_mb") is not None:
            detail += f", 공유 메모리 잡음 {cal['shared_growth_noise_mb']} MB"
        stale = age_days is not None and age_days > VRAM_TABLE_STALE_DAYS
        if stale:
            detail += f" — {VRAM_TABLE_STALE_DAYS}일이 지났습니다. 드라이버·앱 구성이 바뀌었으면 `gtab bench gpu` 로 다시 재세요."
        if n == 0:
            detail += " — 비어 있어 기본 필요량을 씁니다."
        return Check("gpu.vram_table", not stale and n > 0, detail, False,
                     {"present": True, "entries": n, "age_days": age_days, "calibration": cal})

    def models_check() -> Check:
        from gtab import models as model_registry  # stdlib + filelock only

        p = model_registry.lock_path()
        if not p.is_file():
            return Check("models.present", False, f"모델 기록 파일이 없습니다: {p} — 모델 파일 위치·SHA256 을 확인할 수 "
                         f"없습니다. Beat This! 는 `gtab models fetch {BEAT_THIS_MODEL}` 로 받습니다.", False,
                         {"present": False})
        try:
            models = model_registry.entries()
        except model_registry.ModelRegistryError as e:  # Korean: unreadable or wrong format (never overwritten)
            return Check("models.present", False, str(e), False)
        missing = [name for name, entry in sorted(models.items())
                   if not entry.path or not model_registry.resolve(entry).is_file()]
        problems = []
        if missing:
            problems.append(f"기록은 있는데 파일이 없음: {_names_summary(missing)}")
        beat_missing = BEAT_THIS_MODEL not in models or BEAT_THIS_MODEL in missing
        if beat_missing:
            problems.append(f"Beat This! 체크포인트 없음 → `gtab models fetch {BEAT_THIS_MODEL}` "
                            "(없으면 `gtab run` 이 시작 전에 멈춥니다)")
        ok = not problems
        detail = f"기록된 모델 {len(models)}개" + (", 모두 있음" if ok else " — " + "; ".join(problems))
        return Check("models.present", ok, detail, False,
                     {"recorded": sorted(models), "missing": missing, "beat_this_missing": beat_missing})

    def alphatab_check() -> Check:
        # Needed for .gp/.gpx ground-truth import (`gtab gt import`, Tier A) and `gt render --engine alphatab`.
        from gtab.eval import node as gnode  # stdlib + gtab.winjob only

        ok = gnode.alphatab_available()
        if ok:
            return Check("node.alphatab", True, f"alphaTab {gnode.ALPHATAB_VERSION} · {gnode.alphatab_dir()}", False,
                         {"version": gnode.ALPHATAB_VERSION})
        return Check("node.alphatab", False, f"alphaTab {gnode.ALPHATAB_VERSION} 이 없습니다({gnode.alphatab_dir()}) "
                     "— .gp/.gpx 참조 탭을 가져올 수 없습니다. node 폴더에서 npm ci 를 실행하세요"
                     "(gtab.eval.node.npm_ci()).", False, {"version": None})

    return [_one("gpu.vram_table", False, vram_table_check), _one("models.present", False, models_check),
            _one("node.alphatab", False, alphatab_check)]


# ---------------------------------------------------------------------------------------------
# Orchestration

GROUPS: tuple[str, ...] = ("gpu", "ffmpeg", "node", "disk", "sysmon", "ytdlp", "global_python", "models")

_GROUP_FUNCS: dict[str, Callable[[Any], list[Check]]] = {
    "gpu": _check_gpu,
    "ffmpeg": _check_ffmpeg,
    "node": _check_node,
    "disk": _check_disk,
    "sysmon": _check_sysmon,
    "ytdlp": _check_ytdlp,
    "global_python": _check_global_python,
    "models": _check_models,
}


def _run_group(group: str, required: bool, cfg: Any) -> list[Check]:
    try:
        checks = _GROUP_FUNCS[group](cfg)
    except Exception as e:  # noqa: BLE001 - one broken probe must never take the doctor down
        return [_exception_check(group, required, e)]
    return [c for c in checks if isinstance(c, Check)]


def _load_config_or_defaults() -> tuple[Any, list[Check]]:
    try:
        from gtab.config import load_config

        return load_config(), []
    except Exception as e:  # noqa: BLE001
        return None, [Check("config", False, f"설정을 불러오지 못해 기본값으로 검사했습니다: {type(e).__name__}: {e}", True)]


def run_doctor(cfg: Any = None, *, skip: Iterable[str] = ()) -> DoctorReport:
    """Run all M0 environment checks.

    cfg: a `gtab.config.Config` (or nested dict); None loads the effective config, falling back to
         built-in defaults (reported as a failed "config" check) if that fails.
    skip: group names from `GROUPS` to leave out (e.g. {"gpu"}); each skipped group is reported as a
          warning so an unverified area is never shown as passing.
    """
    t0 = time.monotonic()
    skip_set = set(skip)
    unknown = skip_set - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown doctor groups: {sorted(unknown)} (known: {list(GROUPS)})")

    pre: list[Check] = []
    if cfg is None:
        cfg, pre = _load_config_or_defaults()
    required_of = {g: True for g in GROUPS}
    required_of["ytdlp"] = bool(_cfg_value(cfg, "youtube.enabled"))
    required_of["global_python"] = False
    required_of["models"] = False

    results: dict[str, list[Check]] = {}
    # NVML/PDH first so the process list and VRAM budget do not include our own selftest worker.
    if "sysmon" not in skip_set:
        results["sysmon"] = _run_group("sysmon", True, cfg)

    background = [g for g in GROUPS if g not in skip_set and g not in ("gpu", "sysmon")]
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gtab-doctor")
    futures: dict[str, Future[list[Check]]] = {}
    try:
        for g in background:
            futures[g] = pool.submit(_run_group, g, required_of[g], cfg)
        if "gpu" not in skip_set:
            # On the calling thread: Ctrl+C must reach run_worker so it can kill the worker's Job.
            results["gpu"] = _run_group("gpu", True, cfg)
        for g, fut in futures.items():
            try:
                results[g] = fut.result()
            except Exception as e:  # noqa: BLE001 - _run_group already guards; belt and braces
                results[g] = [_exception_check(g, required_of[g], e)]
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    checks = list(pre)
    for g in GROUPS:
        if g in skip_set:
            checks.append(Check(g, False, "건너뜀(검사하지 않음)", False, {"skipped": True}))
        else:
            checks.extend(results.get(g, []))
    report = DoctorReport(checks=checks, duration_s=time.monotonic() - t0)
    log.info("doctor: %s (%d checks, %d fail, %d warn, %.1f s)", "OK" if report.ok else "FAIL",
             len(checks), len(report.failures), len(report.warnings), report.duration_s)
    return report
