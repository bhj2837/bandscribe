"""VRAM budget, wait policy and GPU-stage orchestration (M1_M2_SPEC 8.1; DESIGN 9.5).

torch-free: imported by the core orchestrator (and harmlessly by workers). Two quantities, two sources, never
mixed (NVML free and the worker's ``torch.cuda.mem_get_info`` free disagreed by ~350 MB on this PC):

- orchestrator (no CUDA context of its own): ``reserved_peak_mb + ctx_mb + margin_mb <= NVML free``;
- worker (context exists, model not loaded yet): ``reserved_peak_mb + worker_headroom_mb <= mem_get_info free``.

``reserved_peak_mb`` is torch ``max_memory_reserved`` for a rung (model + activations); ``ctx_mb`` is the CUDA
context (NVML used after context creation minus before), both measured by ``bandscribe bench gpu`` into
``data/bench/vram_table.json``. Until the bench has run, ``DEFAULT_NEEDS`` (from the 2026-09-30 critique probes)
and ``gpu.ctx_mb_default`` are used. ``gpu.vram_margin_gb`` only covers other apps' fluctuation and
fragmentation - the context is counted separately.

Every GPU stage calls ``run_gpu_stage`` (wait for budget -> ``bandscribe.gpu.run_worker`` -> on the worker's own
``insufficient_vram`` answer wait and retry, bounded) and then ``write_gpu_run`` so the rung actually used is
recorded next to the outputs (the rung is provenance, not part of the stage key, M1_M2_SPEC 3.1 / A16).

Additive to the M1_M2_SPEC 7.1 envelope: ``vram.start_rung``. If the NVML pre-check finds that only a lower rung
fits, the worker is told to start there. Without it the pre-check's choice would be lost: on WDDM the worker's
``mem_get_info`` barely moves with other processes' usage (bench 2026-09-30), so the worker would attempt
rung 0 while a game holds the VRAM and spill or OOM first.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths

log = logging.getLogger(__name__)

GPU_RUN_FILE = "gpu_run.json"
GPU_RUN_FORMAT = "bandscribe.gpu_run/1"
VRAM_TABLE_FORMAT = "bandscribe.vram_table/1"

# reserved_peak_mb seeded from the 2026-09-30 critique measurements (M1_M2_SPEC facts); the bench overwrites them.
DEFAULT_NEEDS: dict[str, dict[str, float]] = {
    "sep_msst": {"chunk=588800": 1800.0, "chunk=352256": 1400.0, "chunk=262144": 1200.0},
    "amt_muscriptor": {"medium": 2400.0, "medium-lean": 1800.0, "small": 1400.0},  # small: estimate, never measured
    "beats_beatthis": {"final0": 500.0},
}

# Mirrors of the [gpu] defaults (M1_M2_SPEC 1.3) for callers without a full Config: ``None`` (GRID's worker
# test, scripts) or a partial dict (tests, bench). A real ``bandscribe.config.Config`` is always read strictly, so a
# misspelled key raises instead of silently returning a default; ``tests/test_vram.py`` keeps these mirrors
# equal to ``Config()``.
GPU_DEFAULTS: dict[str, Any] = {
    "gpu.insufficient_vram_policy": "wait",
    "gpu.vram_margin_gb": 0.7,
    "gpu.wait_poll_s": 30.0,
    "gpu.wait_max_s": 3600.0,
    "gpu.wait_total_max_s": 3600.0,
    "gpu.max_worker_retries": 3,
    "gpu.worker_headroom_mb": 256.0,
    "gpu.ctx_mb_default": 150.0,
    "gpu.shared_growth_fail_mb": 64.0,
    "gpu.load_spill_fail_mb": 48.0,
    "gpu.rtf_fail_ratio": 0.5,
    "gpu.cpu_backends": ["beats_beatthis"],
    "gpu.worker_timeout_s": 7200.0,
    "gpu.lock_timeout_s": 7200.0,
}

_MISSING: Any = object()


def _is_config(cfg: Any) -> bool:
    # Duck-typed so this module never imports bandscribe.config (pydantic) itself: a pydantic model with ``get``.
    return hasattr(type(cfg), "model_fields") and callable(getattr(cfg, "get", None))


def cfg_get(cfg: Any, dotted: str, default: Any = _MISSING) -> Any:
    """``a.b`` from a ``bandscribe.config.Config``, a nested mapping or None.

    A real Config is read strictly: every key used here exists in it (P0 config, M1_M2_SPEC 1.3), so an unknown
    key is a programming error and raises KeyError rather than quietly using a default. A mapping (tests,
    partial configs) or None falls back to ``default`` / ``GPU_DEFAULTS``.
    """
    if _is_config(cfg):
        value = cfg.get(dotted)  # KeyError for a misspelled key
        if value is not None:
            return value
    if default is _MISSING:
        default = GPU_DEFAULTS.get(dotted)
    if cfg is None or _is_config(cfg):
        return default
    if not isinstance(cfg, Mapping):
        getter = getattr(cfg, "get", None)
        if callable(getter):
            try:
                v = getter(dotted, None)
            except TypeError:  # a getter without a default parameter
                try:
                    v = getter(dotted)
                except KeyError:
                    v = None
            return default if v is None else v
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif not isinstance(node, Mapping) and hasattr(node, part):
            node = getattr(node, part)
        else:
            return default
    return default if node is None else node


def vram_table_path() -> Path:
    """``paths.VRAM_TABLE``, read at call time (tests repoint it)."""
    return Path(paths.VRAM_TABLE)


# --------------------------------------------------------------------------------------------- budget


@dataclass
class GpuSnapshot:
    total_mb: float
    used_mb: float
    free_mb: float
    processes: list[dict] = field(default_factory=list)


def snapshot() -> GpuSnapshot | None:
    """NVML view of device 0 (the CUDA device our workers use); None if NVML is unavailable."""
    from bandscribe import sysmon

    info = sysmon.nvml_info()
    if not info or not info.get("devices"):
        return None
    dev = info["devices"][0]
    procs = [p for p in info.get("processes", []) if p.get("device", 0) == dev.get("index", 0)]
    return GpuSnapshot(total_mb=float(dev["total_mb"]), used_mb=float(dev["used_mb"]),
                       free_mb=float(dev["free_mb"]), processes=procs)


def budget_mb(snap: GpuSnapshot, margin_mb: float) -> float:
    """NVML free minus the margin for other apps' fluctuation (the CUDA context is part of the need)."""
    return snap.free_mb - margin_mb


@dataclass(frozen=True)
class RungNeed:
    reserved_peak_mb: float
    ctx_mb: float


def orchestrator_need_mb(n: RungNeed) -> float:
    """Compared with NVML free - margin (the orchestrator has no context of its own)."""
    return n.reserved_peak_mb + n.ctx_mb


def worker_need_mb(n: RungNeed, headroom_mb: float) -> float:
    """Compared with ``mem_get_info`` free inside the worker (its context already exists)."""
    return n.reserved_peak_mb + headroom_mb


def read_table(path: Path | None = None) -> dict[str, Any] | None:
    """The raw ``vram_table.json`` or None (absent or unreadable - never fatal)."""
    p = vram_table_path() if path is None else Path(path)
    if not p.is_file():
        return None
    try:
        data = atomic.read_json(p)
    except (OSError, ValueError) as e:
        log.warning("ignoring unreadable VRAM table %s: %s", p, e)
        return None
    if not isinstance(data, dict) or data.get("format") != VRAM_TABLE_FORMAT:
        log.warning("ignoring VRAM table %s with unexpected format", p)
        return None
    return data


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0 else None


def load_table(path: Path | None = None, *, ctx_default_mb: float) -> dict[str, dict[str, RungNeed]]:
    """Needs per backend and rung label: measured entries where present, ``DEFAULT_NEEDS`` otherwise.

    ``path`` defaults to ``paths.VRAM_TABLE`` (resolved at call time, not at import, so tests can repoint it).

    A measured entry without ``ctx_mb`` gets ``ctx_default_mb`` (``gpu.ctx_mb_default``).
    """
    table = read_table(path)
    entries = table.get("entries", {}) if table else {}
    out: dict[str, dict[str, RungNeed]] = {}
    for backend in sorted(set(DEFAULT_NEEDS) | set(entries)):
        rungs: dict[str, RungNeed] = {}
        for label, reserved in DEFAULT_NEEDS.get(backend, {}).items():
            rungs[label] = RungNeed(float(reserved), float(ctx_default_mb))
        for label, e in (entries.get(backend) or {}).items():
            if not isinstance(e, Mapping):
                continue
            reserved = _num(e.get("reserved_peak_mb"))
            if reserved is None:
                continue
            ctx = _num(e.get("ctx_mb"))
            rungs[label] = RungNeed(reserved, float(ctx_default_mb) if ctx is None else ctx)
        out[backend] = rungs
    return out


def rtf_refs(backend: str, labels: list[str], path: Path | None = None) -> dict[str, float | None]:
    """Measured real-time factor per rung (``None`` = not benched -> the worker skips its slowness check)."""
    table = read_table(path)
    entries = ((table or {}).get("entries") or {}).get(backend) or {}
    return {lab: _num((entries.get(lab) or {}).get("rtf")) for lab in labels}


def choose_rung(labels: list[str], budget: float, needs: Mapping[str, RungNeed]) -> int | None:
    """Index of the first (largest) rung whose orchestrator need fits the budget; unknown labels never fit."""
    for i, label in enumerate(labels):
        need = needs.get(label)
        if need is not None and orchestrator_need_mb(need) <= budget:
            return i
    return None


def _gb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB"


def _proc_names(procs: list[dict], limit: int = 6) -> str:
    names = []
    for p in procs:
        name = p.get("name") or f"pid {p.get('pid')}"
        name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
        used = p.get("used_mb")
        names.append(f"{name}({used} MB)" if used else name)
    if not names:
        return "없음"
    return ", ".join(names[:limit]) + (f" 외 {len(names) - limit}개" if len(names) > limit else "")


class VramInsufficient(RuntimeError):
    """Not enough VRAM for any GPU rung (Korean message: need / free / other GPU processes)."""

    def __init__(self, backend: str, need_mb: float | None, free_mb: float | None, processes: list[dict],
                 *, reason: str = "") -> None:
        self.backend, self.need_mb, self.free_mb, self.processes = backend, need_mb, free_mb, processes
        need = "?" if need_mb is None else _gb(need_mb)
        free = "?" if free_mb is None else _gb(free_mb)
        msg = (f"VRAM 부족({backend}): 필요 약 {need}, 여유 {free} "
               f"(다른 GPU 사용 프로그램: {_proc_names(processes)}).")
        if reason:
            msg += f" {reason}"
        msg += " 게임·브라우저 등 GPU를 쓰는 앱을 닫고 다시 실행하세요."
        super().__init__(msg)


def _policy(cfg: Any) -> str:
    p = str(cfg_get(cfg, "gpu.insufficient_vram_policy"))
    return p if p in ("wait", "error", "cpu") else "wait"


def cpu_allowed(backend: str, cfg: Any) -> bool:
    """CPU rungs exist only under policy ``cpu`` and only for ``gpu.cpu_backends`` (M1_M2_SPEC A17)."""
    return _policy(cfg) == "cpu" and backend in list(cfg_get(cfg, "gpu.cpu_backends") or [])


WAIT_NOTICE_EVERY_S = 300.0  # while waiting for VRAM, say so again this often (a silent wait looks like a hang)


def wait_for_budget(backend: str, labels: list[str], cfg: Any, *, clock: Callable[[], float] = time.monotonic,
                    sleep: Callable[[float], None] = time.sleep, notify: Callable[[str], None] | None = None,
                    snap_fn: Callable[[], GpuSnapshot | None] | None = None,
                    needs: Mapping[str, RungNeed] | None = None, max_wait_s: float | None = None,
                    elapsed0: float = 0.0, total_max_s: float | None = None) -> tuple[int | None, float]:
    """(first GPU rung that fits | None = use the CPU rung, waited_s).

    policy "wait": poll every ``gpu.wait_poll_s`` up to ``gpu.wait_max_s`` (``max_wait_s``: the caller's
    smaller remainder of ``gpu.wait_total_max_s``), then VramInsufficient; "error": raise at once; "cpu":
    return None iff backend in ``gpu.cpu_backends``, else behave like "wait". While waiting, ``notify`` gets the
    Korean reason once and a "still waiting" line every ``WAIT_NOTICE_EVERY_S``; ``elapsed0`` (time this stage
    already waited) and ``total_max_s`` only shape those messages.
    Without NVML the check cannot be made: returns (0, 0) and the worker's own check decides.
    """
    snap_fn = snapshot if snap_fn is None else snap_fn
    if needs is None:
        needs = load_table(ctx_default_mb=float(cfg_get(cfg, "gpu.ctx_mb_default"))).get(backend, {})
    margin = float(cfg_get(cfg, "gpu.vram_margin_gb")) * 1024.0
    poll = max(0.0, float(cfg_get(cfg, "gpu.wait_poll_s")))
    max_wait = float(cfg_get(cfg, "gpu.wait_max_s")) if max_wait_s is None else max(0.0, float(max_wait_s))
    policy = _policy(cfg)
    smallest = min((orchestrator_need_mb(needs[lab]) for lab in labels if lab in needs), default=None)

    t0 = clock()
    announced = False
    next_notice = WAIT_NOTICE_EVERY_S
    while True:
        snap = snap_fn()
        waited = clock() - t0
        if snap is None:
            log.warning("NVML unavailable: skipping the VRAM pre-check for %s (the worker checks itself)", backend)
            return 0, waited
        budget = budget_mb(snap, margin)
        idx = choose_rung(labels, budget, needs)
        if idx is not None:
            need = orchestrator_need_mb(needs[labels[idx]])
            msg = (f"VRAM 사전 점검({backend}): 예산 {_gb(budget)} (NVML 여유 {_gb(snap.free_mb)} - 여유분 "
                   f"{_gb(margin)}) -> {labels[idx]} (필요 약 {_gb(need)})")
            log.info(msg)
            if announced and notify is not None:
                notify(f"VRAM 여유가 생겨 {labels[idx]} 으로 이어서 진행합니다.")
            return idx, waited
        if policy == "error":
            raise VramInsufficient(backend, smallest, snap.free_mb, snap.processes,
                                   reason="(gpu.insufficient_vram_policy=error)")
        if policy == "cpu" and cpu_allowed(backend, cfg):
            if notify is not None:
                notify(f"VRAM 부족({backend}): 필요 약 {_gb(smallest or 0)}, 여유 {_gb(budget)}. CPU로 실행합니다(느림).")
            return None, waited
        if waited >= max_wait:
            total = elapsed0 + waited
            if total_max_s is not None and total >= total_max_s - 1e-9:
                reason = (f"이 단계에서 모두 {total / 60:.0f}분 기다려도 여유가 생기지 않았습니다"
                          f"(최대 {total_max_s / 60:.0f}분, gpu.wait_total_max_s).")
            else:
                reason = f"{max_wait / 60:.0f}분 기다려도 여유가 생기지 않았습니다."
            raise VramInsufficient(backend, smallest, snap.free_mb, snap.processes, reason=reason)
        if not announced:  # say it once in full; the wait itself can take minutes
            msg = (f"VRAM 부족: 필요 약 {_gb(smallest or 0)}, 여유 {_gb(max(budget, 0.0))} "
                   f"(다른 GPU 사용 프로그램: {_proc_names(snap.processes)}). 앱을 닫으면 이어서 진행합니다. 대기 중...")
            log.warning(msg)
            if notify is not None:
                notify(msg)
            announced = True
        elif waited >= next_notice:  # ... and then every few minutes, so a long wait never looks like a hang
            limit = total_max_s if total_max_s is not None else elapsed0 + max_wait
            msg = (f"아직 VRAM 여유를 기다리는 중입니다({backend}): {(elapsed0 + waited) / 60:.0f}분째, "
                   f"최대 {limit / 60:.0f}분. 필요 약 {_gb(smallest or 0)}, 여유 {_gb(max(budget, 0.0))} "
                   f"(다른 GPU 사용 프로그램: {_proc_names(snap.processes)}).")
            log.warning(msg)
            if notify is not None:
                notify(msg)
            while next_notice <= waited:
                next_notice += WAIT_NOTICE_EVERY_S
        # A zero poll interval (tests) must still make progress on the clock: never sleep past the deadline.
        sleep(min(poll, max(0.0, max_wait - waited)) if poll > 0 else 0.0)


# ------------------------------------------------------------------------------------- stage runner


class GpuWorkerFailed(RuntimeError):
    """A GPU worker ended without a usable result (Korean message names ``_worker/stderr.log``)."""

    def __init__(self, backend: str, res: Any) -> None:
        self.backend, self.result = backend, res
        err = (res.result or {}).get("error") if getattr(res, "result", None) is not None else None
        why = []
        if getattr(res, "timed_out", False):
            why.append("시간 초과")
        if getattr(res, "lock_released_dirty", False):
            why.append("GPU 락을 깨끗하게 풀지 못함(종료되지 않는 프로세스)")
        if err:
            why.append(str(err))
        stderr = getattr(res, "stderr_path", None)
        super().__init__(f"GPU 워커 {backend} 실패: {'; '.join(why) or '알 수 없는 오류'}. "
                         f"자세한 내용: {stderr}")


def build_vram_request(backend: str, labels: list[str], cfg: Any, *, force_rung: str | None = None,
                       cap_mb: float | None = None, precheck: bool = True) -> dict[str, Any]:
    """The ``vram`` block of the worker envelope (M1_M2_SPEC 7.1), computed on the orchestrator side."""
    ctx_default = float(cfg_get(cfg, "gpu.ctx_mb_default"))
    needs = load_table(ctx_default_mb=ctx_default).get(backend, {})
    return {
        "policy": _policy(cfg),
        "cpu_allowed": cpu_allowed(backend, cfg),
        "precheck": bool(precheck),
        "worker_headroom_mb": float(cfg_get(cfg, "gpu.worker_headroom_mb")),
        "need_table": {lab: {"reserved_peak_mb": needs[lab].reserved_peak_mb, "ctx_mb": needs[lab].ctx_mb}
                       for lab in labels if lab in needs},
        "shared_growth_fail_mb": float(cfg_get(cfg, "gpu.shared_growth_fail_mb")),
        "load_spill_fail_mb": float(cfg_get(cfg, "gpu.load_spill_fail_mb")),
        "rtf_ref": rtf_refs(backend, labels),
        "rtf_fail_ratio": float(cfg_get(cfg, "gpu.rtf_fail_ratio")),
        "cap_mb": cap_mb,
        "force_rung": force_rung,
    }


def run_gpu_stage(backend: str, labels: list[str], request: dict, cfg: Any, work_dir: Path, *,
                  notify: Callable[[str], None] | None = None, run_worker: Callable[..., Any] | None = None,
                  clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                  snap_fn: Callable[[], GpuSnapshot | None] | None = None, check: bool = True) -> Any:
    """wait_for_budget -> run_worker -> on ``output.status == "insufficient_vram"`` wait and retry.

    ``labels`` are the GPU rung labels of the configured ladder (largest first). The request's ``vram`` block
    is filled here (``build_vram_request``) unless the caller supplied one (bench). A ``force_rung`` in it
    restricts the pre-check to that rung. When the pre-check picks a lower rung than the top one, the request
    carries ``vram.start_rung`` and the worker starts there (``gpu_common.run_ladder``; b2). Returns the ``bandscribe.gpu.WorkerResult``, with ``waited_s`` (seconds
    spent waiting for VRAM) attached as an attribute for ``write_gpu_run``.
    check=True raises GpuWorkerFailed for a failed worker (``ok=False`` or a dirty lock release); after
    ``gpu.max_worker_retries`` ``insufficient_vram`` answers (or at once under policy ``error``) raises
    VramInsufficient. All the waiting together (pre-checks and retry pauses) is bounded by
    ``gpu.wait_total_max_s``: each pre-check gets only what is left of it (review 2026-10-05; before, every
    pre-check could wait the full ``gpu.wait_max_s``, about 4 h in all with the default 3 retries).
    """
    if run_worker is None:
        from bandscribe import gpu

        run_worker = gpu.run_worker
    req = copy.deepcopy(request)
    vram = req.get("vram")
    if not isinstance(vram, dict):
        vram = build_vram_request(backend, labels, cfg)
        req["vram"] = vram
    force = vram.get("force_rung")
    check_labels = [force] if force else list(labels)
    needs = load_table(ctx_default_mb=float(cfg_get(cfg, "gpu.ctx_mb_default"))).get(backend, {})
    max_retries = int(cfg_get(cfg, "gpu.max_worker_retries"))
    poll = max(0.0, float(cfg_get(cfg, "gpu.wait_poll_s")))
    wait_max = float(cfg_get(cfg, "gpu.wait_max_s"))
    total_max = float(cfg_get(cfg, "gpu.wait_total_max_s"))
    policy = _policy(cfg)
    precheck = bool(vram.get("precheck", True))

    waited_total = 0.0
    retries = 0
    while True:
        idx: int | None = 0
        if precheck and (not force or force in needs):
            idx, waited = wait_for_budget(backend, check_labels, cfg, clock=clock, sleep=sleep, notify=notify,
                                          snap_fn=snap_fn, needs=needs,
                                          max_wait_s=min(wait_max, max(0.0, total_max - waited_total)),
                                          elapsed0=waited_total, total_max_s=total_max)
            waited_total += waited
        attempt_req = copy.deepcopy(req)
        if idx is None:  # policy "cpu" and a CPU-capable backend: skip the CUDA context entirely
            attempt_req["device"] = "cpu"
            attempt_req["vram"]["cpu_allowed"] = True
        elif idx > 0 and not force:
            # b2: the NVML pre-check chose a lower rung, so the worker starts there (it may still step down).
            # The worker's own mem_get_info check cannot replace this: on WDDM it barely moves with other
            # processes' usage (bench 2026-09-30: 5098 MB free at NVML 1.65 GB used, ~4700 MB at 6.0 GB used),
            # so it would attempt rung 0 and spill or OOM while a game holds the VRAM.
            attempt_req["vram"]["start_rung"] = check_labels[idx]
        res = run_worker(backend, attempt_req, Path(work_dir),
                         timeout_s=float(cfg_get(cfg, "gpu.worker_timeout_s")),
                         lock_timeout_s=float(cfg_get(cfg, "gpu.lock_timeout_s")))
        output = res.output if isinstance(getattr(res, "output", None), dict) else {}
        if res.ok and output.get("status") == "insufficient_vram":
            retries += 1
            vr = output.get("vram") or {}
            free = _num(vr.get("start_free_mb"))
            smallest = min((worker_need_mb(needs[lab], float(cfg_get(cfg, "gpu.worker_headroom_mb")))
                            for lab in check_labels if lab in needs), default=None)
            if policy == "error" or retries > max_retries:
                snap = (snap_fn or snapshot)()
                raise VramInsufficient(backend, smallest, free, snap.processes if snap else [],
                                       reason=f"워커가 VRAM 부족을 {retries}번 알렸습니다.")
            if waited_total + poll > total_max + 1e-9:
                snap = (snap_fn or snapshot)()
                raise VramInsufficient(backend, smallest, free, snap.processes if snap else [],
                                       reason=f"워커가 VRAM 부족을 {retries}번 알렸고, 이 단계에서 기다린 시간이 모두 합쳐 "
                                              f"{total_max / 60:.0f}분(gpu.wait_total_max_s)을 넘게 됩니다.")
            if notify is not None:
                notify(f"워커가 VRAM 부족을 알렸습니다({backend}, 여유 {_gb(free or 0)}). "
                       f"{poll:.0f}초 뒤 다시 시도합니다({retries}/{max_retries}).")
            log.warning("worker %s answered insufficient_vram (free %s MB); retry %d/%d",
                        backend, free, retries, max_retries)
            sleep(poll)
            waited_total += poll
            continue
        res.waited_s = round(waited_total, 3)
        if check and (not res.ok or getattr(res, "lock_released_dirty", False)):
            raise GpuWorkerFailed(backend, res)
        return res


# ----------------------------------------------------------------------------------- gpu_run.json


def write_gpu_run(out_dir: Path, backend: str, labels: list[str], worker_output: Mapping[str, Any],
                  waited_s: float = 0.0) -> dict[str, Any]:
    """Record the rung actually used at the stage root (M1_M2_SPEC 3.1). Returns the written dict.

    ``degraded`` = a lower rung than the top of the configured ladder, or the CPU. The runner warns about
    cached degraded results; experiments record the rung in their metric rows instead. ``slow`` (an attempt
    ran slower than its benched realtime factor) and the worker's Korean ``warnings`` are kept too, so a run
    that reuses this stage from the cache can still show them (review 2026-10-05).
    """
    vr = worker_output.get("vram") or {}
    label = vr.get("rung_label")
    be = worker_output.get("backend") or {}
    device = be.get("device_used")
    if device is None:
        for a in vr.get("attempts") or []:
            if a.get("status") == "ok":
                device = a.get("device")
    device = "cpu" if str(device or "").startswith("cpu") else ("cuda" if device else None)
    if label in labels:
        index = labels.index(label)
    else:  # a CPU rung (beyond the GPU ladder) or an unknown label
        index = len(labels)
    doc = {
        "format": GPU_RUN_FORMAT,
        "backend": backend,
        "ladder": list(labels),
        "rung": {"index": index, "label": label, "device": device},
        "degraded": bool(index > 0 or device == "cpu"),
        "waited_s": round(float(waited_s or 0.0), 3),
        "slow": any(bool(a.get("slow")) for a in vr.get("attempts") or [] if isinstance(a, Mapping)),
        "warnings": [str(w) for w in worker_output.get("warnings") or []],
    }
    atomic.write_json(Path(out_dir) / GPU_RUN_FILE, doc)
    return doc


def read_gpu_run(stage_dir: Path) -> dict | None:
    p = Path(stage_dir) / GPU_RUN_FILE
    if not p.is_file():
        return None
    try:
        doc = atomic.read_json(p)
    except (OSError, ValueError) as e:
        log.warning("unreadable %s: %s", p, e)
        return None
    return doc if isinstance(doc, dict) and doc.get("format") == GPU_RUN_FORMAT else None
