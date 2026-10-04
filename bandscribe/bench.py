"""``bandscribe bench gpu``: VRAM and speed per backend and ladder rung (M1_M2_SPEC 5, 9.1 item 10; DESIGN M2 b1-b5, b9).

For each backend x rung the real worker runs on a clip with ``precheck: false`` (the bench never pre-checks,
so the worker really attempts the rung) and either ``force_rung`` (``--rungs all``: one run per rung) or the
normal ladder from rung 0 (``--rungs auto``). Every attempt of every run becomes a row of ``bench.csv``;
``bench.json`` keeps the full worker outputs plus the NVML process list at the start (the Claude app itself
holds VRAM and cannot be closed while it drives bandscribe, so b1/b9 evidence states what was running).

Two ways to make a rung run short of memory - they test different things:

- ``--cap-mb N``: PyTorch allocator limit (``set_per_process_memory_fraction``). Validates the OOM ladder (b4).
  The allocator raises before real memory runs out, so the driver's Sysmem Fallback never engages.
- ``--ballast-mb N|auto``: a separate ``vram_ballast`` worker, in its own Job Object and outside the GPU lock,
  really occupies N MB before the measured worker starts (b3). ``auto`` = NVML free - (orchestrator need of
  rung 0 - 400 MB): rung 0 does not fit, lower rungs may. With the NVIDIA "Sysmem Fallback" setting off rung 0
  should fail with ``oom``; with it on, with ``shared_growth`` (this pid's shared usage after model load).
  The ballast **rewrites its blocks every 0.2 s**: measured 2026-09-30, an *idle* ballast of 3.2 or even 4.5 GB
  was simply paged out by WDDM (NVML at 98 % of 6 GB) and SW's top rung ran fully in dedicated VRAM with no
  shared growth - an idle ballast creates no pressure. A hot one behaves like a game: with ``auto`` (4.3 GB)
  rung 0 tripped ``shared_growth`` (66 MB) at 1.5x realtime and rung 1 finished (flagged slow). The bench holds
  the GPU lock from before the ballast allocates until it is released, so it never starves another bandscribe worker.

On WDDM ``torch.cuda.mem_get_info`` free barely moves with other processes' usage (5098 MB free with NVML at
1.65 GB used; 4666-4723 MB with NVML at 6.0 GB used), so the worker-side pre-check rarely skips a rung; the
orchestrator's NVML pre-check and the runtime OOM / shared-growth detection are what protect a run.

Only uncapped, ballast-free attempts with status ``ok`` are merged into ``data/bench/vram_table.json`` (median
over ``--repeat``); capped / ballasted runs are evidence for b3 / b4 only. Over all ``--fallback off`` attempts
the maximum ``pdh_self_shared_growth_mb`` is the shared-growth noise floor; the recommended
``gpu.shared_growth_fail_mb`` is ``max(64, 2 x noise)``.

Backends ``amt`` and ``beats`` are driven with their documented requests (M1_M2_SPEC 7.3, 7.5) - this module
does not import AMT's or GRID's code.
"""

from __future__ import annotations

import contextlib
import csv
import io
import logging
import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import bandscribe
from bandscribe import atomic, paths, vram

log = logging.getLogger(__name__)

BENCH_FORMAT = "bandscribe.bench/1"
SR = 44100
BACKENDS = {"sep": "sep_msst", "amt": "amt_muscriptor", "beats": "beats_beatthis"}
BALLAST_AUTO_SLACK_MB = 400.0
BALLAST_TOLERANCE = 0.10
# The ballast rewrites its blocks this often (s): an idle ballast is simply paged out by WDDM (measured).
BALLAST_TOUCH_S = 0.2
MIN_SHARED_GROWTH_FAIL_MB = 64.0

CSV_COLUMNS = ("backend", "run", "repeat", "mode", "rung_index", "rung_label", "device", "status",
               "max_allocated_mb", "max_reserved_mb", "nvml_used_peak_mb", "nvml_delta_peak_mb",
               "pdh_self_shared_growth_mb", "pdh_adapter_shared_delta_mb", "rtf", "slow", "process_s", "load_s",
               "ctx_mb", "start_free_mb", "cap_mb", "ballast_mb", "fallback", "clip_s", "worker_ok", "error")

MT3_GUITAR = ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"]  # MT3 ids 4, 5, 6


class BenchError(RuntimeError):
    """User-facing (Korean) bench failure."""


@dataclass
class BenchOptions:
    clip: str | None = None
    seconds: float | None = 240.0
    backends: list[str] = field(default_factory=lambda: ["sep"])
    rungs: str = "all"  # all | auto
    cap_mb: float | None = None
    ballast_mb: int | str | None = None  # N | "auto" | None
    fallback: str = "unknown"  # on | off | unknown (recorded only; the program never changes it)
    repeat: int = 1

    def validate(self) -> None:
        bad = [b for b in self.backends if b not in BACKENDS]
        if bad:
            raise BenchError(f"알 수 없는 백엔드: {bad} (가능: sep, amt, beats, all)")
        if self.rungs not in ("all", "auto"):
            raise BenchError(f"--rungs 는 all 또는 auto 입니다 (입력값: {self.rungs!r})")
        if self.fallback not in ("on", "off", "unknown"):
            raise BenchError(f"--fallback 은 on, off, unknown 중 하나입니다 (입력값: {self.fallback!r})")
        if self.repeat < 1:
            raise BenchError("--repeat 는 1 이상이어야 합니다")
        if self.cap_mb is not None and self.cap_mb <= 0:
            raise BenchError("--cap-mb 는 양수여야 합니다")
        if self.ballast_mb is not None and self.ballast_mb != "auto":
            try:
                if int(self.ballast_mb) <= 0:
                    raise ValueError
            except ValueError as e:
                raise BenchError(f"--ballast-mb 는 양의 정수 또는 auto 입니다 (입력값: {self.ballast_mb!r})") from e


# ----------------------------------------------------------------------------------------- helpers


def bench_root() -> Path:
    """``paths.BENCH``, read at call time (tests repoint it)."""
    return Path(paths.BENCH)


def git_sha7() -> str:
    from bandscribe import winjob

    try:
        run = winjob.run_captured(["git", "-C", str(paths.ROOT), "rev-parse", "--short=7", "HEAD"],
                                  timeout_s=15, env=paths.child_env())
    except OSError:
        return "nogit"
    sha = run.stdout.strip()
    return sha if run.returncode == 0 and sha else "nogit"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _worker_env() -> dict[str, str]:
    """Same environment as bandscribe.gpu.run_worker gives its workers (src tree first on PYTHONPATH)."""
    src_root = str(Path(bandscribe.__file__).resolve().parents[1])
    env = paths.child_env()
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_root, env.get("PYTHONPATH", "")) if p)
    return env


def synth_clip(seconds: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Deterministic band-like stereo clip (n, 2): plucked harmonic notes, a bass line, noise hits."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros((n, 2), dtype=np.float64)
    t_note = 0.25
    k = int(t_note * sr)
    env = np.exp(-np.arange(k) / (0.12 * sr))
    tt = np.arange(k) / sr
    for start in range(0, n - k, k):
        f = 110.0 * 2 ** (rng.integers(0, 24) / 12)
        tone = sum(np.sin(2 * np.pi * f * h * tt) / h for h in (1, 2, 3, 4)) * env * 0.15
        pan = rng.uniform(0.2, 0.8)
        out[start:start + k, 0] += tone * (1 - pan)
        out[start:start + k, 1] += tone * pan
        if (start // k) % 2 == 0:
            fb = 55.0 * 2 ** (rng.integers(0, 12) / 12)
            out[start:start + k, :] += (np.sin(2 * np.pi * fb * tt) * env * 0.2)[:, None]
        hit = rng.standard_normal(min(k, 2000)) * np.exp(-np.arange(min(k, 2000)) / 300.0) * 0.2
        out[start:start + len(hit), :] += hit[:, None]
    return out.astype(np.float32)


def _find_job_mix(key: str) -> Path | None:
    root = Path(paths.JOBS)
    exact = root / key / "input" / "mix_44k_f32.wav"
    if exact.is_file():
        return exact
    matches = sorted(p for p in root.glob(f"{key}*/input/mix_44k_f32.wav")) if root.is_dir() else []
    return matches[0] if len(matches) == 1 else None


def prepare_clip(clip: str | None, seconds: float | None, out_dir: Path) -> tuple[Path, float, str]:
    """Stereo 44.1 kHz float WAV of at most ``seconds`` s in ``out_dir``: (path, duration_s, description)."""
    import soundfile as sf

    from bandscribe.audio import write_f32

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "clip.wav"
    if clip is None:
        secs = float(seconds or 240.0)
        write_f32(dst, synth_clip(secs), SR)
        return dst, secs, f"synthetic {secs:.0f}s"
    src = Path(clip)
    desc = str(clip)
    if src.is_file():
        from bandscribe.ingest.decode import decode_to_f32

        tmp = out_dir / "clip_full.wav"
        decode_to_f32(src, tmp, SR)
    else:
        mix = _find_job_mix(clip)
        if mix is None:
            raise BenchError(f"--clip 을 찾지 못했습니다: {clip} (파일 경로나 작업 키를 주세요)")
        tmp = mix
        desc = f"job {mix.parents[1].name}"
    info = sf.info(str(tmp))
    frames = info.frames if not seconds else min(info.frames, int(float(seconds) * SR))
    data, sr = sf.read(str(tmp), frames=frames, dtype="float32", always_2d=True)
    if sr != SR:
        raise BenchError(f"clip 샘플레이트가 {sr} Hz 입니다 ({SR} Hz 필요)")
    write_f32(dst, data, SR)
    if tmp.name == "clip_full.wav":
        tmp.unlink(missing_ok=True)
    return dst, frames / SR, desc


# ------------------------------------------------------------------------------------------- ballast


class Ballast:
    """A ``vram_ballast`` worker in its own Job Object, outside the GPU lock (b3)."""

    def __init__(self, mb: int, work_dir: Path, *, python: Path | None = None, mode: str = "hold",
                 max_s: float = 3 * 3600.0, ready_timeout_s: float = 120.0, touch_s: float = BALLAST_TOUCH_S,
                 verify: bool = True, snap_fn: Callable[[], vram.GpuSnapshot | None] | None = None) -> None:
        self.mb = int(mb)
        self.work_dir = Path(work_dir)
        self.python = Path(python) if python is not None else paths.GPU_PYTHON
        self.mode = mode
        self.max_s = max_s
        self.ready_timeout_s = ready_timeout_s
        self.touch_s = touch_s
        self.verify = verify
        self.snap_fn = snap_fn or vram.snapshot
        self.held_mb: int | None = None
        self.nvml_rise_mb: float | None = None
        self.job: Any = None
        self.proc: Any = None

    def start(self) -> int:
        from bandscribe.winjob import JobObject, spawn_in_job

        self.work_dir.mkdir(parents=True, exist_ok=True)
        for name in ("ready.json", "stop", "result.json"):
            (self.work_dir / name).unlink(missing_ok=True)
        req = self.work_dir / "request.json"
        atomic.write_json(req, {"mode": self.mode, "mb": self.mb, "block_mb": 256, "max_s": self.max_s,
                                "parent_pid": os.getpid(), "touch_s": self.touch_s})
        before = self.snap_fn() if self.verify else None
        self.job = JobObject(kill_on_close=True)
        args = [str(self.python), "-m", "bandscribe.workers.vram_ballast", str(req), str(self.work_dir / "result.json")]
        with open(self.work_dir / "stdout.log", "wb") as out, open(self.work_dir / "stderr.log", "wb") as err:
            self.proc = spawn_in_job(args, self.job, env=_worker_env(), cwd=self.work_dir, stdout=out, stderr=err)
        deadline = time.monotonic() + self.ready_timeout_s
        ready = self.work_dir / "ready.json"
        while not ready.exists():
            if self.proc.poll() is not None:
                self.close()
                raise BenchError(f"VRAM 밸러스트가 준비 전에 끝났습니다 (자세한 내용: {self.work_dir / 'stderr.log'})")
            if time.monotonic() > deadline:
                self.close()
                raise BenchError(f"VRAM 밸러스트가 {self.ready_timeout_s:.0f}초 안에 준비되지 않았습니다")
            time.sleep(0.1)
        doc = atomic.read_json(ready)
        self.held_mb = int(doc.get("held_mb", 0))
        if self.verify:
            time.sleep(0.5)  # let NVML catch up with the allocation
            after = self.snap_fn()
            if before is not None and after is not None:
                self.nvml_rise_mb = round(after.used_mb - before.used_mb, 1)
                if abs(self.nvml_rise_mb - self.mb) > BALLAST_TOLERANCE * self.mb:
                    self.close()
                    raise BenchError(f"VRAM 밸러스트 확인 실패: {self.mb} MB 를 잡았어야 하는데 NVML 사용량이 "
                                     f"{self.nvml_rise_mb:.0f} MB 늘었습니다(±10 % 밖). 다른 앱이 VRAM 을 바꾸는 중일 "
                                     "수 있습니다. 잠시 뒤 다시 실행하세요.")
        log.info("ballast holds %s MB (NVML rise %s MB)", self.held_mb, self.nvml_rise_mb)
        return self.held_mb

    def stop(self, timeout_s: float = 30.0) -> None:
        if self.proc is None:
            return
        try:
            (self.work_dir / "stop").write_bytes(b"")
            try:
                self.proc.wait(timeout=timeout_s)
            except Exception:
                log.warning("ballast did not stop within %.0fs; killing its job", timeout_s)
        finally:
            self.close()

    def close(self) -> None:
        if self.job is not None:
            try:
                self.job.terminate(1)
            except OSError:
                pass
            try:
                self.job.wait_empty(10.0)
            except OSError:
                pass
            self.job.close()  # kill-on-close: nothing of the ballast survives
            self.job = None
        self.proc = None

    def __enter__(self) -> Ballast:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# ------------------------------------------------------------------------------------ requests


def _envelope(backend: str, labels: list[str], cfg: Any, out_dir: Path, *, force_rung: str | None,
              cap_mb: float | None) -> dict[str, Any]:
    v = vram.build_vram_request(backend, labels, cfg, force_rung=force_rung, cap_mb=cap_mb, precheck=False)
    v["cpu_allowed"] = False  # the bench measures GPU rungs only
    return {"format": "bandscribe.worker/1", "job": None, "out_dir": str(Path(out_dir).resolve()), "device": "auto",
            "vram": v, "determinism": True}


def sep_request(clip: Path, out_dir: Path, cfg: Any, *, force_rung: str | None, cap_mb: float | None
                ) -> tuple[dict, list[str]]:
    from bandscribe.sep import run as seprun

    params = seprun.sep_params(cfg)
    labels = seprun.rung_labels(params["chunk_ladder"])
    req = seprun.build_request(clip, out_dir, params, cfg, gain_db=0.0, force_rung=force_rung, cap_mb=cap_mb,
                               precheck=False)
    req["vram"]["cpu_allowed"] = False
    return req, labels


def _hf_cached(size: str) -> bool:
    """model.safetensors of ``MuScriptor/muscriptor-<size>`` is in the local HF cache (any snapshot)."""
    snaps = paths.HF_HOME / "hub" / f"models--MuScriptor--muscriptor-{size}" / "snapshots"
    return any(p.is_file() for p in snaps.glob("*/model.safetensors")) if snaps.is_dir() else False


def amt_labels(cfg: Any) -> list[str]:
    loader = str(vram.cfg_get(cfg, "amt.muscriptor_loader", "lean"))
    size = str(vram.cfg_get(cfg, "amt.muscriptor_model", "medium"))
    top = f"{size}-lean" if loader == "lean" else size
    fallback = [str(s) for s in (vram.cfg_get(cfg, "amt.muscriptor_fallback", ["small"]) or [])]
    return [top] + [s for s in fallback if s != top]


def amt_request(clip: Path, out_dir: Path, cfg: Any, *, force_rung: str | None, cap_mb: float | None
                ) -> tuple[dict, list[str]]:
    """M1_M2_SPEC 7.3 request for one mono view of the clip, all classes (worst case for decoding length)."""
    import soundfile as sf

    from bandscribe.audio import pcm_sha256, write_f32

    labels = amt_labels(cfg)
    size = str(vram.cfg_get(cfg, "amt.muscriptor_model", "medium"))
    rev = str(vram.cfg_get(cfg, "amt.muscriptor_revision", "f32236969308476e01fd3aae67357de5feb05a2d"))
    weights = paths.HF_HOME / "hub" / f"models--MuScriptor--muscriptor-{size}" / "snapshots" / rev / "model.safetensors"
    if not weights.is_file():
        raise BenchError(f"MuScriptor-{size} 가중치가 없습니다: {weights}")
    if force_rung is not None and force_rung != labels[0] and not _hf_cached(force_rung):
        # A fallback size whose weights are not in the HF cache (e.g. the gated small model was never accepted):
        # the worker would only answer "unavailable" after taking the GPU lock; it never downloads.
        raise BenchError(f"MuScriptor-{force_rung} 가중치가 HF 캐시에 없어 이 칸은 잴 수 없습니다 "
                         f"({paths.HF_HOME / 'hub'}; 받지 않습니다)")
    from bandscribe import models

    entry = models.get(f"muscriptor-{size}")
    sha = entry.sha256 if entry is not None else None
    data, sr = sf.read(str(clip), dtype="float32", always_2d=True)
    mono = Path(out_dir) / "view_mono.wav"
    write_f32(mono, ((data[:, 0] + data[:, 1]) * np.float32(0.5)).astype(np.float32), sr)
    del data
    req = _envelope("amt_muscriptor", labels, cfg, out_dir, force_rung=force_rung, cap_mb=cap_mb)
    req.update({
        "model": {"size": size, "weights_path": str(weights), "sha256": sha, "revision": rev,
                  "loader": str(vram.cfg_get(cfg, "amt.muscriptor_loader", "lean")),
                  "fallback": [{"size": s, "repo": f"MuScriptor/muscriptor-{s}", "weights_path": None}
                               for s in (vram.cfg_get(cfg, "amt.muscriptor_fallback", ["small"]) or [])]},
        "decode": {"prelude_forcing": bool(vram.cfg_get(cfg, "amt.prelude_forcing", True)),
                   "beam_size": int(vram.cfg_get(cfg, "amt.beam_size", 1)),
                   "cfg_coef": float(vram.cfg_get(cfg, "amt.cfg_coef", 1.0)), "use_sampling": False,
                   "batch_size": int(vram.cfg_get(cfg, "amt.batch_size", 1))},
        "dtype": str(vram.cfg_get(cfg, "amt.dtype", "upstream")),
        "views": [{"id": "bench_mono", "wav": str(mono.resolve()), "instruments": None,
                   "out": str((Path(out_dir) / "raw" / "bench_mono__muscriptor.json").resolve()),
                   "view_sha256": pcm_sha256(mono)}],
    })
    return req, labels


def beats_request(clip: Path, out_dir: Path, cfg: Any, *, force_rung: str | None, cap_mb: float | None
                  ) -> tuple[dict, list[str]]:
    """M1_M2_SPEC 7.5 request; the checkpoint must have been fetched (``bandscribe models fetch beat_this.final0``)."""
    from bandscribe import models

    name = "beat_this." + str(vram.cfg_get(cfg, "grid.checkpoint", "final0"))
    try:
        ckpt = models.local_path(name)
    except (models.ModelMissing, models.ModelRegistryError) as e:  # Korean hint naming `bandscribe models fetch`
        raise BenchError(str(e)) from e
    labels = [name.split(".", 1)[1]]
    req = _envelope("beats_beatthis", labels, cfg, out_dir, force_rung=force_rung, cap_mb=cap_mb)
    req.update({"wav": str(Path(clip).resolve()), "checkpoint_path": str(ckpt),
                "checkpoint_sha256": models.sha256_cached(ckpt), "dbn": False, "float16": False,
                "out_beats": "beats.json", "out_activations": "activations.npz"})
    return req, labels


REQUEST_BUILDERS: dict[str, Callable[..., tuple[dict, list[str]]]] = {
    "sep": sep_request, "amt": amt_request, "beats": beats_request}


def _ladder_labels(key: str, cfg: Any) -> list[str]:
    if key == "sep":
        from bandscribe.sep import run as seprun

        return seprun.rung_labels(list(seprun.sep_value(cfg, "sep.chunk_ladder")))
    if key == "amt":
        return amt_labels(cfg)
    return [str(vram.cfg_get(cfg, "grid.checkpoint", "final0"))]


# -------------------------------------------------------------------------------------------- rows


def _f(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def rows_from_result(backend: str, run_id: str, repeat: int, mode: str, res: Any, *, cap_mb: float | None,
                     ballast_mb: float | None, fallback: str, clip_s: float) -> list[dict[str, Any]]:
    """One row per ladder attempt of a worker run (a failed worker gives one row with its error)."""
    out = res.output if isinstance(getattr(res, "output", None), dict) else {}
    vr = out.get("vram") or {}
    timing = out.get("timing") or {}
    start_nvml = _f(vr.get("start_nvml_used_mb"))
    base = {"backend": backend, "run": run_id, "repeat": repeat, "mode": mode, "cap_mb": cap_mb,
            "ballast_mb": ballast_mb, "fallback": fallback, "clip_s": round(clip_s, 3),
            "load_s": _f(timing.get("load_s")), "ctx_mb": _f(vr.get("ctx_mb")),
            "start_free_mb": _f(vr.get("start_free_mb")), "worker_ok": bool(getattr(res, "ok", False))}
    attempts = vr.get("attempts") or []
    if not attempts:
        err = (getattr(res, "result", None) or {}).get("error") or out.get("status") or "no attempts"
        return [dict(base, rung_index=None, rung_label=None, device=None, status="error", error=str(err)[:300])]
    rows = []
    for a in attempts:
        peak = _f(a.get("nvml_used_peak_mb"))
        rows.append(dict(base, rung_index=a.get("rung_index"), rung_label=a.get("rung_label"), device=a.get("device"),
                         status=a.get("status"), max_allocated_mb=_f(a.get("max_allocated_mb")),
                         max_reserved_mb=_f(a.get("max_reserved_mb")), nvml_used_peak_mb=peak,
                         nvml_delta_peak_mb=None if peak is None or start_nvml is None else round(peak - start_nvml, 1),
                         pdh_self_shared_growth_mb=_f(a.get("pdh_self_shared_growth_mb")),
                         pdh_adapter_shared_delta_mb=_f(a.get("pdh_adapter_shared_delta_mb")),
                         rtf=_f(a.get("rtf")), slow=bool(a.get("slow")), process_s=_f(a.get("process_s")),
                         error=a.get("error")))
    return rows


def csv_text(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS})
    return buf.getvalue()


def mergeable(row: dict[str, Any]) -> bool:
    """Only uncapped, ballast-free, successful attempts describe normal operation."""
    return (row.get("status") == "ok" and not row.get("cap_mb") and not row.get("ballast_mb")
            and row.get("device") not in (None, "cpu") and _f(row.get("max_reserved_mb")) is not None)


def merge_table(table: dict[str, Any] | None, rows: list[dict[str, Any]], *, gpu: str | None, driver: str | None,
                measured_utc: str) -> dict[str, Any]:
    """New vram_table document: per (backend, rung) the median of this bench's mergeable rows replaces the entry;
    the calibration takes the max shared growth over every ``fallback == "off"`` attempt ever seen."""
    t = dict(table or {})
    t["format"] = vram.VRAM_TABLE_FORMAT
    t["gpu"] = gpu or t.get("gpu")
    t["driver"] = driver or t.get("driver")
    entries: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (t.get("entries") or {}).items()}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if mergeable(r):
            groups.setdefault((r["backend"], r["rung_label"]), []).append(r)

    def med(rs: list[dict[str, Any]], key: str) -> float | None:
        vals = [v for v in (_f(r.get(key)) for r in rs) if v is not None]
        return round(statistics.median(vals), 3) if vals else None

    # The CUDA context does not depend on the rung, and one NVML before/after reading is noisy (other apps
    # allocate meanwhile; one measured 6 MB next to 89 / 97 MB): use the median over all of a backend's rows.
    ctx_by_backend: dict[str, list[dict[str, Any]]] = {}
    for (backend, _label), rs in groups.items():
        ctx_by_backend.setdefault(backend, []).extend(rs)

    for (backend, label), rs in sorted(groups.items()):
        fallbacks = sorted({str(r.get("fallback")) for r in rs})
        entries.setdefault(backend, {})[label] = {
            "reserved_peak_mb": med(rs, "max_reserved_mb"), "ctx_mb": med(ctx_by_backend[backend], "ctx_mb"),
            "nvml_delta_peak_mb": med(rs, "nvml_delta_peak_mb"), "rtf": med(rs, "rtf"),
            "fallback": fallbacks[0] if len(fallbacks) == 1 else "mixed", "measured_utc": measured_utc,
            "clip_s": med(rs, "clip_s"), "n": len(rs)}
    t["entries"] = entries
    cal = dict(t.get("calibration") or {})
    growths = [g for g in (_f(r.get("pdh_self_shared_growth_mb")) for r in rows if r.get("fallback") == "off")
               if g is not None]
    if growths:
        previous = _f(cal.get("shared_growth_noise_mb"))
        noise = max(growths) if previous is None else max(max(growths), previous)
        cal["shared_growth_noise_mb"] = round(noise, 1)
        cal["runs"] = int(cal.get("runs") or 0) + len(growths)
        cal["recommended_shared_growth_fail_mb"] = recommended_fail_mb(noise)
        cal["updated_utc"] = measured_utc
    t["calibration"] = cal
    return t


def recommended_fail_mb(noise_mb: float | None) -> float:
    return float(max(MIN_SHARED_GROWTH_FAIL_MB, 2.0 * (noise_mb or 0.0)))


# ---------------------------------------------------------------------------------------------- run


@contextlib.contextmanager
def gpu_lock(timeout_s: float) -> Any:
    """Hold data/gpu.lock (the one-model-on-the-GPU lock of bandscribe.gpu) for a ballasted measurement."""
    from filelock import FileLock, Timeout

    lock = FileLock(str(paths.GPU_LOCK))
    try:
        lock.acquire(timeout=timeout_s)
    except Timeout as e:
        raise BenchError(f"GPU 락을 {timeout_s:.0f}초 안에 얻지 못했습니다(다른 GPU 작업 중)") from e
    try:
        yield lock
    finally:
        lock.release()


def _gpu_identity() -> tuple[str | None, str | None, list[dict]]:
    from bandscribe import sysmon

    info = sysmon.nvml_info()
    if not info:
        return None, None, []
    dev = (info.get("devices") or [{}])[0]
    return dev.get("name"), info.get("driver"), list(info.get("processes") or [])


def auto_ballast_mb(key: str, cfg: Any, snap: vram.GpuSnapshot | None) -> int:
    """NVML free - (orchestrator need of rung 0 - 400 MB), so rung 0 does not fit but lower rungs may."""
    if snap is None:
        raise BenchError("NVML 을 읽지 못해 --ballast-mb auto 를 계산할 수 없습니다. 숫자로 주세요.")
    backend = BACKENDS[key]
    labels = _ladder_labels(key, cfg)
    needs = vram.load_table(ctx_default_mb=float(vram.cfg_get(cfg, "gpu.ctx_mb_default"))).get(backend, {})
    if labels[0] not in needs:
        raise BenchError(f"{backend} {labels[0]} 의 필요량을 몰라 auto 를 계산할 수 없습니다")
    return int(snap.free_mb - (vram.orchestrator_need_mb(needs[labels[0]]) - BALLAST_AUTO_SLACK_MB))


def run_bench(opts: BenchOptions, cfg: Any, *, out_root: Path | None = None,
              run_worker: Callable[..., Any] | None = None, ballast_factory: Callable[..., Any] | None = None,
              snap_fn: Callable[[], vram.GpuSnapshot | None] | None = None,
              progress: Callable[[str], None] | None = None, update_table: bool = True,
              lock_factory: Callable[[float], Any] | None = None) -> dict[str, Any]:
    """Run the bench; returns the bench.json document (also written, with bench.csv, into the run dir)."""
    opts.validate()
    lock_factory = lock_factory or gpu_lock
    if run_worker is None:
        from bandscribe import gpu

        run_worker = gpu.run_worker
    snap_fn = snap_fn or vram.snapshot
    ballast_factory = ballast_factory or Ballast
    say = progress or (lambda m: log.info("%s", m))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root or bench_root()) / f"{stamp}_{git_sha7()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_name, driver, procs = _gpu_identity()
    clip, clip_s, clip_desc = prepare_clip(opts.clip, opts.seconds, run_dir)
    doc: dict[str, Any] = {"format": BENCH_FORMAT, "started_utc": _utc(), "gpu": gpu_name, "driver": driver,
                           "processes_at_start": procs, "options": asdict(opts), "clip": {"path": clip.name,
                           "seconds": round(clip_s, 3), "source": clip_desc}, "runs": [], "rows": []}
    atomic.write_json(run_dir / "bench.json", doc)
    timeout = float(vram.cfg_get(cfg, "gpu.worker_timeout_s"))
    lock_timeout = float(vram.cfg_get(cfg, "gpu.lock_timeout_s"))

    for key in opts.backends:
        backend = BACKENDS[key]
        labels = _ladder_labels(key, cfg)
        forced: list[str | None] = list(labels) if opts.rungs == "all" else [None]
        for force in forced:
            for rep in range(opts.repeat):
                run_id = f"{key}-{(force or 'auto').replace('=', '')}-r{rep + 1}"
                work = run_dir / run_id
                try:
                    req, _labels = REQUEST_BUILDERS[key](clip, work, cfg, force_rung=force, cap_mb=opts.cap_mb)
                except BenchError as e:
                    say(f"{key}: 건너뜀 - {e}")
                    doc["runs"].append({"run": run_id, "backend": backend, "skipped": str(e)})
                    break
                ballast = None
                ballast_mb: float | None = None
                t0 = time.monotonic()
                # With a ballast, the bench itself holds the GPU lock from before the ballast allocates until it
                # is released: the ballast must not starve another bandscribe worker that holds the lock meanwhile,
                # and the "auto" size must be computed while no other bandscribe worker moves NVML free. The measured
                # worker then runs inside that lock (use_lock=False).
                held = lock_factory(lock_timeout) if opts.ballast_mb is not None else contextlib.nullcontext()
                with held:
                    if opts.ballast_mb is not None:
                        mb = (auto_ballast_mb(key, cfg, snap_fn()) if opts.ballast_mb == "auto"
                              else int(opts.ballast_mb))
                        if mb > 0:
                            ballast = ballast_factory(mb, work / "_ballast")
                            say(f"{run_id}: VRAM 밸러스트 {mb} MB 잡는 중...")
                            ballast.start()
                            ballast_mb = float(getattr(ballast, "held_mb", mb) or mb)
                        else:
                            say(f"{run_id}: 여유가 이미 부족해 밸러스트를 쓰지 않습니다 (계산값 {mb} MB)")
                    say(f"{run_id}: {backend} {force or '사다리(auto)'} 실행 중...")
                    kw: dict[str, Any] = {"timeout_s": timeout, "lock_timeout_s": lock_timeout}
                    if opts.ballast_mb is not None:
                        kw["use_lock"] = False  # we hold it
                    try:
                        res = run_worker(backend, req, work / "_worker", **kw)
                    finally:
                        if ballast is not None:
                            ballast.stop()
                rows = rows_from_result(backend, run_id, rep + 1, "force" if force else "auto", res,
                                        cap_mb=opts.cap_mb, ballast_mb=ballast_mb, fallback=opts.fallback,
                                        clip_s=clip_s)
                doc["rows"].extend(rows)
                doc["runs"].append({"run": run_id, "backend": backend, "force_rung": force, "repeat": rep + 1,
                                    "ballast_mb": ballast_mb, "wall_s": round(time.monotonic() - t0, 3),
                                    "ok": bool(getattr(res, "ok", False)), "result": getattr(res, "result", {}),
                                    "stderr": str(getattr(res, "stderr_path", ""))})
                atomic.write_json(run_dir / "bench.json", doc)
                atomic.write_text(run_dir / "bench.csv", csv_text(doc["rows"]))
                if getattr(res, "lock_released_dirty", False):
                    raise BenchError(f"GPU 워커가 깨끗하게 끝나지 않았습니다(락 강제 해제). 벤치를 멈춥니다: "
                                     f"{getattr(res, 'stderr_path', '')}")

    doc["finished_utc"] = _utc()
    if update_table:
        table_path = vram.vram_table_path()
        table = merge_table(vram.read_table(table_path), doc["rows"], gpu=gpu_name, driver=driver,
                            measured_utc=doc["finished_utc"])
        atomic.write_json(table_path, table)
        doc["vram_table"] = str(table_path)
        doc["calibration"] = table.get("calibration")
    atomic.write_json(run_dir / "bench.json", doc)
    atomic.write_text(run_dir / "bench.csv", csv_text(doc["rows"]))
    doc["run_dir"] = str(run_dir)
    return doc
