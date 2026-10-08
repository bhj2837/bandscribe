"""Bass pipeline stages (M5; ``bandscribe.pipeline.graph`` owns names and deps):

- ``bass_f0`` (GPU, gpu env worker ``f0_bass`` + pyin on the CPU): ``f0.npz`` - CREPE, PESTO and pyin F0 tracks of
  the bass view (``stems/views/bass_mono.wav``) on one 10 ms grid (``bandscribe.bass.f0``).
- ``bass`` (CPU): ``bass.json`` - the pass-1 bass after the silence gate (``notes/bass_raw.json``) cleaned up with
  the F0 tracks (``bandscribe.bass.clean``: octave/pitch anchor, fragment and slide-chain merges, kick rejection
  with the drums stem, the F0 gap fill), with technique suggestions per note; ``bass_report.json`` counts what
  each step did. The ``quant`` stage reads the bass from here.

Stage functions read only their dep dirs, the job input and ``ctx.params`` (M1_M2_SPEC 3.1).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic
from bandscribe.bass import clean as CL
from bandscribe.bass import f0 as F0

log = logging.getLogger(__name__)

F0_BACKEND = "f0_bass"
F0_FILE = "f0.npz"
BASS_FILE = "bass.json"
REPORT_FILE = "bass_report.json"
BASS_VIEW = "bass_mono"


class BassStageError(RuntimeError):
    """User-facing (Korean) problem of a bass stage."""


def _notifier(ctx: Any, event: str = "vram_wait") -> Callable[[str], None]:
    from bandscribe.amt.stage import _notifier as amt_notifier

    return amt_notifier(ctx, event)


# ------------------------------------------------------------------------------------------- trackers


def tracker_versions() -> dict[str, str | None]:
    """Installed versions that change the F0 tracks: torchcrepe / pesto-pitch in the gpu env (metadata, no
    import), librosa (pyin) in this env."""
    from importlib.metadata import PackageNotFoundError, version

    from bandscribe import paths
    from bandscribe.amt.stage import dist_version

    try:
        librosa_v: str | None = version("librosa")
    except PackageNotFoundError:
        librosa_v = None
    return {"torchcrepe": dist_version(paths.GPU_PYTHON, "torchcrepe"),
            "pesto-pitch": dist_version(paths.GPU_PYTHON, "pesto-pitch"), "librosa": librosa_v}


def f0_params(cfg: Any = None) -> dict[str, Any]:
    p = {k: json.loads(json.dumps(v)) for k, v in F0.DEFAULT_PARAMS.items() if k in ("hop_s", "crepe", "pesto", "pyin")}
    return {**p, "versions": tracker_versions()}


def _pcm_sha(path: Path) -> str:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    h = hashlib.sha256()
    h.update(str(sr).encode())
    h.update(np.ascontiguousarray(data).tobytes())
    return h.hexdigest()


def cache_key(view: Mapping[str, Any], params: Mapping[str, Any]) -> str:
    sha = view.get("pcm_sha256") or _pcm_sha(Path(view["wav"]))
    blob = json.dumps({"pcm": sha, "params": params, "code": F0.CODE_VERSION}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def eval_cache_root() -> Path:
    from bandscribe import paths

    return paths.EVAL / "f0_cache"


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def run_trackers(views: Sequence[Mapping[str, Any]], params: Mapping[str, Any], cfg: Any, *, work_dir: Path,
                 cache_root: Path | None = None, force_rung: str | None = None, job: dict | None = None,
                 notify: Callable[[str], None] | None = None) -> tuple[dict[str, F0.F0Track], dict[str, Any]]:
    """{view id: F0Track (crepe, pesto, pyin)} for ``views`` = [{"id", "wav", "pcm_sha256"?}].

    The GPU trackers of every view not in ``cache_root`` (``None`` = no cache: the ``bass_f0`` stage, whose output
    is cached by the runner) run in one ``f0_bass`` worker call; pyin runs here meanwhile, in a thread."""
    from bandscribe import vram

    work_dir = Path(work_dir).resolve()  # the worker runs in its own directory
    out: dict[str, F0.F0Track] = {}
    todo: list[tuple[Mapping[str, Any], str | None]] = []
    for v in views:
        key = cache_key(v, params) if cache_root is not None else None
        if key is not None and (Path(cache_root) / f"{key}.npz").is_file():
            out[str(v["id"])] = F0.load_npz(Path(cache_root) / f"{key}.npz")
        else:
            todo.append((v, key))
    info: dict[str, Any] = {"cached": len(out), "computed": len(todo), "rung_label": None, "worker": None}
    if not todo:
        return out, info
    model = str(params["crepe"]["model"])
    labels = [f"crepe-{model}"]
    worker_dir = work_dir / "_worker"  # left out of a stage's data outputs (``runner.data_outputs``)
    gpu_dir = worker_dir / "gpu"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    req = {"format": "bandscribe.worker/1", "job": job, "out_dir": str(gpu_dir), "device": "auto",
           "determinism": True, "mode": "track",
           "views": [{"id": str(v["id"]), "wav": str(Path(v["wav"]).resolve()), "out": str(gpu_dir / f"{i:03d}.npz")}
                     for i, (v, _k) in enumerate(todo)],
           "params": {k: params[k] for k in ("hop_s", "crepe", "pesto")}}
    if force_rung is not None:
        req["vram"] = vram.build_vram_request(F0_BACKEND, labels, cfg, force_rung=force_rung)

    def pyin_all() -> list[F0.F0Track]:
        out_py = []
        for v, _k in todo:
            mono, sr = _read_mono(Path(v["wav"]))
            out_py.append(F0.pyin_track(mono, sr, params, duration_s=len(mono) / float(sr)))
        return out_py

    # pyin (CPU, this process) runs while the worker has the GPU: it is the slower half on a 3-4 min song
    with ThreadPoolExecutor(max_workers=1) as pool:
        py_future = pool.submit(pyin_all)
        res = vram.run_gpu_stage(F0_BACKEND, labels, req, cfg, worker_dir, notify=notify)
        py_tracks = py_future.result()
    output = res.output if isinstance(getattr(res, "output", None), dict) else {}
    if output.get("status") != "ok":
        raise BassStageError(f"F0 워커가 결과를 내지 못했습니다(status={output.get('status')}). "
                             f"자세한 내용: {worker_dir / 'stderr.log'}")
    info.update(worker=output, waited_s=getattr(res, "waited_s", 0.0),
                rung_label=(output.get("vram") or {}).get("rung_label"))
    for i, ((v, key), py) in enumerate(zip(todo, py_tracks)):
        gpu = F0.load_npz(gpu_dir / f"{i:03d}.npz")
        tr = F0.merge(gpu, py, n=gpu.n)
        if key is not None:
            Path(cache_root).mkdir(parents=True, exist_ok=True)
            F0.save_npz(Path(cache_root) / f"{key}.npz", tr)
        out[str(v["id"])] = tr
    return out, info


# -------------------------------------------------------------------------------------------- bass_f0


def bass_f0_params(cfg: Any) -> dict[str, Any]:
    return f0_params(cfg)


def run_bass_f0(ctx: Any) -> None:
    from bandscribe import vram

    out_dir = Path(ctx.out_dir)
    wav = Path(ctx.dep_dirs["stems"]) / "views" / f"{BASS_VIEW}.wav"
    if not wav.is_file():
        raise BassStageError(f"stems 단계에 베이스 뷰가 없습니다: {wav}")
    job = {"song_key": ctx.song_key, "stage": ctx.stage, "stage_key": ctx.key}
    tracks, info = run_trackers([{"id": BASS_VIEW, "wav": str(wav)}], dict(ctx.params), ctx.config,
                                work_dir=out_dir, job=job, notify=_notifier(ctx))
    F0.save_npz(out_dir / F0_FILE, tracks[BASS_VIEW])
    worker = info.get("worker") or {}
    labels = [f"crepe-{ctx.params['crepe']['model']}"]
    vram.write_gpu_run(out_dir, F0_BACKEND, labels, worker, info.get("waited_s", 0.0))
    atomic.write_json(out_dir / "f0_info.json", {
        "format": F0.FORMAT, "view": BASS_VIEW, "frames": tracks[BASS_VIEW].n, "hop_s": tracks[BASS_VIEW].hop_s,
        "backend": worker.get("backend"), "timing": worker.get("timing"), "versions": ctx.params.get("versions")})
    warn = _notifier(ctx, "warning")
    for w in worker.get("warnings") or []:
        warn(str(w))


# ----------------------------------------------------------------------------------------------- bass

BASS_FORMAT = "bandscribe.bass/1"
VOTE_KEYS = ("conf_min", "evidence_s", "min_frames", "agree_st", "min_votes")


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg.get(key, default) if cfg is not None and hasattr(cfg, "get") else default
    except Exception:  # noqa: BLE001 - a plain dict / partial config in tests
        return default


def bass_params(cfg: Any) -> dict[str, Any]:
    """The clean-up switches from ``[bass]`` (E13 decides the defaults) + the fixed thresholds of
    ``bandscribe.bass.clean`` and the vote settings of ``bandscribe.bass.f0``."""
    clean = dict(CL.DEFAULT_PARAMS)
    for k in CL.STEPS + ("techniques",):
        clean[k] = _cfg_get(cfg, f"bass.{k}", CL.DEFAULT_PARAMS[k])
    clean["policy"] = str(clean["policy"])
    for k in ("anchor", "kick", "fragments", "slides", "gap_fill", "techniques"):
        clean[k] = bool(clean[k])
    return {"clean": clean, "f0": {k: json.loads(json.dumps(F0.DEFAULT_PARAMS[k])) for k in VOTE_KEYS}}


def run_bass(ctx: Any) -> None:
    from bandscribe.amt import midi

    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    p = dict(ctx.params)
    raw_path = dd["notes"] / "bass_raw.json"
    raw = atomic.read_json(raw_path) if raw_path.is_file() else {"notes": []}
    track = F0.load_npz(dd["bass_f0"] / F0_FILE)
    wav = dd["stems"] / "views" / f"{BASS_VIEW}.wav"
    mono, sr = _read_mono(wav)
    attack = CL.Attack(mono, sr)
    kicks = None
    drums = dd["sep"] / "stems" / "drums.wav"
    if p["clean"]["kick"] and drums.is_file():
        dm, dsr = _read_mono(drums)
        kicks = CL.kick_onsets(dm, dsr)
    notes, rep = CL.clean(raw.get("notes") or [], track, p["clean"], f0_params=p["f0"], attack=attack, kicks=kicks)
    out = Path(ctx.out_dir)
    doc = {"format": BASS_FORMAT, "view": BASS_VIEW, "source": "notes/bass_raw.json",
           "audio_duration_s": raw.get("audio_duration_s"), "params": p, "notes": notes}
    atomic.write_json(out / BASS_FILE, doc)
    atomic.write_json(out / REPORT_FILE, rep)
    grid = atomic.read_json(dd["grid"] / "grid.json") if "grid" in dd else None
    midi.write_performance_midi(out / "midi" / "bass.mid", [("Bass", 33, notes)], grid)
    moved = rep.get("moved") or 0
    sl = rep.get("slides") or {}
    log.info("bass: %d -> %d notes (moved %d, slides %s)", rep.get("in", 0), rep.get("out", 0), moved, sl)


STAGES: dict[str, dict] = {
    "bass_f0": {"run": run_bass_f0, "code_version": F0.CODE_VERSION, "params": bass_f0_params, "device": "gpu",
                "models": lambda cfg: {}},
    # 3: review fixes 2026-10-08 (glide crossing on the tuning-corrected contour, fragments need no attack,
    # duplicates keep their length)
    "bass": {"run": run_bass, "code_version": "3", "params": bass_params, "device": "cpu", "models": lambda cfg: {}},
}
