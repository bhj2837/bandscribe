"""AMT pipeline stages: ``amt_ms1``, ``amt_bp``, ``instr``, ``amt_gtr``, ``s35``, ``notes`` (M1_M2_SPEC 3.2 rows 8-13).

``STAGES`` follows the stage contract of M1_M2_SPEC 3.1; ``gtab.pipeline.graph`` (GRID) owns names and deps and
imports this module lazily. Stage functions depend only on ``ctx.params``, their dep dirs and the job input;
``ctx.config`` is read only for non-semantic knobs (timeouts, VRAM policy) through ``gtab.vram``.

- ``amt_ms1`` (GPU, ``amt_muscriptor``): pass-1 views ``mix_mono`` (quality only; all classes = baseline B0),
  ``bass_mono`` (bass classes), ``piano_other_mono`` (keys/synth [+ guitar], M1_M2_SPEC A5).
- ``amt_bp`` (CPU, bp310 ``amt_basicpitch``, no GPU lock; **optional**): ``guitar_mono`` and ``bass_mono``. Without
  the bp310 env it writes ``skipped.json`` and a Korean warning; its models() then return
  ``{"basic_pitch": "absent"}`` so the key changes once the env is installed (M1_M2_SPEC A21).
- ``instr`` (CPU): ``instrumentation.json`` (``gtab.amt.instrumentation``).
- ``amt_gtr`` (GPU): the ``amt.guitar_view`` transcription with the guitar-pass mask from ``instr``.
- ``s35`` (CPU): ``s35.json`` index of the S3.5 artifacts (paths relative to the job dir).
- ``notes`` (CPU): ``guitar_all.json`` + ``midi/*.mid`` + ``notes_summary.json``.

Transcriptions go through the shared cache (``gtab.amt.cache``): views without an entry are sent to the worker,
which writes raw NoteSets under ``_worker/raw/`` (provenance); the stage stores them in the cache under the rung
actually used and writes its ``raw/`` outputs from the cache with the latency constant applied
(``onset_s = raw - amt.latency_s``, clamped >= 0; ``time_offset_applied_s = -latency_s``). GPU stages write
``gpu_run.json`` (``gtab.vram.write_gpu_run``); when every view was cached, it records the cached top rung.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from gtab import atomic, models, paths, vram
from gtab.amt import cache as amt_cache
from gtab.amt import instruments as I
from gtab.config import Config, ConfigError

log = logging.getLogger(__name__)

CACHE_CODE_VERSION = "1"
MS_BACKEND = "amt_muscriptor"
BP_BACKEND = "amt_basicpitch"
BP_RUNG = "onnx-cpu"
HF_REPO = "MuScriptor/muscriptor-{size}"

BP_MODEL_NAME = "basic_pitch.icassp2022"

# Mirrors of the [run]/[hints]/[amt]/[instr] defaults (M1_M2_SPEC 1.3) for plain-dict configs (tests, experiments
# with overrides). A real ``gtab.config.Config`` is always read strictly (``cfg.get(key)``): a misspelt key raises
# instead of silently falling back here. ``tests/test_amt_stages.py`` checks these equal ``Config()``.
AMT_DEFAULTS: dict[str, Any] = {
    "run.profile": "quality",
    "hints.instruments": "auto",
    "amt.muscriptor_model": "medium",
    "amt.muscriptor_revision": "f32236969308476e01fd3aae67357de5feb05a2d",
    "amt.muscriptor_fallback": ["small"],
    "amt.muscriptor_loader": "lean",
    "amt.dtype": "upstream",
    "amt.prelude_forcing": True,
    "amt.beam_size": 1,
    "amt.cfg_coef": 1.0,
    "amt.batch_size": 1,
    "amt.latency_s": -0.003,  # latency experiment 2026-10-03 (docs/decisions.md)
    "amt.piano_other_instruments": "keys+guitar",
    "amt.guitar_view": "guitar_mono",
    "amt.guitar_mask": "guitar+present",
    "amt.bp_onset_threshold": 0.6,  # E23 2026-10-03
    "amt.bp_frame_threshold": 0.4,
    "amt.bp_min_note_frames": 7,
    "amt.bp_min_freq_hz": 0.0,
    "amt.bp_max_freq_hz": 0.0,
    "amt.bp_melodia_trick": True,
    "amt.bp_threads": 4,
    "instr.threshold": 0.5,
    "instr.guitar_threshold": 0.15,
    "instr.guitar_stem_active_ratio": 0.05,
    "instr.active_rel_db": -40.0,
    "instr.mix_class_min_notes": 8,
}
BP_WORKER_TIMEOUT_S = 3600.0


class AmtStageError(RuntimeError):
    """A transcription stage could not produce its outputs (Korean, user-facing)."""


def amt_value(cfg: Any, key: str) -> Any:
    if isinstance(cfg, Config):
        return cfg.get(key)
    return vram.cfg_get(cfg, key, AMT_DEFAULTS[key])


# ------------------------------------------------------------------------------------ environments


def bp310_python() -> Path:
    return Path(paths.BP310_PYTHON)  # read at call time (tests repoint it)


def _site_packages(python: Path) -> Path:
    return python.parent.parent / "Lib" / "site-packages"


def dist_version(python: Path, dist: str) -> str | None:
    """Installed version of ``dist`` in another venv, read from its metadata (no import, no subprocess)."""
    from importlib import metadata

    sp = _site_packages(Path(python))
    if not sp.is_dir():
        return None
    for d in metadata.distributions(path=[str(sp)]):
        name = (d.metadata.get("Name") or "").lower().replace("_", "-")
        if name == dist.lower().replace("_", "-"):
            return d.version
    return None


def bp_model_path(python: Path | None = None) -> Path:
    py = bp310_python() if python is None else Path(python)
    return _site_packages(py) / "basic_pitch" / "saved_models" / "icassp_2022" / "nmp.onnx"


def muscriptor_weights(size: str = "medium", revision: str | None = None) -> Path | None:
    """``model.safetensors`` of a MuScriptor variant in the local HF cache (``paths.HF_HOME``), or None.

    Pure path logic (no huggingface_hub in core): ``hub/models--MuScriptor--muscriptor-<size>/snapshots/<rev>/``,
    with ``<rev>`` = the pinned revision or ``refs/main``.
    """
    repo = paths.HF_HOME / "hub" / f"models--MuScriptor--muscriptor-{size}"
    rev = revision
    if not rev:
        ref = repo / "refs" / "main"
        try:
            rev = ref.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    p = repo / "snapshots" / rev / "model.safetensors"
    return p if p.is_file() else None


def _sha256(path: Path) -> str:
    return models.sha256_cached(Path(path))


def record_bp_model() -> models.ModelEntry | None:
    """Record the ONNX model bundled with the bp310 env's basic-pitch wheel as ``basic_pitch.icassp2022``.

    The model ships inside the wheel that ``tools/uvw.cmd sync`` installs in envs/bp310 (pinned by its uv.lock),
    so gtab never downloads it; the registry entry keeps its licence and hash traceable like every other model
    (M1_M2_SPEC 0, 9.2 item 2). Idempotent: rewrites the entry only when the file or its hash changed (a re-sync
    of the env). Returns the entry, or None when the env / file is absent.
    """
    onnx = bp_model_path()
    if not onnx.is_file():
        return None
    sha = _sha256(onnx)
    try:
        rel = onnx.resolve().relative_to(Path(paths.GTAB_ROOT).resolve()).as_posix()
    except ValueError:
        rel = str(onnx)
    cur = models.get(BP_MODEL_NAME)
    if cur is not None and cur.sha256 == sha and cur.path == rel:
        return cur
    version = dist_version(bp310_python(), "basic-pitch") or "?"
    entry = models.ModelEntry(
        name=BP_MODEL_NAME, path=rel,
        url=f"https://pypi.org/project/basic-pitch/{version}/ (bundled in the wheel: "
            "basic_pitch/saved_models/icassp_2022/nmp.onnx)",
        bytes=onnx.stat().st_size, sha256=sha,
        license="Apache-2.0 (spotify/basic-pitch, model included; Copyright 2022 Spotify AB)",
        source=f"PyPI wheel basic-pitch=={version}, installed in envs/bp310 by `tools/uvw.cmd sync` (uv.lock)",
        notes="Basic Pitch ICASSP 2022 model (note/onset/contour), ONNX, run with onnxruntime on the CPU. Training "
              "data: GuitarSet, iKala, MAESTRO, MedleyDB-Pitch, Slakh (EGDB not included; GuitarSet and MedleyDB "
              "are). Not downloaded by gtab.",
        recorded_utc=models.utc_now())
    models.record(entry)
    log.info("recorded %s (%s) in models.lock.json", BP_MODEL_NAME, sha[:12])
    return entry


# ----------------------------------------------------------------------------------- params / models


def ms_params(cfg: Any) -> dict[str, Any]:
    """MuScriptor keys shared by amt_ms1 / amt_gtr (every value that changes a transcription).

    Value checks (dtype, loader, ...) live in ``gtab.config``; the worker re-checks dtype/loader and refuses bf16.
    """
    dtype = str(amt_value(cfg, "amt.dtype"))
    loader = str(amt_value(cfg, "amt.muscriptor_loader"))
    return {
        "muscriptor_model": str(amt_value(cfg, "amt.muscriptor_model")),
        "muscriptor_revision": str(amt_value(cfg, "amt.muscriptor_revision")),
        "muscriptor_fallback": [str(s) for s in amt_value(cfg, "amt.muscriptor_fallback") or []],
        "muscriptor_loader": loader,
        "muscriptor_version": dist_version(paths.GPU_PYTHON, "muscriptor"),
        "dtype": dtype,
        "decode": {"prelude_forcing": bool(amt_value(cfg, "amt.prelude_forcing")),
                   "beam_size": int(amt_value(cfg, "amt.beam_size")),
                   "cfg_coef": float(amt_value(cfg, "amt.cfg_coef")),
                   "use_sampling": False,
                   "batch_size": int(amt_value(cfg, "amt.batch_size"))},
        "latency_s": float(amt_value(cfg, "amt.latency_s")),
    }


def pass1_views(profile: str) -> list[str]:
    return (["mix_mono"] if profile != "fast" else []) + ["bass_mono", "piano_other_mono"]


def amt_ms1_params(cfg: Any) -> dict[str, Any]:
    profile = str(amt_value(cfg, "run.profile"))
    mode = str(amt_value(cfg, "amt.piano_other_instruments"))
    I.mask_piano_other(mode)  # validates
    return {**ms_params(cfg), "profile": profile, "piano_other_instruments": mode, "views": pass1_views(profile)}


def amt_gtr_params(cfg: Any) -> dict[str, Any]:
    view = str(amt_value(cfg, "amt.guitar_view"))
    if view not in I.GUITAR_VIEWS:  # Config already restricts it; dict configs (experiments) land here
        raise ConfigError(f"amt.guitar_view 는 {' | '.join(I.GUITAR_VIEWS)} 중 하나여야 합니다 (입력값: {view!r})")
    policy = str(amt_value(cfg, "amt.guitar_mask"))
    I.mask_guitar_view({}, policy)  # validates the policy name (Korean ConfigError)
    return {**ms_params(cfg), "guitar_view": view, "guitar_mask": policy}


def ms_models(cfg: Any) -> dict[str, str]:
    size = str(amt_value(cfg, "amt.muscriptor_model"))
    rev = str(amt_value(cfg, "amt.muscriptor_revision"))
    w = muscriptor_weights(size, rev)
    if w is None:
        hint = (f"MuScriptor-{size} 가중치가 없습니다 ({paths.HF_HOME / 'hub'} 의 리비전 {rev[:12]}). "
                f"HF 게이트를 수락한 계정으로 `tools\\hf-login.cmd` 로그인 뒤 모델을 받아 두세요"
                f"(gtab run 중에는 내려받지 않습니다).")
        raise models.ModelMissing(f"muscriptor-{size}", hint)
    return {f"muscriptor-{size}": _sha256(w)}


def bp_params(cfg: Any) -> dict[str, Any]:
    """``amt_bp`` params; ``min_note_frames`` is the shortest note *kept* (the worker passes frames - 1)."""
    from gtab.workers.amt_basicpitch import normalize_params

    p = normalize_params({
        "onset_threshold": amt_value(cfg, "amt.bp_onset_threshold"),
        "frame_threshold": amt_value(cfg, "amt.bp_frame_threshold"),
        "min_note_frames": amt_value(cfg, "amt.bp_min_note_frames"),
        "minimum_frequency": amt_value(cfg, "amt.bp_min_freq_hz"),
        "maximum_frequency": amt_value(cfg, "amt.bp_max_freq_hz"),
        "melodia_trick": amt_value(cfg, "amt.bp_melodia_trick"),
        "multiple_pitch_bends": False,
    })
    threads = int(amt_value(cfg, "amt.bp_threads"))
    return {"params": p, "threads": threads, "views": ["guitar_mono", "bass_mono"],
            "basic_pitch_version": dist_version(bp310_python(), "basic-pitch")}


def bp_models(cfg: Any) -> dict[str, str]:
    if not bp310_python().is_file():
        return {"basic_pitch": "absent"}
    onnx = bp_model_path()
    if not onnx.is_file():
        return {"basic_pitch": "absent"}
    return {"basic_pitch.icassp2022": _sha256(onnx)}


def instr_params(cfg: Any) -> dict[str, Any]:
    hint = amt_value(cfg, "hints.instruments")
    I.parse_hint(hint)  # validate at plan time (Korean ConfigError)
    return {
        "threshold": float(amt_value(cfg, "instr.threshold")),
        "guitar_threshold": float(amt_value(cfg, "instr.guitar_threshold")),
        "guitar_stem_active_ratio": float(amt_value(cfg, "instr.guitar_stem_active_ratio")),
        "active_rel_db": float(amt_value(cfg, "instr.active_rel_db")),
        "mix_class_min_notes": int(amt_value(cfg, "instr.mix_class_min_notes")),
        "hints_instruments": hint if isinstance(hint, str) else list(hint),
        "profile": str(amt_value(cfg, "run.profile")),
        # the masks inside instrumentation.json depend on these too
        "guitar_view": str(amt_value(cfg, "amt.guitar_view")),
        "guitar_mask": str(amt_value(cfg, "amt.guitar_mask")),
        "piano_other_instruments": str(amt_value(cfg, "amt.piano_other_instruments")),
    }


# ------------------------------------------------------------------------------------- helpers


def _progress(ctx: Any) -> Callable[[str, dict], None] | None:
    for holder in (ctx, getattr(ctx, "config", None)):
        cb = getattr(holder, "progress", None)
        if callable(cb):
            return cb
    return None


def _notifier(ctx: Any, event: str = "vram_wait") -> Callable[[str], None]:
    cb = _progress(ctx)

    def notify(msg: str) -> None:
        log.warning("%s: %s", getattr(ctx, "stage", "?"), msg)
        if cb is not None:
            try:
                cb(event, {"stage": getattr(ctx, "stage", None), "message": msg})
            except Exception as e:  # a display problem must not fail the stage
                log.debug("progress callback failed: %s", e)

    return notify


def read_views(stems_dir: Path) -> dict[str, dict[str, Any]]:
    """{view id: {"path": abs Path, "pcm_sha256": ...}} from the ``stems`` stage's views.json."""
    doc = atomic.read_json(Path(stems_dir) / "views.json")
    out = {}
    for vid, v in (doc.get("views") or {}).items():
        out[vid] = {"path": Path(stems_dir) / v["path"], "pcm_sha256": v["pcm_sha256"]}
    return out


def validate_noteset(doc: dict[str, Any]) -> dict[str, Any]:
    """Core-side validation with the P0 schema (workers never import it); returns the (unchanged) dict."""
    from gtab.schema.notes import NoteSet

    NoteSet.model_validate(doc)
    return doc


def apply_latency(doc: dict[str, Any], latency_s: float) -> dict[str, Any]:
    """Copy of a raw NoteSet with ``onset/offset - latency_s`` (clamped >= 0), re-sorted, offset recorded."""
    lat = float(latency_s)
    out = dict(doc)
    notes = []
    for n in doc.get("notes") or []:
        m = dict(n)
        on = max(0.0, round(float(n["onset_s"]) - lat, 6))
        off = max(on, round(float(n["offset_s"]) - lat, 6))
        m["onset_s"], m["offset_s"] = on, off
        notes.append(m)
    notes.sort(key=lambda n: (n["onset_s"], n["pitch"], n["offset_s"], n.get("instrument") or ""))
    out["notes"] = notes
    out["time_offset_applied_s"] = round(float(doc.get("time_offset_applied_s") or 0.0) - lat, 6)
    return out


def _rel(job_dir: Path, p: Path) -> str:
    return Path(os.path.relpath(Path(p), Path(job_dir))).as_posix()


# ---------------------------------------------------------------------------- MuScriptor transcription


def ms_ladder(params: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """(GPU rung labels, request "model" block) for the configured MuScriptor ladder.

    ``small`` (fallback) joins the ladder only if its weights are already in the HF cache: the worker never
    downloads, so a rung that cannot load is not offered to the VRAM pre-check either.
    """
    from gtab.workers.amt_muscriptor import rung_labels

    size = params["muscriptor_model"]
    weights = muscriptor_weights(size, params["muscriptor_revision"])
    fallback = []
    for fs in params.get("muscriptor_fallback") or []:
        fw = muscriptor_weights(fs)
        if fw is not None:
            fallback.append({"size": fs, "repo": HF_REPO.format(size=fs), "weights_path": str(fw)})
    model = {"size": size, "weights_path": None if weights is None else str(weights), "sha256": None,
             "revision": params["muscriptor_revision"], "loader": params["muscriptor_loader"],
             "repo": HF_REPO.format(size=size), "fallback": fallback}
    return rung_labels(model, params["muscriptor_loader"]), model


def _ms_cache_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {"dtype": params["dtype"], "decode": dict(params["decode"]), "revision": params["muscriptor_revision"]}


def transcribe_views_ms(views: list[dict[str, Any]], params: Mapping[str, Any], cfg: Any, *, work_dir: Path,
                        cache: amt_cache.AmtCache, model_sha256: str | None = None, job: dict | None = None,
                        notify: Callable[[str], None] | None = None, force_rung: str | None = None,
                        run_gpu_stage: Callable[..., Any] | None = None) -> tuple[dict[str, dict], dict[str, Any]]:
    """Raw NoteSets for ``views`` = [{"id", "wav", "pcm_sha256", "instruments"}] via the cache and the worker.

    Returns ({view id: raw NoteSet dict}, info) where info has ``labels``, ``worker_output`` (None if every
    view was cached), ``waited_s``, ``rung_label`` (actual rung of the fresh views, or the looked-up rung).
    ``force_rung`` (experiments) pins the rung and is also the lookup label.
    """
    labels, model = ms_ladder(params)
    if model["weights_path"] is None:
        raise AmtStageError(f"MuScriptor-{params['muscriptor_model']} 가중치가 없습니다(HF 캐시). "
                            "먼저 모델을 받아 두세요.")
    if model_sha256 is None:
        model_sha256 = _ms_model_sha(params)
    model["sha256"] = model_sha256
    lookup_label = force_rung or labels[0]
    cparams = _ms_cache_params(params)
    version = params.get("muscriptor_version")
    results: dict[str, dict] = {}
    missing: list[dict[str, Any]] = []
    for v in views:
        mask = None if v.get("instruments") is None else I.sort_mt3(v["instruments"])
        key = amt_cache.amt_key("muscriptor", version, model_sha256, v["pcm_sha256"], mask, cparams,
                                CACHE_CODE_VERSION, lookup_label)
        hit = cache.get(key)
        if hit is not None:
            results[v["id"]] = hit
        else:
            missing.append({**v, "instruments": mask})
    info: dict[str, Any] = {"labels": labels, "worker_output": None, "waited_s": 0.0, "rung_label": lookup_label,
                            "cached_views": sorted(results), "fresh_views": [v["id"] for v in missing]}
    if not missing:
        return results, info

    raw_dir = Path(work_dir) / "raw"
    req = {
        "format": "gtab.worker/1", "job": job, "out_dir": str(Path(work_dir).parent), "device": "auto",
        "model": model, "decode": dict(params["decode"]), "dtype": params["dtype"],
        "views": [{"id": v["id"], "wav": str(v["wav"]), "instruments": v["instruments"],
                   "out": str(raw_dir / f"{v['id']}__muscriptor.json"), "view_sha256": v["pcm_sha256"]}
                  for v in missing],
        "determinism": True,
    }
    if force_rung is not None:
        req["vram"] = vram.build_vram_request(MS_BACKEND, labels, cfg, force_rung=force_rung)
    runner = vram.run_gpu_stage if run_gpu_stage is None else run_gpu_stage
    res = runner(MS_BACKEND, labels, req, cfg, Path(work_dir), notify=notify)
    out = res.output
    if out.get("status") != "ok":
        raise AmtStageError(f"MuScriptor 워커가 결과를 내지 못했습니다(status={out.get('status')}). "
                            f"자세한 내용: {getattr(res, 'stderr_path', work_dir)}")
    rung_label = (out.get("vram") or {}).get("rung_label") or lookup_label
    be = out.get("backend") or {}
    actual_sha = be.get("model_sha256") or model_sha256
    for v in missing:
        doc = atomic.read_json(raw_dir / f"{v['id']}__muscriptor.json")
        validate_noteset(doc)
        key = amt_cache.amt_key("muscriptor", version, actual_sha, v["pcm_sha256"], v["instruments"], cparams,
                                CACHE_CODE_VERSION, rung_label)
        cache.put(key, doc)
        results[v["id"]] = doc
    info.update(worker_output=out, waited_s=float(getattr(res, "waited_s", 0.0) or 0.0), rung_label=rung_label)
    return results, info


def _write_ms_outputs(ctx: Any, raws: Mapping[str, dict], params: Mapping[str, Any], info: Mapping[str, Any]) -> None:
    out_dir = Path(ctx.out_dir)
    for vid in sorted(raws):
        doc = apply_latency(raws[vid], params["latency_s"])
        validate_noteset(doc)
        atomic.write_json(out_dir / "raw" / f"{vid}__muscriptor.json", doc)
    worker_out = info.get("worker_output")
    if worker_out is None:  # every view came from the cache: record the rung it was looked up with
        worker_out = {"vram": {"rung_label": info["rung_label"]}, "backend": {"device_used": "cuda"}}
    vram.write_gpu_run(out_dir, MS_BACKEND, list(info["labels"]), worker_out, info.get("waited_s", 0.0))


def _job(ctx: Any) -> dict[str, Any]:
    return {"song_key": ctx.song_key, "stage": ctx.stage, "stage_key": ctx.key}


def _ms_model_sha(params: Mapping[str, Any]) -> str:
    w = muscriptor_weights(params["muscriptor_model"], params["muscriptor_revision"])
    if w is None:
        raise AmtStageError(f"MuScriptor-{params['muscriptor_model']} 가중치가 없습니다(HF 캐시, 리비전 "
                            f"{params['muscriptor_revision'][:12]}).")
    return _sha256(w)


def run_amt_ms1(ctx: Any) -> None:
    params = dict(ctx.params)
    views = read_views(Path(ctx.dep_dirs["stems"]))
    masks = I.pass1_masks(params["piano_other_instruments"])
    todo = [{"id": vid, "wav": views[vid]["path"], "pcm_sha256": views[vid]["pcm_sha256"], "instruments": masks[vid]}
            for vid in params["views"]]
    cache = amt_cache.AmtCache(amt_cache.job_cache_root(ctx.store, ctx.song_key))
    raws, info = transcribe_views_ms(todo, params, ctx.config, work_dir=Path(ctx.out_dir) / "_worker", cache=cache,
                                     model_sha256=_ms_model_sha(params), job=_job(ctx),
                                     notify=_notifier(ctx))
    _write_ms_outputs(ctx, raws, params, info)


def run_amt_gtr(ctx: Any) -> None:
    params = dict(ctx.params)
    views = read_views(Path(ctx.dep_dirs["stems"]))
    inst = atomic.read_json(Path(ctx.dep_dirs["instr"]) / "instrumentation.json")
    masks = inst.get("masks") or {}
    mask = masks["guitar_pass"] if "guitar_pass" in masks else I.mask_guitar_view(inst, params["guitar_mask"])
    if I.guitar_omissions(mask):  # never: masks always keep the guitar classes; guard the contract anyway
        raise AmtStageError("기타 전사 마스크에서 기타 클래스가 빠졌습니다(instrumentation.json 을 확인하세요).")
    vid = params["guitar_view"]
    if vid not in views:
        raise AmtStageError(f"stems 단계에 '{vid}' 뷰가 없습니다.")
    todo = [{"id": vid, "wav": views[vid]["path"], "pcm_sha256": views[vid]["pcm_sha256"], "instruments": mask}]
    cache = amt_cache.AmtCache(amt_cache.job_cache_root(ctx.store, ctx.song_key))
    raws, info = transcribe_views_ms(todo, params, ctx.config, work_dir=Path(ctx.out_dir) / "_worker", cache=cache,
                                     model_sha256=_ms_model_sha(params), job=_job(ctx),
                                     notify=_notifier(ctx))
    _write_ms_outputs(ctx, raws, params, info)


# ---------------------------------------------------------------------------------------- Basic Pitch


def _bp_cache_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {"params": dict(params["params"])}


def transcribe_views_bp(views: list[dict[str, Any]], params: Mapping[str, Any], cfg: Any, *, work_dir: Path,
                        cache: amt_cache.AmtCache, model_output_cache: Path | None, job: dict | None,
                        run_worker: Callable[..., Any] | None = None) -> tuple[dict[str, dict], dict[str, Any]]:
    """Raw Basic Pitch NoteSets for ``views`` = [{"id", "wav", "pcm_sha256"}] via the cache and the bp310 worker."""
    onnx = bp_model_path()
    model_sha = _sha256(onnx) if onnx.is_file() else None
    version = params.get("basic_pitch_version")
    cparams = _bp_cache_params(params)
    results: dict[str, dict] = {}
    missing = []
    for v in views:
        key = amt_cache.amt_key("basicpitch", version, model_sha, v["pcm_sha256"], None, cparams,
                                CACHE_CODE_VERSION, BP_RUNG)
        hit = cache.get(key)
        if hit is not None:
            results[v["id"]] = hit
        else:
            missing.append((v, key))
    info = {"cached_views": sorted(results), "fresh_views": [v["id"] for v, _k in missing], "worker_output": None}
    if not missing:
        return results, info
    raw_dir = Path(work_dir) / "raw"
    req = {"format": "gtab.worker/1", "job": job, "out_dir": str(Path(work_dir).parent), "device": "cpu",
           "mode": "transcribe", "threads": int(params["threads"]),
           "model_output_cache": None if model_output_cache is None else str(model_output_cache),
           "views": [{"id": v["id"], "wav": str(v["wav"]), "out": str(raw_dir / f"{v['id']}__basicpitch.json"),
                      "view_sha256": v["pcm_sha256"]} for v, _k in missing],
           "params": dict(params["params"]), "allow_default_min_len": False, "determinism": True}
    if run_worker is None:
        from gtab.gpu import run_worker as _rw

        run_worker = _rw
    timeout = float(vram.cfg_get(cfg, "gpu.worker_timeout_s", BP_WORKER_TIMEOUT_S))
    res = run_worker("amt_basicpitch", req, Path(work_dir), timeout_s=timeout, python=bp310_python(), use_lock=False)
    if not res.ok:
        raise AmtStageError(f"Basic Pitch 워커 실패: {res.error or '알 수 없는 오류'}. "
                            f"자세한 내용: {getattr(res, 'stderr_path', Path(work_dir) / 'stderr.log')}")
    for v, _lookup in missing:
        doc = atomic.read_json(raw_dir / f"{v['id']}__basicpitch.json")
        validate_noteset(doc)
        key = amt_cache.amt_key("basicpitch", version, (doc.get("backend") or {}).get("model_sha256") or model_sha,
                                v["pcm_sha256"], None, cparams, CACHE_CODE_VERSION, BP_RUNG)
        cache.put(key, doc)
        results[v["id"]] = doc
    info["worker_output"] = res.output
    return results, info


def run_amt_bp(ctx: Any) -> None:
    params = dict(ctx.params)
    out_dir = Path(ctx.out_dir)
    if not bp310_python().is_file() or not bp_model_path().is_file():
        atomic.write_json(out_dir / "skipped.json", {"reason": "bp310 env missing"})
        _notifier(ctx, "warning")("Basic Pitch(bp310) 환경이 없어 보조 전사를 건너뜁니다. "
                                  "envs\\bp310 에서 `tools\\uvw.cmd sync` 로 만들 수 있습니다.")
        return
    try:  # keep the registry in step with the env (a re-sync can change the bundled model); never fatal
        record_bp_model()
    except Exception as e:
        log.warning("could not record %s in models.lock.json: %s", BP_MODEL_NAME, e)
    views = read_views(Path(ctx.dep_dirs["stems"]))
    todo = [{"id": vid, "wav": views[vid]["path"], "pcm_sha256": views[vid]["pcm_sha256"]} for vid in params["views"]]
    root = amt_cache.job_cache_root(ctx.store, ctx.song_key)
    cache = amt_cache.AmtCache(root)
    raws, _info = transcribe_views_bp(todo, params, ctx.config, work_dir=out_dir / "_worker", cache=cache,
                                      model_output_cache=root.parent / "bp_model_output", job=_job(ctx))
    for vid in sorted(raws):  # Basic Pitch is not latency-corrected (its lag is only reported, M1_M2_SPEC 3.2)
        atomic.write_json(out_dir / "raw" / f"{vid}__basicpitch.json", raws[vid])


def bp_skipped(stage_dir: Path) -> bool:
    return (Path(stage_dir) / "skipped.json").is_file()


# ---------------------------------------------------------------------------------------------- instr


def _load_energy(stems_dir: Path) -> tuple[dict[str, Any], float]:
    import numpy as np

    p = Path(stems_dir) / "energy.npz"
    if not p.is_file():
        return {}, 10.0
    with np.load(p) as z:
        data = {k: np.array(z[k]) for k in z.files}
    fps = float(data.pop("fps")[0]) if "fps" in data else 10.0
    return data, fps


def run_instr(ctx: Any) -> None:
    from gtab.amt import instrumentation as S

    params = dict(ctx.params)
    grid = atomic.read_json(Path(ctx.dep_dirs["grid"]) / "grid.json")
    sections_doc = atomic.read_json(Path(ctx.dep_dirs["sections"]) / "sections.json")
    mix_doc = None
    mix_path = Path(ctx.dep_dirs["amt_ms1"]) / "raw" / "mix_mono__muscriptor.json"
    if params["profile"] != "fast" and mix_path.is_file():
        mix_doc = atomic.read_json(mix_path)
    energy, fps = _load_energy(Path(ctx.dep_dirs["stems"]))
    doc = S.estimate(profile=params["profile"], bars=S.bars_from_grid(grid),
                     mix_notes=None if mix_doc is None else mix_doc.get("notes", []), energy=energy,
                     sections=sections_doc.get("sections") or [], hint=params["hints_instruments"],
                     params=params, guitar_view=params["guitar_view"], guitar_mask=params["guitar_mask"],
                     piano_other_mode=params["piano_other_instruments"], energy_fps=fps)
    from gtab.schema.instrumentation import Instrumentation

    Instrumentation.model_validate(doc)
    atomic.write_json(Path(ctx.out_dir) / "instrumentation.json", doc)
    for w in doc["warnings"]:
        _notifier(ctx, "warning")(w)


# ------------------------------------------------------------------------------------------------ s35


def _raw_files(stage_dir: Path) -> list[Path]:
    d = Path(stage_dir) / "raw"
    return sorted(d.glob("*.json")) if d.is_dir() else []


def run_s35(ctx: Any) -> None:
    """``s35.json``: the S3.5 artifacts of this job, by producing stage (paths relative to the job dir).

    ``raw`` is grouped by stage because the guitar pass may transcribe a pass-1 view again with another mask
    (E2's ``mix_mono`` option): both files are called ``mix_mono__muscriptor.json``. Basic Pitch entries are
    null when its env was missing (``skipped.json``).
    """
    job_dir = Path(ctx.store.job_dir(ctx.song_key))
    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    raw: dict[str, dict[str, str | None]] = {}
    for stage in ("amt_ms1", "amt_gtr", "amt_bp"):
        raw[stage] = {f.stem: _rel(job_dir, f) for f in _raw_files(dd[stage])}
    bp_skip = bp_skipped(dd["amt_bp"])
    if bp_skip:
        for vid in ("guitar_mono", "bass_mono"):
            raw["amt_bp"].setdefault(f"{vid}__basicpitch", None)
        raw["amt_bp"] = dict(sorted(raw["amt_bp"].items()))
    gtr = [f for f in _raw_files(dd["amt_gtr"]) if f.name.endswith("__muscriptor.json")]
    b0 = dd["amt_ms1"] / "raw" / "mix_mono__muscriptor.json"
    resid_wav = dd["resid1"] / "repeat_resid_pass1.wav"
    doc = {
        "format": "gtab.s35/1",
        "instrumentation": _rel(job_dir, dd["instr"] / "instrumentation.json"),
        "raw": raw,
        "guitar_pass": _rel(job_dir, gtr[0]) if len(gtr) == 1 else None,
        "b0": _rel(job_dir, b0) if b0.is_file() else None,
        "vocal_activity": _rel(job_dir, dd["vocal"] / "vocal_activity.json"),
        "repeat_resid_pass1": {"wav": _rel(job_dir, resid_wav) if resid_wav.is_file() else None,
                               "json": _rel(job_dir, dd["resid1"] / "resid1.json")},
        "basic_pitch": {"skipped": bp_skip,
                        "reason": atomic.read_json(dd["amt_bp"] / "skipped.json").get("reason") if bp_skip else None},
    }
    atomic.write_json(Path(ctx.out_dir) / "s35.json", doc)


# ---------------------------------------------------------------------------------------------- notes


def _only(doc: Mapping[str, Any], classes: tuple[str, ...]) -> dict[str, Any]:
    out = dict(doc)
    out["notes"] = [n for n in doc.get("notes") or [] if n.get("instrument") in classes]
    return out


def run_notes(ctx: Any) -> None:
    from gtab.amt import midi

    out_dir = Path(ctx.out_dir)
    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    s35 = atomic.read_json(dd["s35"] / "s35.json")
    grid = atomic.read_json(dd["grid"] / "grid.json")
    gtr_files = [f for f in _raw_files(dd["amt_gtr"]) if f.name.endswith("__muscriptor.json")]
    if len(gtr_files) != 1:
        raise AmtStageError(f"amt_gtr 단계의 기타 전사 파일을 찾지 못했습니다: {[f.name for f in gtr_files]}")
    gtr = atomic.read_json(gtr_files[0])
    guitar_all = _only(gtr, I.GUITAR_CLASSES)
    validate_noteset(guitar_all)
    atomic.write_json(out_dir / "guitar_all.json", guitar_all)

    files: dict[str, dict[str, Any]] = {}
    tracks = midi.class_tracks(guitar_all, I.GUITAR_CLASSES)
    summ = midi.write_performance_midi(out_dir / "midi" / "guitar_all.mid", tracks, grid)
    files["midi/guitar_all.mid"] = {"n_notes": summ["n_notes"], "tracks": [t[0] for t in tracks]}

    bass_path = dd["amt_ms1"] / "raw" / "bass_mono__muscriptor.json"
    bass = atomic.read_json(bass_path) if bass_path.is_file() else {"notes": []}
    btracks = midi.class_tracks(bass, tuple(reversed(I.BASS_CLASSES)))  # electric first
    s = midi.write_performance_midi(out_dir / "midi" / "bass_raw.mid", btracks, grid)
    files["midi/bass_raw.mid"] = {"n_notes": s["n_notes"], "tracks": [t[0] for t in btracks]}

    mix_path = dd["amt_ms1"] / "raw" / "mix_mono__muscriptor.json"
    if mix_path.is_file():  # quality profile: baseline B0 (mix transcription, guitar classes, one track)
        b0 = midi.merged_track(atomic.read_json(mix_path), I.GUITAR_CLASSES, "b0_guitar", 27)
        s = midi.write_performance_midi(out_dir / "midi" / "b0_guitar.mid", [b0], grid)
        files["midi/b0_guitar.mid"] = {"n_notes": s["n_notes"], "tracks": ["b0_guitar"]}

    bp_path = dd["amt_bp"] / "raw" / "guitar_mono__basicpitch.json"
    if not bp_skipped(dd["amt_bp"]) and bp_path.is_file():
        bp = atomic.read_json(bp_path)
        s = midi.write_performance_midi(out_dir / "midi" / "guitar_basicpitch.mid",
                                        [("guitar_basicpitch", 27, bp.get("notes") or [])], grid)
        files["midi/guitar_basicpitch.mid"] = {"n_notes": s["n_notes"], "tracks": ["guitar_basicpitch"]}

    counts: dict[str, int] = {}
    for n in guitar_all["notes"]:
        counts[n["instrument"]] = counts.get(n["instrument"], 0) + 1
    summary = {
        "format": "gtab.notes_summary/1",
        "guitar_view": gtr.get("view"),
        "time_offset_applied_s": gtr.get("time_offset_applied_s"),
        "midi_bar_offset": summ["midi_bar_offset"],
        "midi_lead_in": summ["lead_in"],
        "ticks_per_beat": summ["ticks_per_beat"],
        "guitar_classes": {c: counts[c] for c in I.GUITAR_CLASSES if c in counts},
        "files": files,
        "basic_pitch": not bp_skipped(dd["amt_bp"]),
        "s35": s35.get("format"),
    }
    atomic.write_json(out_dir / "notes_summary.json", summary)


# ---------------------------------------------------------------------------------------------- table


STAGES: dict[str, dict] = {
    "amt_ms1": {"run": run_amt_ms1, "code_version": "1", "params": amt_ms1_params, "device": "gpu",
                "models": ms_models},
    "amt_bp": {"run": run_amt_bp, "code_version": "1", "params": bp_params, "device": "cpu", "models": bp_models},
    "instr": {"run": run_instr, "code_version": "2", "params": instr_params, "device": "cpu",
              "models": lambda cfg: {}},
    "amt_gtr": {"run": run_amt_gtr, "code_version": "1", "params": amt_gtr_params, "device": "gpu",
                "models": ms_models},
    "s35": {"run": run_s35, "code_version": "2", "params": lambda cfg: {}, "device": "cpu",
            "models": lambda cfg: {}},
    "notes": {"run": run_notes, "code_version": "1", "params": lambda cfg: {}, "device": "cpu",
              "models": lambda cfg: {}},
}
