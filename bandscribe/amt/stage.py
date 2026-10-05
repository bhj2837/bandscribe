"""AMT pipeline stages: ``amt_ms1``, ``amt_bp``, ``instr``, ``amt_gtr``, ``s35``, ``notes`` (M1_M2_SPEC 3.2 rows 8-13).

``STAGES`` follows the stage contract of M1_M2_SPEC 3.1; ``bandscribe.pipeline.graph`` (GRID) owns names and deps and
imports this module lazily. Stage functions depend only on ``ctx.params``, their dep dirs and the job input;
``ctx.config`` is read only for non-semantic knobs (timeouts, VRAM policy) through ``bandscribe.vram``.

- ``amt_ms1`` (GPU, ``amt_muscriptor``): pass-1 views ``bass_mono`` (bass classes) and the piano+other presence
  pass (keys/synth/strings [+ guitar], M1_M2_SPEC A5): ``piano_other_presence`` = sampled windows of
  ``piano_other_mono`` (``bandscribe.amt.presence``; ``amt.presence_pass = "full"``: the full-length view instead).
  ``eval`` adds two **extra outputs** that no later stage of the job reads: ``mix_mono`` (all classes = baseline
  B0) and the full-length ``piano_other_mono`` (S7/M6 and E3b material). Writes ``presence.json``.
- ``amt_bp`` (CPU, bp310 ``amt_basicpitch``, no GPU lock; **optional**): ``guitar_mono`` and ``bass_mono``. Without
  the bp310 env it writes ``skipped.json`` and a Korean warning; its models() then return
  ``{"basic_pitch": "absent"}`` so the key changes once the env is installed (M1_M2_SPEC A21).
- ``instr`` (CPU): ``instrumentation.json`` (``bandscribe.amt.instrumentation``) from the presence pass, the bass pass,
  stem energy and hints; identical in every profile (``eval``'s mix pass is reported, not used: review 2026-10-04,
  ``eval`` must not change the system it evaluates).
- ``amt_gtr`` (GPU): the ``amt.guitar_view`` transcription with the guitar-pass mask from ``instr``.
- ``s35`` (CPU): ``s35.json`` index of the S3.5 artifacts (paths relative to the job dir). ``guitar_pass`` is the
  **raw** guitar view: with ``guitar+present`` it may hold notes labelled with a present keys/synth class
  (counted in ``guitar_pass_non_guitar``); S4+ read ``notes/guitar_all.json`` (guitar classes only).
- ``notes`` (CPU): ``guitar_all.json`` + ``midi/*.mid`` + ``notes_summary.json``; a Korean warning when the guitar
  pass labelled >= 10 % of its notes non-guitar.

Transcriptions go through the shared cache (``bandscribe.amt.cache``): views without an entry are sent to the worker,
which writes raw NoteSets under ``_worker/raw/`` (provenance); the stage stores them in the cache under the rung
actually used and writes its ``raw/`` outputs from the cache with the latency constant applied
(``onset_s = raw - amt.latency_s``, clamped >= 0; ``time_offset_applied_s = -latency_s``). GPU stages write
``gpu_run.json`` (``bandscribe.vram.write_gpu_run``); when every view was cached, it records the cached top rung.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bandscribe import atomic, models, paths, vram
from bandscribe.amt import cache as amt_cache
from bandscribe.amt import instruments as I
from bandscribe.amt import presence as P
from bandscribe.config import RUN_PROFILES, Config, ConfigError

log = logging.getLogger(__name__)

CACHE_CODE_VERSION = "1"
MS_BACKEND = "amt_muscriptor"
BP_BACKEND = "amt_basicpitch"
BP_RUNG = "onnx-cpu"
HF_REPO = "MuScriptor/muscriptor-{size}"

BP_MODEL_NAME = "basic_pitch.icassp2022"

# Mirrors of the [run]/[hints]/[amt]/[instr] defaults (M1_M2_SPEC 1.3) for plain-dict configs (tests, experiments
# with overrides). A real ``bandscribe.config.Config`` is always read strictly (``cfg.get(key)``): a misspelt key raises
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
    "amt.presence_pass": "auto",
    "amt.presence_window_s": 10.0,
    "amt.presence_max_windows": 6,
    "amt.presence_max_total_s": 60.0,
    "amt.presence_max_fraction": 0.2,
    "amt.presence_min_active": 0.5,
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
    "instr.presence_min_notes": 8,
}
PROFILES = RUN_PROFILES  # fast | quality | eval (2026-10-04)
BP_WORKER_TIMEOUT_S = 3600.0
DROPPED_WARN_SHARE = 0.1  # notes: warn when the guitar pass labelled this share of its notes non-guitar


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
    """Installed version of ``dist`` in another venv, read from its metadata (no import, no subprocess).

    Reads the ``*.dist-info/METADATA`` headers directly: ``importlib.metadata`` caches directory listings by
    the directory's mtime, so an upgrade within the mtime resolution (same second on Windows) returned the
    removed version's listing and no version at all (flaky test, 2026-10-05)."""
    sp = _site_packages(Path(python))
    if not sp.is_dir():
        return None
    want = dist.lower().replace("_", "-")
    for info in sorted(sp.glob("*.dist-info")):
        if info.name.split("-")[0].lower().replace("_", "-") != want:
            continue
        try:
            text = (info / "METADATA").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        head: dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip():
                break  # end of the header block
            key, _, value = line.partition(":")
            head.setdefault(key.strip().lower(), value.strip())
        if head.get("name", "").lower().replace("_", "-") == want:
            return head.get("version") or None
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
    so bandscribe never downloads it; the registry entry keeps its licence and hash traceable like every other model
    (M1_M2_SPEC 0, 9.2 item 2). Idempotent: rewrites the entry only when the file or its hash changed (a re-sync
    of the env). Returns the entry, or None when the env / file is absent.
    """
    onnx = bp_model_path()
    if not onnx.is_file():
        return None
    sha = _sha256(onnx)
    try:
        rel = onnx.resolve().relative_to(Path(paths.ROOT).resolve()).as_posix()
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
              "are). Not downloaded by bandscribe.",
        recorded_utc=models.utc_now())
    models.record(entry)
    log.info("recorded %s (%s) in models.lock.json", BP_MODEL_NAME, sha[:12])
    return entry


# ----------------------------------------------------------------------------------- params / models


def ms_params(cfg: Any) -> dict[str, Any]:
    """MuScriptor keys shared by amt_ms1 / amt_gtr (every value that changes a transcription).

    Value checks (dtype, loader, ...) live in ``bandscribe.config``; the worker re-checks dtype/loader and refuses bf16.
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


def _check_profile(profile: str) -> str:
    if profile not in PROFILES:  # Config already restricts it; dict configs (experiments, tests) land here
        raise ConfigError(f"run.profile 은 {' | '.join(PROFILES)} 중 하나여야 합니다 (입력값: {profile!r})")
    return profile


def presence_mode(profile: str, presence_pass: str = "auto") -> str:
    """The presence source: ``sampled`` (windows) or ``full`` (whole piano+other view). ``auto`` = sampled in
    every profile, so ``eval`` estimates the instrumentation exactly like ``quality`` (review 2026-10-04)."""
    if presence_pass not in ("auto", "sampled", "full"):
        raise ConfigError(f"amt.presence_pass 는 auto | sampled | full 중 하나여야 합니다 "
                          f"(입력값: {presence_pass!r})")
    _check_profile(profile)
    return "sampled" if presence_pass == "auto" else presence_pass


def pass1_views(profile: str, presence_pass: str = "auto") -> list[str]:
    """Views of ``amt_ms1``: bass and the presence source; ``eval`` adds the B0 mix pass and the full-length
    piano+other pass as extra outputs (first and last)."""
    _check_profile(profile)
    po = P.FULL_VIEW if presence_mode(profile, presence_pass) == "full" else P.PRESENCE_VIEW
    views = ["bass_mono", po]
    if profile == "eval":
        views = ["mix_mono"] + views + ([P.FULL_VIEW] if po != P.FULL_VIEW else [])
    return views


def presence_settings(cfg: Any, profile: str) -> dict[str, Any]:
    """The sampled-window budget (``bandscribe.amt.presence``); ``fast`` halves the windows and the audio cap."""
    max_windows = int(amt_value(cfg, "amt.presence_max_windows"))
    max_total = float(amt_value(cfg, "amt.presence_max_total_s"))
    if profile == "fast":
        max_windows, max_total = max(1, max_windows // 2), max_total / 2
    return {"window_s": float(amt_value(cfg, "amt.presence_window_s")), "max_windows": max_windows,
            "max_total_s": max_total, "max_fraction": float(amt_value(cfg, "amt.presence_max_fraction")),
            "min_active": float(amt_value(cfg, "amt.presence_min_active")),
            "active_rel_db": float(amt_value(cfg, "instr.active_rel_db"))}


def amt_ms1_params(cfg: Any) -> dict[str, Any]:
    profile = _check_profile(str(amt_value(cfg, "run.profile")))
    mode = str(amt_value(cfg, "amt.piano_other_instruments"))
    I.mask_piano_other(mode)  # validates
    pmode = presence_mode(profile, str(amt_value(cfg, "amt.presence_pass")))
    presence: dict[str, Any] = {"mode": pmode}
    if pmode == "sampled":
        presence.update(presence_settings(cfg, profile))
    return {**ms_params(cfg), "profile": profile, "piano_other_instruments": mode,
            "views": pass1_views(profile, str(amt_value(cfg, "amt.presence_pass"))), "presence": presence}


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
                f"(bandscribe run 중에는 내려받지 않습니다).")
        raise models.ModelMissing(f"muscriptor-{size}", hint)
    return {f"muscriptor-{size}": _sha256(w)}


def bp_params(cfg: Any) -> dict[str, Any]:
    """``amt_bp`` params; ``min_note_frames`` is the shortest note *kept* (the worker passes frames - 1)."""
    from bandscribe.workers.amt_basicpitch import normalize_params

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
        "presence_min_notes": int(amt_value(cfg, "instr.presence_min_notes")),
        "hints_instruments": hint if isinstance(hint, str) else list(hint),
        "profile": _check_profile(str(amt_value(cfg, "run.profile"))),
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
    from bandscribe.schema.notes import NoteSet

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
    from bandscribe.workers.amt_muscriptor import rung_labels

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


def _view_cparams(cparams: Mapping[str, Any], view: Mapping[str, Any]) -> dict[str, Any]:
    """Cache params of one view: whole-view entries keep the M2 key; a segmented view adds its segments."""
    if not view.get("segments"):
        return dict(cparams)
    return {**cparams, "segments": [[round(float(a), 6), round(float(b), 6)] for a, b in view["segments"]]}


def transcribe_views_ms(views: list[dict[str, Any]], params: Mapping[str, Any], cfg: Any, *, work_dir: Path,
                        cache: amt_cache.AmtCache, model_sha256: str | None = None, job: dict | None = None,
                        notify: Callable[[str], None] | None = None, force_rung: str | None = None,
                        run_gpu_stage: Callable[..., Any] | None = None) -> tuple[dict[str, dict], dict[str, Any]]:
    """Raw NoteSets for ``views`` = [{"id", "wav", "pcm_sha256", "instruments", ["segments"]}] via the cache
    and the worker. ``segments`` ([[start_s, end_s], ...], the presence windows) transcribes only those parts of
    the view (one decode per segment, times in view time); they are part of that view's cache key.

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
        key = amt_cache.amt_key("muscriptor", version, model_sha256, v["pcm_sha256"], mask,
                                _view_cparams(cparams, v), CACHE_CODE_VERSION, lookup_label)
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
        "format": "bandscribe.worker/1", "job": job, "out_dir": str(Path(work_dir).parent), "device": "auto",
        "model": model, "decode": dict(params["decode"]), "dtype": params["dtype"],
        "views": [{"id": v["id"], "wav": str(v["wav"]), "instruments": v["instruments"],
                   "out": str(raw_dir / f"{v['id']}__muscriptor.json"), "view_sha256": v["pcm_sha256"],
                   **({"segments": [list(map(float, sg)) for sg in v["segments"]]} if v.get("segments") else {})}
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
        key = amt_cache.amt_key("muscriptor", version, actual_sha, v["pcm_sha256"], v["instruments"],
                                _view_cparams(cparams, v), CACHE_CODE_VERSION, rung_label)
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
    # the worker's Korean warnings (slow rung, a view that ran out of tokens without EOS) reach the console
    # instead of staying in _worker/result.json (review 2026-10-05); gpu_run.json keeps them for cached runs
    warn = _notifier(ctx, "warning")
    for w in worker_out.get("warnings") or []:
        warn(str(w))


def _job(ctx: Any) -> dict[str, Any]:
    return {"song_key": ctx.song_key, "stage": ctx.stage, "stage_key": ctx.key}


def _ms_model_sha(params: Mapping[str, Any]) -> str:
    w = muscriptor_weights(params["muscriptor_model"], params["muscriptor_revision"])
    if w is None:
        raise AmtStageError(f"MuScriptor-{params['muscriptor_model']} 가중치가 없습니다(HF 캐시, 리비전 "
                            f"{params['muscriptor_revision'][:12]}).")
    return _sha256(w)


def _song_s(grid: Mapping[str, Any] | None, energy: Mapping[str, Any], fps: float) -> float:
    if grid and grid.get("duration_s"):
        return float(grid["duration_s"])
    mix = energy.get("mix")
    return 0.0 if mix is None else len(mix) / float(fps)


def select_presence_windows(ctx: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    """The presence windows of this job (``bandscribe.amt.presence.select_windows`` on the stems/grid/sections deps)."""
    from bandscribe.amt import instrumentation as S

    dd = ctx.dep_dirs
    energy, fps = _load_energy(Path(dd["stems"]))
    grid = atomic.read_json(Path(dd["grid"]) / "grid.json") if "grid" in dd else None
    sections = []
    if "sections" in dd:
        sections = atomic.read_json(Path(dd["sections"]) / "sections.json").get("sections") or []
    bars = S.bars_from_grid(grid) if grid else []
    return P.select_windows(bars=bars, sections=sections, energy=energy, fps=fps, song_s=_song_s(grid, energy, fps),
                            **{k: settings[k] for k in P.DEFAULTS})


def run_amt_ms1(ctx: Any) -> None:
    params = dict(ctx.params)
    views = read_views(Path(ctx.dep_dirs["stems"]))
    masks = I.pass1_masks(params["piano_other_instruments"])
    presence = dict(params.get("presence") or {"mode": "full"})
    po_view = P.FULL_VIEW if presence["mode"] == "full" else P.PRESENCE_VIEW
    todo = []
    pres_doc: dict[str, Any] = {"format": "bandscribe.presence/1", "mode": presence["mode"], "source_view": P.SOURCE_VIEW,
                                "instruments": masks[P.SOURCE_VIEW],
                                "evidence_classes": I.presence_evidence_classes(params["piano_other_instruments"])}
    for vid in params["views"]:
        if vid == P.PRESENCE_VIEW:
            sel = select_presence_windows(ctx, presence)
            segs = P.spans(sel)
            pres_doc.update(view=vid, selection=sel, skipped=None if segs else sel.get("reason") or "no_window",
                            file=f"raw/{vid}__muscriptor.json" if segs else None)
            if not segs:  # piano+other is quiet: nothing to transcribe (instr counts it as coverage, e = 0)
                continue
            src = views[P.SOURCE_VIEW]
            todo.append({"id": vid, "wav": src["path"], "pcm_sha256": src["pcm_sha256"],
                         "instruments": masks[vid], "segments": segs})
            continue
        if vid == po_view:  # the full-length view as the presence source (eval's extra full view is not)
            pres_doc.update(view=vid, selection=None, skipped=None, file=f"raw/{vid}__muscriptor.json")
        todo.append({"id": vid, "wav": views[vid]["path"], "pcm_sha256": views[vid]["pcm_sha256"],
                     "instruments": masks[vid]})
    cache = amt_cache.AmtCache(amt_cache.job_cache_root(ctx.store, ctx.song_key))
    raws, info = transcribe_views_ms(todo, params, ctx.config, work_dir=Path(ctx.out_dir) / "_worker", cache=cache,
                                     model_sha256=_ms_model_sha(params), job=_job(ctx),
                                     notify=_notifier(ctx))
    _write_ms_outputs(ctx, raws, params, info)
    if "view" in pres_doc:
        atomic.write_json(Path(ctx.out_dir) / "presence.json", pres_doc)


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
    """Transcription-cache params of a Basic Pitch view. ``threads`` belongs here as it does in the ``amt_bp``
    stage key: the ONNX session's intra-op thread count changes the model output in the last float bit (1 vs 4
    threads: ~1,700 of the note/onset/contour values of a 30 s clip differ by 1.2e-7, measured 2026-10-05), which
    can flip a threshold decision; the worker pins it for exactly that reason."""
    return {"params": dict(params["params"]), "threads": int(params["threads"])}


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
    req = {"format": "bandscribe.worker/1", "job": job, "out_dir": str(Path(work_dir).parent), "device": "cpu",
           "mode": "transcribe", "threads": int(params["threads"]),
           "model_output_cache": None if model_output_cache is None else str(model_output_cache),
           "views": [{"id": v["id"], "wav": str(v["wav"]), "out": str(raw_dir / f"{v['id']}__basicpitch.json"),
                      "view_sha256": v["pcm_sha256"]} for v, _k in missing],
           "params": dict(params["params"]), "allow_default_min_len": False, "determinism": True}
    if run_worker is None:
        from bandscribe.gpu import run_worker as _rw

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


def load_presence(ms1_dir: Path) -> dict[str, Any] | None:
    """The piano+other presence transcription of an ``amt_ms1`` stage dir, as ``instrumentation.estimate`` wants it.

    ``presence.json`` names the file and the windows; a stage dir from before it (M2: full-length pass, no
    presence.json) falls back to ``raw/piano_other_mono__muscriptor.json`` as a full pass.
    """
    ms1_dir = Path(ms1_dir)
    meta_path = ms1_dir / "presence.json"
    if meta_path.is_file():
        meta = atomic.read_json(meta_path)
    elif (ms1_dir / "raw" / f"{P.FULL_VIEW}__muscriptor.json").is_file():
        meta = {"mode": "full", "view": P.FULL_VIEW, "file": f"raw/{P.FULL_VIEW}__muscriptor.json", "skipped": None}
    else:
        return None
    sel = meta.get("selection") or {}
    wins = sel.get("windows") or []
    summary = {"view": meta.get("view"), "source_view": meta.get("source_view", P.SOURCE_VIEW),
               "windows": [[w["start_s"], w["end_s"]] for w in wins] if sel else None,
               "window_sections": [w.get("section") for w in wins] if sel else None,
               "transcribed_s": sel.get("transcribed_s") if sel else None,
               "fraction": sel.get("fraction") if sel else None, "song_s": sel.get("song_s") if sel else None}
    if meta.get("skipped") or not meta.get("file"):
        return {"mode": meta.get("mode"), "skipped": meta.get("skipped") or "no_file", "notes": [], "spans": None,
                "classes": meta.get("evidence_classes"), "summary": summary}
    doc = atomic.read_json(ms1_dir / meta["file"])
    # A segmented NoteSet is a sampled pass even when its segment list is empty (every window clipped away):
    # that must not read as a full pass with the song-wide saturation (review 2026-10-04).
    sampled = "segments" in doc or meta.get("mode") == "sampled"
    spans = [list(sg) for sg in doc.get("segments") or []] if sampled else None
    win_secs = [w.get("section") for w in wins] if sampled and len(wins) == len(spans or []) else None
    return {"mode": "sampled" if sampled else "full", "skipped": None, "notes": doc.get("notes") or [],
            "spans": spans, "window_sections": win_secs, "classes": meta.get("evidence_classes"),
            "summary": summary}


def run_instr(ctx: Any) -> None:
    from bandscribe.amt import instrumentation as S

    params = dict(ctx.params)
    grid = atomic.read_json(Path(ctx.dep_dirs["grid"]) / "grid.json")
    sections_doc = atomic.read_json(Path(ctx.dep_dirs["sections"]) / "sections.json")
    ms1 = Path(ctx.dep_dirs["amt_ms1"])

    def notes_of(view: str) -> list[dict] | None:
        f = ms1 / "raw" / f"{view}__muscriptor.json"
        return atomic.read_json(f).get("notes", []) if f.is_file() else None

    energy, fps = _load_energy(Path(ctx.dep_dirs["stems"]))
    # eval's mix pass (B0) is reported in the document, never used as evidence: every profile estimates alike
    doc = S.estimate(profile=params["profile"], bars=S.bars_from_grid(grid), mix_notes=notes_of("mix_mono"),
                     energy=energy, sections=sections_doc.get("sections") or [], hint=params["hints_instruments"],
                     params=params, guitar_view=params["guitar_view"], guitar_mask=params["guitar_mask"],
                     piano_other_mode=params["piano_other_instruments"], energy_fps=fps,
                     presence=load_presence(ms1), bass_notes=notes_of("bass_mono"))
    from bandscribe.schema.instrumentation import Instrumentation

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
    gtr_doc = atomic.read_json(gtr[0]) if len(gtr) == 1 else {}
    b0 = dd["amt_ms1"] / "raw" / "mix_mono__muscriptor.json"
    pres = dd["amt_ms1"] / "presence.json"
    resid_wav = dd["resid1"] / "repeat_resid_pass1.wav"
    doc = {
        "format": "bandscribe.s35/1",
        "instrumentation": _rel(job_dir, dd["instr"] / "instrumentation.json"),
        "raw": raw,
        # raw guitar view: may hold notes labelled with a present keys/synth class (guitar+present mask); S4+ read
        # notes/guitar_all.json, which keeps the guitar classes only
        "guitar_pass": _rel(job_dir, gtr[0]) if len(gtr) == 1 else None,
        "guitar_pass_mask": (gtr_doc.get("backend") or {}).get("instruments") if gtr_doc else None,
        "guitar_pass_non_guitar": non_guitar_counts(gtr_doc),
        "b0": _rel(job_dir, b0) if b0.is_file() else None,  # eval profile only
        "presence": _rel(job_dir, pres) if pres.is_file() else None,
        "vocal_activity": _rel(job_dir, dd["vocal"] / "vocal_activity.json"),
        "repeat_resid_pass1": {"wav": _rel(job_dir, resid_wav) if resid_wav.is_file() else None,
                               "json": _rel(job_dir, dd["resid1"] / "resid1.json")},
        "basic_pitch": {"skipped": bp_skip,
                        "reason": atomic.read_json(dd["amt_bp"] / "skipped.json").get("reason") if bp_skip else None},
    }
    atomic.write_json(Path(ctx.out_dir) / "s35.json", doc)


# ---------------------------------------------------------------------------------------------- notes


def non_guitar_counts(doc: Mapping[str, Any]) -> dict[str, int]:
    """{class: notes} of a guitar-pass NoteSet's notes outside the guitar classes, in MT3 order."""
    out: dict[str, int] = {}
    for n in doc.get("notes") or []:
        if n.get("instrument") not in I.GUITAR_CLASSES:
            out[str(n.get("instrument"))] = out.get(str(n.get("instrument")), 0) + 1
    return {c: out[c] for c in sorted(out, key=lambda c: (I.MT3_GROUPS.get(c, 99), c))}


def _only(doc: Mapping[str, Any], classes: tuple[str, ...]) -> dict[str, Any]:
    out = dict(doc)
    out["notes"] = [n for n in doc.get("notes") or [] if n.get("instrument") in classes]
    return out


def run_notes(ctx: Any) -> None:
    from bandscribe.amt import midi

    out_dir = Path(ctx.out_dir)
    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    s35 = atomic.read_json(dd["s35"] / "s35.json")
    grid = atomic.read_json(dd["grid"] / "grid.json")
    gtr_files = [f for f in _raw_files(dd["amt_gtr"]) if f.name.endswith("__muscriptor.json")]
    if len(gtr_files) != 1:
        raise AmtStageError(f"amt_gtr 단계의 기타 전사 파일을 찾지 못했습니다: {[f.name for f in gtr_files]}")
    gtr = atomic.read_json(gtr_files[0])
    # Only guitar classes reach guitar_all: the guitar pass may label stem bleed with a present keys/synth class
    # (guitar+present mask), and those notes must not end up in the guitar MIDI.
    guitar_all = _only(gtr, I.GUITAR_CLASSES)
    dropped = non_guitar_counts(gtr)
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
    if mix_path.is_file():  # eval profile: baseline B0 (mix transcription, guitar classes, one track)
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
    n_view = len(gtr.get("notes") or [])
    n_drop = sum(dropped.values())
    if n_view and n_drop / n_view >= DROPPED_WARN_SHARE:
        names = ", ".join(f"{c} {k}" for c, k in sorted(dropped.items(), key=lambda kv: -kv[1]))
        _notifier(ctx, "warning")(
            f"기타 전사 음의 {100.0 * n_drop / n_view:.0f}%({n_drop}/{n_view})가 기타가 아닌 악기로 표시되어 "
            f"guitar_all 에서 뺐습니다({names}). 기타 스템에 다른 악기가 많이 섞였을 수 있으니 들어 보세요.")
    summary = {
        "format": "bandscribe.notes_summary/1",
        "guitar_view": gtr.get("view"),
        "time_offset_applied_s": gtr.get("time_offset_applied_s"),
        "midi_bar_offset": summ["midi_bar_offset"],
        "midi_lead_in": summ["lead_in"],
        "ticks_per_beat": summ["ticks_per_beat"],
        "guitar_view_notes": n_view,
        "guitar_classes": {c: counts[c] for c in I.GUITAR_CLASSES if c in counts},
        "non_guitar_dropped": dropped,
        "files": files,
        "basic_pitch": not bp_skipped(dd["amt_bp"]),
        "s35": s35.get("format"),
    }
    atomic.write_json(out_dir / "notes_summary.json", summary)


# ---------------------------------------------------------------------------------------------- table


STAGES: dict[str, dict] = {
    # code versions (2026-10-04): amt_ms1 2 = profiles, presence windows, strings in the piano+other mask;
    # 3 = review fixes (window gap/home section/stem coverage, sampled presence in eval too, eval's extra full
    # view). instr 3 = presence source + renormalised weights; 4 = guards (sections, stem agreement), stem
    # attribution, bass pass evidence, mix pass reported only, guitar+present back to keys/synth. amt_gtr 2.
    # s35 3 = presence entry; 4 = guitar_pass_mask / guitar_pass_non_guitar. notes 2 = non_guitar_dropped;
    # 3 = guitar_view_notes + dropped-share warning; 4 (2026-10-05) = MIDI lead-in tempo kept within 30-300 BPM
    # (eighth/sixteenth lead-in bars, tiny lead-ins absorbed into the first beat).
    "amt_ms1": {"run": run_amt_ms1, "code_version": "3", "params": amt_ms1_params, "device": "gpu",
                "models": ms_models},
    "amt_bp": {"run": run_amt_bp, "code_version": "1", "params": bp_params, "device": "cpu", "models": bp_models},
    "instr": {"run": run_instr, "code_version": "4", "params": instr_params, "device": "cpu",
              "models": lambda cfg: {}},
    "amt_gtr": {"run": run_amt_gtr, "code_version": "2", "params": amt_gtr_params, "device": "gpu",
                "models": ms_models},
    "s35": {"run": run_s35, "code_version": "4", "params": lambda cfg: {}, "device": "cpu",
            "models": lambda cfg: {}},
    "notes": {"run": run_notes, "code_version": "4", "params": lambda cfg: {}, "device": "cpu",
              "models": lambda cfg: {}},
}
