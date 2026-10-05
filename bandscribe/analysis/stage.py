"""GRID pipeline stages: ``beats``, ``grid``, ``sections``, ``vocal``, ``resid1`` (M1_M2_SPEC 3.1-3.2, 9.3).

``STAGES`` follows the stage contract of M1_M2_SPEC 3.1; ``bandscribe.pipeline.graph`` owns names and deps.
Every stage depends only on ``ctx.params``, its dep dirs and the job input (the mix); ``ctx.config`` is read
only for non-semantic GPU knobs (timeouts, VRAM policy) inside ``bandscribe.vram.run_gpu_stage``.

- ``beats`` (GPU, gpu env): Beat This! worker ``beats_beatthis`` with the **local** checkpoint recorded as
  ``beat_this.<grid.checkpoint>`` in ``models.lock.json`` (``bandscribe models fetch beat_this.final0``). A missing
  model raises ``bandscribe.models.ModelMissing`` from ``models()`` at plan time, before anything runs.
- ``grid`` (CPU): percussive onsets of the mix + :func:`bandscribe.analysis.grid.derive_grid`.
- ``sections`` (CPU): :func:`bandscribe.analysis.sections.analyze`.
- ``vocal`` (CPU): vocal activity from the ``sep`` stage's ``stems/vocals.wav`` (the ``stems`` stage does not
  re-export the vocals stem, so ``vocal`` depends on ``sep`` directly as well).
- ``resid1`` (CPU): repeat residual pass 1 on the ``stems`` stage's guitar mono view.

CPU stages run under ``threadpoolctl.threadpool_limits(1)`` and write only deterministic data files (JSON via
``bandscribe.atomic.write_json``, npz via :func:`bandscribe.analysis.onsets.write_npz`, WAV via ``bandscribe.audio.write_f32``).
``beats.json``, ``grid.json``, ``sections.json`` and ``repeat_map.json`` are validated against the P0 pydantic
schemas (M1_M2_SPEC 6.2-6.3) before they are written; the S3.5 files (6.6) have no pydantic model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.schema.grid import BeatsRaw, Grid
from bandscribe.schema.sections import RepeatMap, Sections

log = logging.getLogger(__name__)

MIX_NAME = "mix_44k_f32.wav"
BEATS_BACKEND = "beats_beatthis"


class GridStageError(RuntimeError):
    """A GRID stage cannot produce its output (Korean, user-facing message)."""


def _conf(cfg: Any) -> Any:
    """A validated ``bandscribe.config.Config``: as is, from a (partial) nested dict (tests), or the defaults (None).

    Params are read from the validated sections, never from a local copy of the defaults: a misspelled key
    fails loudly instead of silently falling back, and ``grid.dbn = true`` is refused by the config itself.
    """
    from bandscribe.config import Config

    if cfg is None:
        return Config()
    if isinstance(cfg, Config):
        return cfg
    if isinstance(cfg, Mapping):
        return Config.model_validate(dict(cfg))
    raise TypeError(f"expected a bandscribe Config, a mapping or None, got {type(cfg).__name__}")


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


def mix_path(ctx: Any) -> Path:
    return Path(ctx.store.input_dir(ctx.song_key)) / MIX_NAME


# ------------------------------------------------------------------------------------------------ beats


def beats_params(cfg: Any) -> dict[str, Any]:
    """``beats`` stage params = key. Like the MuScriptor / Basic Pitch stages, the installed package version is
    part of it (read from the gpu env's metadata, no import): a beat-this upgrade (its pre/postprocessing) must
    recompute the beats rather than reuse them (review 2026-10-05; this changed every job's beats key once)."""
    from bandscribe import paths
    from bandscribe.amt.stage import dist_version

    g = _conf(cfg).grid  # grid.dbn = true never gets here: the config refuses it (needs madmom)
    return {"checkpoint": str(g.checkpoint), "dbn": bool(g.dbn), "float16": False,
            "beat_this_version": dist_version(paths.GPU_PYTHON, "beat-this")}


def beats_model_name(cfg: Any) -> str:
    return f"beat_this.{_conf(cfg).grid.checkpoint}"


def beats_models(cfg: Any) -> dict[str, str]:
    """``{"beat_this.final0": sha256}``; raises ``bandscribe.models.ModelMissing`` when it was never fetched."""
    from bandscribe import models

    name = beats_model_name(cfg)
    return {name: models.sha256_cached(models.local_path(name))}


def run_beats(ctx: Any) -> None:
    from bandscribe import models, vram

    params = dict(ctx.params)
    name = f"beat_this.{params['checkpoint']}"
    ckpt = models.local_path(name)
    sha = models.sha256_cached(ckpt)
    labels = [params["checkpoint"]]
    out_dir = Path(ctx.out_dir)
    request = {
        "format": "bandscribe.worker/1",
        "job": {"song_key": ctx.song_key, "stage": ctx.stage, "stage_key": ctx.key},
        "out_dir": str(out_dir),
        "device": "auto",
        "determinism": True,
        "wav": str(mix_path(ctx)),
        "checkpoint": params["checkpoint"],
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha,
        "dbn": False,
        "float16": False,
        "out_beats": "beats.json",
        "out_activations": "activations.npz",
    }
    res = vram.run_gpu_stage(BEATS_BACKEND, labels, request, ctx.config, out_dir / "_worker",
                             notify=_notifier(ctx))
    output = res.output
    if output.get("status") != "ok" or not (out_dir / "beats.json").is_file():
        raise GridStageError(f"Beat This! 워커가 결과를 내지 못했습니다(status={output.get('status')}). "
                             f"자세한 내용: {out_dir / '_worker' / 'stderr.log'}")
    vram.write_gpu_run(out_dir, BEATS_BACKEND, labels, output, getattr(res, "waited_s", 0.0))
    BeatsRaw.model_validate(atomic.read_json(out_dir / "beats.json"))
    warn = _notifier(ctx, "warning")  # the worker's Korean warnings reach the console (review 2026-10-05)
    for w in output.get("warnings") or []:
        warn(str(w))


# ------------------------------------------------------------------------------------------------- grid


def grid_params(cfg: Any) -> dict[str, Any]:
    return _conf(cfg).grid.model_dump(mode="json")


def load_activations(path: Path) -> dict[str, Any] | None:
    import numpy as np

    if not Path(path).is_file():
        return None
    with np.load(str(path), allow_pickle=False) as z:
        return {"beat": z["beat"], "downbeat": z["downbeat"],
                "fps": float(z["fps"]) if "fps" in z.files else 50.0}


def run_grid(ctx: Any) -> None:
    from threadpoolctl import threadpool_limits

    from bandscribe.analysis import grid as gridmod
    from bandscribe.analysis import onsets

    beats_dir = Path(ctx.dep_dirs["beats"])
    beats_raw = atomic.read_json(beats_dir / "beats.json")
    acts = load_activations(beats_dir / "activations.npz")
    with threadpool_limits(limits=1):
        ons = onsets.compute(mix_path(ctx))
        onsets.write_npz(Path(ctx.out_dir) / "onsets.npz", onsets.to_arrays(ons))
        doc = gridmod.derive_grid(beats_raw, acts, ons, dict(ctx.params))
    Grid.model_validate(doc)
    atomic.write_json(Path(ctx.out_dir) / "grid.json", doc)


# --------------------------------------------------------------------------------------------- sections


def sections_params(cfg: Any) -> dict[str, Any]:
    return _conf(cfg).sections.model_dump(mode="json")


def run_sections(ctx: Any) -> None:
    from bandscribe.analysis import onsets, sections

    grid_dir = Path(ctx.dep_dirs["grid"])
    grid = atomic.read_json(grid_dir / "grid.json")
    ons = onsets.load(grid_dir / "onsets.npz")
    sec, rep = sections.analyze(mix_path(ctx), grid, ons, dict(ctx.params))
    Sections.model_validate(sec)
    RepeatMap.model_validate(rep)
    atomic.write_json(Path(ctx.out_dir) / "sections.json", sec)
    atomic.write_json(Path(ctx.out_dir) / "repeat_map.json", rep)


# ------------------------------------------------------------------------------------------------ vocal


def vocal_params(cfg: Any) -> dict[str, Any]:
    return {"vocal_active_rel_db": float(_conf(cfg).s35.vocal_active_rel_db)}


def _sep_stem(ctx: Any, name: str) -> Path:
    sep_dir = Path(ctx.dep_dirs["sep"])
    p = sep_dir / "stems" / f"{name}.wav"
    if not p.is_file():
        raise GridStageError(f"sep 단계에 {name} 스템이 없습니다: {p}")
    return p


def run_vocal(ctx: Any) -> None:
    from threadpoolctl import threadpool_limits

    from bandscribe.analysis import vocal

    grid = atomic.read_json(Path(ctx.dep_dirs["grid"]) / "grid.json")
    with threadpool_limits(limits=1):
        doc = vocal.vocal_activity(_sep_stem(ctx, "vocals"), mix_path(ctx), grid["bars"],
                                   float(ctx.params["vocal_active_rel_db"]))
    atomic.write_json(Path(ctx.out_dir) / "vocal_activity.json", doc)


# ----------------------------------------------------------------------------------------------- resid1


def resid1_params(cfg: Any) -> dict[str, Any]:
    return {"resid_min_occurrences": int(_conf(cfg).s35.resid_min_occurrences)}


def guitar_view(stems_dir: Path) -> Path:
    """The ``guitar_mono`` view of the ``stems`` stage (path from ``views.json``, default location otherwise)."""
    views = Path(stems_dir) / "views.json"
    rel = "views/guitar_mono.wav"
    if views.is_file():
        entry = (atomic.read_json(views).get("views") or {}).get("guitar_mono") or {}
        rel = entry.get("path") or rel
    p = Path(stems_dir) / rel
    if not p.is_file():
        raise GridStageError(f"stems 단계에 guitar_mono 뷰가 없습니다: {p}")
    return p


def run_resid1(ctx: Any) -> None:
    from threadpoolctl import threadpool_limits

    from bandscribe import audio
    from bandscribe.analysis import repetition

    grid = atomic.read_json(Path(ctx.dep_dirs["grid"]) / "grid.json")
    sec_dir = Path(ctx.dep_dirs["sections"])
    secs = atomic.read_json(sec_dir / "sections.json")
    rep = atomic.read_json(sec_dir / "repeat_map.json")
    x, sr = repetition.read_mono(guitar_view(Path(ctx.dep_dirs["stems"])))
    with threadpool_limits(limits=1):
        out, doc = repetition.residual(x, sr, grid, secs, rep, int(ctx.params["resid_min_occurrences"]))
    del x
    audio.write_f32(Path(ctx.out_dir) / "repeat_resid_pass1.wav", out, sr)
    atomic.write_json(Path(ctx.out_dir) / "resid1.json", doc)


STAGES: dict[str, dict] = {
    "beats": {"run": run_beats, "code_version": "1", "params": beats_params, "device": "gpu",
              "models": beats_models},
    "grid": {"run": run_grid, "code_version": "2", "params": grid_params, "device": "cpu",
             "models": lambda cfg: {}},
    "sections": {"run": run_sections, "code_version": "2", "params": sections_params, "device": "cpu",
                 "models": lambda cfg: {}},
    "vocal": {"run": run_vocal, "code_version": "1", "params": vocal_params, "device": "cpu",
              "models": lambda cfg: {}},
    "resid1": {"run": run_resid1, "code_version": "1", "params": resid1_params, "device": "cpu",
               "models": lambda cfg: {}},
}
