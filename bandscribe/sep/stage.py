"""Pipeline stages ``sep`` (GPU, sep_msst worker) and ``stems`` (CPU derivations) - M1_M2_SPEC 3.1 / 3.2 rows 4-5.

``STAGES`` follows the stage contract of M1_M2_SPEC 3.1; ``bandscribe.pipeline.graph`` (GRID) owns names and deps
(``stems`` depends on ``sep``) and imports this module lazily.

- ``sep``: params exactly ``ckpt_sha256, config_sha256, msst_commit, chunk_ladder, num_overlap, batch_size,
  dtype, input_lufs``; models = sha256 of the checkpoint and the yaml (``ModelMissing`` at plan time if absent).
  Outputs ``stems/{bass,drums,other,vocals,guitar,piano}.wav``, ``sep.json``, ``gpu_run.json`` (+ ``_worker/``).
- ``stems``: params ``leftover_warn_db`` only; outputs of ``bandscribe.sep.derive``. Byte-deterministic data outputs.

The mix is ``store.input_dir(song_key)/mix_44k_f32.wav``. VRAM waits and the leftover warning go to the
runner's progress callback when the context carries one (``ctx.progress(event, info)``), and to the log.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.sep import derive
from bandscribe.sep import run as seprun

log = logging.getLogger(__name__)

MIX_NAME = "mix_44k_f32.wav"


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


# -------------------------------------------------------------------------------------------- sep


def sep_params(cfg: Any) -> dict[str, Any]:
    return seprun.sep_params(cfg)


def sep_models(cfg: Any) -> dict[str, str]:
    return seprun.sep_models(cfg)


def run_sep(ctx: Any) -> None:
    params = dict(ctx.params)
    job = {"song_key": ctx.song_key, "stage": ctx.stage, "stage_key": ctx.key}
    seprun.run_separation(mix_path(ctx), Path(ctx.out_dir), ctx.config, params, job=job,
                          notify=_notifier(ctx))


# ------------------------------------------------------------------------------------------ stems


def stems_params(cfg: Any) -> dict[str, Any]:
    return {"leftover_warn_db": float(seprun.sep_value(cfg, "sep.leftover_warn_db"))}


def run_stems(ctx: Any) -> None:
    sep_dir = Path(ctx.dep_dirs["sep"])
    stats = derive.derive(mix_path(ctx), sep_dir / "stems", Path(ctx.out_dir),
                          leftover_warn_db=float(ctx.params["leftover_warn_db"]))
    msg = derive.leftover_warning(stats)
    if msg:
        _notifier(ctx, "warning")(msg)


def stem_stats(stage_dir: Path) -> dict[str, Any] | None:
    """Read ``stem_stats.json`` of a finished ``stems`` stage (for the run summary)."""
    p = Path(stage_dir) / "stem_stats.json"
    return atomic.read_json(p) if p.is_file() else None


STAGES: dict[str, dict] = {
    "sep": {"run": run_sep, "code_version": "1", "params": sep_params, "device": "gpu", "models": sep_models},
    "stems": {"run": run_stems, "code_version": "1", "params": stems_params, "device": "cpu",
              "models": lambda cfg: {}},
}
