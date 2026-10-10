"""Router stage (M6; ``bandscribe.pipeline.graph`` owns names and deps):

- ``router`` (CPU, a few seconds): ``router.json`` - for every window of the grid (4 bars, cut at sections) the
  stereo features of the separated guitar stem, the scenario label (R0-R3, ``mixed`` or ``silent``) with its rule
  scores, p and margin, and the solo gate v0 (``bandscribe.router.route``). Reads the ``sep`` stage's stereo
  ``stems/guitar.wav``, ``grid.json``, ``sections.json``, the ``vocal`` stage's per-bar activity and the ``resid1``
  stage's per-bar repetition residual.

Nothing downstream reads it yet (M6b's views and part split will), so a default ``run`` does not compute it:
``bandscribe run <song> --until router``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.router import route as RT

log = logging.getLogger(__name__)

ROUTER_FILE = "router.json"
FORMAT = "bandscribe.router/1"
CODE_VERSION = "1"


def router_params(cfg: Any) -> dict[str, Any]:
    """``[router]`` -> route params (thresholds merged over the synthetic starting values)."""
    from bandscribe.config import Config

    r = (cfg if isinstance(cfg, Config) else Config.model_validate(cfg)).router
    return {"window_bars": int(r.window_bars), "silence_db": float(r.silence_db),
            "thresholds": {**RT.THRESHOLDS, **dict(r.thresholds)},
            "solo_coh_rise": float(r.solo_coh_rise), "solo_resid_db": float(r.solo_resid_db),
            "solo_vocal_max": float(r.solo_vocal_max)}


def _per_bar(doc: dict[str, Any] | None, key: str) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for row in (doc or {}).get("per_bar") or []:
        v = row.get(key)
        out[int(row["bar"])] = None if v is None else float(v)
    return out


def run_router(ctx: Any) -> None:
    import soundfile as sf

    dep = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    stem = dep["sep"] / "stems" / "guitar.wav"
    grid = atomic.read_json(dep["grid"] / "grid.json")
    sections = atomic.read_json(dep["sections"] / "sections.json")
    vocal_path = dep["vocal"] / "vocal_activity.json"
    resid_path = dep["resid1"] / "resid1.json"
    vocal = atomic.read_json(vocal_path) if vocal_path.is_file() else None
    resid = atomic.read_json(resid_path) if resid_path.is_file() else None
    x, sr = sf.read(str(stem), dtype="float32", always_2d=True)
    if x.shape[1] == 1:  # a mono stem has no stereo image: every window is narrow by construction
        x = x.repeat(2, axis=1)
    params = dict(ctx.params)
    body = RT.route_song(x[:, :2], int(sr), grid.get("bars") or [], sections.get("sections") or [], params,
                         vocal_per_bar=_per_bar(vocal, "active_ratio"),
                         resid_per_bar=_per_bar(resid, "resid_rel_db"))
    doc = {"format": FORMAT, "params": params, **body}
    atomic.write_json(Path(ctx.out_dir) / ROUTER_FILE, doc)
    s = body["summary"]
    log.info("router: %s (solo windows %d)", ", ".join(f"{k} {v}" for k, v in sorted(s["windows"].items())),
             s["solo_windows"])


STAGES: dict[str, dict] = {
    "router": {"run": run_router, "code_version": CODE_VERSION, "params": router_params, "device": "cpu",
               "models": lambda cfg: {}},
}
