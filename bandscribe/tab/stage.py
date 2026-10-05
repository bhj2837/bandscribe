"""Tab pipeline stages (M3; ``bandscribe.pipeline.graph`` owns names and deps):

- ``a4`` (CPU): ``a4.json`` - the recording's A4 offset in cents from the pitched stem views
  (``bandscribe.tab.a4``); a Korean warning from 25 cents (varispeed is E16, not applied).
- ``quant`` (CPU): ``quant.json`` - ``notes/guitar_all.json`` (track ``guitar_all``) and the pass-1 bass
  transcription (track ``bass_raw``) quantised on the grid with one subdivision decision per beat, the meter
  re-read (shuffle vs 12/8) and the triplet feel per section (``bandscribe.tab.quantize``).

Stage functions read only their dep dirs and ``ctx.params`` (M1_M2_SPEC 3.1).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.tab import a4 as A4
from bandscribe.tab import quantize as Q

log = logging.getLogger(__name__)

A4_VIEWS = ("guitar_mono", "bass_mono", "piano_other_mono")
BASS_RAW = "raw/bass_mono__muscriptor.json"


class TabStageError(RuntimeError):
    """User-facing (Korean) problem of a tab stage."""


def _warn(ctx: Any, msg: str) -> None:
    from bandscribe.amt.stage import _notifier

    _notifier(ctx, "warning")(msg)


# ------------------------------------------------------------------------------------------------- a4


def a4_params(cfg: Any) -> dict[str, Any]:
    return dict(A4.DEFAULT_PARAMS)


def run_a4(ctx: Any) -> None:
    views_dir = Path(ctx.dep_dirs["stems"]) / "views"
    views = {v: A4.read_mono(views_dir / f"{v}.wav") for v in A4_VIEWS if (views_dir / f"{v}.wav").is_file()}
    if not views:
        raise TabStageError(f"stems 단계에 기준음을 잴 뷰가 없습니다: {views_dir}")
    doc = A4.estimate(views, dict(ctx.params))
    atomic.write_json(Path(ctx.out_dir) / "a4.json", doc)
    if doc["warning"]:
        _warn(ctx, doc["warning"])


# ---------------------------------------------------------------------------------------------- quant


def quant_params(cfg: Any) -> dict[str, Any]:
    return dict(Q.DEFAULT_PARAMS)


def quant_tracks(dep_dirs: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    """{track id: notes} the quantiser reads: guitar classes of the guitar pass, and the pass-1 bass."""
    gtr = atomic.read_json(Path(dep_dirs["notes"]) / "guitar_all.json").get("notes") or []
    bass_path = Path(dep_dirs["amt_ms1"]) / BASS_RAW
    bass = (atomic.read_json(bass_path).get("notes") or []) if bass_path.is_file() else []
    return {"guitar_all": gtr, "bass_raw": bass}


def run_quant(ctx: Any) -> None:
    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    grid = atomic.read_json(dd["grid"] / "grid.json")
    sections = atomic.read_json(dd["sections"] / "sections.json").get("sections") or []
    doc = Q.quantize(quant_tracks(dd), grid, sections, dict(ctx.params))
    atomic.write_json(Path(ctx.out_dir) / "quant.json", doc)


STAGES: dict[str, dict] = {
    "a4": {"run": run_a4, "code_version": "1", "params": a4_params, "device": "cpu", "models": lambda cfg: {}},
    "quant": {"run": run_quant, "code_version": "1", "params": quant_params, "device": "cpu",
              "models": lambda cfg: {}},
}
