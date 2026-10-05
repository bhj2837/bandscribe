"""The M2 stage graph for ``bandscribe run ... --until notes`` (M1_M2_SPEC 3.1-3.2).

This module owns stage **names and deps**; the implementations come from the ``STAGES`` dicts of
``bandscribe.analysis.stage`` (GRID), ``bandscribe.sep.stage`` (SEP) and ``bandscribe.amt.stage`` (AMT), imported lazily and
only for the stages a run needs, so ``bandscribe run --until grid`` works before separation or transcription
exist, and a stage whose provider is missing fails with a Korean message instead of an ImportError.

List order = DAG order; with M0's depth-first ``topo_order`` it puts the GPU stages in the order
beats -> sep -> amt_ms1 -> amt_gtr (DESIGN 9.5).

Deviation from the §3.2 table, deliberate: ``vocal`` also depends on ``sep``. It reads ``stems/vocals.wav``
from the ``sep`` stage dir, and a stage may only read its own dep dirs (``stems`` does not re-export the vocals
stem). The key is unaffected in substance: ``stems`` already depends on ``sep``.

``amt_ms1`` also depends on ``grid`` and ``sections`` (2026-10-04): the sampled piano+other presence pass picks
its windows at bar starts, spread over the sections (``bandscribe.amt.presence``). Whole-view transcriptions (bass,
eval's mix and full piano+other) come from the shared transcription cache, so a new grid does not re-transcribe
them.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from typing import Any

from bandscribe.jobs.dag import Stage

log = logging.getLogger(__name__)

# (name, deps, provider module) in DAG order.
STAGE_TABLE: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("beats", (), "bandscribe.analysis.stage"),
    ("grid", ("beats",), "bandscribe.analysis.stage"),
    ("sections", ("grid",), "bandscribe.analysis.stage"),
    ("sep", (), "bandscribe.sep.stage"),
    ("stems", ("sep",), "bandscribe.sep.stage"),
    ("vocal", ("stems", "grid", "sep"), "bandscribe.analysis.stage"),
    ("resid1", ("stems", "grid", "sections"), "bandscribe.analysis.stage"),
    ("amt_ms1", ("stems", "grid", "sections"), "bandscribe.amt.stage"),
    ("amt_bp", ("stems",), "bandscribe.amt.stage"),
    ("instr", ("amt_ms1", "stems", "sections", "grid"), "bandscribe.amt.stage"),
    ("amt_gtr", ("stems", "instr"), "bandscribe.amt.stage"),
    ("s35", ("instr", "vocal", "resid1", "amt_ms1", "amt_gtr", "amt_bp"), "bandscribe.amt.stage"),
    ("notes", ("amt_gtr", "amt_ms1", "amt_bp", "grid", "s35"), "bandscribe.amt.stage"),
    # M3 (CPU only): A4 reference, quantisation
    ("a4", ("stems",), "bandscribe.tab.stage"),
    ("quant", ("notes", "amt_ms1", "grid", "sections"), "bandscribe.tab.stage"),
)

NAMES: tuple[str, ...] = tuple(n for n, _, _ in STAGE_TABLE)
DEPS: dict[str, tuple[str, ...]] = {n: d for n, d, _ in STAGE_TABLE}
PROVIDER: dict[str, str] = {n: m for n, _, m in STAGE_TABLE}
DEFAULT_UNTIL = "notes"


class StageNotImplemented(RuntimeError):
    """A needed stage has no implementation yet (Korean message)."""


class UnknownStage(ValueError):
    pass


def check_name(name: str, what: str = "단계") -> str:
    if name not in DEPS:
        raise UnknownStage(f"알 수 없는 {what} '{name}' 입니다. 가능한 단계: {', '.join(NAMES)}")
    return name


def ancestors(name: str) -> set[str]:
    """``name`` and everything it depends on, transitively."""
    check_name(name)
    seen: set[str] = set()
    todo = [name]
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo.extend(DEPS[n])
    return seen


def descendants(names: Iterable[str]) -> set[str]:
    """``names`` and every stage downstream of them."""
    out = set(names)
    changed = True
    while changed:
        changed = False
        for n, deps in DEPS.items():
            if n not in out and out.intersection(deps):
                out.add(n)
                changed = True
    return out


def selected(until: str = DEFAULT_UNTIL) -> list[str]:
    """Stage names needed for ``until``, in DAG order."""
    want = ancestors(until)
    return [n for n in NAMES if n in want]


def _provider(module: str) -> dict[str, dict]:
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        log.debug("stage provider %s not importable: %s", module, e)
        return {}
    stages = getattr(mod, "STAGES", None)
    return stages if isinstance(stages, dict) else {}


def implementations(names: Iterable[str]) -> dict[str, dict]:
    """``STAGES`` entries for ``names``; raises StageNotImplemented listing every missing one."""
    cache: dict[str, dict[str, dict]] = {}
    out: dict[str, dict] = {}
    missing: list[str] = []
    for n in names:
        mod = PROVIDER[check_name(n)]
        if mod not in cache:
            cache[mod] = _provider(mod)
        impl = cache[mod].get(n)
        if impl is None:
            missing.append(f"단계 '{n}' 가 아직 구현되지 않았습니다 ({mod})")
        else:
            out[n] = impl
    if missing:
        raise StageNotImplemented("\n".join(missing))
    return out


def build_stages(cfg: Any, until: str = DEFAULT_UNTIL, *, with_models: bool = False) -> list[Stage]:
    """``list[Stage]`` for the stages ``until`` needs (params evaluated from cfg).

    with_models=True also evaluates each stage's ``models(cfg)`` (may raise ``bandscribe.models.ModelMissing``);
    the runner does that itself so it can report all missing models at once.
    """
    names = selected(until)
    impls = implementations(names)
    stages = []
    for n in names:
        impl = impls[n]
        stages.append(Stage(name=n, deps=list(DEPS[n]), run=impl["run"], code_version=str(impl.get("code_version", "1")),
                            params=dict(impl["params"](cfg)), device=impl.get("device", "cpu"),
                            models=dict(impl["models"](cfg)) if with_models and impl.get("models") else {}))
    return stages
