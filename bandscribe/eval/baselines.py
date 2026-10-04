"""Baselines included in every report (DESIGN 8.5).

- ``majority``: every assigned note goes to the largest GT line — the floor of macro assignment accuracy.
- ``no_split``: the guitar-stem transcription as one line (the "분할 없음" baseline).
- ``b0``: guitar classes of the full-mix transcription as one line (B0, the floor; S3.5's mix pass).
The symbolic-split baseline (TabForge-style) comes with M6.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from bandscribe.eval.metrics import UNASSIGNED, Event

GUITAR_CLASSES: tuple[str, ...] = ("acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar")


def largest_line(gt_events: Sequence[Event]) -> str:
    """GT line with the most events (an event counts for every line it is valid for); ties -> first by name."""
    c: Counter[str] = Counter()
    for e in gt_events:
        c.update(e.lines)
    if not c:
        raise ValueError("no GT lines")
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def majority(pred: Any, target: str | Sequence[Event], *, system: str = "majority") -> Any:
    """Copy of ``pred`` with every assigned note moved to one line (``target`` or the largest GT line)."""
    from bandscribe.schema.evalio import PredNote, Prediction

    line = target if isinstance(target, str) else largest_line(target)
    notes = [PredNote(**{**n.model_dump(), "line": line if n.line != UNASSIGNED else UNASSIGNED, "shared": False})
             for n in pred.notes]
    return Prediction(format="bandscribe.prediction/1", system=system, item=pred.item,
                      lines=[{"id": line, "name": line}], notes=notes, meta={**pred.meta, "baseline": "majority"})


def prediction_from_notes(notes: Iterable[Any], *, system: str, item: str, line: str,
                          instruments: Iterable[str] | None = None, meta: dict | None = None) -> Any:
    """One-line Prediction from NoteEvents (optionally only those whose ``instrument`` is listed)."""
    from bandscribe.schema.evalio import PredNote, Prediction

    keep = set(instruments) if instruments is not None else None
    pn = [PredNote(onset_s=n.onset_s, offset_s=max(n.offset_s, n.onset_s), pitch=n.pitch, line=line,
                   posterior=getattr(n, "confidence", None))
          for n in notes if keep is None or getattr(n, "instrument", None) in keep]
    pn.sort(key=lambda n: (n.onset_s, n.pitch, n.offset_s))
    return Prediction(format="bandscribe.prediction/1", system=system, item=item, lines=[{"id": line, "name": line}],
                      notes=pn, meta=dict(meta or {}))


def no_split(noteset: Any, *, item: str, line: str = "all", system: str = "no_split") -> Any:
    """Guitar-stem transcription as a single line (guitar classes only when the backend tags classes)."""
    tagged = any(getattr(n, "instrument", None) for n in noteset.notes)
    return prediction_from_notes(noteset.notes, system=system, item=item, line=line,
                                 instruments=GUITAR_CLASSES if tagged else None,
                                 meta={"baseline": "no_split", "view": noteset.view})


def b0(noteset_mix: Any, *, item: str, line: str = "all", system: str = "b0") -> Any:
    """Guitar classes of the mix transcription merged into one line (baseline B0)."""
    return prediction_from_notes(noteset_mix.notes, system=system, item=item, line=line, instruments=GUITAR_CLASSES,
                                 meta={"baseline": "b0", "view": noteset_mix.view})
