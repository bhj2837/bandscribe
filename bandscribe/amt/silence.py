"""Silence gate of the ``notes`` stage (2026-10-05): transcribed notes that start where their stem is silent.

MuScriptor writes notes over a separated stem that is (near) digital silence, often the same note or chord over
and over. Measured on the five local songs (``docs/decisions.md``, 2026-10-05):

- seisyun complex: 81 G1 over bars 5-17, where the Guitar Pro score's bass rests and the bass stem sits at
  -109 dBFS (the mix at -13). The onsets do not follow the kick (12 % within 40 ms of one, chance 13 %).
- 怪獣の花唄: 92 bass notes and ~690 guitar notes (one A chord, 79 times) in stretches where that stem is
  60-100 dB below the mix; 青と夏: 33 E1 after the song has ended and ~800 guitar notes in guitar-less stretches.
- Real notes are far louder: every bass note of seisyun complex that matches the score starts within 12 dB of
  the mix's loud level; no bass note of any song lies between -50 and -40 dB.

A note is dropped when its view's stem level (RMS dB at 10 Hz from the ``stems`` stage's ``energy.npz``) stays
below ``reference_db + gate_db`` over ``onset + window_s[0]`` .. ``onset + window_s[1]``. The reference is the
mix's 95th-percentile frame level, so a song-long silence or a gap in a quiet song is caught as well (a
reference relative to the local mix would call silence over silence "active").
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

# view id -> the stems (energy.npz series) it is made of (``bandscribe.sep.derive`` views.json "source")
VIEW_STEMS: dict[str, tuple[str, ...]] = {
    "bass_mono": ("bass",),
    "guitar_mono": ("guitar",),
    "guitar_other_mono": ("guitar", "other"),
    "piano_other_mono": ("piano", "other"),
    "nonvox_mono": ("nonvox",),
    "mix_mono": ("mix",),
}
DEFAULT_GATE_DB = -50.0
DEFAULT_WINDOW_S = (-0.03, 0.12)
REFERENCE_PERCENTILE = 95.0
FLOOR_DB = -200.0  # energy.npz's floor (RMS of digital silence)


def reference_db(energy: Mapping[str, Any]) -> float | None:
    """The mix's loud level: 95th percentile of its 10 Hz frame levels (None without a mix series)."""
    mix = energy.get("mix")
    if mix is None or len(mix) == 0:
        return None
    return float(np.percentile(np.asarray(mix, dtype=np.float64), REFERENCE_PERCENTILE))


def view_level_db(energy: Mapping[str, Any], view: str) -> np.ndarray | None:
    """Frame levels (dB) of a view: the power sum of its stems' series (None: unknown view or missing stem)."""
    stems = VIEW_STEMS.get(view)
    if not stems or any(energy.get(s) is None for s in stems):
        return None
    power = sum(10.0 ** (np.asarray(energy[s], dtype=np.float64) / 10.0) for s in stems)
    return 10.0 * np.log10(power + 1e-20)


def onset_levels(notes: Iterable[Mapping[str, Any]], level_db: np.ndarray, fps: float,
                 window_s: Sequence[float] = DEFAULT_WINDOW_S) -> list[float]:
    """Per note: the loudest frame of ``level_db`` overlapping ``onset + window_s[0]`` .. ``onset + window_s[1]``
    (``FLOOR_DB`` past the end of the series)."""
    n = len(level_db)
    out = []
    for x in notes:
        t = float(x["onset_s"])
        a = max(0, math.floor((t + float(window_s[0])) * fps))
        z = min(n, max(a + 1, math.ceil((t + float(window_s[1])) * fps)))
        out.append(float(np.max(level_db[a:z])) if z > a else FLOOR_DB)
    return out


def gate(notes: Sequence[Mapping[str, Any]], energy: Mapping[str, Any], view: str, fps: float, *,
         gate_db: float = DEFAULT_GATE_DB, window_s: Sequence[float] = DEFAULT_WINDOW_S
         ) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """(kept notes in input order, report). Nothing is dropped when the view's level is unknown (``report["gated"]``
    is then False and ``report["reason"]`` says why)."""
    if float(gate_db) <= FLOOR_DB:
        return list(notes), {"gated": False, "reason": "off", "dropped": 0}
    ref = reference_db(energy)
    level = view_level_db(energy, view)
    if ref is None or level is None:
        why = "no mix level" if ref is None else f"no stem level for view {view!r}"
        return list(notes), {"gated": False, "reason": why, "dropped": 0}
    floor = ref + float(gate_db)
    lv = onset_levels(notes, level, fps, window_s)
    kept = [x for x, v in zip(notes, lv) if v >= floor]
    dropped = [(x, v) for x, v in zip(notes, lv) if v < floor]
    return kept, {"gated": True, "view": view, "reference_db": round(ref, 2), "floor_db": round(floor, 2),
                  "dropped": len(dropped), "spans_s": _spans([float(x["onset_s"]) for x, _ in dropped])}


def _spans(onsets: list[float], gap_s: float = 2.0) -> list[list[float]]:
    """Dropped onsets merged into spans (a gap over ``gap_s`` starts a new span), for the summary."""
    out: list[list[float]] = []
    for t in sorted(onsets):
        if out and t - out[-1][1] <= gap_s:
            out[-1][1] = round(t, 3)
        else:
            out.append([round(t, 3), round(t, 3)])
    return out
