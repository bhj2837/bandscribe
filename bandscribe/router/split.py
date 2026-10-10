"""Rule-based two-Line split (DESIGN 10 M6: "P 는 L/R, A 는 centre−S / S∪(L∩R), N 창은 합친 트랙"; M6b).

Each guitar note goes by the router label of the window its onset falls in and by its own stereo signature
(``bandscribe.router.notesig``):

- ``R3`` (P, parts at different positions): ILD >= ``p_ild_db`` -> line ``L``, <= -``p_ild_db`` -> line ``R``;
  in between the note sounds from both sides (a unison, or one part's bleed at a near-centre pan) and goes to
  both lines, ``shared`` (DESIGN R3: for P, a note heard in both channels is written in both Lines).
- ``R2`` (A, a wide same-part double + a centred part): a note of the double is two takes, so low coherence or a
  low mid/side ratio -> line ``double``; a coherent, centred note -> line ``centre``. A note that the double and
  the centred part both play (unison) shows the double's signature and stays in ``double`` (S8's job, R3).
- every other window (``R0`` narrow, ``R1`` ADT, ``mixed``, ``silent``) and every note without a signature: the
  merged line ``all`` (DESIGN: N windows stay merged; ADT is one part by R2 of the output contract).

``summary`` says how much of the song each scenario covers, so the caller can decide whether the split is worth
two tracks (DESIGN: separate tracks only when confident).
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_PARAMS: dict[str, float] = {
    "p_ild_db": 6.0,  # synthetic +-70 % pan = 11.8 dB, hard = 30 dB; bleed of a hard-panned part sits near 0
    "a_coh": 0.8,  # two takes of one note: 0.56 (synthetic, 8 ms / 6 cents apart); one centred take: 1.0
    "a_ms_db": 10.0,  # mid/side of a centred note: > 30 dB; of a double: ~3 dB
}
SPLIT_LINES = {"R3": ("L", "R"), "R2": ("double", "centre")}


def _window_of(starts: Sequence[float], t: float) -> int | None:
    i = bisect.bisect_right(starts, t) - 1
    return i if i >= 0 else None


def assign(notes: Sequence[Mapping[str, Any]], sigs: Sequence[Mapping[str, Any]],
           windows: Sequence[Mapping[str, Any]], params: Mapping[str, float] | None = None
           ) -> list[dict[str, Any]]:
    """-> per note {lines: [line ids], scenario: R3 | R2 | None, shared: bool} (module doc)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    wins = sorted(windows, key=lambda w: float(w["start_s"]))
    starts = [float(w["start_s"]) for w in wins]
    out = []
    for n, s in zip(notes, sigs):
        k = _window_of(starts, float(n["onset_s"]))
        w = wins[k] if k is not None and float(n["onset_s"]) < float(wins[k]["end_s"]) else None
        lab = w.get("label") if w else None
        ild, coh, ms = s.get("ild_db"), s.get("coh"), s.get("ms_db")
        if lab == "R3" and ild is not None:
            if ild >= p["p_ild_db"]:
                out.append({"lines": ["L"], "scenario": "R3", "shared": False})
            elif ild <= -p["p_ild_db"]:
                out.append({"lines": ["R"], "scenario": "R3", "shared": False})
            else:
                out.append({"lines": ["L", "R"], "scenario": "R3", "shared": True})
        elif lab == "R2" and coh is not None and ms is not None:
            centred = coh >= p["a_coh"] and ms >= p["a_ms_db"]
            out.append({"lines": ["centre" if centred else "double"], "scenario": "R2", "shared": False})
        else:
            out.append({"lines": ["all"], "scenario": None, "shared": False})
    return out


def summary(assigned: Sequence[Mapping[str, Any]], windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Notes and seconds per scenario, notes per line."""
    notes_by: dict[str, int] = {}
    lines: dict[str, int] = {}
    for a in assigned:
        sc = a["scenario"] or "merged"
        notes_by[sc] = notes_by.get(sc, 0) + 1
        for ln in a["lines"]:
            lines[ln] = lines.get(ln, 0) + 1
    secs: dict[str, float] = {}
    for w in windows:
        lab = w.get("label")
        sc = lab if lab in SPLIT_LINES else "merged"
        secs[sc] = round(secs.get(sc, 0.0) + float(w["end_s"]) - float(w["start_s"]), 3)
    total = len(assigned) or 1
    return {"notes": notes_by, "lines": lines, "seconds": secs,
            "share_split": round(sum(v for k, v in notes_by.items() if k in SPLIT_LINES) / total, 4),
            "shared": sum(1 for a in assigned if a["shared"])}
