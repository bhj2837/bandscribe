"""Repeat consensus (DESIGN S7 "반복 합의", M8; E29 in docs/decisions.md).

The occurrences of a repeat group (``repeat_map.json``) are laid side by side bar by bar: bar ``k`` of every
occurrence, a note's position = its beat position inside that bar (the grid's beats, the occurrence's ``offset_s``
taken off its onsets). Two notes are the same when they have the same pitch and lie within ``tol_beats``.

- **Voters** of occurrence ``o`` at bar ``k``: the other occurrences that hold at least ``sim_min`` of ``o``'s notes
  in that bar (``o`` needs ``min_notes`` notes there). The repeat map comes from the whole mix, so this checks that
  the bars really are the same guitar music before they vote.
- **Added:** a note position that ``o`` lacks and at least ``max(2, ceil(add_frac x voters))`` voters hold (two
  voters at least) is added to ``o`` at the voters' median position and length, with their most common
  instrument (``added: "repeat"``, ``support`` = holders / voters), unless ``o`` already has that pitch within
  ``dup_s`` of it.
- **Lone:** a note of ``o`` that none of its (two or more) voters holds is a variation: DESIGN keeps it and only
  marks it (``lone`` indices) for the review queue and the confidence model (M10).

Bars whose beat counts differ between occurrences (a pickup) are skipped; groups need ``min_occurrences``.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULTS: dict[str, Any] = {"min_occurrences": 3, "min_notes": 3, "sim_min": 0.7, "add_frac": 1.0, "tol_beats": 0.125,
                            "dup_s": 0.05}
ADDED = "repeat"


class BeatMap:
    """Seconds <-> fractional beat index over the grid's beats (linear between beats, edge spacing outside)."""

    def __init__(self, grid: Mapping[str, Any]) -> None:
        self.t = [float(b["t_s"]) for b in grid.get("beats") or []]

    def pos(self, t: float) -> float:
        ts = self.t
        i = min(max(bisect.bisect_right(ts, t) - 1, 0), len(ts) - 2)
        return i + (t - ts[i]) / (ts[i + 1] - ts[i])

    def time(self, p: float) -> float:
        ts = self.t
        i = min(max(int(math.floor(p)), 0), len(ts) - 2)
        return ts[i] + (p - i) * (ts[i + 1] - ts[i])


def _holds(cell: Sequence[tuple], pitch: int, pos: float, tol: float) -> bool:
    return any(c[1] == pitch and abs(c[2] - pos) <= tol for c in cell)


def consensus(notes: Sequence[Mapping[str, Any]], grid: Mapping[str, Any], repeat: Mapping[str, Any] | None,
              params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """-> {"added": [note dicts], "lone": [indices into ``notes``], "report": {...}}; ``notes`` are not changed."""
    p = {**DEFAULTS, **(params or {})}
    tol = float(p["tol_beats"])
    out: dict[str, Any] = {"added": [], "lone": [],
                           "report": {"groups": 0, "cells": 0, "voted_cells": 0, "added": 0, "lone": 0}}
    bm = BeatMap(grid)
    bars = {int(b["bar"]): b for b in grid.get("bars") or []}
    if len(bm.t) < 2 or not repeat:
        return out
    order = sorted(range(len(notes)), key=lambda i: float(notes[i]["onset_s"]))
    onsets = [float(notes[i]["onset_s"]) for i in order]
    lone: set[int] = set()
    for g in repeat.get("groups") or []:
        occs = list(g.get("occurrences") or [])
        if len(occs) < int(p["min_occurrences"]):
            continue
        out["report"]["groups"] += 1
        common = min(int(o["n_bars"]) for o in occs)
        for k in range(common):
            bs = [bars.get(int(o["start_bar"]) + k) for o in occs]
            if any(b is None for b in bs) or len({int(b.get("n_beats") or 0) for b in bs}) != 1:
                continue
            cells = []
            for o, b in zip(occs, bs):
                off = float(o.get("offset_s") or 0.0)
                start, end = float(b["start_s"]), float(b["end_s"])
                p0 = bm.pos(start)
                lo, hi = bisect.bisect_left(onsets, start + off), bisect.bisect_left(onsets, end + off)
                cell = []
                for j in order[lo:hi]:
                    n = notes[j]
                    a = bm.pos(float(n["onset_s"]) - off)
                    z = bm.pos(max(float(n["onset_s"]), float(n.get("offset_s") or n["onset_s"])) - off)
                    cell.append((j, int(n["pitch"]), a - p0, max(0.0, z - a), n.get("instrument")))
                cells.append({"off": off, "p0": p0, "notes": cell})
            for oi, c in enumerate(cells):
                out["report"]["cells"] += 1
                mine = c["notes"]
                if len(mine) < int(p["min_notes"]):
                    continue
                voters = []
                for vi, v in enumerate(cells):
                    if vi == oi or not v["notes"]:
                        continue
                    hit = sum(1 for x in mine if _holds(v["notes"], x[1], x[2], tol))
                    if hit / len(mine) >= float(p["sim_min"]):
                        voters.append(vi)
                if len(voters) < 2:
                    continue
                out["report"]["voted_cells"] += 1
                for x in mine:
                    if not any(_holds(cells[vi]["notes"], x[1], x[2], tol) for vi in voters):
                        lone.add(x[0])
                need = max(2, math.ceil(float(p["add_frac"]) * len(voters) - 1e-9))
                missing = sorted(((x[1], x[2], vi, x) for vi in voters for x in cells[vi]["notes"]
                                  if not _holds(mine, x[1], x[2], tol)), key=lambda r: (r[0], r[1]))
                placed: list[tuple] = []
                i = 0
                while i < len(missing):
                    pitch, anchor = missing[i][0], missing[i][1]
                    grp = []
                    while i < len(missing) and missing[i][0] == pitch and missing[i][1] - anchor <= tol:
                        grp.append(missing[i])
                        i += 1
                    by_voter = {}
                    for r in grp:
                        by_voter.setdefault(r[2], r[3])
                    if len(by_voter) < need:
                        continue
                    pos = statistics.median(x[2] for x in by_voter.values())
                    if _holds(placed, pitch, pos, tol):
                        continue
                    dur = statistics.median(x[3] for x in by_voter.values())
                    inst = Counter(x[4] for x in by_voter.values()).most_common(1)[0][0]
                    on = bm.time(c["p0"] + pos) + c["off"]
                    off_s = bm.time(c["p0"] + pos + dur) + c["off"]
                    # review 2026-10-18: a note the occurrence already has within dup_s (the matching tolerance of
                    # the evaluation) is not missing - the beat tolerance alone let 24 % of E29's additions double one
                    lo = bisect.bisect_left(onsets, on - float(p["dup_s"]))
                    hi = bisect.bisect_right(onsets, on + float(p["dup_s"]))
                    if any(int(notes[j]["pitch"]) == pitch for j in order[lo:hi]):
                        continue
                    placed.append((None, pitch, pos))
                    note = {"onset_s": round(on, 6), "offset_s": round(max(off_s, on + 0.01), 6), "pitch": pitch,
                            "added": ADDED, "support": round(len(by_voter) / len(voters), 3)}
                    if inst is not None:
                        note["instrument"] = inst
                    out["added"].append(note)
    out["added"].sort(key=lambda n: (n["onset_s"], n["pitch"]))
    out["lone"] = sorted(lone)
    out["report"]["added"] = len(out["added"])
    out["report"]["lone"] = len(out["lone"])
    return out
