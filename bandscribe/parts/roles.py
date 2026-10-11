"""Order and names of the Part view tracks (DESIGN 4.9 "Part 뷰", S9; M7a v0).

- ``Gtr 1`` is the Line active in the most bars (usually the rhythm part); the others are numbered by their first
  note, ties by position left -> right (DESIGN: "동률이면 팬 위치 왼쪽 → 오른쪽").
- A name is "number – role (class, position)", e.g. ``Gtr 1 – Rhythm (dist., L)``. Role: ``Rhythm`` when at least
  ``RHYTHM_CHORD_SHARE`` of the Line's notes are chord notes, ``Lead`` when at least ``LEAD_SHARE`` are lead-layer
  notes or most notes are single and high (median at or above ``LEAD_PITCH``), else ``Riff``. Class: the most
  frequent MuScriptor guitar class. Position (from the Line's seeds in ``R3`` windows): ``L`` / ``R`` beyond
  ``POS_SIDE``, else none is written (a part not heard at a side has no position worth naming).
- One Line (one guitar) is ``Gtr`` with no number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

RHYTHM_CHORD_SHARE = 0.4
LEAD_SHARE = 0.3
LEAD_PITCH = 64  # E4
POS_SIDE = 0.3
CLASS_SHORT = {"distorted_electric_guitar": "dist.", "clean_electric_guitar": "clean", "acoustic_guitar": "acoustic"}


def role_of(idx: Sequence[int], notes: Sequence[Mapping[str, Any]], feats: Sequence[Mapping[str, Any]]) -> str:
    if not idx:
        return "Rhythm"
    chord = sum(1 for i in idx if feats[i]["chord"]) / len(idx)
    lead = sum(1 for i in idx if feats[i]["lead"]) / len(idx)
    single = sum(1 for i in idx if feats[i]["size"] == 1) / len(idx)
    med = float(np.median([int(notes[i]["pitch"]) for i in idx]))
    if chord >= RHYTHM_CHORD_SHARE:
        return "Rhythm"
    if lead >= LEAD_SHARE or (single >= 0.5 and med >= LEAD_PITCH):
        return "Lead"
    return "Riff"


def order_and_names(profiles: Sequence[Mapping[str, Any]], line: Sequence[int | None],
                    notes: Sequence[Mapping[str, Any]], feats: Sequence[Mapping[str, Any]],
                    bar_of: Sequence[int]) -> list[dict[str, Any]]:
    """Per Line (in ``profiles`` order): {number, name, short, role, cls, position, notes, bars, first_s}, numbered
    by the module rules. ``bar_of`` = each note's bar (or any time cell)."""
    k = len(profiles)
    idx: list[list[int]] = [[] for _ in range(k)]
    for i, j in enumerate(line):
        if j is not None:
            idx[j].append(i)
    rows = []
    for j, pr in enumerate(profiles):
        ii = idx[j]
        classes = [str(notes[i].get("instrument")) for i in ii]
        top = sorted(set(classes), key=lambda c: (-classes.count(c), c))[0] if classes else None
        pos = float(pr.get("position") or 0.0)
        rows.append({"j": j, "key": pr["key"], "role": role_of(ii, notes, feats),
                     "cls": CLASS_SHORT.get(str(top), str(top) if top else ""),
                     "position": "L" if pos > POS_SIDE else "R" if pos < -POS_SIDE else "",
                     "pos_value": pos, "notes": len(ii), "bars": len({bar_of[i] for i in ii}),
                     "first_s": min((float(notes[i]["onset_s"]) for i in ii), default=float("inf"))})
    if not rows:
        return []
    first = max(rows, key=lambda r: (r["bars"], r["notes"], -r["first_s"], r["pos_value"]))
    rest = sorted((r for r in rows if r is not first), key=lambda r: (r["first_s"], -r["pos_value"]))
    out = []
    for num, r in enumerate([first, *rest], start=1):
        short = "Gtr" if k == 1 else f"Gtr {num}"
        detail = ", ".join(x for x in (r["cls"], r["position"]) if x)
        out.append({**r, "number": num, "short": short,
                    "name": f"{short} – {r['role']}" + (f" ({detail})" if detail else "")})
    return sorted(out, key=lambda r: r["j"])
