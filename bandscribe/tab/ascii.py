"""Plain-text tab (for a quick look and the report; the real outputs are .gp5 / alphaTex, M3 export).

One column per 16th of a quarter beat (``TICK_RES`` = 12 ticks; triplet and 32nd positions go to the nearest
column, so the text is only a sketch of the rhythm), bars separated by ``|``, ``bars_per_line`` bars per system,
string names from the tuning (string 1 = highest on top).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

TICK_RES = 12
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def string_names(tuning: Sequence[int]) -> list[str]:
    """Names of strings 1..n (highest first): lower-case for the top string as tab convention ("e")."""
    names = [NAMES[int(p) % 12] for p in reversed(tuning)]
    if len(names) == 6 and names[0] == "E" and names[-1] == "E":
        names[0] = "e"
    return names


def render(notes: Sequence[Mapping[str, Any]], bars: Sequence[Mapping[str, Any]], tuning: Sequence[int], *,
           first_bar: int | None = None, last_bar: int | None = None, bars_per_line: int = 4,
           title: str = "") -> str:
    """``notes`` = [{"bar", "tick", "string", "fret"}] (unfretted notes skipped); ``bars`` = quant.json bars."""
    n = len(tuning)
    names = string_names(tuning)
    width = max(len(x) for x in names)
    blen = {int(b["bar"]): int(b["n_beats"]) * 48 for b in bars}
    cells: dict[int, dict[int, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    for nt in notes:
        if nt.get("string") is None:
            continue
        col = int(round(int(nt["tick"]) / TICK_RES))
        prev = cells[int(nt["bar"])][col].get(int(nt["string"]))
        txt = str(int(nt["fret"]))
        cells[int(nt["bar"])][col][int(nt["string"])] = txt if prev is None else prev
    used = sorted(cells)
    if not used:
        return (title + "\n" if title else "") + "(음 없음)\n"
    lo = first_bar if first_bar is not None else used[0]
    hi = last_bar if last_bar is not None else used[-1]
    out = [title] if title else []
    bar_list = list(range(lo, hi + 1))
    for k in range(0, len(bar_list), bars_per_line):
        chunk = bar_list[k:k + bars_per_line]
        rows = [f"{names[s - 1]:>{width}}|" for s in range(1, n + 1)]
        head = " " * (width + 1)
        for b in chunk:
            ncol = max(1, blen.get(b, 192) // TICK_RES)
            seg = ["" for _ in range(n)]
            for c in range(ncol):
                here = cells.get(b, {}).get(c, {})
                w = max([len(v) for v in here.values()] + [1])
                for s in range(1, n + 1):
                    v = here.get(s)
                    seg[s - 1] += (v if v is not None else "-" * w).ljust(w, "-") + "-"
            label = str(b)
            head += label + " " * (len(seg[0]) + 1 - len(label))
            for s in range(n):
                rows[s] += seg[s] + "|"
        out.append(head.rstrip())
        out.extend(rows)
        out.append("")
    return "\n".join(out)
