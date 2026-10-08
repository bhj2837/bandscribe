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


def _marks(notes: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    """Text marks after a fret (M5 bass techniques): ``~`` vibrato, a backslash / ``/`` for a slide down / up into
    the next note on the same string (or out of the note)."""
    order = sorted((i for i, n in enumerate(notes) if n.get("string") is not None),
                   key=lambda i: (int(notes[i]["bar"]), int(notes[i]["tick"])))
    nxt: dict[int, int] = {}
    last: dict[int, int] = {}
    for i in order:
        s = int(notes[i]["string"])
        if s in last:
            nxt[last[s]] = i
        last[s] = i
    out: dict[int, str] = {}
    for i in order:
        t = notes[i].get("tech") or {}
        m = "~" if t.get("vibrato") else ""
        if t.get("slide") == "legato" and i in nxt:
            m += "\\" if int(notes[nxt[i]]["fret"]) < int(notes[i]["fret"]) else "/"
        elif t.get("slide") in ("out_down", "out_up"):
            m += "\\" if t["slide"] == "out_down" else "/"
        if m:
            out[i] = m
    return out


def render(notes: Sequence[Mapping[str, Any]], bars: Sequence[Mapping[str, Any]], tuning: Sequence[int], *,
           first_bar: int | None = None, last_bar: int | None = None, bars_per_line: int = 4,
           title: str = "") -> str:
    """``notes`` = [{"bar", "tick", "string", "fret"}] (unfretted notes skipped); ``bars`` = quant.json bars."""
    n = len(tuning)
    names = string_names(tuning)
    width = max(len(x) for x in names)
    blen = {int(b["bar"]): int(b["n_beats"]) * 48 for b in bars}
    cells: dict[int, dict[int, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    marks = _marks(notes)
    for k, nt in enumerate(notes):
        if nt.get("string") is None:
            continue
        # nearest column, but never past the bar's last one (a 32nd at tick 186 of a 4/4 bar rounds to column 16,
        # which a 16-column bar does not draw; review 2026-10-05)
        ncol = max(1, blen.get(int(nt["bar"]), 192) // TICK_RES)
        col = min(int(round(int(nt["tick"]) / TICK_RES)), ncol - 1)
        prev = cells[int(nt["bar"])][col].get(int(nt["string"]))
        txt = ("x" if (nt.get("tech") or {}).get("dead") else str(int(nt["fret"]))) + marks.get(k, "")
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
