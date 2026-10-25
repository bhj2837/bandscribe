"""Read tab bars as a sheet part (the JSON of the extracted band scores, so ``bandscribe.eval.sheets`` can score against
it), and a check of one tab against another (docs/TAB_VIDEO.md)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

GUITAR = (40, 45, 50, 55, 59, 64)


def to_sheet_part(bars: Sequence[dict[str, Any]], *, n_strings: int = 6, numerator: int = 4,
                  source: str = "") -> dict[str, Any]:
    """The sheet JSON of ``eval.sheets`` (bars -> columns -> notes with string, fret, dead, tied). A video does not show
    the time signature to this reader: every bar gets ``numerator`` (the sheet evaluation only uses it for the bar's
    time span). Numbers missing from the video become empty bars so the bar count stays right."""
    by = {b["number"]: b for b in bars}
    out = []
    for num in range(min(by), max(by) + 1) if by else []:
        b = by.get(num)
        cols = []
        for i, col in enumerate(b["cols"] if b else []):
            ph = b["phases"][i] if b and i < len(b["phases"]) else (i + 0.5) / max(1, len(b["cols"]))
            cols.append({"phase": ph, "notes": [{"string": int(s), "fret": None if f == "x" else int(f),
                                                 "dead": f == "x", "tied": False} for s, f in col]})
        out.append({"bar": num, "numerator": numerator, "rest_only": not cols, "columns": cols,
                    "read": b is not None, "agree": b["agree"] if b else None})
    return {"source": source, "n_strings": n_strings, "bars": out}


def _bag(cols: Sequence[Sequence[tuple[int, Any]]], tuning: Sequence[int]) -> Counter:
    n = len(tuning)
    return Counter(tuning[n - int(s)] + int(f) for c in cols for s, f in c if str(f).isdigit())


def compare(a: dict[str, Any], b: dict[str, Any], *, tuning: Sequence[int] = GUITAR,
            offsets: range = range(-8, 9)) -> dict[str, Any]:
    """Bar-bag agreement of two sheet parts (pitch multisets per bar, dead and tied notes left out) at the best bar
    offset (bar n of ``a`` against bar n + offset of ``b``)."""
    def cols(part: dict[str, Any]) -> dict[int, list[list[tuple[int, int]]]]:
        return {x["bar"]: [[(n["string"], n["fret"]) for n in c["notes"] if not n["dead"] and not n["tied"]
                            and n["fret"] is not None] for c in x["columns"]] for x in part["bars"]}

    ca, cb = cols(a), cols(b)
    best = None
    for off in offsets:
        tp = fp = fn = same = 0
        for num, c in ca.items():
            have, want = _bag(c, tuning), _bag(cb.get(num + off, []), tuning)
            k = sum((have & want).values())
            tp, fp, fn = tp + k, fp + sum(have.values()) - k, fn + sum(want.values()) - k
            same += have == want
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if best is None or f1 > best["F1"]:
            best = {"offset": off, "F1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "bars": len(ca),
                    "bars_same": same}
    return best or {}
