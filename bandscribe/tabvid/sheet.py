"""Read tab bars as a sheet part (the JSON of the extracted band scores, so ``bandscribe.eval.sheets`` can score against
it), and a check of one tab against another (docs/TAB_VIDEO.md)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

GUITAR = (40, 45, 50, 55, 59, 64)
TUNINGS = {"guitar": GUITAR, "bass": (28, 33, 38, 43), "bass-dropd": (26, 33, 38, 43), "bass5": (23, 28, 33, 38, 43)}


def tuning(spec: str, n_strings: int) -> tuple[int, ...]:
    """MIDI pitches of the open strings, low to high: a name of ``TUNINGS``, numbers ('26,33,38,43'), or '' for the
    standard tuning of ``n_strings`` (6 guitar, 4 bass, 5 five-string bass)."""
    spec = spec.strip().lower()
    if not spec:
        spec = {6: "guitar", 4: "bass", 5: "bass5"}.get(n_strings, "")
    if spec in TUNINGS:
        out = TUNINGS[spec]
    else:
        try:
            out = tuple(int(x) for x in spec.split(","))
        except ValueError:
            raise ValueError(f"튜닝을 모릅니다: '{spec}' ({', '.join(TUNINGS)} 또는 MIDI 번호)") from None
    if len(out) != n_strings:
        raise ValueError(f"튜닝 {out} 은 {len(out)}줄인데 탭은 {n_strings}줄입니다")
    return out


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


def score(part: dict[str, Any], tempo: Any, est: Sequence[tuple[float, int, str]], *, tuning: Sequence[int] = GUITAR,
          bars: set[int] | None = None, block: int = 8) -> dict[str, Any]:
    """Our notes ``est`` ((onset s, pitch, line), by onset, on ``tempo``: a job's TempoMap) against a tab-video part, by
    the sheet evaluation's bar bags (``bandscribe.eval.sheets``): the best beat offset in -16..16 (a video's bar 1 may
    be a count-in), then drifting offsets. ``bars``: the part's bars to score (read, agreeing frames). A video tab often
    covers one player's version only (one electric guitar for two), so its notes missed by us (recall) mean more than
    our notes it lacks. ``blocks``: recall and precision per ``block`` bars."""
    from bandscribe.eval import sheets as E

    n = len(tuning)
    b0 = E.beat0(part)
    sheet = [E.SheetBar(b["bar"], b0[b["bar"]], b["numerator"],
                        [tuning[n - x["string"]] + x["fret"] for c in b["columns"] for x in c["notes"]
                         if not x["dead"] and not x["tied"] and x["fret"] is not None])
             for b in part["bars"] if bars is None or b["bar"] in bars]
    if not sheet:
        raise ValueError("채점할 마디가 없습니다")
    f1 = {o: E.bag_counts(tempo, [sheet], est, lambda _b, o=o: o).f1 for o in range(-16, 17)}
    best = max(f1, key=lambda o: (f1[o], -abs(o)))
    local, changes = E.drifting_offsets(tempo, [sheet], est, best)
    single = E.bag_counts(tempo, [sheet], est, lambda _b: best)
    drift = E.bag_counts(tempo, [sheet], est, lambda b: local.get(b, best))
    out = []
    for i in range(0, len(sheet), block):
        c = E.bag_counts(tempo, [sheet[i:i + block]], est, lambda b: local.get(b, best))
        out.append({"bar": sheet[i].bar, "last": sheet[min(i + block, len(sheet)) - 1].bar, "tab_notes": c.tp + c.fn,
                    "R": c.doc()["R"], "P": c.doc()["P"]})
    return {"bars": len(sheet), "offset": best, "single": single.doc(), "drifting": drift.doc(), "changes": changes,
            "blocks": out}
