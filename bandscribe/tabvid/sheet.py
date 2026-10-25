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
            bent = bool(b.get("bend") and i < len(b["bend"]) and b["bend"][i])
            slid = bool(b.get("slide") and i < len(b["slide"]) and b["slide"][i])
            cols.append({"phase": ph, "bend": bent, "slide": slid,
                         "notes": [{"string": int(s), "fret": None if f == "x" else int(f), "dead": f == "x",
                                    "tied": False} for s, f in col]})
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
          bars: set[int] | None = None, block: int = 8, scale: float = 1.0, search: int = 16) -> dict[str, Any]:
    """Our notes ``est`` ((onset s, pitch, line), by onset, on ``tempo``: a job's TempoMap) against a tab-video part, by
    the sheet evaluation's bar bags (``bandscribe.eval.sheets``): the best beat offset in ``-search..search`` (a
    video's bar 1 may be a count-in; a cut version starts later), then drifting offsets. ``scale``: the tab's beats per
    beat of ours (2 when the tab counts a beat at twice our tempo). ``bars``: the part's bars to score (read, agreeing
    frames). A video tab often covers one player's version only (one electric guitar for two), so its notes missed by
    us (recall) mean more than our notes it lacks. ``blocks``: recall and precision per ``block`` bars."""
    from bandscribe.eval import sheets as E

    n = len(tuning)
    b0 = E.beat0(part)
    sheet = [E.SheetBar(b["bar"], b0[b["bar"]] / scale, b["numerator"] / scale,  # type: ignore[arg-type]
                        [tuning[n - x["string"]] + x["fret"] for c in b["columns"] for x in c["notes"]
                         if not x["dead"] and not x["tied"] and x["fret"] is not None])
             for b in part["bars"] if bars is None or b["bar"] in bars]
    if not sheet:
        raise ValueError("채점할 마디가 없습니다")
    f1 = {o: E.bag_counts(tempo, [sheet], est, lambda _b, o=o: o).f1 for o in range(-search, search + 1)}
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


def score_time(part: dict[str, Any], times: dict[int, float], path: Sequence[tuple[float, float]], tempo: Any,
               est: Sequence[tuple[float, int, str]], *, tuning: Sequence[int] = GUITAR, bars: set[int] | None = None,
               block: int = 8) -> dict[str, Any]:
    """``score`` by time instead of a beat offset: tab bar n plays from video second ``times[n]`` to ``times[n + 1]``
    (``video.bar_times``), the alignment ``path`` (``align.align``) carries that to song seconds and our tempo map to
    our beats. A cut version, a song played twice in one video and a tab counted at another tempo need nothing more.
    Page times are only as fine as the frames: they give the bar length in our beats (of the two ``BAR_BEATS`` values
    nearest the measured one, the one that fits better) and where each ``block`` of bars lies; the block is then laid
    evenly from the start within four beats that fits best, or - when that fits worse, as with unsure page numbering -
    each bar stays at its own time, moved by up to a beat."""
    import numpy as np

    from bandscribe.eval import sheets as E
    from bandscribe.tabvid import align as A

    n = len(tuning)
    seq = sorted(x for x, t in times.items() if t is not None)
    steps = [times[b] - times[a] for a, b in zip(seq, seq[1:]) if b == a + 1 and times[b] > times[a]]
    dur = float(np.median(steps)) if steps else 2.0

    def bars_at(lead: float) -> list:
        out = []
        for b in part["bars"]:
            num = b["bar"]
            t0 = times.get(num)
            if t0 is None or (bars is not None and num not in bars):
                continue
            t1 = times.get(num + 1)
            if t1 is None or t1 <= t0:
                t1 = t0 + dur
            g0 = float(tempo.time_to_beat(A.song_time(list(path), t0 + lead)))
            g1 = float(tempo.time_to_beat(A.song_time(list(path), t1 + lead)))
            if g1 <= g0:
                continue
            out.append(E.SheetBar(num, g0, g1 - g0,  # type: ignore[arg-type]
                                  [tuning[n - x["string"]] + x["fret"] for c in b["columns"] for x in c["notes"]
                                   if not x["dead"] and not x["tied"] and x["fret"] is not None]))
        return out

    # pages turn a little before (or after) their bars sound: the one lead over the whole video that fits best
    leads = [round(0.25 * k, 2) for k in range(-12, 13)]
    fit = {}
    for lead in leads:
        sh = bars_at(lead)
        fit[lead] = E.bag_counts(tempo, [sh], est, lambda _b: 0).f1 if sh else -1.0
    lead = max(leads, key=lambda x: (fit[x], -abs(x)))
    raw = bars_at(lead)
    if not raw:
        raise ValueError("채점할 마디가 없습니다(마디 시간이 없음)")
    # page times are coarse: they give the bar length in our beats (rounded to a plain value) and where each block of
    # bars lies; within a block the bars are then laid evenly from the start that fits best within two beats
    measured = float(np.median([b.beats for b in raw]))

    def laid(blk: list, start: float, length: float) -> list:
        return [E.SheetBar(b.bar, start + (b.bar - blk[0].bar) * length, length, b.pitches)  # type: ignore[arg-type]
                for b in blk]

    def fitted(length: float) -> tuple[float, list, list, dict]:
        plain, sheet, moved = [], [], {}
        for i in range(0, len(raw), block):
            blk = raw[i:i + block]
            base = float(np.median([b.beat0 - (b.bar - blk[0].bar) * length for b in blk]))
            # laid evenly from a start within four beats, or each bar at its own time moved by up to a beat (when
            # the page numbering is unsure, the bars' own times are the better guess)
            cands = [(abs(0.5 * k), 0.5 * k, laid(blk, base + 0.5 * k, length)) for k in range(-8, 9)]
            cands += [(abs(0.5 * k) + 0.25, 0.5 * k, [E.SheetBar(b.bar, b.beat0 + 0.5 * k, b.beats, b.pitches)
                                                       for b in blk]) for k in range(-2, 3)]
            f1 = [E.bag_counts(tempo, [c[2]], est, lambda _b: 0).f1 for c in cands]
            pick = max(range(len(cands)), key=lambda q: (f1[q], -cands[q][0]))
            plain += laid(blk, base, length)
            sheet += cands[pick][2]
            for b in blk:
                moved[b.bar] = cands[pick][1]
        return E.bag_counts(tempo, [sheet], est, lambda _b: 0).f1, plain, sheet, moved

    # the bar length in our beats: the two plain values nearest the measured one, the one that fits better
    near = sorted(BAR_BEATS, key=lambda x: abs(x - measured))[:2]
    tries = {length: fitted(length) for length in near}
    length = max(near, key=lambda x: (tries[x][0], -abs(x - measured)))
    _f, plain, sheet, moved = tries[length]
    single = E.bag_counts(tempo, [plain], est, lambda _b: 0)
    drift = E.bag_counts(tempo, [sheet], est, lambda _b: 0)
    out = []
    for i in range(0, len(sheet), block):
        c = E.bag_counts(tempo, [sheet[i:i + block]], est, lambda _b: 0)
        out.append({"bar": sheet[i].bar, "last": sheet[min(i + block, len(sheet)) - 1].bar, "tab_notes": c.tp + c.fn,
                    "R": c.doc()["R"], "P": c.doc()["P"]})
    return {"bars": len(sheet), "offset": None, "lead_s": lead, "single": single.doc(), "drifting": drift.doc(),
            "changes": [{"bar": b, "shift": d} for b, d in sorted(moved.items()) if d], "blocks": out,
            "beats_per_bar": length, "measured_beats_per_bar": round(measured, 2)}


BAR_BEATS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
