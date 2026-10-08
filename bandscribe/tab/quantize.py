"""Quantisation of performed notes onto the bar/beat grid (DESIGN 6.2; M3).

Positions are integer ticks, ``TPB`` (48) per beat of the bar's beat unit (``grid.json``), counted on the global
beat index of the grid's TempoMap: ``gtick = 48 * beat_index + tick_in_beat`` ("global tick"); bar and tick in
the bar come from the TempoMap's bars. Durations are ticks too.

**Subdivision per beat (Viterbi).** Every beat picks a family: binary (``bin``: k/8 of a quarter beat, the
32nd points, 6 ticks apart), ternary (``ter``: k/6, the 16th-triplet points, 8 ticks apart) or, in a
triplet-feel section, swung binary (``swing``: the binary points matched at their swung times, see below).
A note costs ``min(r², cap) + level penalty`` at the point of the family where ``r² + penalty`` is smallest,
r = onset residual / ``sigma_ms``; the penalty grows with the point's subdivision level (a beat point is free,
an 8th cheap, a 32nd or a 16th triplet dear), so a note takes a finer point only when its timing asks for it.
A beat whose notes use a non-beat point also pays the family's prior (ternary in simple meter, binary in
compound meter), and changing between binary and ternary from one beat to the next costs ``switch``. With
``joint`` (default) all tracks share one decision per beat (guitar and bass agree on triplets). A note belongs
to the beat ``floor(beat position)``; point 1 of a beat is the next beat's point 0.

**Durations** run to the note's offset quantised on the family of the beat holding the offset, and are at
least the finest step of the onset beat's family (a 32nd in binary quarter beats, DESIGN 6.2).

**Shuffle vs compound (song level).** The grid calls a beat compound when onset energy sits at 1/3 and 2/3 of
it rather than at 1/2 (``bandscribe.analysis.grid`` step 6), which a 4/4 shuffle (onsets at 0 and 2/3, never
1/3) triggers as well. So when the grid gives dotted-quarter beats in 4-beat bars (12/8) with a compound or
ambiguous decision, and the transcribed onsets play 2/3 but rarely 1/3 (<= ``third_max`` x the 2/3 count) and
rarely 1/2, those bars become 4/4 with triplet feel; otherwise compound stands. The other reading is kept in
``meter.alternatives`` (UI toggle, E14). 6/8 and 9/8 are never re-read (a long-short 6/8 stays 6/8).

**Triplet feel (per section, quarter beats).** Off-beat onsets that crowd at 2/3 of the beat (>= ``swing_min``
of them, >= ``swing_ratio`` x those at 1/2, and those at 1/3 <= ``third_max`` x) give the section the
``triplet8th`` feel: it is written straight (the swung 8th at tick 24) and its binary points are matched at
their swung times (beat fraction f of a straight point is played at 4/3 f up to 1/2, 2/3 + 2/3 (f - 1/2) after).
A shuffle-read song keeps the feel in every section that does not clearly play straight 8ths.

Thresholds are provisional (E14). Pure numpy, deterministic.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bandscribe.schema.grid import Grid, TempoMap

TPB = 48
FAMILIES = ("bin", "ter", "swing")
KIND = {"bin": 0, "swing": 0, "ter": 1}  # switching penalty applies between kinds
# points per beat (k / D of the beat, k = 0..D) of each family, by beat unit
DENOM: dict[str, dict[str, int]] = {
    "quarter": {"bin": 8, "ter": 6, "swing": 8},
    "dotted_quarter": {"bin": 4, "ter": 6, "swing": 4},
    "eighth": {"bin": 4, "ter": 3, "swing": 4},
}
# the subdivision levels of each family: a point's level is the first one that contains it (the middle of a
# ternary beat, 3/6, is a sextuplet point, level 6, not an 8th)
HIERARCHY: dict[str, tuple[int, ...]] = {"bin": (1, 2, 4, 8), "swing": (1, 2, 4, 8), "ter": (1, 3, 6)}
# penalty of a point by its subdivision level, by beat unit
LEVEL_PEN: dict[str, dict[int, float]] = {
    "quarter": {1: 0.0, 2: 0.3, 4: 0.9, 8: 2.5, 3: 0.6, 6: 2.0},
    "dotted_quarter": {1: 0.0, 3: 0.3, 6: 0.9, 2: 1.2, 4: 2.5},
    "eighth": {1: 0.0, 2: 0.6, 4: 2.0, 3: 1.5},
}
# prior of a family for a beat that uses a non-beat point
FAMILY_PRIOR: dict[str, dict[str, float]] = {
    "quarter": {"bin": 0.0, "swing": 0.0, "ter": 1.0},
    "dotted_quarter": {"bin": 1.5, "swing": 1.5, "ter": 0.0},
    "eighth": {"bin": 0.0, "swing": 0.0, "ter": 1.5},
}
DEFAULT_PARAMS: dict[str, Any] = {
    # onset timing noise (performance + transcription). 30 ms (2026-10-05, provisional until E14): synthetic
    # ±20 ms exact (bar, tick) 0.990 (22 ms: 0.994), while the real songs' p90 residual is 23-36 ms and 22 ms
    # turned 6-18 % of their guitar notes into 32nds (30 ms: 0-11 %, the rest mostly a half-tempo grid)
    "sigma_ms": 30.0,
    "cap": 9.0,           # a residual beyond 3 sigma costs no more (outliers)
    "switch": 1.0,        # binary <-> ternary between neighbouring beats (an isolated triplet beat pays twice)
    "joint": True,        # one family per beat for all tracks
    "swing_min": 8,       # off-beat onsets at 2/3 needed to call triplet feel / shuffle
    "swing_ratio": 2.0,   # ... at least this many times those at 1/2
    "third_max": 0.35,    # ... and those at 1/3 at most this share of them
    "feel_window": 0.06,  # beat fraction counted as "at" 1/3, 1/2, 2/3
    "dedupe_ms": 30.0,    # onsets closer than this (a chord, both tracks) count once for feel and meter
}


def swung(f: Any) -> np.ndarray:
    """Played beat fraction of a straight (written) one under 8th triplet feel."""
    f = np.asarray(f, dtype=float)
    return np.where(f <= 0.5, f * 4.0 / 3.0, 2.0 / 3.0 + (f - 0.5) * 2.0 / 3.0)


@dataclass(frozen=True)
class _Points:
    ticks: np.ndarray   # written tick in the beat of each point (k * TPB / D)
    played: np.ndarray  # beat fraction the point is played at
    pen: np.ndarray
    level: np.ndarray


def _level(k: int, d: int, fam: str) -> int:
    return next(h for h in HIERARCHY[fam] if (k * h) % d == 0)


def _points(fam: str, unit: str) -> _Points:
    d = DENOM[unit][fam]
    k = np.arange(d + 1)
    level = np.array([_level(int(x), d, fam) for x in k])
    pen = np.array([LEVEL_PEN[unit].get(int(lv), 3.0) for lv in level])
    frac = k / d
    return _Points(ticks=(k * (TPB // d)).astype(int), played=swung(frac) if fam == "swing" else frac, pen=pen,
                   level=level)


_POINTS = {(f, u): _points(f, u) for f in FAMILIES for u in DENOM}


def _allowed(unit: str, swing: bool) -> tuple[str, ...]:
    return ("swing", "ter") if unit == "quarter" and swing else ("bin", "ter")


def default_family(unit: str, swing: bool = False) -> str:
    """Family of a beat no Viterbi decision covers (before the first or after the last onset): the one without
    a prior for the beat unit (``bandscribe.tab.notation`` assumes the same)."""
    if unit == "dotted_quarter":
        return "ter"
    return "swing" if unit == "quarter" and swing else "bin"


def _choose(fr: np.ndarray, ms: float, pts: _Points, sigma: float) -> np.ndarray:
    """Index of the point each beat fraction goes to (lowest r² + penalty; ties to the lower point)."""
    r = (pts.played[None, :] - fr[:, None]) * ms / sigma
    return np.argmin(r * r + pts.pen[None, :], axis=1)


def _beat_cost(fr: np.ndarray, ms: float, fam: str, unit: str, p: Mapping[str, Any]) -> float:
    if fr.size == 0:
        return 0.0
    pts = _POINTS[(fam, unit)]
    sigma = float(p["sigma_ms"])
    c = _choose(fr, ms, pts, sigma)
    r = (pts.played[c] - fr) * ms / sigma
    cost = float(np.sum(np.minimum(r * r, float(p["cap"])) + pts.pen[c]))
    if np.any(pts.level[c] > 1):
        cost += FAMILY_PRIOR[unit][fam]
    return cost


# ------------------------------------------------------------------------------------------- bar context


class _Bars:
    """Bar -> (numerator, denominator, beat unit, n_beats) after the meter decision, with nearest-bar fallback
    for bars outside the grid (notes before the first or after the last beat)."""

    def __init__(self, bars: Sequence[Mapping[str, Any]]) -> None:
        self.rows = {int(b["bar"]): dict(b) for b in bars}
        self.keys = sorted(self.rows)

    def get(self, bar: int) -> dict[str, Any]:
        if bar in self.rows:
            return self.rows[bar]
        if not self.keys:
            return {"bar": bar, "numerator": 4, "denominator": 4, "beat_unit": "quarter", "n_beats": 4}
        i = bisect.bisect_left(self.keys, bar)
        return self.rows[self.keys[min(max(i - 1, 0), len(self.keys) - 1) if i > 0 else 0]]


def _onset_fracs(tm: TempoMap, onsets: np.ndarray, dedupe_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(beat index, fraction in the beat) of onsets merged within ``dedupe_s`` (a chord counts once)."""
    t = np.sort(np.asarray(onsets, dtype=float))
    keep: list[float] = []
    for x in t:
        if not keep or x - keep[-1] > dedupe_s:
            keep.append(float(x))
    g = np.asarray(tm.time_to_beat(np.asarray(keep)), dtype=float) if keep else np.zeros(0)
    b = np.floor(g)
    return b.astype(int), g - b


def _counts(fr: np.ndarray, win: float) -> dict[str, int]:
    return {"third": int(np.sum(np.abs(fr - 1 / 3) < win)), "half": int(np.sum(np.abs(fr - 0.5) < win)),
            "two_thirds": int(np.sum(np.abs(fr - 2 / 3) < win))}


def _swings(c: Mapping[str, int], p: Mapping[str, Any]) -> bool:
    t23 = c["two_thirds"]
    return (t23 >= int(p["swing_min"]) and t23 >= float(p["swing_ratio"]) * c["half"]
            and c["third"] <= float(p["third_max"]) * t23)


def _straight(c: Mapping[str, int], p: Mapping[str, Any]) -> bool:
    return c["half"] >= int(p["swing_min"]) and c["half"] >= float(p["swing_ratio"]) * c["two_thirds"]


def decide_meter_and_feel(grid: Mapping[str, Any], tm: TempoMap, onsets: np.ndarray,
                          sections: Sequence[Mapping[str, Any]] | None, params: Mapping[str, Any] | None = None
                          ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """(bars with the decided meter and ``feel``, meter document, section feel rows)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    win = float(p["feel_window"])
    bars = [dict(b) for b in grid["bars"]]
    beat_idx, fr = _onset_fracs(tm, onsets, float(p["dedupe_ms"]) / 1000.0)
    bar_of_beat = np.array([tm._bar_of_beat(float(b))[0] for b in beat_idx], dtype=int) if beat_idx.size else \
        np.zeros(0, dtype=int)
    off = (fr > 0.15) & (fr < 0.85)

    comp = ((grid.get("hypotheses") or {}).get("compound") or {})
    dq = [b for b in bars if b["beat_unit"] == "dotted_quarter"]
    full = [b["n_beats"] for b in dq if not b.get("pickup")]
    typical = max(set(full), key=lambda n: (full.count(n), n)) if full else None
    meter_doc: dict[str, Any] = {"grid_decision": comp.get("decision"), "grid_triple_ratio": comp.get("triple_ratio"),
                                 "read_as": None, "counts": None, "alternatives": []}
    if dq and typical == 4 and comp.get("decision") in ("compound", "ambiguous"):
        sel = np.isin(bar_of_beat, [b["bar"] for b in dq]) & off
        c = _counts(fr[sel], win)
        meter_doc["counts"] = c
        if _swings(c, p):
            meter_doc["read_as"] = "shuffle"
            for b in bars:
                if b["beat_unit"] == "dotted_quarter":
                    b.update(numerator=int(b["n_beats"]), denominator=4, beat_unit="quarter")
            meter_doc["alternatives"] = [{"numerator": 12, "denominator": 8, "beat_unit": "dotted_quarter"}]
        else:
            meter_doc["read_as"] = "compound"
            meter_doc["alternatives"] = [{"numerator": 4, "denominator": 4, "beat_unit": "quarter",
                                          "feel": "triplet8th"}]

    # triplet feel per section (quarter-beat bars only)
    secs = [dict(s) for s in (sections or [])]
    if not secs and bars:
        secs = [{"id": "all", "start_bar": bars[0]["bar"], "end_bar": bars[-1]["bar"] + 1}]
    shuffle = meter_doc["read_as"] == "shuffle"
    rows: list[dict[str, Any]] = []
    feel_of_bar: dict[int, str | None] = {}
    for s in secs:
        a, e = int(s["start_bar"]), int(s["end_bar"])
        quarter = [b["bar"] for b in bars if a <= b["bar"] < e and b["beat_unit"] == "quarter"]
        sel = np.isin(bar_of_beat, quarter) & off
        c = _counts(fr[sel], win)
        feel = None
        if quarter and (_swings(c, p) or (shuffle and not _straight(c, p))):
            feel = "triplet8th"
        rows.append({"id": s.get("id"), "start_bar": a, "end_bar": e, "feel": feel, "counts": c})
        for b in range(a, e):
            feel_of_bar[b] = feel
    # bars outside every section (before the first one, a trailing bar): the nearest section's feel
    keys = sorted(feel_of_bar)
    for b in bars:
        if b["bar"] in feel_of_bar:
            b["feel"] = feel_of_bar[b["bar"]]
        elif keys:
            i = bisect.bisect_left(keys, b["bar"])
            b["feel"] = feel_of_bar[keys[min(i, len(keys) - 1)]]
        else:
            b["feel"] = None
        if b["beat_unit"] != "quarter":
            b["feel"] = None
    return bars, meter_doc, rows


# ------------------------------------------------------------------------------------------------ main


def _beat_ctx(tm: TempoMap, bars: _Bars, b: int) -> tuple[str, bool, float]:
    """(beat unit, swing allowed, beat length ms) of global beat ``b``."""
    bar, _ = tm._bar_of_beat(float(b))
    row = bars.get(int(bar))
    t0, t1 = (float(x) for x in tm.beat_to_time(np.array([b, b + 1], dtype=float)))
    return str(row["beat_unit"]), row.get("feel") == "triplet8th", (t1 - t0) * 1000.0


def _viterbi(costs: list[dict[str, float]], switch: float) -> list[str]:
    """Family per beat minimising Σ beat cost + switch x (kind changes)."""
    n = len(costs)
    if n == 0:
        return []
    inf = float("inf")
    dp = [{f: costs[0].get(f, inf) for f in FAMILIES}]
    back: list[dict[str, str]] = [{}]
    for i in range(1, n):
        row: dict[str, float] = {}
        bk: dict[str, str] = {}
        for f in FAMILIES:
            c = costs[i].get(f, inf)
            if c == inf:
                row[f] = inf
                continue
            best, arg = inf, FAMILIES[0]
            for g in FAMILIES:  # fixed order: ties go to bin, then ter, then swing
                v = dp[-1][g] + (switch if KIND[g] != KIND[f] else 0.0)
                if v < best:
                    best, arg = v, g
            row[f], bk[f] = best + c, arg
        dp.append(row)
        back.append(bk)
    last = min(FAMILIES, key=lambda f: (dp[-1][f], FAMILIES.index(f)))
    path = [last]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return path[::-1]


# note fields the quantiser carries over unchanged (M5 bass: techniques, provenance, the anchor's move)
PASS_KEYS = ("tech", "src", "conf", "f0", "from_pitch", "merged")


def quantize(tracks: Mapping[str, Sequence[Mapping[str, Any]]], grid: Mapping[str, Any] | Grid,
             sections: Sequence[Mapping[str, Any]] | None = None, params: Mapping[str, Any] | None = None
             ) -> dict[str, Any]:
    """``quant.json`` document for ``tracks`` = {track id: [{"onset_s", "offset_s", "pitch", ...}]}."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    gdoc = grid.model_dump(mode="json") if isinstance(grid, Grid) else dict(grid)
    tm = TempoMap.from_dict(gdoc)
    all_on = np.array([float(n["onset_s"]) for ns in tracks.values() for n in ns], dtype=float)
    bars, meter_doc, sec_rows = decide_meter_and_feel(gdoc, tm, all_on, sections, p)
    bmap = _Bars(bars)

    # beat of every onset, per track
    per_track: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tid, ns in tracks.items():
        on = np.array([float(n["onset_s"]) for n in ns], dtype=float)
        g = np.asarray(tm.time_to_beat(on), dtype=float) if on.size else np.zeros(0)
        per_track[tid] = (np.floor(g).astype(int), g - np.floor(g))
    used = sorted({int(b) for bi, _f in per_track.values() for b in bi})
    decisions: dict[str, dict[int, str]] = {}
    groups = [list(tracks)] if p["joint"] else [[t] for t in tracks]
    for grp in groups:
        if not used:
            continue
        lo, hi = used[0], used[-1]
        costs: list[dict[str, float]] = []
        for b in range(lo, hi + 1):
            unit, sw, ms = _beat_ctx(tm, bmap, b)
            fr = np.concatenate([per_track[t][1][per_track[t][0] == b] for t in grp]) if grp else np.zeros(0)
            costs.append({f: _beat_cost(fr, ms, f, unit, p) for f in _allowed(unit, sw)})
        path = _viterbi(costs, float(p["switch"]))
        for t in grp:
            decisions[t] = {b: path[b - lo] for b in range(lo, hi + 1)}

    sigma = float(p["sigma_ms"])
    out_tracks: dict[str, Any] = {}
    for tid, ns in tracks.items():
        bi, fr = per_track[tid]
        fams = decisions.get(tid, {})
        rows: list[dict[str, Any]] = []
        for k, n in enumerate(ns):
            b = int(bi[k])
            unit, sw, ms = _beat_ctx(tm, bmap, b)
            fam = fams.get(b) or default_family(unit, sw)
            pts = _POINTS[(fam, unit)]
            c = int(_choose(fr[k:k + 1], ms, pts, sigma)[0])
            g_on = b * TPB + int(pts.ticks[c])
            resid_ms = (fr[k] - float(pts.played[c])) * ms
            # offset on the family grid of its own beat
            g_off_f = float(np.asarray(tm.time_to_beat(float(n["offset_s"]))))
            bo = int(math.floor(g_off_f))
            u2, sw2, ms2 = _beat_ctx(tm, bmap, bo)
            fam2 = fams.get(bo) or default_family(u2, sw2)
            pts2 = _POINTS[(fam2, u2)]
            c2 = int(_choose(np.array([g_off_f - bo]), ms2, pts2, sigma)[0])
            g_off = bo * TPB + int(pts2.ticks[c2])
            # the shortest note is one step of the family of the beat that HOLDS the onset: a note rounded up to
            # the next beat's first point lives on that beat, whose family can differ (a binary 6-tick end inside
            # a triplet beat cannot be written; review 2026-10-05)
            b_on = g_on // TPB
            if b_on == b:
                u_on, f_on = unit, fam
            else:
                u_on, sw_on, _ms_on = _beat_ctx(tm, bmap, b_on)
                f_on = fams.get(b_on) or default_family(u_on, sw_on)
            step = TPB // DENOM[u_on][f_on]
            dur = max(g_off - g_on, step)
            bar, start_beat = tm._bar_of_beat(g_on / TPB)
            rows.append({"i": k, "onset_s": round(float(n["onset_s"]), 6), "offset_s": round(float(n["offset_s"]), 6),
                         "pitch": int(n["pitch"]), "instrument": n.get("instrument"), "gtick": int(g_on),
                         "dur": int(dur), "bar": int(bar), "tick": int(g_on - round(start_beat * TPB)),
                         "family": fam, "level": int(pts.level[c]), "resid_ms": round(float(resid_ms), 2),
                         **{key: n[key] for key in PASS_KEYS if key in n}})
        res = np.array([abs(r["resid_ms"]) for r in rows]) if rows else np.zeros(0)
        lv = [r["level"] for r in rows]
        out_tracks[tid] = {
            "notes": rows,
            "stats": {"n": len(rows), "mean_abs_resid_ms": round(float(res.mean()), 2) if res.size else None,
                      "p90_abs_resid_ms": round(float(np.percentile(res, 90)), 2) if res.size else None,
                      "levels": {str(v): lv.count(v) for v in sorted(set(lv))}},
        }
    # family of every decided beat, with or without an onset: a note END can fall in a beat that holds no onset,
    # and the speller must use the grid that end was quantised on ("*" = all tracks when joint)
    fam_rows: dict[str, list[list[Any]]] = {}
    n_beats = n_ter = 0
    for grp in groups:
        if not grp or grp[0] not in decisions:
            continue
        held = {int(b) for t in grp for b in per_track[t][0]}
        fam_rows["*" if p["joint"] else grp[0]] = [[b, f] for b, f in sorted(decisions[grp[0]].items())]
        n_beats += len(held)
        n_ter += sum(1 for b, f in decisions[grp[0]].items() if b in held and f == "ter")
    return {
        "format": "bandscribe.quant/1",
        "tpb": TPB,
        "params": {k: p[k] for k in DEFAULT_PARAMS},
        "meter": meter_doc,
        "sections": sec_rows,
        "bars": [{k: b[k] for k in ("bar", "start_s", "end_s", "n_beats", "numerator", "denominator", "beat_unit",
                                    "pickup", "feel") if k in b} for b in bars],
        "beat_families": fam_rows,
        "summary": {"beats_with_notes": n_beats, "ternary_beats": n_ter},
        "tracks": out_tracks,
    }


def bar_tick(tm: TempoMap, gtick: int) -> tuple[int, int]:
    """(bar, tick in the bar) of a global tick."""
    bar, start = tm._bar_of_beat(gtick / TPB)
    return int(bar), int(gtick - round(start * TPB))


def notes_of(doc: Mapping[str, Any], track: str) -> list[dict[str, Any]]:
    return list(((doc.get("tracks") or {}).get(track) or {}).get("notes") or [])


def exact_ratio(pairs: Iterable[tuple[tuple[int, int], tuple[int, int]]]) -> float:
    """Share of (estimated (bar, tick), true (bar, tick)) pairs that are equal."""
    pairs = list(pairs)
    return sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else float("nan")
