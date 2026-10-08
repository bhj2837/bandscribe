"""Rhythm spelling: quantised events -> bars of written beats (note values, dots, tuplets, ties) (DESIGN 6.2).

One voice per track: every event (the notes starting on one global tick) lasts until its own end or the next
event, whichever comes first; gaps become rests. Segments are cut at bar lines and beat boundaries and spelled
inside each beat by its subdivision family (``quant.json`` ``beat_families``):
- binary quarter beat: 8 units of 6 ticks, split recursively at halves (a dotted value when a piece starts the
  half it fills 3/4 of): 16th then 8th, never a tied 8th across the middle of a beat;
- ternary quarter beat: a whole beat is a quarter; otherwise 8th triplets (``(3, 2)``, quarter triplets where
  aligned) or, when a 16th-triplet point is used, 16th sextuplets (``(6, 4)``);
- compound (dotted-quarter) beat: 8ths and 16ths without tuplets (dotted quarter, quarter, dotted 8th, 8th, 16th);
  its binary family (duplets) is written as dotted 8ths / dotted 16ths.
Full beats of one segment merge into a half, dotted half or whole (dotted half / dotted whole in compound meter)
when they start on the bar's first or a strong beat; a piece that starts on a beat and lasts 1.5 beats is a dotted
quarter. Later pieces of a split segment carry ``tie`` on every note.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

TPB = 48


@dataclass
class WNote:
    pitch: int
    string: int | None = None
    fret: int | None = None
    tie: bool = False  # tie destination (continues the same note from the previous beat)
    src: tuple[str, ...] = ()  # ids of the notes this written note stands for (M4a: a click finds its notes)
    fx: tuple[str, ...] = ()  # alphaTex note effects (M5 bass): x dead, g ghost, v vibrato, sl / sod / sou slides


@dataclass
class WBeat:
    start: int  # tick in the bar
    dur: int    # ticks
    value: int  # 1, 2, 4, 8, 16, 32, 64
    dots: int = 0
    tuplet: tuple[int, int] | None = None
    notes: list[WNote] = field(default_factory=list)  # empty = rest


@dataclass
class WBar:
    bar: int            # musical bar number (quant.json)
    start_gtick: int
    n_beats: int
    beat_unit: str
    numerator: int
    denominator: int
    start_s: float
    feel: str | None
    section: str | None = None
    beats: list[WBeat] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.n_beats * TPB


# ----------------------------------------------------------------------------------------- within a beat


def _bin_pieces(a: int, b: int, lo: int = 0, hi: int = 8) -> list[tuple[int, int, bool]]:
    """(start, end, dotted) in 6-tick units of a binary quarter beat."""
    if a >= b:
        return []
    if a == lo and b == hi:
        return [(lo, hi, False)]
    if a == lo and hi - lo >= 4 and (b - a) * 4 == (hi - lo) * 3:
        return [(a, b, True)]
    mid = (lo + hi) // 2
    out = []
    if a < mid:
        out += _bin_pieces(a, min(b, mid), lo, mid)
    if b > mid:
        out += _bin_pieces(max(a, mid), b, mid, hi)
    return out


_BIN_VALUE = {8: 4, 4: 8, 2: 16, 1: 32}           # units of 6 ticks -> note value
_BIN_DOTTED = {6: 8, 3: 16}                       # dotted 8th, dotted 16th


def _spell_binary(a: int, b: int) -> list[tuple[int, int, int, int, tuple[int, int] | None]]:
    """[(start tick, dur, value, dots, tuplet)] of the piece [a, b) ticks inside a binary quarter beat."""
    out = []
    for s, e, dotted in _bin_pieces(a // 6, b // 6):
        n = e - s
        if dotted:
            out.append((s * 6, n * 6, _BIN_DOTTED[n], 1, None))
        else:
            out.append((s * 6, n * 6, _BIN_VALUE[n], 0, None))
    return out


def _spell_ternary(a: int, b: int, fine: bool) -> list[tuple[int, int, int, int, tuple[int, int] | None]]:
    """Inside a ternary quarter beat: 8th triplets (16 ticks) or, with ``fine``, 16th sextuplets (8 ticks)."""
    if a == 0 and b == TPB:
        return [(0, TPB, 4, 0, None)]
    out = []
    if fine:
        for t in range(a, b, 8):
            out.append((t, 8, 16, 0, (6, 4)))
        return out
    t = a
    while t < b:
        if t == 0 and b - t >= 32:
            out.append((t, 32, 4, 0, (3, 2)))
            t += 32
        elif t == 16 and b - t >= 32:
            out.append((t, 32, 4, 0, (3, 2)))
            t += 32
        else:
            out.append((t, 16, 8, 0, (3, 2)))
            t += 16
    return out


def _spell_compound(a: int, b: int, binary: bool) -> list[tuple[int, int, int, int, tuple[int, int] | None]]:
    """Inside a dotted-quarter beat (48 ticks = three 8ths of 16)."""
    if binary:  # duplets: 24 = dotted 8th, 12 = dotted 16th
        out, t = [], a
        while t < b:
            step = 24 if t % 24 == 0 and b - t >= 24 else 12
            out.append((t, step, 8 if step == 24 else 16, 1, None))
            t += step
        return out
    if a == 0 and b == TPB:
        return [(0, TPB, 4, 1, None)]
    out, t = [], a
    while t < b:
        if t == 0 and b - t >= 32:
            out.append((t, 32, 4, 0, None))
            t += 32
        elif t % 16 == 0 and b - t >= 24 and t == 0:
            out.append((t, 24, 8, 1, None))
            t += 24
        elif t % 16 == 0 and b - t >= 16:
            out.append((t, 16, 8, 0, None))
            t += 16
        else:
            out.append((t, 8, 16, 0, None))
            t += 8
    return out


# --------------------------------------------------------------------------------------------- spelling


def bars_from_quant(quant: Mapping[str, Any], tm: Any, first_bar: int, last_bar: int,
                    sections: Sequence[Mapping[str, Any]] | None = None) -> list[WBar]:
    """Bars ``first_bar..last_bar`` with meter, feel and start time (``tm`` = the grid's TempoMap).

    Every bar's start and length come from the TempoMap, which also numbered the quantised notes, so the bars tile
    the global ticks exactly; bars outside the grid (notes before the first or after the last beat) are
    extrapolated the TempoMap's way (before a pickup bar: bars of the pickup's length) and take the nearest grid
    bar's beat unit and feel. A short bar (pickup, extrapolated) gets a time signature of its own length."""
    rows = {int(b["bar"]): b for b in quant["bars"]}
    keys = sorted(rows)
    sec_at = {int(s["start_bar"]): str(s.get("label") or s.get("id") or "") for s in (sections or [])}
    out = []
    for bar in range(first_bar, last_bar + 1):
        row = rows.get(bar) or rows[min(keys, key=lambda k: (abs(k - bar), k))]
        start_beat = tm._bar_start_beat(bar)
        n = int(round(tm._bar_start_beat(bar + 1) - start_beat))
        start_s = float(tm.beat_to_time(start_beat))
        unit = str(row["beat_unit"])
        num = int(row["numerator"]) if bar in rows and not row.get("pickup") and int(row["n_beats"]) == n else \
            (3 * n if unit == "dotted_quarter" else n)
        out.append(WBar(bar=bar, start_gtick=int(round(start_beat * TPB)), n_beats=n, beat_unit=unit,
                        numerator=num, denominator=int(row["denominator"]), start_s=max(0.0, start_s),
                        feel=row.get("feel"), section=sec_at.get(bar)))
    return out


def _merge_full_beats(bar: WBar, k: int, m: int) -> list[tuple[int, int, int, int]]:
    """(start tick, dur, value, dots) for m full beats starting at beat k of ``bar``."""
    compound = bar.beat_unit == "dotted_quarter"
    out, i = [], k
    while i < k + m:
        left = k + m - i
        if compound:
            if left >= 4 and i == 0 and bar.n_beats == 4:
                out.append((i * TPB, 4 * TPB, 1, 1))
                i += 4
            elif left >= 2 and i % 2 == 0:
                out.append((i * TPB, 2 * TPB, 2, 1))
                i += 2
            else:
                out.append((i * TPB, TPB, 4, 1))
                i += 1
            continue
        if left >= 4 and i == 0 and bar.n_beats == 4:
            out.append((0, 4 * TPB, 1, 0))
            i += 4
        elif left >= 3 and i == 0 and bar.n_beats in (3, 4):
            out.append((0, 3 * TPB, 2, 1))
            i += 3
        elif left >= 2 and i % 2 == 0:
            out.append((i * TPB, 2 * TPB, 2, 0))
            i += 2
        else:
            out.append((i * TPB, TPB, 4, 0))
            i += 1
    return out


def _grid_step(beat_unit: str, fam: str) -> int:
    """Ticks between the points of a family in a beat of ``beat_unit`` (the quantiser's grids)."""
    if beat_unit == "dotted_quarter":
        return 12 if fam == "bin" else 8
    if beat_unit == "eighth":
        return 16 if fam == "ter" else 12
    return 8 if fam == "ter" else 6


def _snap_segments(bar: WBar, segments: Sequence[tuple[int, int, list[WNote] | None, bool]],
                   family_of: Mapping[int, str]) -> list[tuple[int, int, list[WNote] | None, bool]]:
    """Every boundary on the grid of its beat's family (a safety net: the quantiser already puts them there).
    A shared boundary snaps the same way for both of its segments, so the bar stays tiled; an emptied segment
    goes."""
    first_beat_g = bar.start_gtick // TPB
    default = "ter" if bar.beat_unit == "dotted_quarter" else "bin"

    def snap(t: int) -> int:
        if t % TPB == 0 or t >= bar.length:
            return min(t, bar.length)
        base = (t // TPB) * TPB
        step = _grid_step(bar.beat_unit, family_of.get(first_beat_g + t // TPB, default))
        return base + min(TPB, int(round((t - base) / step)) * step)

    out = []
    for a, b, notes, cont in segments:
        sa, sb = snap(a), snap(b)
        if sb > sa:
            out.append((sa, sb, notes, cont))
    return out


def spell_bar(bar: WBar, segments: Sequence[tuple[int, int, list[WNote] | None, bool]],
              family_of: Mapping[int, str]) -> list[WBeat]:
    """``segments`` = [(start tick, end tick, notes or None for a rest, continues a previous piece)] covering
    the bar; ``family_of`` = global beat index -> "bin" | "ter" | "swing" (beats it does not list: binary, or
    ternary in compound meter, as ``quantize.default_family``)."""
    beats: list[WBeat] = []
    compound = bar.beat_unit == "dotted_quarter"
    first_beat_g = bar.start_gtick // TPB
    segments = _snap_segments(bar, segments, family_of)
    # a ternary beat is written in 16th sextuplets when any boundary inside it is a 16th-triplet point
    fine_beat = {int(t // TPB) for a, b, _n, _c in segments for t in (a, b) if t % TPB and (t % TPB) % 16}
    for a, b, notes, cont in segments:
        pieces: list[tuple[int, int, int, int, tuple[int, int] | None]] = []
        t = a
        while t < b:
            k = t // TPB
            beat_start, beat_end = k * TPB, (k + 1) * TPB
            if t == beat_start and b >= beat_end:
                m = (b - t) // TPB
                if not compound and m == 1 and b - t >= TPB + TPB // 2 and (b - t - TPB) < TPB and \
                        family_of.get(first_beat_g + k + 1, "bin") != "ter" and t + 72 <= b:
                    pieces.append((t, 72, 4, 1, None))  # dotted quarter
                    t += 72
                    continue
                for s, d, v, dots in _merge_full_beats(bar, k, m):
                    pieces.append((s, d, v, dots, None))
                t += m * TPB
                continue
            end = min(b, beat_end)
            fam = family_of.get(first_beat_g + k, "ter" if compound else "bin")
            ra, rb = t - beat_start, end - beat_start
            if compound:
                sub = _spell_compound(ra, rb, binary=fam == "bin" and (ra % 8 or rb % 8) != 0)
            elif bar.beat_unit == "eighth":
                sub = [(ra, rb - ra, 16 if rb - ra <= 24 else 8, 0, None)]
            elif fam == "ter":
                sub = _spell_ternary(ra, rb, k in fine_beat)
            else:
                sub = _spell_binary(ra, rb)
            pieces += [(beat_start + s, d, v, dots, tu) for s, d, v, dots, tu in sub]
            t = end
        for j, (s, d, v, dots, tu) in enumerate(pieces):
            tie = cont or j > 0
            ns = [] if notes is None else [WNote(n.pitch, n.string, n.fret, tie, n.src, n.fx) for n in notes]
            beats.append(WBeat(start=s, dur=d, value=v, dots=dots, tuplet=tu, notes=ns))
    return beats


def spell_track(events: Sequence[tuple[int, int, list[WNote]]], bars: Sequence[WBar],
                family_of: Mapping[int, str]) -> list[list[WBeat]]:
    """``events`` = [(gtick, duration ticks, notes)] sorted by gtick -> written beats per bar of ``bars``."""
    timeline: list[tuple[int, int, list[WNote] | None]] = []
    t0 = bars[0].start_gtick
    t_end = bars[-1].start_gtick + bars[-1].length
    cur = t0
    evs = [e for e in events if t0 <= e[0] < t_end]
    for i, (g, dur, notes) in enumerate(evs):
        end = g + max(1, dur)
        if i + 1 < len(evs):
            end = min(end, evs[i + 1][0])
        end = min(end, t_end)
        if g > cur:
            timeline.append((cur, g, None))
        timeline.append((g, end, notes))
        cur = end
    if cur < t_end:
        timeline.append((cur, t_end, None))
    out = []
    j = 0
    for bar in bars:
        a_bar, b_bar = bar.start_gtick, bar.start_gtick + bar.length
        segs: list[tuple[int, int, list[WNote] | None, bool]] = []
        while j < len(timeline) and timeline[j][1] <= a_bar:
            j += 1
        k = j
        while k < len(timeline) and timeline[k][0] < b_bar:
            s, e, notes = timeline[k]
            a, b = max(s, a_bar), min(e, b_bar)
            if b > a:
                segs.append((a - a_bar, b - a_bar, notes, s < a_bar and notes is not None))
            k += 1
        out.append(spell_bar(bar, segs, family_of))
    place_effects(out)
    return out


HEAD_FX = ("x", "g")  # on the first written piece of a note only
TAIL_FX = ("sl", "sod", "sou")  # on its last piece only (a slide leaves the note where it ends)


def place_effects(spelled: Sequence[Sequence[WBeat]]) -> None:
    """A note written as tied pieces keeps a dead / ghost mark on its first piece and a slide on its last one;
    vibrato stays on every piece."""
    chains: dict[tuple[Any, ...], list[WNote]] = {}
    done: list[list[WNote]] = []
    for beats in spelled:
        for be in beats:
            for n in be.notes:
                if not n.fx and not n.tie:
                    continue
                key = (n.src, n.string, n.pitch)
                if n.tie and key in chains:
                    chains[key].append(n)
                else:
                    if key in chains:
                        done.append(chains[key])
                    chains[key] = [n]
    done.extend(chains.values())
    for chain in done:
        last = len(chain) - 1
        for k, n in enumerate(chain):
            n.fx = tuple(f for f in n.fx if (f not in HEAD_FX or k == 0) and (f not in TAIL_FX or k == last))


def written_ticks(value: int, dots: int, tuplet: tuple[int, int] | None, beat_unit: str) -> int:
    """Ticks of a written value in a bar of ``beat_unit`` (48 ticks per beat: a quarter is 48 ticks in quarter
    beats, 32 in dotted-quarter beats, 96 in eighth beats)."""
    quarter = {"quarter": TPB, "dotted_quarter": TPB * 2 // 3, "eighth": TPB * 2}[beat_unit]
    t = quarter * 4 // value
    if dots == 1:
        t = t * 3 // 2
    elif dots == 2:
        t = t * 7 // 4
    if tuplet:
        t = t * tuplet[1] // tuplet[0]
    return t


def check_bar(bar: WBar, beats: Sequence[WBeat]) -> list[str]:
    """Problems of a spelled bar (durations must tile the bar exactly)."""
    errs = []
    t = 0
    for be in beats:
        if be.start != t:
            errs.append(f"bar {bar.bar}: beat at {be.start}, expected {t}")
        t = be.start + be.dur
        want = written_ticks(be.value, be.dots, be.tuplet, bar.beat_unit)
        if want != be.dur:
            errs.append(f"bar {bar.bar}: value 1/{be.value}{'.' * be.dots} {be.tuplet} = {want} ticks, not {be.dur}")
    if t != bar.length:
        errs.append(f"bar {bar.bar}: beats fill {t} of {bar.length} ticks")
    return errs
