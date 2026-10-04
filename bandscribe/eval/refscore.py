"""Neutral reference-score format (``refscore.json``) and its expansion into ground-truth notes.

Two importers produce a ``RefScore``: PyGuitarPro for ``.gp3/.gp4/.gp5`` (``bandscribe.eval.gp_import``) and alphaTab in
Node for ``.gp/.gpx`` and as a cross-check (``node/bandscribe_score.mjs import``). One expansion implementation
(here) turns either into ``GtNotes``, so both importers are held to the same result.

Conventions fixed by RefScore (spec §6): string 1 = highest-pitched string; ``tuning[i]`` = open pitch of
string ``i+1``; ``pitch = tuning + capo + fret`` stored explicitly per note (alphaTab's own pitch is the
reference for capo handling: PyGuitarPro's ``Note.realValue`` omits the capo). Durations and positions are MIDI
ticks at 960 per quarter, positions relative to the written bar. ``repeat_count`` = total play count of a
repeat (0 = no repeat close); ``alternate`` = bitmask of the passes that play the bar (bit 0 = 1st ending).

Expansion: repeats and alternate endings are unrolled (or ``bar_order`` from gt.yaml is used verbatim, for
D.S./coda); expanded bars are numbered 1.. with 0 for a pickup bar; ticks are 48 per beat of the bar's beat
unit (quarter, dotted quarter in 6/8·9/8·12/8, eighth in other x/8); tied notes are merged into the note they
continue; dead (x) notes are dropped (no pitch); notes with the same (pitch, bar, tick) in two lines are
flagged ``shared`` in both (unison, DESIGN R3).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from bandscribe import atomic
from bandscribe.formats import normalize_doc

log = logging.getLogger(__name__)

TICKS_PER_QUARTER = 960
TPB = 48
BeatUnit = Literal["quarter", "dotted_quarter", "eighth"]
_UNIT_QUARTERS = {"quarter": 1.0, "dotted_quarter": 1.5, "eighth": 0.5}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _legacy_format_id(cls, data: Any) -> Any:
        return normalize_doc(data)  # "gtab.refscore/1" from before the rename (bandscribe.formats)


class RsTrack(_Model):
    index: int                         # 1-based track number in the file
    name: str
    percussion: bool = False
    n_strings: int
    tuning: list[int]                  # open pitch of string 1 (highest) .. string n
    capo: int = 0
    program: int | None = None         # GM program (0-based)


class RsMasterBar(_Model):
    index: int                         # 1-based written bar
    numerator: int
    denominator: int
    repeat_open: bool = False
    repeat_count: int = 0              # total plays of the repeat ending here (0 = no close)
    alternate: int = 0                 # bitmask of passes that play this bar
    tempo_bpm: float | None = None     # tempo at the bar start if it changes here
    start: int | None = None           # ticks from the start of the written score (info)
    length: int | None = None
    anacrusis: bool = False


class RsNote(_Model):
    track: int
    bar: int                           # written bar (1-based)
    voice: int = 1
    start: float                       # ticks within the bar
    dur: float
    string: int
    fret: int
    pitch: int
    tie: bool = False                  # continues the previous note on the same string
    dead: bool = False
    tuplet: list[int] | None = None


class RefScore(_Model):
    format: Literal["bandscribe.refscore/1"] = "bandscribe.refscore/1"
    source: dict[str, Any]
    title: str = ""
    artist: str = ""
    tempo_bpm: float = 120.0
    tracks: list[RsTrack]
    masterbars: list[RsMasterBar]
    notes: list[RsNote]
    warnings: list[str] = []

    def save(self, path: Path) -> None:
        atomic.write_text(Path(path), self.model_dump_json(indent=1) + "\n")

    @staticmethod
    def load(path: Path) -> RefScore:
        return RefScore.model_validate_json(Path(path).read_bytes())


# ------------------------------------------------------------------------------------------ meter


def beat_unit(numerator: int, denominator: int) -> tuple[BeatUnit, int]:
    """(beat unit, beats per bar) of a time signature (DESIGN 6.2 compound meters use the dotted quarter)."""
    if denominator == 8 and numerator >= 6 and numerator % 3 == 0:
        return "dotted_quarter", numerator // 3
    if denominator == 8:
        return "eighth", numerator
    if denominator == 4:
        return "quarter", numerator
    if denominator == 2:
        return "quarter", numerator * 2
    if denominator == 1:
        return "quarter", numerator * 4
    if denominator == 16:
        return "eighth", max(1, (numerator + 1) // 2)
    raise ValueError(f"unsupported time signature {numerator}/{denominator}")


def ticks960_to_tpb(ticks: float, unit: BeatUnit, tpb: int = TPB) -> float:
    return float(ticks) / (TICKS_PER_QUARTER * _UNIT_QUARTERS[unit]) * tpb


def bar_length960(mb: RsMasterBar) -> float:
    return mb.numerator * TICKS_PER_QUARTER * 4 / mb.denominator


# ------------------------------------------------------------------------------------------ repeats


def play_order(masterbars: Sequence[RsMasterBar], bar_order: Sequence[int] | None = None) -> list[int]:
    """Written bar numbers (1-based) in play order.

    ``bar_order`` (gt.yaml ``reference.bar_order``) is returned as given (D.S./coda are mapped by hand).
    Otherwise repeats are unrolled: a repeat starts at the last ``repeat_open`` (or the song start / the bar
    after the previous completed repeat) and plays ``repeat_count`` times; a bar with ``alternate`` bits plays
    only on those passes.
    """
    n = len(masterbars)
    if bar_order:
        bad = [b for b in bar_order if not 1 <= int(b) <= n]
        if bad:
            raise ValueError(f"bar_order has bars outside 1..{n}: {bad[:5]}")
        return [int(b) for b in bar_order]
    order: list[int] = []
    i, start, pass_no, jumped = 0, 0, 1, False
    guard = 0
    while i < n:
        guard += 1
        if guard > 100_000:
            raise ValueError("repeat structure does not terminate")
        mb = masterbars[i]
        if mb.repeat_open and not jumped:
            start, pass_no = i, 1
        jumped = False
        if mb.alternate and not (mb.alternate >> (pass_no - 1)) & 1:
            i += 1
            continue
        order.append(mb.index)
        if mb.repeat_count > 1 and pass_no < mb.repeat_count:
            pass_no += 1
            i, jumped = start, True
            continue
        if mb.repeat_count > 0:
            start, pass_no = i + 1, 1
        i += 1
    return order


def detect_pickup(masterbars: Sequence[RsMasterBar], order: Sequence[int]) -> bool:
    """A first bar marked anacrusis (GP7) or shorter than the next one and not repeated is a pickup."""
    if len(masterbars) < 2 or not order:
        return False
    first = masterbars[order[0] - 1]
    if first.anacrusis:
        return True
    return order.count(first.index) == 1 and bar_length960(first) < bar_length960(masterbars[1])


# ----------------------------------------------------------------------------------------- expansion


def expand(rs: RefScore, *, song_id: str, track_lines: Mapping[int, str], bar_order: Sequence[int] | None = None,
           pickup: bool | None = None, tpb: int = TPB) -> Any:
    """RefScore -> ``bandscribe.schema.gt.GtNotes`` (expanded). ``track_lines`` maps ref track number -> line id;
    tracks without a line (drums, keys, …) are left out. ``onset_s``/``offset_s`` stay None until ``gt align``."""
    from bandscribe.schema.gt import GtNote, GtNotes

    mbs = {mb.index: mb for mb in rs.masterbars}
    order = play_order(rs.masterbars, bar_order)
    has_pickup = detect_pickup(rs.masterbars, order) if pickup is None else bool(pickup)
    first_bar = 0 if has_pickup else 1
    bars_meta = []
    tempo = float(rs.tempo_bpm)
    for k, wb in enumerate(order):
        mb = mbs[wb]
        if mb.tempo_bpm:
            tempo = float(mb.tempo_bpm)
        unit, _n = beat_unit(mb.numerator, mb.denominator)
        bars_meta.append({"bar": first_bar + k, "numerator": mb.numerator, "denominator": mb.denominator,
                          "beat_unit": unit, "nominal_bpm": round(tempo * _UNIT_QUARTERS[unit] ** -1, 4)})
    by_bar: dict[int, list[RsNote]] = defaultdict(list)
    for nt in rs.notes:
        if nt.track in track_lines:
            by_bar[nt.bar].append(nt)
    tracks = {t.index: t for t in rs.tracks}
    out: list[dict[str, Any]] = []
    last_on_string: dict[tuple[int, int], dict[str, Any]] = {}
    for k, wb in enumerate(order):
        mb = mbs[wb]
        unit, _n = beat_unit(mb.numerator, mb.denominator)
        bar_no = first_bar + k
        for nt in sorted(by_bar.get(wb, ()), key=lambda x: (x.start, x.track, x.voice, x.string)):
            if nt.dead:
                continue
            tr = tracks[nt.track]
            if tr.percussion:
                continue
            tick = ticks960_to_tpb(nt.start, unit, tpb)
            dur = ticks960_to_tpb(nt.dur, unit, tpb)
            key = (nt.track, nt.string)
            prev = last_on_string.get(key)
            if nt.tie and prev is not None and prev["pitch"] == nt.pitch:
                prev["dur_ticks"] += dur
                prev["tied"] = True
                continue
            rec = {"line": track_lines[nt.track], "ref_track": nt.track, "bar": bar_no, "tick": round(tick, 4),
                   "dur_ticks": round(dur, 4), "pitch": nt.pitch, "string": nt.string, "fret": nt.fret,
                   "tied": False, "shared": False}
            out.append(rec)
            last_on_string[key] = rec
    lines_at: dict[tuple[int, int, float], set[str]] = defaultdict(set)
    for r in out:
        lines_at[(r["pitch"], r["bar"], r["tick"])].add(r["line"])
    for r in out:
        r["shared"] = len(lines_at[(r["pitch"], r["bar"], r["tick"])]) > 1
        r["dur_ticks"] = round(r["dur_ticks"], 4)
    out.sort(key=lambda r: (r["bar"], r["tick"], r["line"], r["pitch"]))
    return GtNotes(format="bandscribe.gtnotes/1", song_id=song_id, tpb=tpb, bars=bars_meta,
                   notes=[GtNote(**r) for r in out])


def nominal_beat_times(gt_notes: Any) -> tuple[list[float], list[int], list[int], dict[int, int]]:
    """Beat grid of the expanded score at its nominal tempo (time 0 = first beat of the first expanded bar).

    Returns (beat_times, bars, beat_numbers, beats_per_bar) with one extra closing beat after the last bar."""
    t = 0.0
    times: list[float] = []
    bars: list[int] = []
    beats: list[int] = []
    bpb: dict[int, int] = {}
    last_dur = 0.5
    for b in gt_notes.bars:
        unit, n = beat_unit(int(b["numerator"]), int(b["denominator"]))
        dur = 60.0 / float(b["nominal_bpm"])
        bpb[int(b["bar"])] = n
        for k in range(n):
            times.append(round(t, 9))
            bars.append(int(b["bar"]))
            beats.append(k + 1)
            t += dur
        last_dur = dur
    if gt_notes.bars:
        nb = int(gt_notes.bars[-1]["bar"]) + 1
        times.append(round(t, 9))
        bars.append(nb)
        beats.append(1)
        bpb.setdefault(nb, bpb[int(gt_notes.bars[-1]["bar"])])
    del last_dur
    return times, bars, beats, bpb


def tempo_map_dict(times: Sequence[float], bars: Sequence[int], beats: Sequence[int], bpb: Mapping[int, int], *,
                   source: str, tpb: int = TPB) -> dict[str, Any]:
    return {"format": "bandscribe.tempomap/1", "source": source, "tpb": tpb,
            "beats": [{"t_s": round(float(t), 6), "bar": int(b), "beat": int(k)} for t, b, k in zip(times, bars, beats)],
            "beats_per_bar": {str(k): int(v) for k, v in sorted(bpb.items())}}


def tempo_map_from_dict(d: Mapping[str, Any]) -> Any:
    """``tempo_map.json`` document -> P0's ``TempoMap`` (one parser for GT and grid: ``TempoMap.from_dict``).
    A dict without ``format`` is read as ``bandscribe.tempomap/1`` (older in-memory dicts)."""
    from bandscribe.schema.grid import TempoMap

    return TempoMap.from_dict({"format": "bandscribe.tempomap/1", **d} if "format" not in d else d)


def nominal_tempo_map(gt_notes: Any) -> Any:
    times, bars, beats, bpb = nominal_beat_times(gt_notes)
    return tempo_map_from_dict(tempo_map_dict(times, bars, beats, bpb, source="nominal", tpb=gt_notes.tpb))


def fill_times(gt_notes: Any, tempo_map: Any) -> Any:
    """Copy of ``gt_notes`` with ``onset_s``/``offset_s`` from a tempo map (bar/tick -> seconds)."""
    import numpy as np

    if not gt_notes.notes:
        return gt_notes
    bars = np.array([n.bar for n in gt_notes.notes])
    ticks = np.array([n.tick for n in gt_notes.notes], dtype=float)
    ends = ticks + np.array([n.dur_ticks for n in gt_notes.notes], dtype=float)
    on = np.asarray(tempo_map.bar_tick_to_time(bars, ticks), dtype=float)
    off = np.asarray(tempo_map.bar_tick_to_time(bars, ends), dtype=float)
    notes = [n.model_copy(update={"onset_s": round(max(0.0, float(a)), 6), "offset_s": round(max(float(a), float(b)), 6)})
             for n, a, b in zip(gt_notes.notes, on, off)]
    return gt_notes.model_copy(update={"notes": notes})
