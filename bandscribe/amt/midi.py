"""Performance MIDI export for the first DAW-usable outputs (M1_M2_SPEC 9.2 item 7, DESIGN M2).

``guitar_all.mid`` / ``bass_raw.mid`` / ``b0_guitar.mid`` / ``guitar_basicpitch.mid`` keep the **audio timeline**:
time 0 of the MIDI file is time 0 of the song, and every note sits at its transcribed second. The grid only
shapes the tempo map so that DAW bar lines follow the song:

- ticks_per_beat = 480 per **quarter note** (MIDI tempo is always microseconds per quarter). A grid beat of unit
  *u* and duration *d* seconds spans ``480 * {quarter: 1, dotted_quarter: 1.5, eighth: 0.5}[u]`` ticks and gets
  one ``set_tempo`` of ``round(d / factor[u] * 1e6)`` us per quarter (6/8: 720 ticks per dotted-quarter beat).
- **Lead-in.** Audio usually does not start on a beat, and a gap such as 0.37 s (= 0.74 beat) cannot be written
  as a partial bar with ``time_signature`` alone. So when the first grid beat ``t0`` is > 0, lead-in bar(s) come
  first: one ``1/4`` bar whose quarter lasts ``t0`` when ``t0 <= 2 x`` the first beat interval, otherwise one
  ``k/4`` bar with ``k = ceil(t0 / interval)`` and ``t0 / k`` s per quarter. The pickup bar (bar 0) then gets
  its own time signature, and bar 1 starts exactly on its downbeat. ``midi_bar_offset`` = MIDI bars before
  musical bar 1 (lead-in + pickup); ``notes_summary.json`` records it.
- Without a grid: constant 120 BPM, 4/4, no lead-in.

Deterministic bytes: format-1 file, conductor track first, one track per name, fixed channel assignment (drum
channel 9 skipped), events sorted by (tick, note-off before note-on, pitch). Same-pitch overlaps in one track are
trimmed at the next onset (MIDI cannot pair overlapping identical notes unambiguously). Velocity: the note's
velocity, else ``round(127 * amplitude)`` (Basic Pitch), else 90 (MuScriptor never sets one).
"""

from __future__ import annotations

import logging
import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bandscribe import atomic

log = logging.getLogger(__name__)

TPQ = 480
DEFAULT_BPM = 120.0
DEFAULT_VELOCITY = 90
MAX_TEMPO_US = 0xFFFFFF  # MIDI set_tempo is 24-bit
UNIT_FACTOR = {"quarter": 1.0, "dotted_quarter": 1.5, "eighth": 0.5}
# eighths or quarters (the time-signature denominator unit) per beat of each beat unit
UNITS_PER_BEAT = {"quarter": 1, "dotted_quarter": 3, "eighth": 1}

GM_PROGRAMS = {  # 0-based General MIDI programs (M1_M2_SPEC 9.2 item 7)
    "acoustic_guitar": 25, "clean_electric_guitar": 27, "distorted_electric_guitar": 30,
    "acoustic_bass": 32, "electric_bass": 33,
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _notes_of(noteset: Any) -> list[Any]:
    if isinstance(noteset, Mapping):
        return list(noteset.get("notes") or [])
    if isinstance(noteset, (list, tuple)):
        return list(noteset)
    return list(getattr(noteset, "notes", []) or [])


# --------------------------------------------------------------------------------------- tempo map


@dataclass(frozen=True)
class _Seg:
    t0: float          # seconds at the segment start
    tick0: float       # ticks at the segment start
    spt: float         # seconds per tick inside the segment
    tempo_us: int      # set_tempo value (us per quarter)


@dataclass
class TempoPlan:
    segments: list[_Seg]
    time_sigs: list[tuple[int, int, int]]   # (tick, numerator, denominator)
    midi_bar_offset: int
    lead_in: dict[str, Any] | None

    def tick_at(self, t: float) -> int:
        t = max(0.0, float(t))
        starts = [s.t0 for s in self.segments]
        i = max(0, bisect_right(starts, t) - 1)
        s = self.segments[i]
        return int(round(s.tick0 + (t - s.t0) / s.spt))

    def time_at(self, tick: float) -> float:
        ticks = [s.tick0 for s in self.segments]
        i = max(0, bisect_right(ticks, float(tick)) - 1)
        s = self.segments[i]
        return s.t0 + (float(tick) - s.tick0) * s.spt


def _tempo_us(sec_per_quarter: float) -> int:
    us = int(round(sec_per_quarter * 1e6))
    return max(1, min(MAX_TEMPO_US, us))


def constant_plan(bpm: float = DEFAULT_BPM) -> TempoPlan:
    spq = 60.0 / bpm
    return TempoPlan([_Seg(0.0, 0.0, spq / TPQ, _tempo_us(spq))], [(0, 4, 4)], 0, None)


def _bar_sig(bar: Any) -> tuple[str, int, int]:
    """(beat unit, numerator, denominator) of a grid bar, numerator taken from its **actual** beat count.

    Every beat gets ``480 * factor[unit]`` ticks, so a bar spans ``n_beats`` of them; writing the signature
    from ``n_beats`` (not from the smoothed ``numerator``) keeps DAW bar lines on the song's downbeats even for
    an odd bar. A pickup bar (bar 0) thus becomes ``n_pickup_beats/4`` (or ``3n/8`` in compound time).
    """
    unit = str(_get(bar, "beat_unit", "quarter") or "quarter")
    if unit not in UNIT_FACTOR:
        raise ValueError(f"unknown beat unit {unit!r}")
    den = int(_get(bar, "denominator", 8 if unit != "quarter" else 4) or 4)
    n_beats = int(_get(bar, "n_beats", 0) or 0)
    num = n_beats * UNITS_PER_BEAT[unit] if n_beats > 0 else int(_get(bar, "numerator", 4) or 4)
    return unit, num, den


def plan_from_grid(grid: Any) -> TempoPlan:
    """Tempo segments + time signatures from a ``Grid`` (model or grid.json dict). See the module docstring."""
    beats = sorted(_get(grid, "beats") or [], key=lambda b: float(_get(b, "t_s")))
    bars = {int(_get(b, "bar")): b for b in (_get(grid, "bars") or [])}
    if len(beats) < 2:
        log.info("grid has fewer than two beats: constant %.0f BPM MIDI", DEFAULT_BPM)
        return constant_plan()
    times = [float(_get(b, "t_s")) for b in beats]
    bar_of = [int(_get(b, "bar")) for b in beats]
    segs: list[_Seg] = []
    sigs: list[tuple[int, int, int]] = []
    tick = 0.0
    t0 = times[0]
    first_iv = times[1] - times[0]
    lead_in = None
    offset = 0
    if t0 > 1e-9 and first_iv > 0:
        k = 1 if t0 <= 2.0 * first_iv else int(math.ceil(t0 / first_iv - 1e-9))
        spq = t0 / k
        segs.append(_Seg(0.0, 0.0, spq / TPQ, _tempo_us(spq)))
        sigs.append((0, k, 4))
        tick = float(k * TPQ)
        lead_in = {"quarters": k, "sec_per_quarter": round(spq, 6), "t0_s": round(t0, 6)}
        offset += 1
    prev_bar: int | None = None
    unit = "quarter"
    for i, t in enumerate(times):
        b = bar_of[i]
        if b != prev_bar:
            bar = bars.get(b)
            if bar is None:  # a beat without its bar entry: count the bar's beats ourselves
                n = sum(1 for x in bar_of if x == b)
                bar = {"n_beats": n, "beat_unit": unit, "denominator": 4 if unit == "quarter" else 8}
            unit, num, den = _bar_sig(bar)
            if b < 1:
                offset += 1  # pickup bar(s) before musical bar 1
            sigs.append((int(round(tick)), num, den))
            prev_bar = b
        d = (times[i + 1] - t) if i + 1 < len(times) else (t - times[i - 1])
        if d <= 0:
            raise ValueError(f"grid beats are not strictly increasing near t={t:.6f}s")
        factor = UNIT_FACTOR[unit]
        spq = d / factor
        segs.append(_Seg(t, tick, spq / TPQ, _tempo_us(spq)))
        tick += TPQ * factor
    final: list[tuple[int, int, int]] = []
    for s in sigs:  # consecutive identical signatures are redundant
        if final and final[-1][1:] == s[1:]:
            continue
        final.append(s)
    return TempoPlan(segs, final, offset, lead_in)


# ------------------------------------------------------------------------------------------ writing


def _velocity(n: Any) -> int:
    v = _get(n, "velocity")
    if v is not None:
        return max(1, min(127, int(v)))
    a = _get(n, "amplitude")
    if a is not None:
        return max(1, min(127, int(round(127 * float(a)))))
    return DEFAULT_VELOCITY


def _trimmed(notes: Sequence[Any]) -> list[tuple[float, float, int, int]]:
    """(onset, offset, pitch, velocity), sorted; same-pitch overlaps cut at the next onset."""
    rows = sorted(((float(_get(n, "onset_s")), float(_get(n, "offset_s")), int(_get(n, "pitch")), _velocity(n))
                   for n in notes), key=lambda r: (r[0], r[2], r[1], r[3]))
    last: dict[int, int] = {}
    out: list[list[Any]] = []
    for on, off, p, v in rows:
        if not 0 <= p <= 127:
            continue
        j = last.get(p)
        if j is not None and out[j][1] > on:
            out[j][1] = on
        last[p] = len(out)
        out.append([on, max(on, off), p, v])
    return [tuple(r) for r in out if r[1] > r[0] or r[1] == r[0]]


def write_performance_midi(path: Path, tracks: Sequence[tuple[str, int, Any]], grid: Any = None) -> dict[str, Any]:
    """Write ``tracks`` = [(name, GM program 0-based, NoteSet-like)] as a format-1 MIDI file; returns a summary."""
    import io

    import mido

    plan = plan_from_grid(grid) if grid is not None else constant_plan()
    mid = mido.MidiFile(type=1, ticks_per_beat=TPQ)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="tempo", time=0))
    ev: list[tuple[int, int, Any]] = []
    for s in plan.segments:
        ev.append((int(round(s.tick0)), 1, mido.MetaMessage("set_tempo", tempo=s.tempo_us, time=0)))
    for tick, num, den in plan.time_sigs:
        ev.append((tick, 0, mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0)))
    ev.sort(key=lambda e: (e[0], e[1]))
    # Consecutive identical tempos are redundant: drop them (bytes stay deterministic either way).
    cur = 0
    last_tempo = None
    for tick, _order, msg in ev:
        if msg.type == "set_tempo":
            if msg.tempo == last_tempo:
                continue
            last_tempo = msg.tempo
        msg.time = tick - cur
        cur = tick
        conductor.append(msg)
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(conductor)

    n_notes = 0
    channels = [c for c in range(16) if c != 9]
    for k, (name, program, noteset) in enumerate(tracks):
        ch = channels[k % len(channels)]
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=str(name), time=0))
        tr.append(mido.Message("program_change", channel=ch, program=int(program) & 0x7F, time=0))
        events: list[tuple[int, int, int, Any]] = []
        for on, off, p, v in _trimmed(_notes_of(noteset)):
            t_on, t_off = plan.tick_at(on), plan.tick_at(off)
            if t_off <= t_on:
                t_off = t_on + 1
            events.append((t_on, 1, p, mido.Message("note_on", channel=ch, note=p, velocity=v, time=0)))
            events.append((t_off, 0, p, mido.Message("note_off", channel=ch, note=p, velocity=0, time=0)))
            n_notes += 1
        events.sort(key=lambda e: (e[0], e[1], e[2]))
        cur = 0
        for tick, _o, _p, msg in events:
            msg.time = tick - cur
            cur = tick
            tr.append(msg)
        tr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr)

    buf = io.BytesIO()
    mid.save(file=buf)
    atomic.write_bytes(Path(path), buf.getvalue())
    return {"ticks_per_beat": TPQ, "midi_bar_offset": plan.midi_bar_offset, "lead_in": plan.lead_in,
            "n_tracks": len(tracks), "n_notes": n_notes}


def class_tracks(noteset: Any, classes: Sequence[str]) -> list[tuple[str, int, list[Any]]]:
    """One (name, program, notes) track per class present in ``noteset`` (fixed ``classes`` order)."""
    notes = _notes_of(noteset)
    out = []
    for cls in classes:
        sel = [n for n in notes if _get(n, "instrument") == cls]
        if sel:
            out.append((cls, GM_PROGRAMS.get(cls, 0), sel))
    return out


def merged_track(noteset: Any, classes: Sequence[str], name: str, program: int) -> tuple[str, int, list[Any]]:
    keep = set(classes)
    return (name, program, [n for n in _notes_of(noteset) if _get(n, "instrument") in keep])
