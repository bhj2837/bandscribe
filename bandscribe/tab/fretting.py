"""String/fret assignment by Viterbi (DESIGN 6.3, M3 step 1).

Input: events = notes sounding together (a chord or a single note), in time order, each with its pitches and
per-note offsets. Every event gets candidate fingerings: one distinct string per pitch with a fret in
``0..max_fret`` (frets are written relative to the capo), fretted notes spanning at most ``max_span`` frets.
A chord with no such fingering (more notes than strings, an impossible stretch) loses notes one at a time - the
one whose removal leaves the cheapest fingering - and the dropped pitches are reported (they cannot be tabbed).

Cost of a path = Σ static(fingering) + Σ transition(previous, fingering):
- static: ``height`` x lowest fretted fret (stay near the nut unless the line climbs), ``high_fret`` x frets
  above 12, ``span`` x (span - 3)² for stretches beyond 4 frets, minus ``open`` per open string, ``skip`` per
  string skipped inside a chord of >= 3 notes;
- transition: ``shift`` x |hand move| (mean fretted fret; open strings do not move the hand), discounted by the
  time between the events (``shift_gap_s``: a shift over a rest is cheaper), ``string_move`` x string distance
  between single notes, and ``cut`` per still-ringing note of the previous event whose string the next fingering
  takes for another fret (an arpeggio wants its notes on different strings).

Viterbi gives the cheapest path; forward and backward passes give every event's min-marginals, and the fret
confidence is ``1 - exp(-margin / conf_scale)`` with margin = second-best minus best min-marginal (DESIGN 6.3
"1위 경로와 2위 경로의 비용 차이"). Weights are E12's (pre-registered in docs/decisions.md). Pure numpy.

Strings are numbered 1 = highest-pitched (tab / Guitar Pro convention); ``tuning`` lists open-string pitches
lowest string first (``schema.score.Line.tuning``), so string s is ``tuning[n - s]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

GUITAR_STD = (40, 45, 50, 55, 59, 64)
BASS_STD = (28, 33, 38, 43)

DEFAULT_WEIGHTS: dict[str, float] = {
    "height": 0.15,
    "high_fret": 0.3,
    "span": 1.0,
    "open": 0.3,
    "skip": 0.5,
    "shift": 0.6,
    "shift_gap_s": 0.5,
    "string_move": 0.1,
    "cut": 1.5,
    "conf_scale": 1.0,
}
MAX_CANDIDATES = 64
RING_TOL_S = 0.02  # a previous note still sounds if it ends later than this before the next onset


@dataclass(frozen=True)
class Instrument:
    tuning: tuple[int, ...] = GUITAR_STD  # lowest string first
    max_fret: int = 24
    capo: int = 0
    max_span: int = 5

    @property
    def n_strings(self) -> int:
        return len(self.tuning)

    def open_pitch(self, string: int) -> int:
        """Sounding pitch of string ``string`` (1 = highest) played open (at the capo)."""
        return int(self.tuning[self.n_strings - string]) + self.capo

    def positions(self, pitch: int) -> list[tuple[int, int]]:
        """(string, written fret) where ``pitch`` can be played."""
        out = []
        for s in range(1, self.n_strings + 1):
            f = int(pitch) - self.open_pitch(s)
            if 0 <= f <= self.max_fret - self.capo:
                out.append((s, f))
        return out


@dataclass
class Event:
    pitches: list[int]
    onset_s: float
    offsets_s: list[float]
    ids: list[Any] = field(default_factory=list)  # caller's note ids, same order as pitches


@dataclass
class Assignment:
    event: int
    pitch: int
    note_id: Any
    string: int | None  # None = dropped (could not be fretted with the rest of its chord)
    fret: int | None
    confidence: float


# ------------------------------------------------------------------------------------------- candidates


def _static(fing: Sequence[tuple[int, int]], w: Mapping[str, float]) -> float:
    frets = [f for _s, f in fing if f > 0]
    c = 0.0
    if frets:
        lo, hi = min(frets), max(frets)
        c += w["height"] * lo + w["high_fret"] * max(0, hi - 12) + w["span"] * max(0, hi - lo - 3) ** 2
    c -= w["open"] * sum(1 for _s, f in fing if f == 0)
    if len(fing) >= 3:
        ss = sorted(s for s, _f in fing)
        c += w["skip"] * ((ss[-1] - ss[0] + 1) - len(ss))
    return c


def _assignments(options: Sequence[Sequence[tuple[int, int]]], max_span: int) -> list[tuple[tuple[int, int], ...]]:
    """Every choice of one position per note on distinct strings whose fretted span is <= max_span."""
    out: list[tuple[tuple[int, int], ...]] = []

    def rec(i: int, used: frozenset[int], lo: int, hi: int, acc: list[tuple[int, int]]) -> None:
        if i == len(options):
            out.append(tuple(acc))
            return
        for s, f in options[i]:
            if s in used:
                continue
            nlo, nhi = (min(lo, f), max(hi, f)) if f > 0 else (lo, hi)
            if f > 0 and nhi - nlo > max_span:
                continue
            acc.append((s, f))
            rec(i + 1, used | {s}, nlo, nhi, acc)
            acc.pop()
            if len(out) > 20000:  # pathological chords: enough to choose from
                return

    rec(0, frozenset(), 10**6, -1, [])
    return out


def candidates(pitches: Sequence[int], inst: Instrument, w: Mapping[str, float],
               limit: int = MAX_CANDIDATES) -> tuple[list[tuple[tuple[int, int], ...]], list[int]]:
    """(fingerings, indices of dropped pitches). Fingerings list one (string, fret) per kept pitch, in the
    order of the kept pitches; cheapest ``limit`` by static cost."""
    keep = list(range(len(pitches)))
    dropped: list[int] = []
    while keep:
        opts = [inst.positions(pitches[i]) for i in keep]
        if all(opts) and len(keep) <= inst.n_strings:
            fings = _assignments(opts, inst.max_span)
            if fings:
                fings.sort(key=lambda f: (_static(f, w), f))
                return fings[:limit], dropped
        # drop the note whose removal leaves the cheapest fingering (unplayable pitches first; ties: a note
        # whose pitch class another note doubles, then the lowest)
        choice: tuple[tuple[float, int, int], int] | None = None
        for j, i in enumerate(keep):
            if not inst.positions(pitches[i]):
                choice = ((-math.inf, 0, 0), j)
                break
        if choice is None:
            for j in range(len(keep)):
                rest = keep[:j] + keep[j + 1:]
                cost = 1e6
                if len(rest) <= inst.n_strings:
                    ro = [inst.positions(pitches[i]) for i in rest]
                    fs = _assignments(ro, inst.max_span) if all(ro) else []
                    cost = min((_static(f, w) for f in fs), default=1e6)
                doubled = any(pitches[k] % 12 == pitches[keep[j]] % 12 for k in rest)
                key = (cost, 0 if doubled else 1, int(pitches[keep[j]]))
                if choice is None or key < choice[0]:
                    choice = (key, j)
        dropped.append(keep.pop(choice[1]))
    return [], sorted(dropped)


# ------------------------------------------------------------------------------------------------ viterbi


@dataclass
class _Cands:
    fings: list[tuple[tuple[int, int], ...]]
    kept: list[int]          # indices into the event's pitches
    static: np.ndarray       # (K,)
    hand: np.ndarray         # (K,) mean fretted fret, nan if all open
    frets: np.ndarray        # (K, n_strings) fret on each string (index s-1), -1 unused
    single: np.ndarray       # (K,) string of a one-note fingering, 0 otherwise


def _pack(fings: list[tuple[tuple[int, int], ...]], kept: list[int], n: int, w: Mapping[str, float]) -> _Cands:
    k = len(fings)
    frets = -np.ones((k, n), dtype=int)
    hand = np.full(k, np.nan)
    single = np.zeros(k, dtype=int)
    for i, f in enumerate(fings):
        fr = [x for _s, x in f if x > 0]
        if fr:
            hand[i] = float(np.mean(fr))
        for s, x in f:
            frets[i, s - 1] = x
        if len(f) == 1:
            single[i] = f[0][0]
    return _Cands(fings, kept, np.array([_static(f, w) for f in fings]), hand, frets, single)


def _transition(a: _Cands, b: _Cands, ev_a: Event, ev_b: Event, w: Mapping[str, float]) -> np.ndarray:
    gap = max(0.0, ev_b.onset_s - max(ev_a.onset_s, max(ev_a.offsets_s, default=ev_a.onset_s)))
    shift = np.abs(a.hand[:, None] - b.hand[None, :])
    shift = np.where(np.isnan(shift), 0.0, shift)
    t = w["shift"] * shift / (1.0 + gap / w["shift_gap_s"])
    both_single = (a.single[:, None] > 0) & (b.single[None, :] > 0)
    t = t + w["string_move"] * np.where(both_single, np.abs(a.single[:, None] - b.single[None, :]), 0)
    # notes of a still ringing at b's onset, on strings b takes for another fret
    ringing = [ev_a.offsets_s[i] > ev_b.onset_s + RING_TOL_S for i in range(len(ev_a.pitches))]
    if any(ringing):
        ring_frets = -np.ones_like(a.frets)
        for ci, f in enumerate(a.fings):
            for (s, x), pi in zip(f, a.kept):
                if ringing[pi]:
                    ring_frets[ci, s - 1] = x
        rf, bf = ring_frets[:, None, :], b.frets[None, :, :]
        cut = ((rf >= 0) & (bf >= 0) & (rf != bf)).sum(axis=2)
        t = t + w["cut"] * cut
    return t


def assign(events: Sequence[Event], inst: Instrument, weights: Mapping[str, float] | None = None
           ) -> list[Assignment]:
    """Fingering of every note of ``events`` (one Assignment per pitch, events in order)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    cands: list[_Cands | None] = []
    drops: list[list[int]] = []
    for ev in events:
        fings, dropped = candidates(ev.pitches, inst, w)
        kept = [i for i in range(len(ev.pitches)) if i not in dropped]
        cands.append(_pack(fings, kept, inst.n_strings, w) if fings else None)
        drops.append(dropped)
    out: list[Assignment] = []
    # independent Viterbi runs over stretches of fingerable events (an unfingerable event breaks the chain)
    idx = [i for i, c in enumerate(cands) if c is not None]
    runs: list[list[int]] = []
    for i in idx:
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    chosen: dict[int, tuple[int, float]] = {}
    for run in runs:
        trans = [_transition(cands[a], cands[b], events[a], events[b], w) for a, b in zip(run, run[1:])]
        fwd = [cands[run[0]].static.copy()]
        back: list[np.ndarray] = []
        for t, i in zip(trans, run[1:]):
            tot = fwd[-1][:, None] + t
            back.append(np.argmin(tot, axis=0))
            fwd.append(tot.min(axis=0) + cands[i].static)
        bwd = [np.zeros(len(cands[run[-1]].fings))]
        for t, i in zip(reversed(trans), reversed(run[1:])):
            bwd.append((t + (cands[i].static + bwd[-1])[None, :]).min(axis=1))
        bwd.reverse()
        k = int(np.argmin(fwd[-1]))
        path = [k]
        for bk in reversed(back):
            path.append(int(bk[path[-1]]))
        path.reverse()
        for pos, (i, k) in enumerate(zip(run, path)):
            mm = fwd[pos] + bwd[pos]
            srt = np.sort(mm)
            margin = float(srt[1] - srt[0]) if srt.size > 1 else 10.0
            chosen[i] = (k, 1.0 - math.exp(-max(margin, 0.0) / w["conf_scale"]))
    for e, ev in enumerate(events):
        ids = ev.ids or list(range(len(ev.pitches)))
        c = cands[e]
        pos_of: dict[int, tuple[int, int]] = {}
        conf = 0.0
        if c is not None and e in chosen:
            k, conf = chosen[e]
            pos_of = {pi: sf for pi, sf in zip(c.kept, c.fings[k])}
        for pi, p in enumerate(ev.pitches):
            sf = pos_of.get(pi)
            out.append(Assignment(e, int(p), ids[pi], sf[0] if sf else None, sf[1] if sf else None,
                                  round(conf, 4) if sf else 0.0))
    return out


# ----------------------------------------------------------------------------------------------- events


def group_events(notes: Sequence[Mapping[str, Any]], *, key: str = "gtick", tol: float = 0.0) -> list[Event]:
    """Events from notes: same ``key`` value (quantised global tick) - or onsets within ``tol`` seconds of the
    event's first note when ``key`` is ``"onset_s"`` - form one event; ids are the notes' indices."""
    order = sorted(range(len(notes)), key=lambda i: (float(notes[i][key]), int(notes[i]["pitch"])))
    events: list[Event] = []
    start = None
    for i in order:
        n = notes[i]
        v = float(n[key])
        if events and start is not None and v - start <= tol + 1e-9:
            ev = events[-1]
        else:
            ev = Event([], float(n["onset_s"]), [], [])
            events.append(ev)
            start = v
        ev.pitches.append(int(n["pitch"]))
        ev.offsets_s.append(float(n.get("offset_s", n["onset_s"])))
        ev.ids.append(i)
        ev.onset_s = min(ev.onset_s, float(n["onset_s"]))
    return events


def playable(assignments: Sequence[Assignment], events: Sequence[Event], inst: Instrument) -> dict[str, Any]:
    """Invariant check (DESIGN M3: frets 0-24, span <= 5 per event, one note per string per event)."""
    bad: list[int] = []
    by_ev: dict[int, list[Assignment]] = {}
    for a in assignments:
        by_ev.setdefault(a.event, []).append(a)
    for e, rows in by_ev.items():
        placed = [a for a in rows if a.string is not None]
        frets = [a.fret for a in placed if a.fret and a.fret > 0]
        strings = [a.string for a in placed]
        ok = all(0 <= (a.fret or 0) <= 24 for a in placed) and len(strings) == len(set(strings))
        ok = ok and (not frets or max(frets) - min(frets) <= inst.max_span)
        ok = ok and all(inst.open_pitch(a.string) + a.fret == a.pitch for a in placed)
        if not ok:
            bad.append(e)
    n = len(assignments)
    placed_n = sum(1 for a in assignments if a.string is not None)
    return {"events": len(by_ev), "bad_events": bad, "notes": n, "placed": placed_n, "dropped": n - placed_n}


__all__ = ["Instrument", "Event", "Assignment", "assign", "candidates", "group_events", "playable", "GUITAR_STD",
           "BASS_STD", "DEFAULT_WEIGHTS"]
