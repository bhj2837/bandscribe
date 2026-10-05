"""Tuning and capo search (DESIGN 6.1 T2, M3: one tuning for the merged guitar, one for the bass).

Hypotheses (open-string pitches, lowest string first):
- guitar: {6-string standard, drop} x downtune {0..-5} (E std, Eb std, Drop D, D std, Drop C#, ...); capo 1..7 on
  standard / drop at downtune 0; open tunings {Open G, Open D, DADGAD} x capo 0..7; 7-string standard / drop x
  downtune, only when the notes go below what every 6-string hypothesis can play.
- bass: {4-string standard, 4-string drop, 5-string standard} x downtune {0..-5}.

Score (lower is better) of a hypothesis on a sample of the track's events:
  ``unplayable`` x share of notes no string can play + mean fretting cost per note (``fretting.assign`` path
  cost: positions, shifts, open strings - drop riffs fall on 0-0-0, open tunings on their open shapes) + the
  prior (E std < Eb std < Drop D < D std < open tunings < Drop C#/C < C std; a capo costs a little per fret).
Guitar and bass are chosen jointly: a downtune that differs between them costs ``downtune_mismatch`` (bands
tune together, DADGP rejects songs whose instruments disagree).

Fake low notes are removed from the evidence first (DESIGN 6.1 "가짜 저음 방어", ``fake_low_notes``): lone guitar
notes below E2 at the exact pitch of a sounding bass note (bleed) and isolated bare octaves below E2 (octave
errors). Otherwise an E-standard song reads as drop or 7-string; but a drop riff's roots (repeated, in power
chords, doubled by the bass) stay.

The best hypothesis comes with the runner-up and the margin; a best cost per note above ``unknown_cost``
marks the tuning unknown (asked in the UI later). Thresholds are provisional.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bandscribe.tab import fretting as F

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
GUITAR_PROFILES: dict[str, tuple[int, ...]] = {
    "std": (40, 45, 50, 55, 59, 64),
    "drop": (38, 45, 50, 55, 59, 64),
    "open_g": (38, 43, 50, 55, 59, 62),
    "open_d": (38, 45, 50, 54, 57, 62),
    "dadgad": (38, 45, 50, 55, 57, 62),
    "std7": (35, 40, 45, 50, 55, 59, 64),
    "drop7": (33, 40, 45, 50, 55, 59, 64),
}
BASS_PROFILES: dict[str, tuple[int, ...]] = {
    "std4": (28, 33, 38, 43),
    "drop4": (26, 33, 38, 43),
    "std5": (23, 28, 33, 38, 43),
}
DEFAULT_PARAMS: dict[str, Any] = {
    "unplayable": 20.0,          # per unit share of unplayable notes
    "downtune_mismatch": 1.0,
    "unknown_cost": 4.0,         # best mean cost per note above this -> tuning unknown
    "max_events": 600,           # events sampled evenly for the search
    "capo_cost": 0.08,           # prior per capo fret
    "bleed_window_s": 0.05,
    "seven_min_share": 0.01,     # guitar notes below D2 (after the fake-low guard) that open the 7-string search
}
# prior by (profile, downtune)
_PRIOR_GUITAR = {("std", 0): 0.0, ("std", -1): 0.15, ("drop", 0): 0.25, ("std", -2): 0.3, ("drop", -1): 0.4,
                 ("drop", -2): 0.45, ("std", -3): 0.55, ("std", -4): 0.55, ("drop", -3): 0.6, ("drop", -4): 0.65}
_PRIOR_BASS = {("std4", 0): 0.0, ("std4", -1): 0.15, ("drop4", 0): 0.25, ("std4", -2): 0.3, ("std5", 0): 0.35}


@dataclass(frozen=True)
class Hypothesis:
    profile: str
    downtune: int
    capo: int
    tuning: tuple[int, ...]  # sounding open pitches (downtune applied), lowest first
    prior: float

    @property
    def label(self) -> str:
        low = NAMES[self.tuning[0] % 12]
        name = {
            "std": f"{low} standard", "std4": f"{low} standard", "std5": f"5-string {low} standard",
            "std7": f"7-string {low} standard", "drop": f"Drop {low}", "drop4": f"Drop {low}",
            "drop7": f"7-string Drop {low}", "open_g": "Open G", "open_d": "Open D", "dadgad": "DADGAD",
        }[self.profile]
        return name + (f", capo {self.capo}" if self.capo else "")


def guitar_hypotheses(allow7: bool = False) -> list[Hypothesis]:
    out = []
    for prof in ("std", "drop"):
        for d in range(0, -6, -1):
            out.append(Hypothesis(prof, d, 0, tuple(p + d for p in GUITAR_PROFILES[prof]),
                                  _PRIOR_GUITAR.get((prof, d), 0.75)))
        for c in range(1, 8):
            out.append(Hypothesis(prof, 0, c, GUITAR_PROFILES[prof], _PRIOR_GUITAR[(prof, 0)] + 0.1 + 0.08 * c))
    for prof in ("open_g", "open_d", "dadgad"):
        for c in range(0, 8):
            # only just above D standard: an open-tuning song differs from D standard by ~0.1 per note (open
            # shapes), while a standard-tuning song in an open tuning loses far more (awkward or unplayable shapes)
            out.append(Hypothesis(prof, 0, c, GUITAR_PROFILES[prof], 0.35 + 0.08 * c))
    if allow7:
        for prof in ("std7", "drop7"):
            for d in range(0, -6, -1):
                out.append(Hypothesis(prof, d, 0, tuple(p + d for p in GUITAR_PROFILES[prof]), 0.5 + 0.1 * -d))
    return out


def bass_hypotheses() -> list[Hypothesis]:
    out = []
    for prof in ("std4", "drop4", "std5"):
        for d in range(0, -6, -1):
            out.append(Hypothesis(prof, d, 0, tuple(p + d for p in BASS_PROFILES[prof]),
                                  _PRIOR_BASS.get((prof, d), 0.6 + (0.2 if prof == "std5" else 0.0))))
    return out


def _sample(events: Sequence[F.Event], k: int) -> list[F.Event]:
    if len(events) <= k:
        return list(events)
    idx = np.unique(np.linspace(0, len(events) - 1, k).round().astype(int))
    return [events[i] for i in idx]


def fake_low_notes(guitar: Sequence[Mapping[str, Any]], bass: Sequence[Mapping[str, Any]],
                   window_s: float = 0.05, isolation_s: float = 2.0, min_low_neighbours: int = 2) -> set[int]:
    """Indices of guitar notes below E2 that are not evidence for a lower tuning (DESIGN 6.1 "가짜 저음 방어"):

    - **bleed**: a lone guitar note (nothing else starts within ``window_s``) at the exact pitch of a sounding
      bass note - the bass leaking into the guitar stem. A bass doubling the guitar's root (an octave below, or
      under a power chord) is normal and does not count (review 2026-10-05: matching the pitch class threw away
      every drop-D root);
    - **octave error**: a bare octave (the note and the note 12 up, nothing else at that onset) that is
      *isolated* - fewer than ``min_low_neighbours`` other guitar notes below E2 within ``isolation_s``. A drop
      riff repeats its low notes; a stray octave error does not."""
    out: set[int] = set()
    b_on = np.array([float(n["onset_s"]) for n in bass]) if bass else np.zeros(0)
    b_off = np.array([float(n["offset_s"]) for n in bass]) if bass else np.zeros(0)
    b_p = np.array([int(n["pitch"]) for n in bass]) if bass else np.zeros(0, dtype=int)
    g_on = np.array([float(n["onset_s"]) for n in guitar])
    g_p = np.array([int(n["pitch"]) for n in guitar])
    low = g_p < 40
    for i in np.flatnonzero(low):
        p, t = int(g_p[i]), float(g_on[i])
        others = np.abs(g_on - t) <= window_s
        others[i] = False
        if not others.any():
            if b_on.size and np.any((b_p == p) & (b_on <= t + window_s) & (b_off >= t - window_s)):
                out.add(int(i))
            continue
        if others.sum() == 1 and np.any(others & (g_p == p + 12)):
            near_low = low & (np.abs(g_on - t) <= isolation_s)
            near_low[i] = False
            if near_low.sum() < min_low_neighbours:
                out.add(int(i))
    return out


def score(events: Sequence[F.Event], hyp: Hypothesis, p: Mapping[str, Any],
          weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    """``unplayable`` = share of notes outside the instrument's range (no string can play them); notes a chord
    cannot hold (more notes than strings, merged parts) are not counted: they would favour more strings."""
    inst = F.Instrument(hyp.tuning, max_fret=24, capo=hyp.capo)
    n = sum(len(e.pitches) for e in events)
    if n == 0:
        return {"cost": hyp.prior, "unplayable": 0.0, "per_note": 0.0}
    out_of_range = sum(1 for e in events for q in e.pitches if not inst.positions(q))
    # v1 model and weights: the open-string bonus is the evidence for drop and open tunings (fretting.V1_WEIGHTS)
    w = {**F.V1_WEIGHTS, **(weights or {})}
    res = F.assign(events, inst, w, model="v1")
    placed = sum(1 for a in res if a.string is not None)
    per_note = _path_cost(res, events, inst, w) / max(1, placed)
    unplayable = out_of_range / n
    return {"cost": float(p["unplayable"]) * unplayable + per_note + hyp.prior, "unplayable": unplayable,
            "per_note": per_note}


def _path_cost(res: Sequence[F.Assignment], events: Sequence[F.Event], inst: F.Instrument,
               w: Mapping[str, float]) -> float:
    """Static + transition cost of the chosen path (recomputed from the assignments)."""
    by_ev: dict[int, list[tuple[int, int]]] = {}
    for a in res:
        if a.string is not None:
            by_ev.setdefault(a.event, []).append((a.string, a.fret))
    total, prev = 0.0, None
    for e in range(len(events)):
        f = by_ev.get(e)
        if not f:
            continue
        total += F._static(f, w)
        if prev is not None:
            hp = [x for _s, x in prev if x > 0]
            hc = [x for _s, x in f if x > 0]
            if hp and hc:
                total += w["shift"] * abs(float(np.mean(hc)) - float(np.mean(hp)))
        prev = f
    return total


def choose(guitar_events: Sequence[F.Event], bass_events: Sequence[F.Event], params: Mapping[str, Any] | None = None,
           *, guitar_notes_low: bool = False) -> dict[str, Any]:
    """{"guitar": {...}, "bass": {...}} with best hypothesis, runner-up, margin and unknown flag per instrument."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    k = int(p["max_events"])
    ge, be = _sample(guitar_events, k), _sample(bass_events, k)
    g_rows = [(h, score(ge, h, p)) for h in guitar_hypotheses()]
    # 7-string only with notes below D2 (DESIGN 6.1: "7현은 D2 아래 음이 있을 때만"), >= ``seven_min_share``
    n_g = sum(len(e.pitches) for e in ge)
    below_d2 = sum(1 for e in ge for q in e.pitches if q < 38)
    if guitar_notes_low or (n_g and below_d2 / n_g >= float(p["seven_min_share"])):
        g_rows += [(h, score(ge, h, p)) for h in guitar_hypotheses(allow7=True) if h.profile in ("std7", "drop7")]
    b_rows = [(h, score(be, h, p)) for h in bass_hypotheses()]
    have_g, have_b = bool(ge), bool(be)
    best = None
    for gh, gs in (g_rows if have_g else [(None, {"cost": 0.0})]):
        for bh, bs in (b_rows if have_b else [(None, {"cost": 0.0})]):
            joint = gs["cost"] + bs["cost"]
            if gh is not None and bh is not None and gh.downtune != bh.downtune:
                joint += float(p["downtune_mismatch"])
            key = (joint, gh.prior if gh else 0.0, bh.prior if bh else 0.0)
            if best is None or key < best[0]:
                best = (key, gh, bh)
    out: dict[str, Any] = {"params": dict(p)}
    for name, rows, chosen in (("guitar", g_rows, best[1] if best else None), ("bass", b_rows, best[2] if best else None)):
        if chosen is None:
            out[name] = None
            continue
        ranked = sorted(rows, key=lambda hs: (hs[1]["cost"], hs[0].prior))
        runner = next(((h, s) for h, s in ranked if h != chosen), None)
        cs = dict(rows)[chosen]
        out[name] = {
            "label": chosen.label, "profile": chosen.profile, "downtune": chosen.downtune, "capo": chosen.capo,
            "tuning": list(chosen.tuning), "cost": round(cs["cost"], 4), "per_note": round(cs["per_note"], 4),
            "unplayable": round(cs["unplayable"], 4),
            "runner_up": None if runner is None else {"label": runner[0].label, "tuning": list(runner[0].tuning),
                                                       "capo": runner[0].capo, "cost": round(runner[1]["cost"], 4)},
            "margin": None if runner is None else round(runner[1]["cost"] - cs["cost"], 4),
            "unknown": bool(cs["per_note"] > float(p["unknown_cost"])),
            "top": [{"label": h.label, "cost": round(s["cost"], 4)} for h, s in ranked[:5]],
        }
    return out
