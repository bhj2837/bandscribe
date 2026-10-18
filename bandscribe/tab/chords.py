"""Chord chart (DESIGN 6.5, M9; E31 in docs/decisions.md) and the bass cross-check (DESIGN 5 item 8).

Per beat of the grid: a duration-weighted pitch-class histogram of the guitar notes sounding in it (overlap seconds
over the beat length) is matched to the templates (``QUALITIES`` x 12 roots, cosine similarity, a small prior for
the plainer chord); a beat whose bass (the pitch class with the longest overlap) is the template's root gets
``bass_bonus``. A Viterbi pass over the beats charges ``change_cost`` per chord change, so a passing note does not
flip the label. A beat with almost no guitar (< ``min_fill`` of its length) prefers "N" (no chord) but may keep the
chord around it.

The cross-check flags every beat whose bass pitch class is not a tone of the beat's chord (``checks``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "5": (0, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10), "sus2": (0, 2, 7), "sus4": (0, 5, 7), "dim": (0, 3, 6), "add9": (0, 2, 4, 7)}
SUFFIX = {"maj": "", "min": "m", "5": "5", "7": "7", "maj7": "maj7", "m7": "m7", "sus2": "sus2", "sus4": "sus4",
          "dim": "dim", "add9": "add9"}
ROOTS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
# the evaluation's classes (DESIGN §10 M9 "근음 + maj/min/5/7"; the rest kept apart)
CLASS = {"maj": "maj", "maj7": "maj", "add9": "maj", "min": "min", "m7": "min", "7": "7", "5": "5",
         "sus2": "sus", "sus4": "sus", "dim": "dim"}
# E33 (opt-in, ``extended``): tension chords, the guitar voicings E31 missed (m9, maj9 ... ), reduced to E31's classes
EXTENDED: dict[str, tuple[int, ...]] = {
    "m9": (0, 3, 7, 10, 2), "maj9": (0, 4, 7, 11, 2), "9": (0, 4, 7, 10, 2), "7sus4": (0, 5, 7, 10),
    "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9), "dim7": (0, 3, 6, 9), "m7b5": (0, 3, 6, 10), "madd9": (0, 3, 7, 2)}
SUFFIX.update({q: q for q in EXTENDED})
CLASS.update({"m9": "min", "maj9": "maj", "9": "7", "7sus4": "sus", "6": "maj", "m6": "min", "dim7": "dim",
              "m7b5": "min", "madd9": "min"})
# bass_weight (E33): the bass's own sounding seconds join the guitar histogram (x weight) - a listener reads the
# harmony from both, and guitar voicings often leave the root to the bass
DEFAULTS: dict[str, Any] = {"change_cost": 0.15, "bass_bonus": 0.1, "size_prior": 0.02, "min_fill": 0.1,
                            "empty_cost": 0.25, "extended": False, "bass_weight": 0.0}
N = "N"


def chord_name(root: int, quality: str) -> str:
    return ROOTS[root % 12] + SUFFIX[quality]


def _intervals(q: str) -> tuple[int, ...]:
    return QUALITIES.get(q) or EXTENDED[q]


def _states(extended: bool = False) -> list[tuple[int, str]]:
    return [(r, q) for q in (*QUALITIES, *(EXTENDED if extended else ())) for r in range(12)]


def _templates(states: Sequence[tuple[int, str]]) -> np.ndarray:
    t = np.zeros((len(states), 12))
    for i, (r, q) in enumerate(states):
        for iv in _intervals(q):
            t[i, (r + iv) % 12] = 1.0
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def histograms(notes: Sequence[Mapping[str, Any]], edges: Sequence[float]) -> np.ndarray:
    """(beats, 12): per beat, each pitch class's sounding seconds over the beat's length."""
    e = np.asarray(edges, dtype=float)
    h = np.zeros((max(0, len(e) - 1), 12))
    if len(e) < 2:
        return h
    for n in notes:
        on = float(n["onset_s"])
        off = max(on, float(n.get("offset_s") or on))
        a = max(0, int(np.searchsorted(e, on, side="right")) - 1)
        b = min(len(e) - 1, int(np.searchsorted(e, off, side="left")))
        pc = int(n["pitch"]) % 12
        for k in range(a, b):
            ov = min(off, e[k + 1]) - max(on, e[k])
            if ov > 0:
                h[k, pc] += ov
    return h / np.maximum(np.diff(e), 1e-9)[:, None]


def bass_roots(bass: Sequence[Mapping[str, Any]], edges: Sequence[float]) -> list[int | None]:
    h = histograms(bass, edges)
    return [int(np.argmax(r)) if r.max() > 0 else None for r in h]


def recognize(guitar: Sequence[Mapping[str, Any]], bass: Sequence[Mapping[str, Any]], edges: Sequence[float],
              params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """-> {"beats": [label per beat ("N" or a name)], "states": [(root, quality) or None], "segments": [{"start",
    "end" (beat indices, end exclusive), "t0", "t1", "root", "quality", "name"}], "checks": [{"beat", "t", "chord",
    "bass"}]} over the beats between consecutive ``edges`` (seconds)."""
    p = {**DEFAULTS, **(params or {})}
    states = _states(bool(p["extended"]))
    tmpl = _templates(states)
    h = histograms(guitar, edges)
    roots = bass_roots(bass, edges)
    if float(p["bass_weight"]) > 0 and len(h):
        h = h + float(p["bass_weight"]) * histograms(bass, edges)
    n = len(h)
    if n == 0:
        return {"beats": [], "states": [], "segments": [], "checks": []}
    size = np.array([len(_intervals(q)) for _r, q in states], dtype=float)
    prior = float(p["size_prior"]) * np.abs(size - 3.0)
    root_of = np.array([r for r, _q in states])
    S = len(states)
    cost = np.empty((n, S + 1))  # last column = N
    for k in range(n):
        tot = float(h[k].sum())
        if tot < float(p["min_fill"]):
            cost[k, :S] = float(p["empty_cost"])
            cost[k, S] = 0.0
            continue
        v = h[k] / (np.linalg.norm(h[k]) + 1e-12)
        c = 1.0 - tmpl @ v + prior
        if roots[k] is not None:
            c = c - float(p["bass_bonus"]) * (root_of == roots[k])
        cost[k, :S] = c
        cost[k, S] = 1.0
    lam = float(p["change_cost"])
    acc = cost[0].copy()
    back = np.zeros((n, S + 1), dtype=np.int32)
    for k in range(1, n):
        j = int(np.argmin(acc))
        switch = acc[j] + lam
        stay = acc
        use_switch = switch < stay
        back[k] = np.where(use_switch, j, np.arange(S + 1))
        acc = np.where(use_switch, switch, stay) + cost[k]
    path = [int(np.argmin(acc))]
    for k in range(n - 1, 0, -1):
        path.append(int(back[k][path[-1]]))
    path.reverse()
    st = [None if s == S else states[s] for s in path]
    labels = [N if s is None else chord_name(*s) for s in st]
    segs: list[dict[str, Any]] = []
    for k, s in enumerate(st):
        if segs and segs[-1]["state"] == s:
            segs[-1]["end"] = k + 1
            segs[-1]["t1"] = float(edges[k + 1])
            continue
        segs.append({"state": s, "start": k, "end": k + 1, "t0": float(edges[k]), "t1": float(edges[k + 1])})
    out_segs = [{"start": g["start"], "end": g["end"], "t0": round(g["t0"], 4), "t1": round(g["t1"], 4),
                 "root": None if g["state"] is None else g["state"][0],
                 "quality": None if g["state"] is None else g["state"][1],
                 "name": N if g["state"] is None else chord_name(*g["state"])} for g in segs]
    checks = []
    for k, s in enumerate(st):
        if s is None or roots[k] is None:
            continue
        tones = {(s[0] + iv) % 12 for iv in _intervals(s[1])}
        if roots[k] not in tones:
            checks.append({"beat": k, "t": round(float(edges[k]), 4), "chord": labels[k], "bass": ROOTS[roots[k]]})
    return {"beats": labels, "states": st, "segments": out_segs, "checks": checks}


def beat_edges(grid: Mapping[str, Any]) -> list[float]:
    """The grid's beat times plus the end of the last beat (its spacing repeated)."""
    t = [float(b["t_s"]) for b in grid.get("beats") or []]
    if len(t) >= 2:
        t.append(t[-1] + (t[-1] - t[-2]))
    return t


def text_chart(segments: Sequence[Mapping[str, Any]], grid: Mapping[str, Any],
               sections: Sequence[Mapping[str, Any]] | None = None, *, bars_per_line: int = 4) -> str:
    """A plain-text chart: one cell per bar, each beat a chord name where it changes or ".", sections as headers."""
    beats = grid.get("beats") or []
    label_at: dict[int, str] = {}
    for g in segments:
        label_at[int(g["start"])] = "N.C." if g["name"] == N else str(g["name"])
    sec_at = {int(s["start_bar"]): str(s.get("label") or s.get("id") or "") for s in sections or []}
    bars: dict[int, list[str]] = {}
    for k, b in enumerate(beats):
        bars.setdefault(int(b["bar"]), []).append(label_at.get(k, "."))
    lines, row = [], []
    for bar in sorted(bars):
        if bar in sec_at and row:
            lines.append("| " + " | ".join(row) + " |")
            row = []
        if bar in sec_at:
            lines.append(f"[{sec_at[bar]}] (bar {bar})")
        row.append(" ".join(bars[bar]))
        if len(row) == bars_per_line:
            lines.append("| " + " | ".join(row) + " |")
            row = []
    if row:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
