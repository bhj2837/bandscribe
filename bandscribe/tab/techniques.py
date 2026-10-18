"""Technique layer 1 (DESIGN 6.4, M9; E32 in docs/decisions.md): f0 rules for bends on single-note lines.

The f0 is one channel (or the mid) of the guitar stem, the side the part sits on (pyin, 10 ms; unvoiced = NaN). A
note of a part is looked at only when no other note of that part overlaps half of it (a chord's f0 means nothing).

**Bend:** inside the note (from ``start_s`` after its onset to its offset, no new onset by construction), the
smoothed f0 rises at least ``min_rise_cents`` above where it started, the top is reached at least ``min_after_s``
after the onset, and either the start or the top is the note's own pitch (within ``pitch_tol_cents``: the
transcriber writes the fretted note or the pitch it is bent to). Rises over ``max_rise_cents`` are a tracker jump,
not a bend. Depth classes (DESIGN): ½ (< 150 cents), full (< 250), 1½. **Release:** the f0 comes back down by
``min_rise_cents`` after the top before the note ends.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

BEND_DEFAULTS: dict[str, Any] = {"min_rise_cents": 80.0, "max_rise_cents": 450.0, "start_s": 0.04,
                                 "min_after_s": 0.05, "pitch_tol_cents": 70.0, "vprob_min": 0.5, "smooth": 5,
                                 "min_frames": 8, "overlap_max": 0.5}
CLASSES = (("half", 150.0), ("full", 250.0), ("one_half", 1e9))
QUARTER_TONES = {"half": 2, "full": 4, "one_half": 6}  # alphaTex / Guitar Pro bend values


def _midi(hz: np.ndarray, vprob: np.ndarray, vmin: float) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    ok = (np.asarray(vprob) > vmin) & (hz > 0)
    return np.where(ok, 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0), np.nan)


def _running_median(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    h = w // 2
    for i in range(len(x)):
        seg = x[max(0, i - h):i + h + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= max(1, h):
            out[i] = np.median(seg)
    return out


def solo_notes(notes: Sequence[Mapping[str, Any]], overlap_max: float = 0.5) -> list[int]:
    """Indices of notes no other note of ``notes`` overlaps for more than ``overlap_max`` of its length."""
    on = np.array([float(n["onset_s"]) for n in notes])
    off = np.array([max(float(n["onset_s"]), float(n.get("offset_s") or n["onset_s"])) for n in notes])
    out = []
    for i in range(len(notes)):
        dur = max(1e-6, off[i] - on[i])
        ov = np.clip(np.minimum(off, off[i]) - np.maximum(on, on[i]), 0, None)
        ov[i] = 0.0
        if float(ov.max(initial=0.0)) <= overlap_max * dur:
            out.append(i)
    return out


def bends(notes: Sequence[Mapping[str, Any]], hz: np.ndarray, vprob: np.ndarray, hop_s: float,
          params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """[{"i", "rise_cents", "class", "release", "start_pitch", "top_pitch", "top_s"}] for the bent notes."""
    p = {**BEND_DEFAULTS, **(params or {})}
    m = _midi(hz, vprob, float(p["vprob_min"]))
    tol = float(p["pitch_tol_cents"]) / 100.0
    out = []
    for i in solo_notes(notes, float(p["overlap_max"])):
        n = notes[i]
        on = float(n["onset_s"])
        off = max(on, float(n.get("offset_s") or on))
        a, b = int((on + float(p["start_s"])) / hop_s), int(off / hop_s)
        if b - a < int(p["min_frames"]) or a >= len(m):
            continue
        seg = _running_median(m[a:b], int(p["smooth"]))
        valid = np.flatnonzero(~np.isnan(seg))
        if len(valid) < int(p["min_frames"]):
            continue
        start = float(np.median(seg[valid[:4]]))
        k_top = int(valid[np.argmax(seg[valid])])
        top = float(seg[k_top])
        rise = 100.0 * (top - start)
        if not float(p["min_rise_cents"]) <= rise <= float(p["max_rise_cents"]):
            continue
        if (k_top * hop_s) + float(p["start_s"]) < float(p["min_after_s"]):
            continue
        pitch = float(n["pitch"])
        if abs(start - pitch) > tol and abs(top - pitch) > tol:
            continue
        after = seg[k_top:]
        after = after[~np.isnan(after)]
        release = bool(len(after) and 100.0 * (top - float(after.min())) >= float(p["min_rise_cents"]))
        cls = next(c for c, lim in CLASSES if rise < lim)
        out.append({"i": i, "rise_cents": round(rise, 1), "class": cls, "release": release,
                    "start_pitch": round(start, 2), "top_pitch": round(top, 2),
                    "top_s": round((a + k_top) * hop_s, 3)})
    return out
