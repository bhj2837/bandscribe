"""Evaluation metrics (DESIGN §8.4 items 1, 2, 3, 4, 7, 10, 11 — the ones M1a names).

- ``note_prf``: onset(+offset) note P/R/F1 with counts from ``mir_eval.transcription.match_notes`` on **Hz**
  pitches (see ``matching``); P/R/F1 are computed from the counts, which the pooled block bootstrap needs.
- ``tick_prf``: predictions are mapped with the **ground-truth** tempo map into (bar, tick); a match needs the
  same pitch and |Δ| within the tick range equivalent to ±50 ms at that position. The system's own grid is never
  used, so a half-beat phase error or a shifted downbeat of the system cannot leak into note scores (#1).
- ``notation_match``: exact (bar, tick) equality of the system's own quantisation after mapping its bars to GT
  bars by nearest downbeat — reported separately (#2).
- ``octave_error_rate`` = chroma F1 − pitch F1 (#4).
- ``assignment``: macro (line-balanced) assignment accuracy with Hungarian line mapping, Unassigned = error
  (#7), ``unassigned_ratio`` (#10), note-weighted accuracy as reference, ``label_swap_raw`` (identity mapping).
- ``coverage_accuracy`` (posterior threshold sweep, #7) and ``shared_prf`` in unison windows (#11).

Line mapping detail: the mapping maximises the macro accuracy itself (exact search over injective maps for
≤ 6 lines per side; beyond that Hungarian on Σ C[p, r] / n_r, which equals it for single-line events), with the
note-weighted accuracy as a 1e-12 tie-break. Consequences, both tested: turning predictions into Unassigned can
never raise macro accuracy (every mapping only loses, so the maximum does too), and putting all notes in one
line equals the majority baseline (ties go to the largest line).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from gtab.eval.matching import match_min_cost, match_note_pairs, max_bipartite

log = logging.getLogger(__name__)

UNASSIGNED = "unassigned"
DEDUPE_TOL_S = 0.0105  # ≈ 1 tick at 48 per beat and 120 BPM (spec: "equal (pitch, onset ±1 tick)")

NOTE_VARIANTS: dict[str, dict[str, Any]] = {
    "onset50": {"onset_tol": 0.05},
    "onset100": {"onset_tol": 0.10},
    "offset50": {"onset_tol": 0.05, "offset_ratio": 0.2, "offset_min_tolerance": 0.05},
}


class PRF(NamedTuple):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def prf(tp: int, fp: int, fn: int) -> PRF:
    """P/R/F1 from counts. Empty reference and prediction -> 1.0 (nothing to find, nothing wrong)."""
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return PRF(int(tp), int(fp), int(fn), float(p), float(r), float(f))


# --------------------------------------------------------------------------------------- note arrays

NoteArrays = tuple[np.ndarray, np.ndarray]


def as_arrays(x: Any, *, lines: Iterable[str] | None = None, tempo_map: Any = None) -> NoteArrays:
    """(intervals (n, 2) s, pitches (n,) MIDI) from arrays, a NoteSet, a Prediction, GtNotes, a DatasetTrack
    or a list of note-like objects. ``lines`` filters Prediction / GtNotes / DatasetTrack lines."""
    keep = set(lines) if lines is not None else None
    if isinstance(x, tuple) and len(x) == 2:
        iv = np.asarray(x[0], dtype=float).reshape(-1, 2)
        return iv, np.asarray(x[1], dtype=float).reshape(-1)
    if hasattr(x, "string_fret") and isinstance(getattr(x, "notes", None), Mapping):  # DatasetTrack
        notes = [n for ln, ns in x.notes.items() if keep is None or ln in keep for n in ns.notes]
        return _from_notes(notes)
    notes = list(getattr(x, "notes", x))
    if notes and hasattr(notes[0], "line") and keep is not None:
        notes = [n for n in notes if n.line in keep]
    if notes and hasattr(notes[0], "tick") and hasattr(notes[0], "bar") and getattr(notes[0], "onset_s", None) is None:
        if tempo_map is None:
            raise ValueError("GT notes have no onset_s yet (run `gtab gt align`) and no tempo map was given")
        notes = _times_from_map(notes, tempo_map)
    return _from_notes(notes)


def _times_from_map(notes: Sequence[Any], tempo_map: Any) -> list[Any]:
    bars = np.array([n.bar for n in notes])
    ticks = np.array([n.tick for n in notes], dtype=float)
    ends = ticks + np.array([n.dur_ticks for n in notes], dtype=float)
    on = np.asarray(tempo_map.bar_tick_to_time(bars, ticks), dtype=float)
    off = np.asarray(tempo_map.bar_tick_to_time(bars, ends), dtype=float)
    return [_N(float(a), float(max(a, b)), int(n.pitch)) for n, a, b in zip(notes, on, off)]


@dataclass(frozen=True)
class _N:
    onset_s: float
    offset_s: float
    pitch: int


def _from_notes(notes: Sequence[Any]) -> NoteArrays:
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[float(n.onset_s), float(n.offset_s if n.offset_s is not None else n.onset_s)] for n in notes])
    return iv, np.array([float(n.pitch) for n in notes])


# -------------------------------------------------------------------------------------------- note F1


def _chroma(p: np.ndarray) -> np.ndarray:
    return 60.0 + np.mod(np.round(p), 12.0)


def note_prf(ref: Any, est: Any, *, onset_tol: float = 0.05, pitch_tol_cents: float = 50.0,
             offset_ratio: float | None = None, offset_min_tolerance: float = 0.05, chroma: bool = False) -> PRF:
    """Time-based note P/R/F1. ``chroma``: pitches mod 12 mapped into MIDI 60–71 before the Hz conversion."""
    r_iv, r_p = as_arrays(ref)
    e_iv, e_p = as_arrays(est)
    if chroma:
        r_p, e_p = _chroma(r_p), _chroma(e_p)
    pairs = match_note_pairs(r_iv, r_p, e_iv, e_p, onset_tol=onset_tol, pitch_tol_cents=pitch_tol_cents,
                             offset_ratio=offset_ratio, offset_min_tolerance=offset_min_tolerance)
    tp = len(pairs)
    return prf(tp, len(e_p) - tp, len(r_p) - tp)


def note_prf_variants(ref: Any, est: Any) -> dict[str, PRF]:
    """The reported variants: 50 ms onset-only, 100 ms onset-only, 50 ms with offsets (ratio 0.2, min 50 ms)."""
    return {name: note_prf(ref, est, **kw) for name, kw in NOTE_VARIANTS.items()}


def octave_error_rate(ref: Any, est: Any, *, onset_tol: float = 0.05) -> float:
    """chroma F1 − pitch F1 (≥ 0; the share of F1 lost only to octave errors)."""
    return note_prf(ref, est, onset_tol=onset_tol, chroma=True).f1 - note_prf(ref, est, onset_tol=onset_tol).f1


# -------------------------------------------------------------------------------------------- tick F1


def _in_sections(bars: np.ndarray, sections: Sequence[tuple[int, int]] | None) -> np.ndarray:
    if not sections:
        return np.ones(len(bars), dtype=bool)
    m = np.zeros(len(bars), dtype=bool)
    for s, e in sections:
        m |= (bars >= int(s)) & (bars < int(e))  # half-open bar ranges everywhere
    return m


def section_ranges(sections: Iterable[Any] | None) -> list[tuple[int, int]] | None:
    """GtSection / Section objects or (start_bar, end_bar) pairs -> list of half-open ranges."""
    if sections is None:
        return None
    out = []
    for s in sections:
        if isinstance(s, (tuple, list)):
            out.append((int(s[0]), int(s[1])))
        else:
            out.append((int(s.start_bar), int(s.end_bar)))
    return out


def _dedupe_positions(pos: np.ndarray, pitch: np.ndarray, tol: float) -> np.ndarray:
    """Indices kept after merging equal-pitch notes whose positions are within ``tol`` of the cluster start."""
    keep: list[int] = []
    for p in np.unique(pitch):
        idx = np.flatnonzero(pitch == p)
        idx = idx[np.argsort(pos[idx], kind="stable")]
        start = None
        for i in idx:
            if start is None or pos[i] - start > tol + 1e-9:
                keep.append(int(i))
                start = pos[i]
    return np.array(sorted(keep), dtype=int)


def tick_prf(gt: Any, est: Any, tempo_map: Any, *, sections: Iterable[Any] | None = None, tol_s: float = 0.05,
             lines: Iterable[str] | None = None) -> PRF:
    """Tick F1 on the ground-truth grid (DESIGN 8.4 #1). ``gt`` = GtNotes; ``est`` = anything ``as_arrays`` reads."""
    keep = set(lines) if lines is not None else None
    rng = section_ranges(sections)
    gnotes = [n for n in gt.notes if keep is None or n.line in keep]
    g_bar = np.array([n.bar for n in gnotes], dtype=int)
    g_sel = _in_sections(g_bar, rng)
    gnotes = [n for n, s in zip(gnotes, g_sel) if s]
    e_iv, e_p = as_arrays(est)
    tpb = float(getattr(tempo_map, "tpb", gt.tpb))
    if gnotes:
        g_t = np.asarray(tempo_map.bar_tick_to_time(np.array([n.bar for n in gnotes]),
                                                      np.array([n.tick for n in gnotes], dtype=float)), dtype=float)
        g_beat = np.asarray(tempo_map.time_to_beat(g_t), dtype=float)
        g_tol = tol_s * np.asarray(tempo_map.ticks_per_second(g_t), dtype=float)
        g_p = np.array([n.pitch for n in gnotes], dtype=int)
    else:
        g_beat = g_tol = np.zeros(0)
        g_p = np.zeros(0, dtype=int)
    if len(e_p):
        e_t = e_iv[:, 0]
        e_bar, _ = tempo_map.time_to_bar_tick(e_t)
        e_sel = _in_sections(np.asarray(e_bar, dtype=int), rng)
        e_beat = np.asarray(tempo_map.time_to_beat(e_t[e_sel]), dtype=float)
        e_pp = np.round(e_p[e_sel]).astype(int)
    else:
        e_beat = np.zeros(0)
        e_pp = np.zeros(0, dtype=int)
    # unison copies (same pitch within ±1 tick, several lines) are one note of the merged content on both sides
    keep_g = _dedupe_positions(g_beat, g_p, 1.0 / tpb)
    g_beat, g_tol, g_p = g_beat[keep_g], g_tol[keep_g], g_p[keep_g]
    keep_e = _dedupe_positions(e_beat, e_pp, 1.0 / tpb)
    e_beat, e_pp = e_beat[keep_e], e_pp[keep_e]
    edges: list[tuple[int, int]] = []
    order = np.argsort(e_beat, kind="stable")
    sb = e_beat[order]
    for i in range(len(g_beat)):
        w = g_tol[i] / tpb  # tolerance in beats at this GT position
        lo, hi = np.searchsorted(sb, g_beat[i] - w - 1e-9), np.searchsorted(sb, g_beat[i] + w + 1e-9, side="right")
        for k in order[lo:hi]:
            if e_pp[k] == g_p[i] and abs(e_beat[k] - g_beat[i]) * tpb <= g_tol[i] + 1e-9:
                edges.append((i, int(k)))
    tp = len(max_bipartite(len(g_beat), len(e_beat), edges))
    return prf(tp, len(e_beat) - tp, len(g_beat) - tp)


def bar_map_by_downbeats(system_downbeats: Mapping[int, float] | Sequence[tuple[int, float]], tempo_map: Any,
                         gt_bars: Iterable[int]) -> dict[int, int]:
    """System bar -> GT bar whose downbeat (GT tempo map) is nearest to the system bar's downbeat time."""
    items = list(system_downbeats.items()) if isinstance(system_downbeats, Mapping) else list(system_downbeats)
    gb = np.array(sorted(set(int(b) for b in gt_bars)))
    if not len(gb) or not items:
        return {}
    gt_t = np.asarray(tempo_map.bar_tick_to_time(gb, np.zeros(len(gb))), dtype=float)
    return {int(b): int(gb[int(np.argmin(np.abs(gt_t - float(t))))]) for b, t in items}


def notation_match(gt: Any, est_bar_tick: Iterable[tuple[int, float, int]], bar_map: Mapping[int, int], *,
                   sections: Iterable[Any] | None = None, lines: Iterable[str] | None = None) -> dict[str, float]:
    """Share of GT notes whose (bar, tick, pitch) the system reproduced exactly after bar mapping (DESIGN 8.4 #2)."""
    keep = set(lines) if lines is not None else None
    rng = section_ranges(sections)
    est = {(int(bar_map[b]), round(float(t), 4), int(p)) for b, t, p in est_bar_tick if b in bar_map}
    notes = [n for n in gt.notes if (keep is None or n.line in keep)]
    sel = _in_sections(np.array([n.bar for n in notes], dtype=int), rng) if notes else []
    notes = [n for n, s in zip(notes, sel) if s]
    hits = sum((n.bar, round(float(n.tick), 4), n.pitch) in est for n in notes)
    return {"exact_ratio": hits / len(notes) if notes else float("nan"), "n": len(notes), "hits": hits}


# ------------------------------------------------------------------------------------ line-aware events


@dataclass(frozen=True)
class Event:
    """Deduplicated note event: GT events carry every line that plays it (``lines`` = valid lines); predicted
    events carry the predicted lines (``UNASSIGNED`` included when a copy was left unassigned)."""

    onset_s: float
    pitch: int
    lines: frozenset[str]
    posterior: float | None = None
    shared: bool = False
    bar: int | None = None


def _dedupe(rows: list[tuple[float, int, str, float | None, bool, int | None]], tol: float) -> list[Event]:
    by_pitch: dict[int, list[tuple[float, int, str, float | None, bool, int | None]]] = defaultdict(list)
    for r in rows:
        by_pitch[r[1]].append(r)
    out: list[Event] = []
    for p in sorted(by_pitch):
        grp: list[tuple[float, int, str, float | None, bool, int | None]] = []
        for r in sorted(by_pitch[p], key=lambda x: (x[0], x[2])):
            if grp and r[0] - grp[0][0] > tol:
                out.append(_merge(grp))
                grp = []
            grp.append(r)
        if grp:
            out.append(_merge(grp))
    out.sort(key=lambda e: (e.onset_s, e.pitch))
    return out


def _merge(grp: list[tuple[float, int, str, float | None, bool, int | None]]) -> Event:
    lines = frozenset(r[2] for r in grp)
    posts = [r[3] for r in grp if r[3] is not None]
    assigned = lines - {UNASSIGNED}
    return Event(onset_s=grp[0][0], pitch=grp[0][1], lines=lines, posterior=max(posts) if posts else None,
                 shared=any(r[4] for r in grp) or len(assigned) > 1, bar=grp[0][5])


def events_from_gt(gt: Any, *, tempo_map: Any = None, lines: Iterable[str] | None = None,
                   windows: Sequence[tuple[float, float]] | None = None, tol_s: float = DEDUPE_TOL_S) -> list[Event]:
    """GT events from ``GtNotes`` (onset_s, or bar/tick through ``tempo_map``) or a ``DatasetTrack``.
    Notes of equal pitch within ±``tol_s`` merge into one event whose ``lines`` are all valid lines."""
    keep = set(lines) if lines is not None else None
    rows: list[tuple[float, int, str, float | None, bool, int | None]] = []
    if hasattr(gt, "string_fret") and isinstance(getattr(gt, "notes", None), Mapping):  # DatasetTrack
        for ln, ns in gt.notes.items():
            if keep is None or ln in keep:
                rows += [(float(n.onset_s), int(n.pitch), ln, None, False, None) for n in ns.notes]
    else:
        notes = [n for n in gt.notes if keep is None or n.line in keep]
        if notes and any(n.onset_s is None for n in notes):
            if tempo_map is None:
                raise ValueError("GT notes have no onset_s yet (run `gtab gt align`) and no tempo map was given")
            timed = _times_from_map(notes, tempo_map)
            rows = [(t.onset_s, n.pitch, n.line, None, bool(n.shared), n.bar) for n, t in zip(notes, timed)]
        else:
            rows = [(float(n.onset_s), int(n.pitch), n.line, None, bool(n.shared), getattr(n, "bar", None)) for n in notes]
    if windows is not None:
        rows = [r for r in rows if any(s <= r[0] < e for s, e in windows)]
    return _dedupe(rows, tol_s)


def events_from_prediction(pred: Any, *, windows: Sequence[tuple[float, float]] | None = None,
                           tol_s: float = DEDUPE_TOL_S) -> list[Event]:
    """Predicted events from a ``Prediction`` (PredNote.line; ``unassigned`` allowed)."""
    rows = [(float(n.onset_s), int(n.pitch), n.line or UNASSIGNED, n.posterior, bool(n.shared), n.bar)
            for n in pred.notes]
    if windows is not None:
        rows = [r for r in rows if any(s <= r[0] < e for s, e in windows)]
    return _dedupe(rows, tol_s)


def match_events(gt_events: Sequence[Event], est_events: Sequence[Event], *, onset_tol: float = 0.05) -> list[tuple[int, int]]:
    """Line-agnostic matching (same pitch, |Δonset| ≤ tol; max pairs then min |Δt|)."""
    return match_min_cost([e.onset_s for e in gt_events], [e.pitch for e in gt_events],
                          [e.onset_s for e in est_events], [e.pitch for e in est_events], onset_tol)


@dataclass
class AssignResult:
    macro: float
    weighted: float
    per_line: dict[str, dict[str, float]]
    mapping: dict[str, str]
    unassigned_ratio: float
    label_swap_raw: float
    confusion: dict[str, Any]
    n_gt: int
    n_est: int
    n_matched: int
    pairs: list[tuple[int, int]] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {"macro": self.macro, "weighted": self.weighted, "per_line": self.per_line, "mapping": self.mapping,
                "unassigned_ratio": self.unassigned_ratio, "label_swap_raw": self.label_swap_raw,
                "confusion": self.confusion, "n_gt": self.n_gt, "n_est": self.n_est, "n_matched": self.n_matched}


def _accuracy(pairs: Sequence[tuple[int, int]], gt: Sequence[Event], est: Sequence[Event], mapping: Mapping[str, str],
              gt_lines: Sequence[str], assigned: Sequence[bool] | None = None) -> tuple[dict[str, dict[str, float]], float, float]:
    per = {r: {"n": 0, "correct": 0} for r in gt_lines}
    tot_n = tot_c = 0
    for k, (gi, ei) in enumerate(pairs):
        valid = gt[gi].lines
        if assigned is not None and not assigned[k]:
            ok = False
        else:
            ok = any(mapping.get(p) in valid for p in est[ei].lines if p != UNASSIGNED)
        tot_n += 1
        tot_c += ok
        for r in valid:
            if r in per:
                per[r]["n"] += 1
                per[r]["correct"] += ok
    accs = []
    for r, d in per.items():
        d["acc"] = d["correct"] / d["n"] if d["n"] else float("nan")
        if d["n"]:
            accs.append(d["acc"])
    macro = float(np.mean(accs)) if accs else float("nan")
    weighted = tot_c / tot_n if tot_n else float("nan")
    return per, macro, weighted


_EXACT_MAX_LINES = 6
_TIE_EPS = 1e-12


def _combo_score(combos: Mapping[tuple[frozenset, frozenset], int], n_r: Mapping[str, int],
                 mapping: Mapping[str, str]) -> tuple[float, float]:
    correct_r = dict.fromkeys(n_r, 0)
    tot = ok_tot = 0
    for (valid, preds), cnt in combos.items():
        ok = any(mapping.get(p) in valid for p in preds)
        tot += cnt
        if ok:
            ok_tot += cnt
            for r in valid:
                correct_r[r] += cnt
    accs = [correct_r[r] / n for r, n in n_r.items() if n]
    return (float(np.mean(accs)) if accs else 0.0), (ok_tot / tot if tot else 0.0)


def _best_mapping(pairs: Sequence[tuple[int, int]], gt: Sequence[Event], est: Sequence[Event],
                  gt_lines: Sequence[str], pred_lines: Sequence[str], conf: np.ndarray, n_r_arr: np.ndarray) -> dict[str, str]:
    """Injective pred-line -> GT-line mapping maximising macro accuracy (tie-break: note-weighted accuracy, then
    enumeration order). Exact search over injective maps when both sides have ≤ 6 lines (a pred line is only
    ever mapped to a GT line it hit); otherwise Hungarian on the n_r-normalised confusion, which is exact for
    single-line events."""
    from scipy.optimize import linear_sum_assignment

    if not conf.size:
        return {}
    if len(pred_lines) <= _EXACT_MAX_LINES and len(gt_lines) <= _EXACT_MAX_LINES:
        combos: dict[tuple[frozenset, frozenset], int] = defaultdict(int)
        for gi, ei in pairs:
            combos[(gt[gi].lines, frozenset(est[ei].lines - {UNASSIGNED}))] += 1
        n_r = {r: int(n_r_arr[i]) for i, r in enumerate(gt_lines)}
        options = [[gt_lines[j] for j in range(len(gt_lines)) if conf[i, j] > 0] for i in range(len(pred_lines))]
        best: tuple[float, dict[str, str]] = (-1.0, {})

        def rec(i: int, used: frozenset, cur: dict[str, str]) -> None:
            nonlocal best
            if i == len(pred_lines):
                m, w = _combo_score(combos, n_r, cur)
                s = m + _TIE_EPS * w
                if s > best[0]:
                    best = (s, dict(cur))
                return
            for r in options[i]:
                if r not in used:
                    cur[pred_lines[i]] = r
                    rec(i + 1, used | {r}, cur)
                    del cur[pred_lines[i]]
            rec(i + 1, used, cur)

        rec(0, frozenset(), {})
        return best[1]
    w = conf / np.maximum(n_r_arr, 1)[None, :] + _TIE_EPS * conf
    rows, cols = linear_sum_assignment(w, maximize=True)
    return {pred_lines[a]: gt_lines[b] for a, b in zip(rows, cols) if conf[a, b] > 0}


def assignment(gt_events: Sequence[Event], est_events: Sequence[Event], *, onset_tol: float = 0.05) -> AssignResult:
    """Macro assignment accuracy (DESIGN 4.11): events matched line-agnostically, Hungarian line mapping,
    per GT line r the share of matched events (r among their valid lines) that went to a mapped pred line in the
    event's valid lines; Unassigned = error; macro = mean over GT lines with ≥ 1 matched event."""
    pairs = match_events(gt_events, est_events, onset_tol=onset_tol)
    gt_lines = sorted({ln for e in gt_events for ln in e.lines})
    pred_lines = sorted({ln for e in est_events for ln in e.lines} - {UNASSIGNED})
    gi_of = {r: i for i, r in enumerate(gt_lines)}
    pi_of = {p: i for i, p in enumerate(pred_lines)}
    conf = np.zeros((len(pred_lines), len(gt_lines)), dtype=float)
    n_r = np.zeros(len(gt_lines), dtype=float)
    for gi, ei in pairs:
        for r in gt_events[gi].lines:
            n_r[gi_of[r]] += 1
            for p in est_events[ei].lines:
                if p in pi_of:
                    conf[pi_of[p], gi_of[r]] += 1
    mapping = _best_mapping(pairs, gt_events, est_events, gt_lines, pred_lines, conf, n_r)
    per, macro, weighted = _accuracy(pairs, gt_events, est_events, mapping, gt_lines)
    ident = {p: p for p in pred_lines if p in gi_of}
    _, swap_raw, _ = _accuracy(pairs, gt_events, est_events, ident, gt_lines)
    n_un = sum(1 for e in est_events if not (e.lines - {UNASSIGNED}))
    return AssignResult(
        macro=macro, weighted=weighted, per_line=per, mapping=mapping,
        unassigned_ratio=n_un / len(est_events) if est_events else 0.0, label_swap_raw=swap_raw,
        confusion={"pred_lines": pred_lines, "gt_lines": gt_lines, "matrix": conf.astype(int).tolist()},
        n_gt=len(gt_events), n_est=len(est_events), n_matched=len(pairs), pairs=pairs)


def coverage_accuracy(gt_events: Sequence[Event], est_events: Sequence[Event],
                      taus: Sequence[float] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9),
                      *, onset_tol: float = 0.05) -> list[tuple[float, float, float]]:
    """(τ, coverage, accuracy | assigned): events with posterior < τ become Unassigned; the line mapping is the
    one of the full prediction. A single point (τ = 0) when the prediction carries no posteriors."""
    res = assignment(gt_events, est_events, onset_tol=onset_tol)
    pairs = res.pairs
    if not pairs:
        return [(0.0, float("nan"), float("nan"))]
    has_post = any(e.posterior is not None for e in est_events)
    out = []
    for tau in (taus if has_post else (0.0,)):
        assigned = []
        for _gi, ei in pairs:
            e = est_events[ei]
            base = bool(e.lines - {UNASSIGNED})
            assigned.append(base and (e.posterior is None or e.posterior >= tau))
        cov = sum(assigned) / len(pairs)
        n_ok = sum(1 for k, (gi, ei) in enumerate(pairs) if assigned[k] and any(
            res.mapping.get(p) in gt_events[gi].lines for p in est_events[ei].lines if p != UNASSIGNED))
        acc = n_ok / sum(assigned) if sum(assigned) else float("nan")
        out.append((float(tau), float(cov), float(acc)))
    return out


def shared_prf(gt_events: Sequence[Event], est_events: Sequence[Event],
               windows: Sequence[tuple[float, float]] | None = None, *, onset_tol: float = 0.05) -> PRF:
    """Precision/recall of ``shared`` (unison) notes inside unison windows (DESIGN 8.4 #11).
    GT positives: events valid for ≥ 2 lines (or flagged shared); predicted positives: shared flag or ≥ 2 lines."""
    def inside(e: Event) -> bool:
        return windows is None or any(s <= e.onset_s < t for s, t in windows)

    g = [e for e in gt_events if inside(e) and (e.shared or len(e.lines) > 1)]
    p = [e for e in est_events if inside(e) and (e.shared or len(e.lines - {UNASSIGNED}) > 1)]
    tp = len(match_events(g, p, onset_tol=onset_tol))
    return prf(tp, len(p) - tp, len(g) - tp)


def lines_f1(gt_events: Sequence[Event], est_events: Sequence[Event], mapping: Mapping[str, str], *,
             onset_tol: float = 0.05) -> dict[str, PRF]:
    """Per GT line onset F1 after the Hungarian mapping (pred line p scored against line mapping[p])."""
    out: dict[str, PRF] = {}
    inv = defaultdict(list)
    for p, r in mapping.items():
        inv[r].append(p)
    for r in sorted({ln for e in gt_events for ln in e.lines}):
        g = [e for e in gt_events if r in e.lines]
        p = [e for e in est_events if any(q in inv[r] for q in e.lines)]
        tp = len(match_events(g, p, onset_tol=onset_tol))
        out[r] = prf(tp, len(p) - tp, len(g) - tp)
    return out


def finite_or_none(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x
