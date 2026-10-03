"""Note matching primitives shared by the metrics.

``match_note_pairs`` is the mir_eval path: **pitches are converted MIDI -> Hz with ``mir_eval.util.midi_to_hz``
before any mir_eval call.** mir_eval expects Hz; given raw MIDI numbers, 61 vs 60 is ≈ 29 "cents" apart, so
semitone errors would count as hits (and the +12 octave identity would still pass, hiding the bug).

``match_min_cost`` is the line-agnostic event matcher of the assignment metrics: per pitch, maximum number of
pairs within the onset tolerance, then minimum total |Δt| (Hungarian on small time-connected components).
``max_bipartite`` is the maximum matching used by tick F1.
mir_eval / scipy are imported lazily (CLI start-up stays light, spec §0).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

_BIG = 1e9


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    import mir_eval

    return np.asarray(mir_eval.util.midi_to_hz(np.asarray(midi, dtype=float)), dtype=float)


def sanitize_intervals(intervals: np.ndarray, min_dur: float = 1e-3) -> np.ndarray:
    """mir_eval rejects non-positive durations; zero-length notes (possible after rounding) get ``min_dur``."""
    iv = np.asarray(intervals, dtype=float).reshape(-1, 2).copy()
    iv[:, 1] = np.maximum(iv[:, 1], iv[:, 0] + min_dur)
    return iv


def match_note_pairs(ref_intervals: np.ndarray, ref_midi: np.ndarray, est_intervals: np.ndarray, est_midi: np.ndarray,
                     *, onset_tol: float = 0.05, pitch_tol_cents: float = 50.0, offset_ratio: float | None = None,
                     offset_min_tolerance: float = 0.05) -> list[tuple[int, int]]:
    """mir_eval.transcription.match_notes on Hz pitches -> list of (ref index, est index)."""
    ref_midi = np.asarray(ref_midi, dtype=float).reshape(-1)
    est_midi = np.asarray(est_midi, dtype=float).reshape(-1)
    if ref_midi.size == 0 or est_midi.size == 0:
        return []
    import mir_eval

    return [(int(i), int(j)) for i, j in mir_eval.transcription.match_notes(
        sanitize_intervals(ref_intervals), midi_to_hz(ref_midi), sanitize_intervals(est_intervals), midi_to_hz(est_midi),
        onset_tolerance=onset_tol, pitch_tolerance=pitch_tol_cents, offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance, strict=False)]


def _components(a: np.ndarray, b: np.ndarray, tol: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split two onset lists into groups that can only match inside themselves (gap > tol between groups)."""
    ev = sorted([(float(t), 0, i) for i, t in enumerate(a)] + [(float(t), 1, j) for j, t in enumerate(b)])
    groups: list[tuple[list[int], list[int]]] = []
    last = -np.inf
    for t, side, idx in ev:
        if t - last > tol or not groups:
            groups.append(([], []))
        groups[-1][side].append(idx)
        last = t
    return [(np.array(g, dtype=int), np.array(h, dtype=int)) for g, h in groups if g and h]


def match_min_cost(ref_onsets: Sequence[float], ref_pitch: Sequence[int], est_onsets: Sequence[float],
                   est_pitch: Sequence[int], tol: float) -> list[tuple[int, int]]:
    """Same pitch, |Δonset| ≤ tol; maximum number of pairs, then minimum total |Δt|. Deterministic."""
    from scipy.optimize import linear_sum_assignment

    ro, eo = np.asarray(ref_onsets, dtype=float), np.asarray(est_onsets, dtype=float)
    by_r: dict[int, list[int]] = defaultdict(list)
    by_e: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(ref_pitch):
        by_r[int(p)].append(i)
    for j, p in enumerate(est_pitch):
        by_e[int(p)].append(j)
    pairs: list[tuple[int, int]] = []
    for p in sorted(set(by_r) & set(by_e)):
        ri, ej = np.array(by_r[p]), np.array(by_e[p])
        for ga, gb in _components(ro[ri], eo[ej], tol):
            r_idx, e_idx = ri[ga], ej[gb]
            d = np.abs(ro[r_idx][:, None] - eo[e_idx][None, :])
            cost = np.where(d <= tol + 1e-12, d, _BIG)
            rr, cc = linear_sum_assignment(cost)
            pairs.extend((int(r_idx[a]), int(e_idx[b])) for a, b in zip(rr, cc) if cost[a, b] < _BIG)
    pairs.sort()
    return pairs


def max_bipartite(n_left: int, n_right: int, edges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Maximum-cardinality matching of a bipartite graph given as (left, right) edges."""
    if not edges or n_left == 0 or n_right == 0:
        return []
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    rows, cols = zip(*edges)
    g = csr_matrix((np.ones(len(edges), dtype=np.int8), (np.array(rows), np.array(cols))), shape=(n_left, n_right))
    m = maximum_bipartite_matching(g, perm_type="column")
    return [(int(i), int(j)) for i, j in enumerate(m) if j >= 0]
