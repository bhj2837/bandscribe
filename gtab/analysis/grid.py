"""Beats -> bar/meter grid (DESIGN S2; M1_M2_SPEC 9.3 item 3, §6.2 ``grid.json``).

``derive_grid(beats_raw, activations, onsets, params)`` turns Beat This! output (``beats.json`` +
``activations.npz``) and the full-mix percussive onsets (:mod:`gtab.analysis.onsets`) into a ``Grid`` dict,
in this order:

1. **Downbeat check.** Beat This!'s minimal postprocessing already snaps every downbeat to its nearest beat,
   so this only verifies that each downbeat coincides with a beat (±1 ms). A violation (e.g. a future DBN
   option) is snapped within ``downbeat_snap_ms`` or dropped; both are counted in the downbeat confidence.

   1b. **One metrical level per song** (added after the first real songs, 2026-10-03). Beat This! switches
   level *within* a song: on three of the five Tier A candidates (AIZO, 絶対零度, 青と夏) the inter-beat
   intervals form two clusters at a 2:1 ratio (sections tracked at 2x or 1/2x the tempo of the rest), which a
   song-wide half/double decision cannot repair and which turned bars into 1/4..9/4 fragments. Every interval
   is classed against the duration-weighted median interval P: ~P/2 (``|log2(d/P) + 1| < 0.25``) = fast,
   ~2P = slow, otherwise left alone (real tempo changes up to ±19 % and swing stay untouched). Runs of >= 2
   fast intervals drop every other beat (the run's boundary beats are kept; the phase keeps the most tracked
   downbeats, then the fewest uneven intervals, then the strongest beat activations); slow intervals get
   their midpoint. Then missed beats (an interval of k x the local interval, k >= 2) are filled.
2. **Tempo octave** (half/double, the dominant remaining failure, DESIGN S2). Candidates: the beats as
   unified in 1b, every other beat (half tempo; counted from each tracked downbeat, so every downbeat is kept
   even where a tracked bar has an odd beat count) and beats plus midpoints (double).
   ``score = F1(onset recall, beat precision) x prior`` where recall = strength-weighted share of percussive
   onsets within ±70 ms of a candidate beat, precision = share of candidate beats with an onset within ±70 ms
   (this is where the onset-density ratio enters: a double-time candidate adds beats without onsets), and
   prior = log-normal tempo prior (centre 120 BPM, sigma 0.5 octave). Switch only if an alternative beats
   "keep" by more than 0.1. A tracked bar of 6/9/12 beats is left to the compound check (step 6): halving an
   eighth-note 6/8 grid would give a wrong 3/4.
3. **Refinement.** Beat This! times are quantised to 20 ms (50 fps), so each beat moves to the strongest
   percussive onset within ±``refine_window_ms`` if that onset is at least 1.5 x the local envelope median;
   otherwise it stays. ``shift_ms`` records the move.
4. **Bars** from downbeats; beats before the first downbeat form the pickup bar 0 (shorter than a bar; whole
   bars' worth of leading beats become full bars). Bar 1 = first full bar (§6 conventions).
5. **Meter.** The bar lines are a minimum-cost segmentation of the beats (dynamic programming): each beat
   costs ``-log p`` if it starts a bar and ``-log(1 - p)`` otherwise, p = Beat This!'s downbeat probability at
   the beat (max over ±2 frames; >= 0.98 on tracked downbeats; clipped to [0.02, 0.995]). A bar whose length
   differs from the meter costs about as much as three bars of contrary downbeat evidence. The meter is the
   mode of the tracked bar lengths; a running mode over ``meter_smooth_bars`` bars overrides it only for a
   consistent run whose length neither divides nor is a multiple of it (a 3/4 passage in 4/4; 2 or 8 there
   means the tracker counted half or double bars, which on real songs comes with the level switches of
   1b), and a song-wide mode of 2 or 8 is taken as half or double bars of 4 (recorded; not when the beats
   divide in three, where 2 dotted-quarter beats are 6/8). So a missed downbeat splits a double-length bar, an
   isolated spurious downbeat is ignored, a run of noisy downbeats (intros, breaks) is re-cut into regular
   bars, and an odd bar (a real 2/4, or a downbeat phase change) is kept only when the following bars confirm
   the new phase. The first bar may be a pickup and the last one may be cut off by the song's end (both free).
   Sections are *not* used here (they are computed from this grid, M1_M2_SPEC A7).
6. **Compound check.** Per beat, onset-envelope energy at 1/3 and 2/3 of the beat interval vs at 1/2 ->
   ``triple_ratio``; >= ``compound_triple_ratio`` -> compound (6/8, 9/8, 12/8, beat unit dotted quarter).
   If Beat This! tracked eighths (6/9/12 beats per bar), the accent pattern decides whether they group in
   threes (regroup 3 tracked beats per dotted-quarter beat) or in twos (regroup 2 -> 3/4). Within
   ±``compound_margin`` of the threshold the decision is ``ambiguous`` and both hypotheses are kept in
   ``alternatives`` (UI toggle later, E14). Triplet *feel* is decided in M3, not here.
7. **Tempo** per bar (median beat interval, 0.1 BPM, BPM of the beat unit) merged into runs; confidences
   from the activations.

Pure numpy; deterministic (no randomness, no threads). Times rounded to 1e-6 s in the output.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

TPB = 48
DOWNBEAT_EXACT_S = 0.001
OCTAVE_TOL_S = 0.070
PRIOR_BPM = 120.0
PRIOR_SIGMA_OCT = 0.5
OCTAVE_SWITCH_MARGIN = 0.1
REFINE_MIN_RATIO = 1.5
REFINE_MEDIAN_S = 1.0
# step 1b (one metrical level)
LEVEL_TOL_OCT = 0.25  # an interval is at level L if |log2(interval / reference) - L| < this (±19 %)
MIN_FAST_RUN = 2  # a single half-length interval is a phase jump of the tracker, not a level switch
# step 5 (meter DP)
P_DOWN_CLIP = 0.02  # downbeat probabilities are clipped to [0.02, 0.995] (bounded per-beat evidence)
P_DOWN_MAX = 0.995
DOWN_WIN_FRAMES = 2  # max of the downbeat activation over +-2 frames (+-40 ms) around a beat
ODD_BAR_EVIDENCE_BARS = 2.5  # an odd bar costs ~2.5 bars of contrary downbeat evidence (x1.1-1.3 by length)
MAX_MODE_BEATS = 12  # longer tracked "bars" (stretches without downbeats) never define the meter
LOCAL_MODE_MIN_BARS = 3  # a local meter other than the song's needs >= 3 tracked bars in the window ...
LOCAL_MODE_SHARE = 0.6  # ... of which >= 60 % have exactly that length
SUBDIV_WIN_S = 0.02  # envelope max within +-20 ms of a subdivision point
MIN_SUBDIV_EVIDENCE = 0.05  # subdivision energy below this share of on-beat energy -> no evidence
FALLBACK_BPM = 120.0
GAP_FILL_MAX_S = 4.0
GAP_FILL_TOL = 0.08
# The flux envelope peaks ~8 ms after an attack (gtab.analysis.onsets.refine_times); envelope lookups at a
# musical time t therefore read around t + ENV_LAG_S.
ENV_LAG_S = 0.008

BEAT_FACTOR = {"quarter": 1.0, "dotted_quarter": 1.5, "eighth": 0.5}  # beat unit in quarter notes

DEFAULT_PARAMS: dict[str, Any] = {
    "refine_window_ms": 35.0,
    "half_double_check": True,
    "compound_triple_ratio": 0.6,
    "compound_margin": 0.1,
    "downbeat_snap_ms": 70.0,
    "meter_smooth_bars": 8,
}


def _r6(x: float) -> float:
    return round(float(x), 6)


def _r4(x: float) -> float:
    return round(float(x), 4)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)))


class _Act:
    """Activation lookup (sigmoid of Beat This! logits at 50 fps); None-safe."""

    def __init__(self, activations: dict[str, Any] | None) -> None:
        self.fps = float((activations or {}).get("fps", 50.0) or 50.0)
        self.beat = None if not activations or activations.get("beat") is None else _sigmoid(activations["beat"])
        self.down = None if not activations or activations.get("downbeat") is None else _sigmoid(activations["downbeat"])

    def _at(self, arr: np.ndarray | None, t: float) -> float | None:
        if arr is None or arr.size == 0:
            return None
        i = int(round(t * self.fps))
        return float(arr[min(max(i, 0), arr.size - 1)])

    def beat_at(self, t: float) -> float | None:
        return self._at(self.beat, t)

    def down_at(self, t: float) -> float | None:
        return self._at(self.down, t)

    def down_max(self, t: float, frames: int) -> float | None:
        """Max downbeat probability within +-``frames`` of t (robust to the 20 ms quantisation)."""
        if self.down is None or self.down.size == 0:
            return None
        i = int(round(t * self.fps))
        a, b = max(0, i - frames), min(self.down.size, i + frames + 1)
        if b <= a:
            return float(self.down[min(max(i, 0), self.down.size - 1)])
        return float(self.down[a:b].max())


class _Onsets:
    def __init__(self, onsets: dict[str, Any] | None) -> None:
        o = onsets or {}
        self.times = np.asarray(o.get("times", []), dtype=np.float64)
        self.strength = np.asarray(o.get("strength", np.ones(self.times.size)), dtype=np.float64)
        if self.strength.size != self.times.size:
            self.strength = np.ones(self.times.size)
        self.env = np.asarray(o.get("env", []), dtype=np.float64)
        self.fps = float(o.get("fps", 86.1328125) or 86.1328125)

    def env_max(self, t: float, half_s: float) -> float:
        if self.env.size == 0:
            return 0.0
        t += ENV_LAG_S
        a = max(0, int(math.floor((t - half_s) * self.fps)))
        b = min(self.env.size, int(math.ceil((t + half_s) * self.fps)) + 1)
        return float(self.env[a:b].max()) if b > a else 0.0

    def env_median(self, t: float, half_s: float) -> float:
        if self.env.size == 0:
            return 0.0
        a = max(0, int(math.floor((t - half_s) * self.fps)))
        b = min(self.env.size, int(math.ceil((t + half_s) * self.fps)) + 1)
        return float(np.median(self.env[a:b])) if b > a else 0.0


# ------------------------------------------------------------------------------------------ step 1


def _check_downbeats(beats: np.ndarray, downbeats: Sequence[float], snap_s: float) -> tuple[np.ndarray, dict]:
    is_down = np.zeros(beats.size, dtype=bool)
    stats = {"n": len(downbeats), "exact": 0, "snapped": 0, "dropped": 0}
    if beats.size == 0:
        stats["dropped"] = len(downbeats)
        return is_down, stats
    for d in downbeats:
        j = int(np.argmin(np.abs(beats - d)))
        dist = abs(float(beats[j]) - float(d))
        if dist <= DOWNBEAT_EXACT_S:
            stats["exact"] += 1
        elif dist <= snap_s:
            stats["snapped"] += 1
        else:
            stats["dropped"] += 1
            continue
        is_down[j] = True
    return is_down, stats


def _fill_gaps(beats: np.ndarray, is_down: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Insert beats the tracker missed: an interval of ~k x the local beat interval (k >= 2, within 8 %, at
    most GAP_FILL_MAX_S long) gets k-1 evenly spaced beats. Irregular breaks are left alone."""
    if beats.size < 4:
        return beats, is_down, 0
    ibi = np.diff(beats)
    out = [float(beats[0])]
    down = [bool(is_down[0])]
    filled = 0
    for i, d in enumerate(ibi):
        local = float(np.median(ibi[max(0, i - 4): i + 5]))
        k = int(round(d / local)) if local > 0 else 1
        if k >= 2 and d <= GAP_FILL_MAX_S and abs(d / k - local) <= GAP_FILL_TOL * local:
            for j in range(1, k):
                out.append(float(beats[i] + d * j / k))
                down.append(False)
            filled += k - 1
        out.append(float(beats[i + 1]))
        down.append(bool(is_down[i + 1]))
    return np.asarray(out), np.asarray(down, dtype=bool), filled


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x, kind="stable")
    cw = np.cumsum(w[order])
    return float(x[order][min(int(np.searchsorted(cw, 0.5 * cw[-1])), x.size - 1)])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True values."""
    out: list[tuple[int, int]] = []
    i, n = 0, int(mask.size)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _move_down(drop_i: int, kept: np.ndarray, beats: np.ndarray, act: _Act) -> int:
    """The kept beat that takes over the downbeat of dropped beat ``drop_i``: the nearest in time; between two
    equally near ones the one with the higher downbeat activation, then the earlier."""
    pos = int(np.searchsorted(kept, drop_i))
    cands = [int(kept[j]) for j in (pos - 1, pos) if 0 <= j < kept.size]
    t = float(beats[drop_i])

    def key(i: int) -> tuple[float, float, int]:
        a = act.down_at(float(beats[i]))
        return (round(abs(float(beats[i]) - t), 6), -(a if a is not None else 0.0), i)

    return min(cands, key=key)


def _unify_level(beats: np.ndarray, is_down: np.ndarray, act: _Act) -> tuple[np.ndarray, np.ndarray, dict]:
    """One metrical level for the whole song (module doc, step 1b)."""
    info: dict[str, Any] = {"reference_bpm": None, "fast_runs": 0, "slow_runs": 0, "dropped": 0, "inserted": 0}
    if beats.size < 8:
        return beats, is_down, info
    d = np.diff(beats)
    ref = _weighted_median(d, d)
    if ref <= 0:
        return beats, is_down, info
    info["reference_bpm"] = round(60.0 / ref, 2)
    r = np.log2(np.maximum(d, 1e-9) / ref)
    fast = np.abs(r + 1.0) < LEVEL_TOL_OCT
    slow = np.abs(r - 1.0) < LEVEL_TOL_OCT
    n = int(beats.size)
    keep = np.ones(n, dtype=bool)
    for s0, e0 in _runs(fast):  # intervals s0..e0-1 = beats s0..e0
        if e0 - s0 < MIN_FAST_RUN:
            continue
        cands = []
        for phase in (0, 1):
            sel = set(range(s0 + phase, e0 + 1, 2))
            if s0 > 0:  # a boundary beat at the normal level stays (only an open end of the song may go)
                sel.add(s0)
            if e0 < n - 1:
                sel.add(e0)
            idx = sorted(sel)
            n_down = sum(1 for i in idx if is_down[i])
            uneven = sum(1 for a, b in zip(idx, idx[1:]) if b - a != 2)
            strength = sum((act.beat_at(float(beats[i])) or 0.0) for i in idx)
            cands.append(((n_down, -uneven, round(strength, 6)), idx))
        chosen = set(cands[0][1] if cands[0][0] >= cands[1][0] else cands[1][1])
        for i in range(s0, e0 + 1):
            if i not in chosen:
                keep[i] = False
        info["fast_runs"] += 1
    kept_idx = np.flatnonzero(keep)
    new_down = is_down & keep
    for i in np.flatnonzero(is_down & ~keep):
        new_down[_move_down(int(i), kept_idx, beats, act)] = True
    slow_runs = _runs(slow)
    mids = [0.5 * (float(beats[i]) + float(beats[i + 1])) for s0, e0 in slow_runs for i in range(s0, e0)]
    info["dropped"] = int(n - kept_idx.size)
    info["slow_runs"] = len(slow_runs)
    info["inserted"] = len(mids)
    if not info["dropped"] and not mids:
        return beats, is_down, info
    t_all = np.concatenate([beats[kept_idx], np.asarray(mids, dtype=np.float64)])
    d_all = np.concatenate([new_down[kept_idx], np.zeros(len(mids), dtype=bool)])
    order = np.argsort(t_all, kind="stable")
    return t_all[order], d_all[order], info


# ------------------------------------------------------------------------------------------ step 2


def _bpm(times: np.ndarray) -> float:
    if times.size < 2:
        return FALLBACK_BPM
    return 60.0 / float(np.median(np.diff(times)))


def _prior(bpm: float) -> float:
    return math.exp(-0.5 * (math.log2(bpm / PRIOR_BPM) / PRIOR_SIGMA_OCT) ** 2)


def _near(sorted_ref: np.ndarray, x: np.ndarray, tol: float) -> np.ndarray:
    """For each x: is some element of sorted_ref within tol?"""
    if sorted_ref.size == 0 or x.size == 0:
        return np.zeros(x.size, dtype=bool)
    idx = np.searchsorted(sorted_ref, x)
    lo = np.clip(idx - 1, 0, sorted_ref.size - 1)
    hi = np.clip(idx, 0, sorted_ref.size - 1)
    d = np.minimum(np.abs(sorted_ref[lo] - x), np.abs(sorted_ref[hi] - x))
    return d <= tol


def _octave_score(cand: np.ndarray, ons: _Onsets) -> dict[str, float]:
    bpm = _bpm(cand)
    if cand.size < 2 or ons.times.size == 0:
        return {"bpm": round(bpm, 2), "recall": 0.0, "precision": 0.0, "prior": _r4(_prior(bpm)), "score": 0.0,
                "onsets_per_beat": 0.0}
    span = (ons.times >= cand[0] - OCTAVE_TOL_S) & (ons.times <= cand[-1] + OCTAVE_TOL_S)
    t, w = ons.times[span], ons.strength[span]
    hit = _near(cand, t, OCTAVE_TOL_S)
    recall = float(w[hit].sum() / w.sum()) if w.sum() > 0 else 0.0
    precision = float(_near(t, cand, OCTAVE_TOL_S).mean()) if t.size else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    prior = _prior(bpm)
    return {"bpm": round(bpm, 2), "recall": _r4(recall), "precision": _r4(precision), "prior": _r4(prior),
            "score": _r4(f1 * prior), "onsets_per_beat": _r4(t.size / cand.size)}


def _double(beats: np.ndarray, is_down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mids = (beats[:-1] + beats[1:]) / 2.0
    out = np.empty(beats.size + mids.size)
    out[0::2] = beats
    out[1::2] = mids
    down = np.zeros(out.size, dtype=bool)
    down[0::2] = is_down
    return out, down


def _half(beats: np.ndarray, is_down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every other beat, counted from each tracked downbeat (and backwards from the first one), so every
    downbeat survives even where a tracked bar has an odd number of beats."""
    d = [int(x) for x in np.flatnonzero(is_down)]
    if not d:
        keep = list(range(0, beats.size, 2))
    else:
        keep = [i for i in range(d[0]) if (d[0] - i) % 2 == 0]
        for k, s0 in enumerate(d):
            e0 = d[k + 1] if k + 1 < len(d) else int(beats.size)
            keep.extend(range(s0, e0, 2))
    idx = np.asarray(keep, dtype=np.int64)
    return beats[idx], is_down[idx].copy()


def _modal_bar_beats(is_down: np.ndarray) -> int | None:
    d = np.flatnonzero(is_down)
    if d.size < 2:
        return None
    return _mode([int(x) for x in np.diff(d)])


def _tempo_octave(beats: np.ndarray, is_down: np.ndarray, ons: _Onsets, *, allow_double: bool = True
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    modal = _modal_bar_beats(is_down)
    cands: dict[str, tuple[np.ndarray, np.ndarray]] = {"keep": (beats, is_down)}
    if beats.size >= 2 and allow_double:
        cands["double"] = _double(beats, is_down)
    if beats.size >= 4 and modal not in (6, 9, 12):
        cands["half"] = _half(beats, is_down)
    scores = {k: _octave_score(v[0], ons) for k, v in cands.items()}
    decision = "keep"
    best = scores["keep"]["score"] + OCTAVE_SWITCH_MARGIN
    for k in ("half", "double"):
        if k in scores and scores[k]["score"] > best:
            decision, best = k, scores[k]["score"]
    span = (beats[-1] - beats[0]) if beats.size >= 2 else 0.0
    hyp = {"decision": decision, "scores": scores,
           "onset_rate_hz": _r4(ons.times.size / span) if span > 0 else 0.0,
           "excluded": ([] if allow_double else ["double: triple subdivision (compound meter)"])
           + (["half: 6/9/12 tracked beats per bar (compound regrouping)"] if modal in (6, 9, 12) else [])}
    b, d = cands[decision]
    return b, d, hyp


# ------------------------------------------------------------------------------------------ step 3


def _refine(beats: np.ndarray, ons: _Onsets, window_s: float) -> np.ndarray:
    out = beats.copy()
    if ons.times.size == 0 or window_s <= 0:
        return out
    for i, t in enumerate(beats):
        lo = np.searchsorted(ons.times, t - window_s, side="left")
        hi = np.searchsorted(ons.times, t + window_s, side="right")
        if hi <= lo:
            continue
        j = lo + int(np.argmax(ons.strength[lo:hi]))
        med = ons.env_median(float(t), REFINE_MEDIAN_S)
        if ons.strength[j] >= REFINE_MIN_RATIO * med and ons.strength[j] > 0:
            out[i] = ons.times[j]
    # keep the beats strictly increasing (two beats may not claim the same onset)
    for i in range(1, out.size):
        if out[i] <= out[i - 1] + 1e-3:
            out[i] = beats[i] if beats[i] > out[i - 1] + 1e-3 else out[i - 1] + 1e-3
    return out


# ------------------------------------------------------------------------------------------ steps 4-5


def _mode(values: Sequence[int], prefer: int | None = None) -> int:
    c = Counter(values)
    top = max(c.values())
    tied = sorted(v for v, n in c.items() if n == top)
    if prefer is not None and prefer in tied:
        return prefer
    return tied[0]


def _segments(n_beats_total: int, down_idx: list[int]) -> list[tuple[int, int]]:
    """(first beat index, n beats) per bar from sorted downbeat indices; the pickup is not included."""
    segs = []
    for k, s in enumerate(down_idx):
        e = down_idx[k + 1] if k + 1 < len(down_idx) else n_beats_total
        if e > s:
            segs.append((s, e - s))
    return segs


def _local_modes(n: int, down_idx: list[int], window: int, *, simple_beats: bool = True
                 ) -> tuple[np.ndarray, int, int | None]:
    """Per beat: the meter (beats per bar) the bar segmentation should follow, the song's meter, and the
    tracked mode it replaces (2 or 8 -> 4) if any.

    The song's meter is the mode of the tracked bar lengths. A local meter (running mode over ``window``
    tracked bars) replaces it only for a consistent run of bars whose length is neither a divisor nor a
    multiple of it: 3 in a 4/4 song is a real 3/4 passage, but 2 or 8 means the tracker counted half or
    double bars there (on real songs this comes with the level switches of step 1b: sections tracked at
    double tempo were also tracked with half-length bars). A song-wide mode of 2 or 8 becomes 4 for the same
    reason (2/4 or 8/4 throughout is rare in this repertoire; a tracker at half tempo that was doubled by the
    octave check leaves 8 beats per tracked bar; the tracked mode is recorded) unless the beats divide in
    three (``simple_beats=False``: two dotted-quarter beats per bar is 6/8).
    """
    segs = _segments(n, down_idx)
    counts = [c for _, c in segs]
    body = counts[:-1] if len(counts) > 1 else counts  # the final bar is usually cut off by the song's end
    # a "bar" of one beat is a tracker artefact (every beat flagged as a downbeat), never a meter; very long
    # ones are stretches without downbeats
    usable = [c for c in body if 2 <= c <= MAX_MODE_BEATS]
    global_mode = _mode(usable) if usable else 4
    replaced = None
    if global_mode in (2, 8) and simple_beats:
        replaced, global_mode = global_mode, 4
    m_beat = np.full(n, global_mode, dtype=np.int64)
    half = max(1, window) // 2
    for i, (s0, c) in enumerate(segs):
        win = body[max(0, i - half): min(len(body), i + half + 1)]
        local = _mode([x for x in win if 2 <= x <= MAX_MODE_BEATS] or [global_mode], prefer=global_mode)
        # a noisy run of downbeats (intros, breaks) has no consistent length and keeps the song's meter
        if local != global_mode and (local % global_mode == 0 or global_mode % local == 0
                                     or len(win) < LOCAL_MODE_MIN_BARS
                                     or sum(1 for x in win if x == local) < LOCAL_MODE_SHARE * len(win)):
            local = global_mode
        m_beat[s0:s0 + c] = local
    if segs:
        m_beat[:segs[0][0]] = m_beat[segs[0][0]]
    return m_beat, global_mode, replaced


def _downbeat_prob(t_raw: np.ndarray, is_down: np.ndarray, act: _Act) -> np.ndarray:
    """Downbeat probability per beat: the activation near the tracker's own beat time (>= 0.98 on tracked
    downbeats; 0.02 / 0.98 by the flags alone without activations), clipped to [0.02, 0.995] so one beat
    never outweighs a few bars of evidence. Above 0.98 the activation still ranks competing downbeats (which
    of two half-bar downbeats starts the bar)."""
    p = np.full(t_raw.size, P_DOWN_CLIP, dtype=np.float64)
    if act.down is not None:
        for i, t in enumerate(t_raw):
            a = act.down_max(float(t), DOWN_WIN_FRAMES)
            if a is not None:
                p[i] = a
    p[is_down] = np.maximum(p[is_down], 1.0 - P_DOWN_CLIP)
    return np.clip(p, P_DOWN_CLIP, P_DOWN_MAX)


def _odd_cost(length: int, mode: int, lam: float) -> float:
    # closer to the mode is cheaper (a +1 phase change becomes a 5/4 bar rather than 1/4); a one-beat bar costs
    # double; the tiny length term prefers 2/4 over 6/4 for a 2-beat phase change
    return lam * (1.0 + 0.1 * abs(length - mode)) + (lam if length == 1 else 0.0) + 0.01 * length


def _meter_dp(p: np.ndarray, m_beat: np.ndarray) -> tuple[int, list[tuple[int, int]], dict[str, Any]]:
    """Minimum-cost bar segmentation (module doc, step 5). Returns (pickup beats, bars as (start, n), info)."""
    n = int(p.size)
    lam = ODD_BAR_EVIDENCE_BARS * 2.0 * -math.log(P_DOWN_CLIP)
    if n == 0:
        return 0, [], {"consistent_ratio": 0.0, "odd_bars": 0, "tracked_agreement": 0.0, "odd_bar_cost": _r4(lam)}
    c_in = -np.log(1.0 - p)
    c_down = -np.log(p)
    cum = np.concatenate([[0.0], np.cumsum(c_in)])
    best = np.full(n + 1, np.inf)
    back: list[tuple[int, str]] = [(0, "")] * (n + 1)
    best[0] = 0.0
    for e in range(1, min(int(m_beat[0]) - 1, n) + 1):  # pickup: [0, e) holds no downbeat
        c = float(cum[e])
        if c < best[e] - 1e-12:
            best[e], back[e] = c, (0, "pickup")
    for s0 in range(n):
        if not np.isfinite(best[s0]):
            continue
        m = int(m_beat[s0])
        # bar [s0, e): downbeat at s0, no downbeat on s0+1 .. e-1
        base = float(best[s0] + c_down[s0] - c_in[s0] - cum[s0])
        for length in range(1, 2 * m):
            e = s0 + length
            if e > n:
                break
            free = length == m or (e == n and length < m)
            c = base + float(cum[e]) + (0.0 if free else _odd_cost(length, m, lam))
            if c < best[e] - 1e-12:
                best[e], back[e] = c, (s0, "bar")
    bars: list[tuple[int, int]] = []
    pickup = 0
    e = n
    while e > 0:
        s0, kind = back[e]
        if kind == "pickup":
            pickup = e
        else:
            bars.append((s0, e - s0))
        e = s0
    bars.reverse()
    body = bars[:-1] if len(bars) > 1 else bars
    ok = sum(1 for s0, c in body if c == int(m_beat[s0]))
    starts = [s0 for s0, _ in bars]
    agree = float(np.mean(p[starts] >= 0.5)) if starts else 0.0
    return pickup, bars, {"consistent_ratio": _r4(ok / max(1, len(body))), "odd_bars": len(body) - ok,
                          "tracked_agreement": _r4(agree), "odd_bar_cost": _r4(lam)}


# ------------------------------------------------------------------------------------------ step 6


def _triple_ratio(times: np.ndarray, ons: _Onsets) -> float | None:
    """Share of subdivision energy at 1/3, 2/3 of the beat interval vs 1/2 (None = no subdivision evidence)."""
    if times.size < 2 or ons.env.size == 0:
        return None
    e3 = e2 = eb = 0.0
    for a, b in zip(times[:-1], times[1:]):
        d = float(b - a)
        if d <= 0:
            continue
        floor = ons.env_median(float(a + d / 2), max(d, 0.5))
        v13 = max(0.0, ons.env_max(float(a + d / 3), SUBDIV_WIN_S) - floor)
        v23 = max(0.0, ons.env_max(float(a + 2 * d / 3), SUBDIV_WIN_S) - floor)
        v12 = max(0.0, ons.env_max(float(a + d / 2), SUBDIV_WIN_S) - floor)
        e3 += (v13 + v23) / 2.0
        e2 += v12
        eb += max(0.0, ons.env_max(float(a), SUBDIV_WIN_S) - floor)
    if e3 + e2 <= 0 or (eb > 0 and (e3 + e2) < MIN_SUBDIV_EVIDENCE * eb):
        return None
    return e3 / (e3 + e2)


def _grouping(times: np.ndarray, segs: list[tuple[int, int]], n: int, ons: _Onsets) -> str:
    """For bars of n tracked (eighth) beats: do the accents group them in threes or twos?"""
    if n % 3 != 0:
        return "twos"
    if n % 2 != 0:
        return "threes"
    only3 = [k for k in range(n) if k % 3 == 0 and k % 2 != 0]  # positions accented only by grouping in threes
    only2 = [k for k in range(n) if k % 2 == 0 and k % 3 != 0]
    s3, s2 = [], []
    for start, cnt in segs:
        if cnt != n:
            continue
        s3 += [ons.env_max(float(times[start + k]), SUBDIV_WIN_S) for k in only3]
        s2 += [ons.env_max(float(times[start + k]), SUBDIV_WIN_S) for k in only2]
    if not s3 or not s2:
        return "twos"
    return "threes" if float(np.mean(s3)) > float(np.mean(s2)) else "twos"


def _regroup(segs: list[tuple[int, int]], n_beats: int, group: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Keep every ``group``-th beat from each bar start; returns (kept original indices, new segs)."""
    keep: list[int] = []
    new_segs: list[tuple[int, int]] = []
    first_bar_start = segs[0][0] if segs else n_beats
    lead = list(range(0, first_bar_start))
    # pickup beats: keep those aligned to the group counted backwards from the first downbeat
    lead_keep = [i for i in lead if (first_bar_start - i) % group == 0]
    keep += lead_keep
    for start, cnt in segs:
        idx = list(range(start, start + cnt, group))
        new_segs.append((len(keep), len(idx)))
        keep += idx
    return np.asarray(keep, dtype=np.int64), new_segs


# ------------------------------------------------------------------------------------------ main


def _fallback_beats(duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    step = 60.0 / FALLBACK_BPM
    t = np.arange(0.0, max(duration_s, step), step)
    down = np.zeros(t.size, dtype=bool)
    down[0::4] = True
    return t, down


def derive_grid(beats_raw: dict[str, Any], activations: dict[str, Any] | None, onsets: dict[str, Any] | None,
                params: dict[str, Any]) -> dict[str, Any]:
    """Build the ``grid.json`` document (M1_M2_SPEC §6.2) from Beat This! output and percussive onsets."""
    p = {**DEFAULT_PARAMS, **{k: v for k, v in params.items() if k in DEFAULT_PARAMS}}
    act = _Act(activations)
    ons = _Onsets(onsets)
    duration = float(beats_raw.get("duration_s") or 0.0)
    raw = np.asarray(sorted(float(t) for t in beats_raw.get("beats_s", [])), dtype=np.float64)
    warnings: list[str] = []

    # 1. downbeat check
    fallback = raw.size < 2
    if fallback:
        warnings.append("비트를 거의 찾지 못해 120 BPM 4/4 격자로 대신합니다")
        raw, is_down = _fallback_beats(duration)
        db_stats = {"n": 0, "exact": 0, "snapped": 0, "dropped": 0}
    else:
        is_down, db_stats = _check_downbeats(raw, beats_raw.get("downbeats_s", []),
                                             float(p["downbeat_snap_ms"]) / 1000.0)
    if not is_down.any():
        warnings.append("다운비트가 없어 첫 비트부터 4박으로 마디를 나눴습니다")
        is_down[0::4] = True

    # 1b. one metrical level, then fill missed beats
    level_info: dict[str, Any] = {"reference_bpm": None, "fast_runs": 0, "slow_runs": 0, "dropped": 0,
                                  "inserted": 0}
    if not fallback:
        raw, is_down, level_info = _unify_level(raw, is_down, act)
        if level_info["dropped"] or level_info["inserted"]:
            warnings.append(f"비트 추적기가 곡 중간에 박 단위를 바꿔 한 단위로 맞췄습니다 "
                            f"(뺀 비트 {level_info['dropped']}개, 넣은 비트 {level_info['inserted']}개)")
    raw, is_down, n_filled = _fill_gaps(raw, is_down) if not fallback else (raw, is_down, 0)
    db_stats["beats_filled"] = n_filled

    # 2. tempo octave (doubling is never offered when the tracked beats already divide in three)
    thr, margin = float(p["compound_triple_ratio"]), float(p["compound_margin"])
    pre_ratio = _triple_ratio(raw, ons)
    if p["half_double_check"] and not fallback:
        beats, is_down, octave = _tempo_octave(
            raw, is_down, ons, allow_double=pre_ratio is None or pre_ratio < thr - margin)
    else:
        beats = raw
        octave = {"decision": "keep", "scores": {}, "onset_rate_hz": 0.0}
    t_raw = beats.copy()

    # 3. refinement
    times = _refine(beats, ons, float(p["refine_window_ms"]) / 1000.0) if not fallback else beats.copy()

    # 4-5. bars and meter
    down_idx = [int(i) for i in np.flatnonzero(is_down)]
    pre_meter_ratio = _triple_ratio(times, ons)
    m_beat, global_mode, replaced = _local_modes(int(times.size), down_idx, int(p["meter_smooth_bars"]),
                                                 simple_beats=pre_meter_ratio is None or pre_meter_ratio <= thr - margin)
    lead, segs, meter_info = _meter_dp(_downbeat_prob(t_raw, is_down, act), m_beat)
    meter_info["mode"] = global_mode
    meter_info["tracked_mode_replaced"] = replaced
    if replaced == 2:
        warnings.append("추적된 마디가 대부분 2박이라 반 마디로 보고 4박 마디로 묶었습니다")
    elif replaced == 8:
        warnings.append("추적된 마디가 대부분 8박이라 두 마디로 보고 4박 마디로 나눴습니다")

    # 6. compound check
    modal = meter_info.get("mode") or 4
    grouped = None
    if modal in (6, 9, 12):
        grouped = _grouping(times, segs, modal, ons)
        kept, segs = _regroup(segs, times.size, 3 if grouped == "threes" else 2)
        times, t_raw = times[kept], t_raw[kept]
        lead = segs[0][0] if segs else times.size
    ratio = _triple_ratio(times, ons)
    if grouped == "threes":
        # tracked eighths grouped in threes: the regrouped beats are dotted quarters by construction
        ratio_eff = 1.0 if ratio is None else max(ratio, thr + margin + 1e-9)
    else:
        ratio_eff = 0.0 if ratio is None else ratio
    if ratio_eff >= thr + margin:
        decision = "compound"
    elif ratio_eff <= thr - margin:
        decision = "simple"
    else:
        decision = "ambiguous"
    primary_compound = ratio_eff >= thr
    compound_hyp = {"triple_ratio": None if ratio is None else _r4(ratio), "decision": decision,
                    "grouping_of_tracked_beats": grouped,
                    "alternatives": []}

    # bar list (pickup = bar 0; the meter step keeps it shorter than a bar)
    first_n = segs[0][1] if segs else 4
    pickup_beats = int(lead) if segs else 0
    bar_segs = ([(0, pickup_beats)] if pickup_beats else []) + list(segs)
    if not segs and times.size:  # fewer beats than a pickup: one partial bar
        bar_segs, pickup_beats = [(0, int(times.size))], 0

    def meter_of(n: int, compound: bool) -> tuple[int, int, str]:
        return (3 * n, 8, "dotted_quarter") if compound else (n, 4, "quarter")

    ibi_all = float(np.median(np.diff(times))) if times.size >= 2 else 60.0 / FALLBACK_BPM
    bars: list[dict[str, Any]] = []
    beats_out: list[dict[str, Any]] = []
    tempo_per_bar: list[tuple[int, float]] = []
    bar_no = 0 if pickup_beats else 1
    for k, (start, n) in enumerate(bar_segs):
        is_pickup = pickup_beats > 0 and k == 0
        num_n = first_n if is_pickup else n
        numerator, denominator, unit = meter_of(num_n, primary_compound)
        end_idx = start + n
        start_s = float(times[start])
        if end_idx < times.size:
            end_s = float(times[end_idx])
        else:
            local = np.diff(times[start:end_idx]) if n >= 2 else np.asarray([ibi_all])
            end_s = float(times[end_idx - 1]) + float(np.median(local) if local.size else ibi_all)
            if duration > 0:
                end_s = max(min(end_s, duration), float(times[end_idx - 1]) + 1e-3)
        bars.append({"bar": bar_no, "start_s": _r6(start_s), "end_s": _r6(end_s), "n_beats": int(n),
                     "numerator": int(numerator), "denominator": int(denominator), "beat_unit": unit,
                     "pickup": bool(is_pickup)})
        ivals = np.diff(np.append(times[start:end_idx], end_s)) if end_idx < times.size else np.diff(times[start:end_idx])
        ibi = float(np.median(ivals)) if ivals.size else ibi_all
        tempo_per_bar.append((bar_no, round(60.0 / ibi, 1) if ibi > 0 else FALLBACK_BPM))
        for j in range(start, end_idx):
            a = act.beat_at(float(t_raw[j]))
            beats_out.append({"t_s": _r6(times[j]), "t_raw_s": _r6(t_raw[j]), "bar": bar_no, "beat": j - start + 1,
                              "is_downbeat": j == start and not is_pickup,
                              "shift_ms": round((float(times[j]) - float(t_raw[j])) * 1000.0, 2),
                              "activation": None if a is None else _r4(a)})
        bar_no += 1

    # alternatives: both hypotheses with their scores
    typical = _mode([b["n_beats"] for b in bars if not b["pickup"]] or [4])
    alt_c = meter_of(typical, True)
    alt_s = meter_of(typical, False)
    compound_hyp["alternatives"] = sorted([
        {"numerator": alt_c[0], "denominator": alt_c[1], "beat_unit": alt_c[2], "score": _r4(ratio_eff)},
        {"numerator": alt_s[0], "denominator": alt_s[1], "beat_unit": alt_s[2], "score": _r4(1.0 - ratio_eff)},
    ], key=lambda a: (-a["score"], a["beat_unit"]))

    # 7. tempo segments and meter runs
    tempo: list[dict[str, Any]] = []
    for bar, bpm in tempo_per_bar:
        if tempo and tempo[-1]["bpm"] == bpm and tempo[-1]["end_bar"] == bar:
            tempo[-1]["end_bar"] = bar + 1
        else:
            tempo.append({"start_bar": bar, "end_bar": bar + 1, "bpm": bpm})
    meter: list[dict[str, Any]] = []
    for b in bars:
        sig = (b["numerator"], b["denominator"], b["beat_unit"])
        if meter and (meter[-1]["numerator"], meter[-1]["denominator"], meter[-1]["beat_unit"]) == sig \
                and meter[-1]["end_bar"] == b["bar"]:
            meter[-1]["end_bar"] = b["bar"] + 1
        else:
            meter.append({"start_bar": b["bar"], "end_bar": b["bar"] + 1, "numerator": sig[0],
                          "denominator": sig[1], "beat_unit": sig[2]})
    unit_counts = Counter(b["beat_unit"] for b in bars if not b["pickup"]) or Counter({"quarter": 1})
    dominant = sorted(unit_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    # confidences
    beat_acts = [b["activation"] for b in beats_out if b["activation"] is not None]
    down_acts = [act.down_at(float(t_raw[s])) for s, _ in segs] if segs else []
    down_acts = [a for a in down_acts if a is not None]
    viol = (db_stats["snapped"] + db_stats["dropped"]) / db_stats["n"] if db_stats["n"] else 0.0
    compound_certainty = 1.0 if decision != "ambiguous" else 0.5
    confidence = {
        "beats": _r4(np.mean(beat_acts)) if beat_acts else None,
        "downbeats": _r4(float(np.mean(down_acts)) * (1.0 - viol)) if down_acts else None,
        "meter": _r4(float(meter_info.get("consistent_ratio", 0.0)) * compound_certainty),
    }
    if fallback:
        confidence = {"beats": 0.0, "downbeats": 0.0, "meter": 0.0}

    return {
        "format": "gtab.grid/1",
        "tpb": TPB,
        "duration_s": _r6(duration if duration > 0 else (bars[-1]["end_s"] if bars else 0.0)),
        "beats": beats_out,
        "bars": bars,
        "tempo": tempo,
        "meter": meter,
        "beat_unit": dominant,
        "pickup": {"present": pickup_beats > 0, "beats": int(pickup_beats)},
        "hypotheses": {"tempo_octave": octave, "compound": compound_hyp, "beat_level": level_info,
                       "downbeat_check": db_stats, "meter_smoothing": meter_info, "warnings": warnings},
        "confidence": confidence,
        "params": dict(p),
    }
