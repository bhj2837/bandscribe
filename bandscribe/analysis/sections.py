"""Sections and repeat map from the full mix (DESIGN S2 "구간과 반복"; M1_M2_SPEC 9.3 item 4, §6.3).

``analyze(mix_wav, grid, onsets, params) -> (sections_doc, repeat_map_doc)``:

1. **Features** (22050 Hz mono mix, hop 512): ``chroma_cens`` and 13 MFCCs (standardised per coefficient),
   both aggregated per grid beat (CENS by median, MFCC by mean).
2. **Laplacian segmentation** (McFee & Ellis 2014, as in the librosa example): beat recurrence
   (``librosa.segment.recurrence_matrix``, affinity, symmetric), smoothed along diagonals, plus a path graph
   weighted by MFCC continuity; normalised Laplacian -> eigenvectors; the number of clusters k in
   [2, ``max_sections``] is the largest eigengap; KMeans on the first k (row-normalised) eigenvectors gives a
   label per beat. The recurrence runs on CENS **and** MFCC (both unit scale) stacked over a centred window
   of two bars' worth of beats: on CENS of single beats the clusters follow chords (a verse's C-G-Am-F bars
   become four clusters and a chorus sharing an F chord joins one of them); the two-bar context and timbre
   make beats cluster by section instead (checked on synthetic ABABCA/AABBAC/ABCABC songs).
3. **Sections**: label changes -> boundaries snapped to the nearest downbeat (bar start); segments shorter
   than ``min_section_bars`` merged into the more similar neighbour; each segment's mean embedding is
   assigned to the nearest KMeans centroid (clustering of segment feature means); adjacent segments with the
   same label merge; labels renamed A, B, C... in order of first appearance. Bar ranges are half-open.
4. **Repeat map**: bar-level self-similarity (cosine of standardised per-bar feature sequences, 4 slots per
   bar) smoothed along diagonals in the lag domain (``recurrence_to_lag`` -> ±1 bar -> back), per-bar best
   matches with score >= ``repeat_min_score``; repeat groups = labels with >= 2 sections, reference = the
   medoid occurrence. ``offset_s`` (timing of an occurrence/bar relative to its reference after aligning bar
   starts, positive = later) comes from cross-correlating the percussive onset envelopes (8x upsampled,
   parabolic peak) over ±``align_max_offset_ms``, summed bar by bar so tempo drift does not blur it.

**Determinism** (eigendecomposition and KMeans can differ in the last bit across runs, especially with
multithreaded OpenBLAS): everything runs under ``threadpoolctl.threadpool_limits(1)``; eigenvectors are
sign-normalised (largest-|x| component positive) and ordered by (eigenvalue, index); KMeans uses
``random_state=0, n_init=10``; scores are rounded to 1e-4 and times to 1e-6 before writing.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SR = 22050
FEAT_HOP = 512
N_MFCC = 13
REC_WIDTH = 3  # beats: ignore self-similarity within +-3 beats
PATH_FILTER = 7  # beats: diagonal median filter of the recurrence
EVEC_MEDIAN = 9
SLOTS_PER_BAR = 4
MAX_MATCHES = 8
UPSAMPLE = 8
MIN_BEATS_FOR_SEGMENTATION = 16
METHOD = "laplacian segmentation (McFee-Ellis) on beat-sync CENS + MFCC; k by eigengap"

DEFAULT_PARAMS: dict[str, Any] = {
    "min_section_bars": 4,
    "max_sections": 16,
    "repeat_min_score": 0.6,
    "align_max_offset_ms": 50.0,
}


def _r4(x: float) -> float:
    return round(float(x), 4)


def _r6(x: float) -> float:
    return round(float(x), 6)


# ------------------------------------------------------------------------------------------ features


def _beat_features(y: np.ndarray, beat_times: np.ndarray, end_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(CENS (12, K), standardised MFCC (13, K)) per beat segment [beat_i, beat_i+1) (last one up to end_s)."""
    import librosa

    cens = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=FEAT_HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=FEAT_HOP, n_mfcc=N_MFCC)
    sd = mfcc.std(axis=1, keepdims=True)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    n_frames = min(cens.shape[1], mfcc.shape[1])
    bounds = np.append(beat_times, end_s)
    fr = np.clip(np.round(bounds * SR / FEAT_HOP).astype(np.int64), 0, n_frames)
    K = len(beat_times)
    C = np.zeros((cens.shape[0], K))
    M = np.zeros((mfcc.shape[0], K))
    for k in range(K):
        a, b = int(fr[k]), int(fr[k + 1])
        if b <= a:  # a beat shorter than one feature frame: take the nearest frame
            a = min(a, n_frames - 1)
            b = a + 1
        C[:, k] = np.median(cens[:, a:b], axis=1)
        M[:, k] = np.mean(mfcc[:, a:b], axis=1)
    return C, M


# ------------------------------------------------------------------------------------------ segmentation


def _stacked(C: np.ndarray, M: np.ndarray, n: int) -> np.ndarray:
    """[CENS / |CENS|; MFCC / sqrt(13)] per beat, stacked over a centred window of n beats (edge padded)."""
    Cn = C / np.maximum(np.linalg.norm(C, axis=0, keepdims=True), 1e-9)
    F = np.vstack([Cn, M / math.sqrt(M.shape[0])])
    lo = n // 2
    P = np.pad(F, ((0, 0), (lo, n - lo - 1)), mode="edge")
    K = F.shape[1]
    return np.vstack([P[:, o:o + K] for o in range(n)])


def _laplacian_embedding(F: np.ndarray, M: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(evecs sorted/sign-normalised (K, K), evals, k by eigengap) for recurrence features F and path MFCC M."""
    import librosa
    import scipy.linalg
    import scipy.ndimage
    import scipy.sparse.csgraph

    K = F.shape[1]
    R = librosa.segment.recurrence_matrix(F, width=REC_WIDTH, mode="affinity", sym=True)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, PATH_FILTER))
    # The lag-domain median filter does not preserve symmetry (up to 0.28 apart on real songs), and eigh only
    # reads one triangle: the "Laplacian" then had eigenvalues down to -0.04. Symmetrise before building it.
    Rf = 0.5 * (Rf + Rf.T)
    path_distance = np.sum(np.diff(M, axis=1) ** 2, axis=0)
    sigma = float(np.median(path_distance)) or 1.0
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    denom = float(np.sum((deg_path + deg_rec) ** 2))
    mu = float(deg_path.dot(deg_path + deg_rec) / denom) if denom > 0 else 0.5
    A = mu * Rf + (1 - mu) * R_path
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)
    order = np.lexsort((np.arange(evals.size), np.round(evals, 12)))
    evals, evecs = evals[order], evecs[:, order]
    # sign normalisation: the largest-|x| component of each eigenvector is positive
    idx = np.argmax(np.abs(evecs), axis=0)
    signs = np.sign(evecs[idx, np.arange(evecs.shape[1])])
    evecs = evecs * np.where(signs == 0, 1.0, signs)
    evecs = scipy.ndimage.median_filter(evecs, size=(EVEC_MEDIAN, 1))
    hi = max(2, min(max_k, K - 1))
    gaps = [(evals[k] - evals[k - 1], k) for k in range(2, hi + 1)]
    k = max(gaps, key=lambda g: (round(float(g[0]), 10), -g[1]))[1] if gaps else 2
    return evecs, evals, int(k)


def _cluster(evecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5
    X = evecs[:, :k] / np.maximum(cnorm[:, k - 1:k], 1e-12)
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
    return km.labels_.astype(np.int64), km.cluster_centers_, X


def _bar_of_beat(grid: dict[str, Any]) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """(bar number per beat, sorted bar numbers, bar -> index of its first beat)."""
    bars_of = np.asarray([b["bar"] for b in grid["beats"]], dtype=np.int64)
    first: dict[int, int] = {}
    for i, b in enumerate(bars_of):
        first.setdefault(int(b), i)
    return bars_of, sorted(first), first


def _segments_from_labels(labels: np.ndarray, bars_of: np.ndarray, bar_list: list[int], first: dict[int, int]
                          ) -> list[list[int]]:
    """Beat label changes -> boundaries snapped to the nearest bar start -> list of [start_bar, end_bar)."""
    starts_beat = np.asarray([first[b] for b in bar_list], dtype=np.int64)
    cuts = {0}
    for i in range(1, labels.size):
        if labels[i] != labels[i - 1]:
            j = int(np.argmin(np.abs(starts_beat - i)))
            cuts.add(j)
    idx = sorted(cuts)
    segs = []
    for n, j in enumerate(idx):
        e = idx[n + 1] if n + 1 < len(idx) else len(bar_list)
        if e > j:
            segs.append([j, e])  # indices into bar_list
    return segs


def _merge_short(segs: list[list[int]], means: Callable[[list[int]], np.ndarray], min_bars: int) -> list[list[int]]:
    segs = [list(s) for s in segs]
    while len(segs) > 1:
        lens = [e - s for s, e in segs]
        short = [i for i, n in enumerate(lens) if n < min_bars]
        if not short:
            break
        i = min(short, key=lambda i: (lens[i], i))
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = i - 1
        else:
            me = means(segs[i])
            dp = float(np.linalg.norm(me - means(segs[i - 1])))
            dn = float(np.linalg.norm(me - means(segs[i + 1])))
            j = i - 1 if dp <= dn else i + 1
        a, b = sorted((i, j))
        segs[a] = [segs[a][0], segs[b][1]]
        del segs[b]
    return segs


# ------------------------------------------------------------------------------------------ repeat map


def _bar_vectors(C: np.ndarray, M: np.ndarray, bars_of: np.ndarray, bar_list: list[int]) -> np.ndarray:
    """Per bar: the beat features resampled to SLOTS_PER_BAR slots and concatenated, then standardised."""
    feats = np.vstack([C, M])  # (25, K)
    out = np.zeros((len(bar_list), feats.shape[0] * SLOTS_PER_BAR))
    for n, bar in enumerate(bar_list):
        cols = np.flatnonzero(bars_of == bar)
        sub = feats[:, cols]
        pos = (np.arange(SLOTS_PER_BAR) + 0.5) * sub.shape[1] / SLOTS_PER_BAR
        pick = np.clip(np.floor(pos).astype(np.int64), 0, sub.shape[1] - 1)
        out[n] = sub[:, pick].T.reshape(-1)
    sd = out.std(axis=0, keepdims=True)
    return (out - out.mean(axis=0, keepdims=True)) / np.where(sd > 0, sd, 1.0)


def _bar_similarity(V: np.ndarray) -> np.ndarray:
    import librosa
    import scipy.ndimage

    norm = np.linalg.norm(V, axis=1, keepdims=True)
    U = V / np.maximum(norm, 1e-12)
    S = np.clip(U @ U.T, 0.0, 1.0)
    np.fill_diagonal(S, 0.0)
    if S.shape[0] >= 3:
        smooth = librosa.segment.timelag_filter(scipy.ndimage.uniform_filter)
        S = smooth(S, size=(1, 3), mode="nearest")
        np.fill_diagonal(S, 0.0)
    return S


class _Env:
    def __init__(self, onsets: dict[str, Any] | None) -> None:
        import scipy.signal

        o = onsets or {}
        self.fps = float(o.get("fps", 22050 / 256) or 22050 / 256)
        env = np.asarray(o.get("env", []), dtype=np.float64)
        self.up = scipy.signal.resample_poly(env, UPSAMPLE, 1) if env.size else env
        self.rate = self.fps * UPSAMPLE

    def curve(self, a_s: float, b_s: float, dur_s: float, max_lag: int) -> np.ndarray | None:
        """Cross-correlation of env[b + lag .. ] with env[a ..] over dur_s, for lag in [-max_lag, max_lag]."""
        n = int(dur_s * self.rate)
        ia, ib = int(round(a_s * self.rate)), int(round(b_s * self.rate))
        if n <= 0 or ia < 0 or ia + n > self.up.size:
            return None
        ref = self.up[ia:ia + n]
        if not np.any(ref):
            return None
        out = np.zeros(2 * max_lag + 1)
        for k, lag in enumerate(range(-max_lag, max_lag + 1)):
            s = ib + lag
            if s < 0 or s + n > self.up.size:
                out[k] = -np.inf
                continue
            out[k] = float(np.dot(ref, self.up[s:s + n]))
        return out

    def best_lag_s(self, curve: np.ndarray | None, max_lag: int) -> float:
        if curve is None or not np.isfinite(curve).any() or float(np.max(curve)) <= 0:
            return 0.0
        k = int(np.argmax(curve))
        off = 0.0
        if 0 < k < curve.size - 1 and np.isfinite(curve[k - 1]) and np.isfinite(curve[k + 1]):
            y0, y1, y2 = curve[k - 1], curve[k], curve[k + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                off = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        return (k - max_lag + off) / self.rate


def _bar_times(grid: dict[str, Any]) -> dict[int, tuple[float, float]]:
    return {int(b["bar"]): (float(b["start_s"]), float(b["end_s"])) for b in grid["bars"]}


# ------------------------------------------------------------------------------------------ main


def analyze(mix_wav: Path, grid: dict[str, Any], onsets: dict[str, Any] | None, params: dict[str, Any]
            ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sections + repeat map documents for a mix (runs single-threaded, see module doc)."""
    from threadpoolctl import threadpool_limits

    from bandscribe.analysis.onsets import load_mono

    with threadpool_limits(limits=1):
        y = load_mono(Path(mix_wav), SR)
        return analyze_signal(y, grid, onsets, params)


def analyze_signal(y: np.ndarray, grid: dict[str, Any], onsets: dict[str, Any] | None, params: dict[str, Any]
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    from threadpoolctl import threadpool_limits

    p = {**DEFAULT_PARAMS, **{k: v for k, v in params.items() if k in DEFAULT_PARAMS}}
    with threadpool_limits(limits=1):
        return _analyze(y, grid, onsets, p)


def _analyze(y: np.ndarray, grid: dict[str, Any], onsets: dict[str, Any] | None, p: dict[str, Any]
             ) -> tuple[dict[str, Any], dict[str, Any]]:
    bars = grid.get("bars") or []
    bt = _bar_times(grid)
    bars_of, bar_list, first = _bar_of_beat(grid)
    beat_times = np.asarray([b["t_s"] for b in grid.get("beats") or []], dtype=np.float64)
    empty_sections = {"format": "bandscribe.sections/1", "method": METHOD, "sections": [], "params": dict(p)}
    empty_map = {"format": "bandscribe.repeat_map/1", "groups": [], "bars": [], "params": dict(p)}
    if not bars or beat_times.size == 0:
        return empty_sections, empty_map
    end_s = float(bars[-1]["end_s"])
    C, M = _beat_features(y, beat_times, end_s)

    # --- segmentation
    if beat_times.size >= MIN_BEATS_FOR_SEGMENTATION and len(bar_list) >= 2:
        bpb = int(np.median([b["n_beats"] for b in bars])) if bars else 4
        F = _stacked(C, M, int(np.clip(2 * bpb, 4, 16)))
        evecs, _evals, k = _laplacian_embedding(F, M, int(p["max_sections"]))
        labels, centers, X = _cluster(evecs, k)
    else:
        k, labels = 1, np.zeros(beat_times.size, dtype=np.int64)
        X = np.zeros((beat_times.size, 1))
        centers = np.zeros((1, 1))

    def beat_cols(seg: Sequence[int]) -> np.ndarray:
        lo_bar, hi_bar = bar_list[seg[0]], bar_list[seg[1] - 1]
        return np.flatnonzero((bars_of >= lo_bar) & (bars_of <= hi_bar))

    def seg_mean(seg: Sequence[int]) -> np.ndarray:
        cols = beat_cols(seg)
        return X[cols].mean(axis=0) if cols.size else np.zeros(X.shape[1])

    segs = _segments_from_labels(labels, bars_of, bar_list, first)
    segs = _merge_short(segs, seg_mean, int(p["min_section_bars"]))
    seg_cluster = [int(np.argmin(np.linalg.norm(centers - seg_mean(s), axis=1))) for s in segs]
    merged: list[list[int]] = []
    merged_cl: list[int] = []
    for s, c in zip(segs, seg_cluster):
        if merged and merged_cl[-1] == c:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
            merged_cl.append(c)
    letters: dict[int, str] = {}
    sections: list[dict[str, Any]] = []
    for n, (s, c) in enumerate(zip(merged, merged_cl)):
        if c not in letters:
            letters[c] = _letter(len(letters))
        cols = beat_cols(s)
        purity = float(np.mean(labels[cols] == c)) if cols.size and k > 1 else 1.0
        sb, eb = bar_list[s[0]], bar_list[s[1] - 1] + 1
        sections.append({"id": f"S{n + 1}", "label": letters[c], "start_bar": int(sb), "end_bar": int(eb),
                         "start_s": _r6(bt[sb][0]), "end_s": _r6(bt[eb - 1][1]), "confidence": _r4(purity)})

    # --- repeat map
    V = _bar_vectors(C, M, bars_of, bar_list)
    S = _bar_similarity(V)
    env = _Env(onsets)
    max_lag = int(math.ceil(float(p["align_max_offset_ms"]) / 1000.0 * env.rate))
    min_score = float(p["repeat_min_score"])
    bar_index = {b: i for i, b in enumerate(bar_list)}

    def bar_curve(ref_bar: int, other_bar: int) -> np.ndarray | None:
        a0, a1 = bt[ref_bar]
        b0, _b1 = bt[other_bar]
        return env.curve(a0, b0, a1 - a0, max_lag)

    bar_matches: list[dict[str, Any]] = []
    for i, bar in enumerate(bar_list):
        order = np.lexsort((np.arange(len(bar_list)), -np.round(S[i], 4)))
        matches = []
        for j in order:
            if j == i:
                continue
            sc = _r4(S[i, j])
            if sc < min_score or len(matches) >= MAX_MATCHES:
                break
            other = bar_list[int(j)]
            off = env.best_lag_s(bar_curve(bar, other), max_lag)
            matches.append({"bar": int(other), "score": sc, "offset_s": _r6(off)})
        bar_matches.append({"bar": int(bar), "matches": matches})

    groups: list[dict[str, Any]] = []
    for c, letter in letters.items():
        occ = [s for s in sections if s["label"] == letter]
        if len(occ) < 2:
            continue

        def occ_sim(a: dict, b: dict) -> float:
            n = min(a["end_bar"] - a["start_bar"], b["end_bar"] - b["start_bar"])
            vals = [S[bar_index[a["start_bar"] + t], bar_index[b["start_bar"] + t]] for t in range(n)
                    if a["start_bar"] + t in bar_index and b["start_bar"] + t in bar_index]
            return float(np.mean(vals)) if vals else 0.0

        mean_sim = [np.mean([occ_sim(a, b) for b in occ if b is not a]) for a in occ]
        ref = occ[int(np.argmax(np.round(mean_sim, 6)))]
        occurrences = []
        for o in occ:
            if o is ref:
                off = 0.0
            else:
                n = min(o["end_bar"] - o["start_bar"], ref["end_bar"] - ref["start_bar"])
                total = None
                for t in range(n):
                    cv = bar_curve(ref["start_bar"] + t, o["start_bar"] + t)
                    if cv is None:
                        continue
                    cv = np.where(np.isfinite(cv), cv, 0.0)
                    total = cv if total is None else total + cv
                off = env.best_lag_s(total, max_lag)
            occurrences.append({"section": o["id"], "start_bar": int(o["start_bar"]),
                                "n_bars": int(o["end_bar"] - o["start_bar"]), "offset_s": _r6(off)})
        groups.append({"label": letter, "occurrences": occurrences, "reference": ref["id"]})

    sections_doc = {"format": "bandscribe.sections/1", "method": METHOD, "sections": sections, "params": dict(p)}
    repeat_doc = {"format": "bandscribe.repeat_map/1", "groups": groups, "bars": bar_matches, "params": dict(p)}
    log.info("sections: %d (%s), %d repeat groups", len(sections), "".join(s["label"] for s in sections), len(groups))
    return sections_doc, repeat_doc


def _letter(n: int) -> str:
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s
