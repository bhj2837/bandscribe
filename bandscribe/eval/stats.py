"""Song-block bootstrap, paired comparison and the adoption rule of DESIGN §8.1.

Units: statistics are returned in the unit the ``stat`` function produces. The metric helpers below return
**points** (F1 × 100) because MDEs and the "−2 points" regression guard of DESIGN §8.1 are stated in points.

Block bootstrap: items (windows, tracks, sections) are grouped by ``group`` (song / performer). A replicate
draws groups with replacement and recomputes the statistic from the **pooled counts** of every item in the
drawn groups (pooling, not averaging per-item F1s, so a song with 3 notes cannot swing the result).
Deterministic: numpy ``Generator(PCG64(seed))`` and a fixed group order (sorted ids).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_ITERATIONS = 10_000
DEFAULT_SEED = 20260930
REGRESSION_POINTS = 2.0  # DESIGN §8.1: no scenario subset may have its CI upper bound below -2 points

Stat = Callable[[Mapping[str, np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class ItemCounts:
    """Counts of one item. ``counts`` holds additive quantities (tp/fp/fn, or sum/n for means)."""

    item: str
    group: str
    counts: Mapping[str, float] = field(default_factory=dict)

    @staticmethod
    def prf(item: str, group: str, tp: float, fp: float, fn: float) -> ItemCounts:
        return ItemCounts(item, group, {"tp": float(tp), "fp": float(fp), "fn": float(fn)})

    @staticmethod
    def mean(item: str, group: str, total: float, n: float) -> ItemCounts:
        return ItemCounts(item, group, {"sum": float(total), "n": float(n)})


# ------------------------------------------------------------------------------------ pooled statistics


def f1_points(c: Mapping[str, np.ndarray]) -> np.ndarray:
    tp, fp, fn = (np.asarray(c[k], dtype=float) for k in ("tp", "fp", "fn"))
    den = 2 * tp + fp + fn
    return np.where(den > 0, 100.0 * 2 * tp / np.where(den > 0, den, 1), np.nan)


def precision_points(c: Mapping[str, np.ndarray]) -> np.ndarray:
    tp, fp = np.asarray(c["tp"], dtype=float), np.asarray(c["fp"], dtype=float)
    den = tp + fp
    return np.where(den > 0, 100.0 * tp / np.where(den > 0, den, 1), np.nan)


def recall_points(c: Mapping[str, np.ndarray]) -> np.ndarray:
    tp, fn = np.asarray(c["tp"], dtype=float), np.asarray(c["fn"], dtype=float)
    den = tp + fn
    return np.where(den > 0, 100.0 * tp / np.where(den > 0, den, 1), np.nan)


def mean_value(c: Mapping[str, np.ndarray]) -> np.ndarray:
    s, n = np.asarray(c["sum"], dtype=float), np.asarray(c["n"], dtype=float)
    return np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)


def ratio_points(c: Mapping[str, np.ndarray]) -> np.ndarray:
    return 100.0 * mean_value(c)


STATS: dict[str, Stat] = {
    "f1": f1_points, "precision": precision_points, "recall": recall_points,
    "mean": mean_value, "ratio": ratio_points,
}


# ------------------------------------------------------------------------------------------- bootstrap


def _group_matrix(items: Sequence[ItemCounts], keys: Sequence[str], groups: Sequence[str]) -> np.ndarray:
    gi = {g: i for i, g in enumerate(groups)}
    m = np.zeros((len(groups), len(keys)), dtype=float)
    for it in items:
        row = gi[it.group]
        for j, k in enumerate(keys):
            m[row, j] += float(it.counts.get(k, 0.0))
    return m


def _keys(items: Iterable[ItemCounts]) -> list[str]:
    ks: set[str] = set()
    for it in items:
        ks.update(it.counts)
    return sorted(ks)


def _draws(n_groups: int, n: int, seed: int) -> np.ndarray:
    """(n, n_groups) multiplicities of each group per replicate, from one seeded PCG64 stream."""
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, n_groups, size=(n, n_groups))
    flat = (idx + (np.arange(n, dtype=np.int64) * n_groups)[:, None]).ravel()
    return np.bincount(flat, minlength=n * n_groups).reshape(n, n_groups).astype(float)


def _percentile_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(v, [alpha / 2, 1 - alpha / 2], method="linear")
    return float(lo), float(hi)


def block_bootstrap(items: Sequence[ItemCounts], stat: Stat, n: int = DEFAULT_ITERATIONS,
                    seed: int = DEFAULT_SEED, *, alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, ci_low, ci_high) of ``stat`` over pooled counts, resampling ``group`` blocks with replacement."""
    if not items:
        return float("nan"), float("nan"), float("nan")
    keys = _keys(items)
    groups = sorted({it.group for it in items})
    gm = _group_matrix(items, keys, groups)
    point = float(stat({k: gm[:, j].sum() for j, k in enumerate(keys)}))
    if len(groups) < 2 or n <= 0:
        return point, float("nan"), float("nan")
    pooled = _draws(len(groups), n, seed) @ gm  # (n, K)
    reps = np.asarray(stat({k: pooled[:, j] for j, k in enumerate(keys)}), dtype=float)
    lo, hi = _percentile_ci(reps, alpha)
    return point, lo, hi


def paired_bootstrap(a: Sequence[ItemCounts], b: Sequence[ItemCounts], stat: Stat,
                     n: int = DEFAULT_ITERATIONS, seed: int = DEFAULT_SEED, *,
                     alpha: float = 0.05) -> tuple[float, float, float]:
    """(delta, lo, hi) of ``stat(b) - stat(a)``: b = candidate, a = baseline, same groups drawn for both."""
    if not a or not b:
        return float("nan"), float("nan"), float("nan")
    keys = sorted(set(_keys(a)) | set(_keys(b)))
    groups = sorted({it.group for it in a} | {it.group for it in b})
    ga, gb = _group_matrix(a, keys, groups), _group_matrix(b, keys, groups)
    tot = lambda m: {k: m[:, j].sum() for j, k in enumerate(keys)}  # noqa: E731
    delta = float(stat(tot(gb)) - stat(tot(ga)))
    if len(groups) < 2 or n <= 0:
        return delta, float("nan"), float("nan")
    w = _draws(len(groups), n, seed)
    pa, pb = w @ ga, w @ gb
    reps = (np.asarray(stat({k: pb[:, j] for j, k in enumerate(keys)}), dtype=float)
            - np.asarray(stat({k: pa[:, j] for j, k in enumerate(keys)}), dtype=float))
    lo, hi = _percentile_ci(reps, alpha)
    return delta, lo, hi


# -------------------------------------------------------------------------------------------- decision

Decision = Literal["adopt", "reject", "hold"]


def decide(delta: float, lo: float, hi: float, mde: float,
           subsets: Mapping[str, tuple[float, float, float]] | Sequence[tuple[float, float, float]] = (),
           *, regression: float = REGRESSION_POINTS) -> Decision:
    """DESIGN §8.1 component gate, all values in points.

    - CI contains 0 (or is undefined)      -> ``hold``  ("판단 보류", keep the default)
    - CI entirely below 0                 -> ``reject``
    - CI above 0 but point < MDE           -> ``reject`` (real but too small to justify the change)
    - some scenario subset CI upper < −2   -> ``reject`` (clear regression somewhere)
    - otherwise                            -> ``adopt``
    ``subsets``: {name: (delta, lo, hi)} or a list of such triples.
    """
    vals = [delta, lo, hi]
    if any(v is None or not np.isfinite(v) for v in vals):
        return "hold"
    if hi < 0:
        return "reject"
    if lo <= 0:
        return "hold"
    triples = subsets.values() if isinstance(subsets, Mapping) else subsets
    for _d, _lo, s_hi in triples:
        if s_hi is not None and np.isfinite(s_hi) and s_hi < -regression:
            return "reject"
    if delta < mde:
        return "reject"
    return "adopt"


def noninferior(lo: float, margin: float) -> bool:
    """Non-inferiority (E24): the CI lower bound of candidate − baseline must lie above ``margin`` (e.g. −0.5)."""
    return bool(np.isfinite(lo) and lo > margin)
