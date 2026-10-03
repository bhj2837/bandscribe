"""Song-block bootstrap and the DESIGN §8.1 decision rule."""

from __future__ import annotations

import numpy as np
import pytest

from gtab.eval import stats
from gtab.eval.stats import ItemCounts


def _items(n_groups: int, per_group: int, p: float, seed: int, *, n_notes: int = 50) -> list[ItemCounts]:
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_groups):
        for k in range(per_group):
            tp = int(rng.binomial(n_notes, p))
            out.append(ItemCounts.prf(f"s{g}_w{k}", f"s{g}", tp, n_notes - tp, n_notes - tp))
    return out


def test_bootstrap_deterministic_with_seed():
    items = _items(12, 5, 0.7, seed=1)
    a = stats.block_bootstrap(items, stats.f1_points, n=2000, seed=7)
    b = stats.block_bootstrap(items, stats.f1_points, n=2000, seed=7)
    c = stats.block_bootstrap(items, stats.f1_points, n=2000, seed=8)
    assert a == b
    assert a[0] == c[0] and a[1:] != c[1:]
    assert a[1] < a[0] < a[2]


def test_bootstrap_point_is_pooled_counts():
    items = [ItemCounts.prf("a", "g1", 1, 0, 0), ItemCounts.prf("b", "g2", 0, 9, 9)]
    point, _lo, _hi = stats.block_bootstrap(items, stats.f1_points, n=100, seed=1)
    assert point == pytest.approx(100 * 2 * 1 / (2 * 1 + 9 + 9))


def test_bootstrap_ci_covers_known_mean_in_simulation():
    """95 % CI of a mean over song blocks should cover the true mean most of the time."""
    covered = 0
    trials = 40
    for t in range(trials):
        rng = np.random.default_rng(100 + t)
        items = [ItemCounts.mean(f"i{g}", f"g{g}", float(rng.normal(5.0, 1.0)), 1.0) for g in range(30)]
        _p, lo, hi = stats.block_bootstrap(items, stats.mean_value, n=2000, seed=t)
        covered += lo <= 5.0 <= hi
    assert covered >= 34  # ≈ 95 % nominal, loose bound for 40 trials


def test_paired_bootstrap_detects_shift_and_is_deterministic():
    base = _items(15, 4, 0.6, seed=3)
    better = [ItemCounts.prf(it.item, it.group, it.counts["tp"] + 5, max(0, it.counts["fp"] - 5),
                             max(0, it.counts["fn"] - 5)) for it in base]
    d1 = stats.paired_bootstrap(base, better, stats.f1_points, n=2000, seed=5)
    d2 = stats.paired_bootstrap(base, better, stats.f1_points, n=2000, seed=5)
    assert d1 == d2
    assert d1[0] > 0 and d1[1] > 0
    same = stats.paired_bootstrap(base, base, stats.f1_points, n=500, seed=5)
    assert same == (0.0, 0.0, 0.0)


def test_single_group_has_no_ci():
    items = [ItemCounts.prf("a", "g", 5, 1, 1), ItemCounts.prf("b", "g", 3, 1, 1)]
    p, lo, hi = stats.block_bootstrap(items, stats.f1_points, n=100)
    assert np.isfinite(p) and np.isnan(lo) and np.isnan(hi)


@pytest.mark.parametrize(("delta", "lo", "hi", "mde", "subsets", "expected"), [
    (2.0, 0.5, 3.5, 1.0, {}, "adopt"),
    (2.0, -0.5, 3.5, 1.0, {}, "hold"),          # CI contains 0
    (0.6, 0.1, 1.1, 1.0, {}, "reject"),         # significant but below MDE
    (-2.0, -3.0, -1.0, 1.0, {}, "reject"),      # clearly worse
    (2.0, 0.5, 3.5, 1.0, {"B": (-4.0, -6.0, -2.5)}, "reject"),  # a scenario subset regresses (hi < -2)
    (2.0, 0.5, 3.5, 1.0, {"B": (-1.0, -3.0, -1.5)}, "adopt"),   # subset upper bound above -2: tolerated
    (float("nan"), float("nan"), float("nan"), 1.0, {}, "hold"),
])
def test_decide_truth_table(delta, lo, hi, mde, subsets, expected):
    assert stats.decide(delta, lo, hi, mde, subsets) == expected


def test_noninferiority():
    assert stats.noninferior(-0.3, -0.5)
    assert not stats.noninferior(-0.7, -0.5)
    assert not stats.noninferior(float("nan"), -0.5)
