"""G0's plans: real takes, different parts, and the remix label (no audio, no GPU)."""

from __future__ import annotations

from bandscribe.parts import eval_g0 as G
from bandscribe.router import tierb as T


def test_every_plan_has_a_part_count_the_scenario_promises():
    want = {"T3": 3, "X2p": 2, "X2c": 2, "N1a": 1}
    for scenarios in G.PLANS.values():
        for sc, pans in scenarios.items():
            assert len(pans) == want[sc], sc
            assert all(-1.0 <= p <= 1.0 for p in pans.values())
            if sc == "T3":
                assert sorted(pans.values()) == [-1.0, 0.0, 1.0]
            if sc == "X2c" or sc == "N1a":
                assert set(pans.values()) == {0.0}


def test_g0_windows_have_no_router_truth():
    truth, shares = T.window_truth("T3", {"lines": [0, 1, 2]}, {0: 1.0, 1: 1.0, 2: 2.0}, -20.0)
    assert truth == "mixed" and shares == {"lines": 1.0}
    assert T.window_truth("T3", {"lines": [0]}, {}, -80.0)[0] == "silent"
