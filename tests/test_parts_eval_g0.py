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


def test_note_side_outside_r3_needs_one_coherent_source_on_one_side():
    from bandscribe.parts import arbitration as A

    left = {"ild_db": 6.0, "coh": 0.95}
    assert A._pan(left, "R3", 1.0) > 0.9  # R3 windows as before
    assert A._pan(left, "R2", 1.0) == 0.0  # off: nothing outside R3
    assert A._pan(left, "R2", 1.0, 3.0, 0.9) > 0.9  # on
    assert A._pan({"ild_db": -6.0, "coh": 0.95}, "R0", 1.0, 3.0, 0.9) < -0.9
    assert A._pan({"ild_db": 6.0, "coh": 0.5}, "R2", 1.0, 3.0, 0.9) == 0.0  # a double's takes are not coherent
    assert A._pan({"ild_db": 1.0, "coh": 0.95}, "R2", 1.0, 3.0, 0.9) == 0.0  # near the centre
    assert A._pan({"ild_db": 6.0, "coh": None}, "R2", 1.0, 3.0, 0.9) == 0.0


def test_note_side_off_keeps_the_parts_params():
    from bandscribe.config import Config
    from bandscribe.parts import stage as PS

    assert "note_side_db" not in PS.parts_params(Config())["count"]
    on = PS.parts_params(Config.model_validate({"parts": {"note_side_db": 3.0}}))["count"]
    assert on["note_side_db"] == 3.0 and on["note_side_coh"] == 0.9

