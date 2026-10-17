"""E27 helpers: identity metrics, the whole-song windows and the registered decision rules."""

from __future__ import annotations

from bandscribe.parts import eval_e27 as E27
from bandscribe.parts import eval_track as ET
from bandscribe.parts import idmetrics as IM


def test_identity_switches_count_block_majority_changes():
    truth = [{"a"}] * 6 + [{"b"}] * 2 + [set()]
    on = [0.1, 0.5, 2.2, 2.4, 4.1, 6.3, 0.2, 4.2, 1.0]
    pred = ["1", "1", "2", "2", "1", None, "2", "2", "1"]
    sw = IM.identity_switches(truth, pred, on, {"1": "a", "2": "b"})
    # a: blocks 0 -> a, 1 -> b, 2 -> a, 3 -> u : three switches; b: blocks 0, 2 both b
    assert sw["per_line"] == {"a": 3, "b": 0} and sw["total"] == 3 and sw["blocks"] == {"a": 4, "b": 2}
    assert IM.identity_switches(truth, ["9"] * 9, on, {})["per_line"]["a"] == 0  # unmatched label: one identity


def test_block_der_counts_missed_false_alarm_and_confusion():
    # block 0: a and b both play, predicted all as a -> one missed; block 1: a only, predicted as b -> confusion
    truth = [{"a"}, {"a"}, {"b"}, {"b"}, {"a"}, {"a"}]
    on = [0.1, 0.2, 0.3, 0.4, 2.1, 2.2]
    pred = ["1", "1", "1", "1", "2", "2"]
    d = IM.block_der(truth, pred, on, {"1": "a", "2": "b"})
    assert (d["missed"], d["false_alarm"], d["confusion"], d["true_blocks"]) == (1, 0, 1, 3)
    assert abs(d["der"] - 2 / 3) < 1e-9


def test_bar_windows_stay_inside_sections():
    grid = {"bars": [{"bar": b, "start_s": 2.0 * b, "end_s": 2.0 * b + 2.0} for b in range(10)]}
    sections = [{"id": "S1", "start_bar": 0, "end_bar": 6}, {"id": "S2", "start_bar": 6, "end_bar": 10}]
    w = ET.bar_windows(grid, sections, 19.0)
    assert [(x["section"], x["start_bar"]) for x in w] == [("S1", 0), ("S1", 4), ("S2", 6)]
    assert w[1]["end_s"] == 12.0 and w[2]["end_s"] == 19.0  # cut at the remix's end


def _row(song, sc, sw0, sw1, m0, m1):
    return {"song": song, "scenario": sc, "v0": {"id_switches": sw0, "assign_macro": m0, "block_der": 0.3},
            "track": {"id_switches": sw1, "assign_macro": m1, "block_der": 0.2}}


def test_decide_follows_the_registered_rules():
    rows = [_row("s1", "P", 6, 2, 0.8, 0.81), _row("s2", "P", 4, 1, 0.7, 0.7), _row("s3", "A1", 3, 2, 0.5, 0.5)]
    d = E27.decide(rows)
    assert d["median_switches"] == {"v0": 4, "track": 2} and d["rule1_switches"] and d["rule2_macro"]
    assert d["verdict"] == "PASS" and d["der_by_song"]["s1"] == {"v0": 0.3, "track": 0.2}
    same = [_row("s1", "P", 0, 0, 0.8, 0.8), _row("s2", "P", 1, 1, 0.7, 0.7)]
    assert E27.decide(same)["rule1_switches"] is False  # not fewer than v0 -> FAIL as registered
    worse = [_row(f"s{i}", "B2", 3, 1, 0.8, 0.7) for i in range(4)]
    w = E27.decide(worse)
    assert not w["rule2_macro"] and w["regressing_scenarios"] == ["B2"] and w["verdict"] == "FAIL"


def test_grid_is_conservative_first():
    assert E27.GRID[0] == {"p_floor": 0.0, "w_link": 1.0, "boundary": 0.5, "w_rep": 0.0} and len(E27.GRID) == 24


def test_bar_windows_merge_a_one_bar_leftover_like_the_router():
    grid = {"bars": [{"bar": b, "start_s": 2.0 * b, "end_s": 2.0 * b + 2.0} for b in range(10)]}
    w = ET.bar_windows(grid, [{"id": "S1", "start_bar": 0, "end_bar": 5}, {"id": "S2", "start_bar": 5, "end_bar": 10}],
                       20.0)
    # S1 = 5 bars: 4 + a 1-bar leftover -> one window of 5; S2 = 5 bars likewise
    assert [(x["section"], x["start_bar"], x["end_bar"]) for x in w] == [("S1", 0, 5), ("S2", 5, 10)]
