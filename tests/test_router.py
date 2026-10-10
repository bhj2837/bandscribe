"""Guitar router S4 (M6): windows, labels, solo gate, Tier B relations and remix plans."""
from __future__ import annotations

import math

import librosa
import numpy as np
import pytest
from scipy import signal

from bandscribe.eval import remix as R
from bandscribe.router import route as RT
from bandscribe.router import tierb as T


def _bars(n, bar_s=2.0, t0=0.0):
    return [{"bar": i, "start_s": t0 + i * bar_s, "end_s": t0 + (i + 1) * bar_s} for i in range(n)]


# ------------------------------------------------------------------------------------------------ windows


def test_windows_cut_at_sections_and_fold_a_single_remaining_bar():
    secs = [{"id": "S1", "start_bar": 0, "end_bar": 9}, {"id": "S2", "start_bar": 9, "end_bar": 21}]
    w = RT.windows(_bars(21), secs)
    assert [(x["start_bar"], x["end_bar"], x["section"]) for x in w] == [
        (0, 4, "S1"), (4, 9, "S1"), (9, 13, "S2"), (13, 17, "S2"), (17, 21, "S2")]
    assert w[1]["start_s"] == 8.0 and w[1]["end_s"] == 18.0


def test_windows_take_in_bars_outside_the_sections():
    secs = [{"id": "S1", "start_bar": 1, "end_bar": 7}]  # bar 0 (pickup) and bars 7-8 lie outside
    w = RT.windows(_bars(9), secs)
    assert [(x["start_bar"], x["end_bar"]) for x in w] == [(0, 4), (4, 9)]  # the lone bar 8 joins 4-8


def test_lone_pickup_window_joins_the_first_window():
    secs = [{"id": "S0", "start_bar": 0, "end_bar": 1}, {"id": "S1", "start_bar": 1, "end_bar": 9}]
    w = RT.windows(_bars(9), secs)
    assert (w[0]["start_bar"], w[0]["end_bar"], w[0]["start_s"]) == (0, 5, 0.0)
    assert len(w) == 2


def test_windows_without_sections_cover_the_song():
    w = RT.windows(_bars(10), None)
    assert [(x["start_bar"], x["end_bar"]) for x in w] == [(0, 4), (4, 8), (8, 10)]
    assert RT.windows([], None) == []


# ------------------------------------------------------------------------------------------------ labels


@pytest.mark.parametrize("scenario, expected", [
    ("A", "R2"), ("A_rhythm_only", "R2"), ("quad", "R2"), ("centre_unison", "R2"),
    ("B", "R0"), ("F", "R0"), ("ADT", "R1"), ("P70", "R3"), ("P_hard", "R3"), ("twin", "R3"),
])
def test_research_scenes_get_their_design_label(scenario, expected):
    """The synthetic scenes the DESIGN S4 thresholds came from (router_exp.py, ported in eval.remix)."""
    g = R.make_scene(scenario, 0).guitar_stereo()
    lab, scores, p, margin = RT.label(RT.features(g, R.SR))
    assert lab == expected, (scenario, scores)
    assert margin > 0.5 and max(p, key=p.get) == expected
    assert abs(sum(p.values()) - 1.0) < 1e-3


def test_no_rule_met_is_mixed_and_undefined_features_never_meet_a_rule():
    f = {"SM": -8.0, "coh": 0.9, "cen": 0.3, "hard": 0.2, "ildcoh": 4.0, "xcoff": 0.3, "lag": 10.0}
    assert RT.label(f)[0] == "mixed"
    g = {**f, "ildcoh": None, "hard": 0.0, "coh": 0.95}  # no coherent bins: the ILD alternative is never met
    assert RT.scores(g)["R3"] == pytest.approx(min(-0.3 / 0.05, (0.6 - 0.95) / 0.05), abs=1e-3)


def test_adt_needs_a_lag_of_at_least_5_ms():
    f = {"SM": -4.0, "r0": 0.3, "coh": 0.9, "cen": 0.8, "hard": 0.0, "ildcoh": 1.0, "xcoff": 0.8, "lag": 2.0}
    assert RT.scores(f)["R1"] < 0
    assert RT.scores({**f, "lag": -12.0})["R1"] > 0  # either side, either polarity
    assert RT.scores({**f, "lag": 12.0, "xcoff": -0.8})["R1"] > 0


def test_a_centred_mono_part_is_not_adt_however_periodic():
    """Identical channels: r0 = 1; a sustained part still has an off-lag peak of 0.5-0.75 (Tier B smoke run)."""
    f = {"SM": -30.0, "r0": 1.0, "coh": 1.0, "cen": 1.0, "hard": 0.0, "ildcoh": 0.1, "xcoff": 0.72, "lag": 18.0}
    assert RT.scores(f)["R1"] < 0 and RT.label(f)[0] == "R0"
    y = np.tile(np.sin(2 * np.pi * 110 * np.arange(int(0.5 * R.SR)) / R.SR), 8)  # a held, periodic note
    assert RT.label(RT.features(np.stack([y, y]), R.SR))[0] == "R0"


def test_thresholds_override_moves_the_boundary():
    f = {"SM": -10.0, "coh": 0.97, "cen": 0.9, "hard": 0.0, "ildcoh": 0.5, "xcoff": 0.1, "lag": 3.0}
    assert RT.label(f)[0] != "R0"  # S/M -10 dB is not under -12
    assert RT.label(f, {"r0_sm_db": -8.0})[0] == "R0"


# ------------------------------------------------------------------------------------------------ song


def _concat(*xs):
    return np.concatenate([np.asarray(x, dtype=np.float64) for x in xs], axis=1)


def test_route_song_labels_each_window_and_skips_silence():
    scenes = [R.make_scene(s, 0).guitar_stereo() for s in ("B", "A", "P_hard", "ADT")]
    quiet = np.random.default_rng(0).normal(0, 1e-6, (2, 8 * R.SR))
    x = _concat(*scenes, quiet)
    bars = _bars(20)
    doc = RT.route_song(x, R.SR, bars, [{"id": "S1", "start_bar": 0, "end_bar": 20}])
    labels = [w["label"] for w in doc["windows"]]
    assert labels == ["R0", "R2", "R3", "R1", "silent"]
    assert doc["summary"]["windows"] == {"R0": 1, "R2": 1, "R3": 1, "R1": 1, "silent": 1}
    assert doc["windows"][4]["features"] == {} and doc["windows"][4]["p"] == {}
    assert doc["windows"][0]["rel_db"] <= 0.0 and max(w["rel_db"] for w in doc["windows"]) == 0.0
    assert [w["i"] for w in doc["windows"]] == list(range(5))
    # (n, 2) input gives the same result
    doc2 = RT.route_song(x.T, R.SR, bars, [{"id": "S1", "start_bar": 0, "end_bar": 20}])
    assert [w["label"] for w in doc2["windows"]] == labels


def test_route_song_is_deterministic():
    x = R.make_scene("A", 0).guitar_stereo()
    a = RT.route_song(x, R.SR, _bars(4), None)
    b = RT.route_song(x, R.SR, _bars(4), None)
    assert a == b


# ------------------------------------------------------------------------------------------------ solo gate


def _win(label, fcoh, a, b):
    return {"start_bar": a, "end_bar": b, "label": label, "features": {"fcoh": fcoh}}


def test_solo_gate_a_path_rise_over_the_songs_rhythm_level():
    wins = [_win("R2", 0.18, 0, 4), _win("R2", 0.19, 4, 8), _win("R2", 0.42, 8, 12),
            _win("R2", 0.17, 12, 16)]
    g = RT.solo_gate(wins, {}, {}, RT.DEFAULT_PARAMS)
    assert [x["on"] for x in g] == [False, False, True, False]
    assert g[2]["path"] == "A" and g[2]["coh_rise"] == pytest.approx(0.42 - 0.185, abs=1e-3)


def test_solo_gate_vocal_veto_and_n_path():
    wins = [_win("R2", 0.18, 0, 4), _win("R2", 0.45, 4, 8), _win("R2", 0.18, 8, 12),
            _win("R0", 0.9, 12, 16), _win("R0", 0.9, 16, 20), _win("R3", 0.5, 20, 24)]
    vocal = {k: (0.9 if 4 <= k < 8 else 0.0) for k in range(24)}
    resid = {k: (-3.0 if 12 <= k < 16 else -15.0) for k in range(24)}
    g = RT.solo_gate(wins, vocal, resid, RT.DEFAULT_PARAMS)
    assert g[1]["on"] is False and g[1].get("vetoed_by_vocal") is True
    assert (g[3]["on"], g[3]["path"]) == (True, "N") and g[4]["on"] is False
    assert g[5]["path"] is None and g[5]["on"] is False  # P windows: roles are S8's
    assert g[3]["resid_rel_db"] == -3.0 and g[1]["vocal"] == 0.9


# ------------------------------------------------------------------------------------------------ Tier B


@pytest.fixture(scope="module")
def takes():
    a, _ = R.rhythm_take(np.random.default_rng(1), dur_s=24)
    b, _ = R.rhythm_take(np.random.default_rng(2), dur_s=24)  # the same part played again (jitter, detune)
    c, _ = R.lead_line(np.random.default_rng(3), dur_s=24)  # another part
    mic2 = 0.6 * np.concatenate([np.zeros(132), a[:-132]])  # the same performance on a second, darker mic
    mic2 = signal.lfilter(*signal.butter(2, 3000 / (R.SR / 2)), mic2)
    rs = {k: librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=R.SR, target_sr=T.SR_REL)
          for k, x in {"a": a, "b": b, "c": c, "mic2": mic2}.items()}
    return {k: T.part_features(v) for k, v in rs.items()}


def test_pair_relations_same_take_double_and_other_part(takes):
    assert T.pair_relation(takes["a"], takes["mic2"])["relation"] == "same"
    d = T.pair_relation(takes["a"], takes["b"])
    assert d["relation"] == "double" and T.DOUBLE_R_MIN <= d["r"] < T.SAME_TAKE_R and d["chroma"] >= T.DOUBLE_CHROMA
    assert T.pair_relation(takes["a"], takes["c"])["relation"] == "different"


def test_no_overlap_is_no_relation(takes):
    silent = {k: (np.zeros_like(v) if k == "act" else v) for k, v in takes["b"].items()}
    assert T.pair_relation(takes["a"], silent)["relation"] == "none"


def test_takes_from_pairs_unions_same_pairs_in_file_order():
    pairs = [{"a": "p1", "b": "p2", "relation": "same"}, {"a": "p2", "b": "p3", "relation": "double"},
             {"a": "p3", "b": "p4", "relation": "same"}, {"a": "p1", "b": "p5", "relation": "different"}]
    assert T.takes_from_pairs(["p1", "p2", "p3", "p4", "p5"], pairs) == [["p1", "p2"], ["p3", "p4"], ["p5"]]


def _rel(n_s=200):
    return {"duration_s": float(n_s),
            "takes": [{"parts": ["p1"], "line": "G1", "active_s": list(range(0, 150))},
                      {"parts": ["p2"], "line": "G2", "active_s": list(range(100, 200))},
                      {"parts": ["p3"], "line": "G3", "active_s": list(range(40, 120))}],
            "links": [{"takes": [0, 1], "relation": "double", "overlap_s": 50.0},
                      {"takes": [0, 2], "relation": "different", "overlap_s": 80.0},
                      {"takes": [1, 2], "relation": "different", "overlap_s": 20.0}]}


def test_plans_pick_the_most_joint_playing_and_the_right_pans():
    rel = _rel()
    a = T.plan(rel, "A")
    assert a.pans == {0: -1.0, 1: 1.0, 2: 0.0} and a.roles == {"double": [0, 1], "centre": [2]}
    assert a.start_s == 90  # both double takes play 100-149; every start 90..100 covers all 50 s: earliest
    p = T.plan(rel, "P", song_index=0)
    assert p.pans == {0: -T.P_PAN, 2: T.P_PAN} and p.roles == {"pair": [0, 2]}
    assert T.plan(rel, "P", song_index=1).pans == {0: -1.0, 2: 1.0}
    b = T.plan(rel, "B")
    assert b.pans == {0: 0.0, 1: 0.0, 2: 0.0} and b.start_s == 0
    adt = T.plan(rel, "ADT")
    assert adt.roles == {"adt": [0]} and adt.delay_ms == T.ADT_DELAY_MS and adt.start_s == 0


def test_plan_without_a_double_or_a_pair_is_none():
    rel = _rel()
    rel["links"] = [ln for ln in rel["links"] if ln["relation"] != "double"]
    assert T.plan(rel, "A") is None
    rel["links"] = []
    assert T.plan(rel, "P") is None
    with pytest.raises(ValueError):
        T.plan(_rel(), "Z")


def test_window_truth_rules():
    roles = {"double": [0, 1], "centre": [2]}
    loud = -20.0
    assert T.window_truth("A", roles, {0: 1.0, 1: 0.8, 2: 0.3}, loud)[0] == "R2"
    assert T.window_truth("A", roles, {0: 1.0, 1: 0.0, 2: 0.1}, loud)[0] == "R3"  # one hard-panned take alone
    assert T.window_truth("A", roles, {0: 0.01, 1: 0.01, 2: 1.0}, loud)[0] == "R0"
    assert T.window_truth("A", roles, {0: 0.25, 1: 0.25, 2: 0.7}, loud)[0] == "mixed"  # 42 % / 58 %
    assert T.window_truth("A", roles, {0: 1.0, 1: 1.0, 2: 0.0}, -80.0)[0] == "silent"
    assert T.window_truth("B", {"centre": [0]}, {0: 1.0}, loud)[0] == "R0"
    assert T.window_truth("P", {"pair": [0, 2]}, {0: 0.0, 2: 1.0}, loud)[0] == "R3"
    assert T.window_truth("ADT", {"adt": [0]}, {0: 1.0}, loud)[0] == "R1"
    truth, shares = T.window_truth("A", roles, {0: 1.0, 1: 1.0, 2: 2.0}, loud)
    assert shares == {"double": 0.5, "centre": 0.5} and truth == "R2"
    assert math.isclose(sum(shares.values()), 1.0)


# ------------------------------------------------------------------------------------------------ E6 helpers


def test_score_matrix_matches_scores_window_by_window():
    rng = np.random.default_rng(3)
    feats = []
    for _ in range(300):
        f = {"SM": float(rng.uniform(-40, 3)), "r0": float(rng.uniform(-0.2, 1.0)), "xcoff": float(rng.uniform(-1, 1)),
             "lag": float(rng.uniform(-40, 40)), "coh": float(rng.uniform(0, 1)), "cen": float(rng.uniform(0, 1)),
             "hard": float(rng.uniform(0, 1)), "ildcoh": float(rng.uniform(0, 20))}
        if rng.random() < 0.1:
            f["ildcoh"] = None
        feats.append(f)
    thr = {"r2_hard": 0.2, "r2_cen": 0.3}
    sm = RT.score_matrix(RT.feature_matrix(feats), thr)
    labs = RT.labels_from_scores(sm)
    for i, f in enumerate(feats):
        sc = RT.scores(f, thr)
        for k in RT.LABELS:
            assert sm[k][i] == pytest.approx(sc[k], abs=1e-3)
        assert labs[i] == RT.label(f, thr)[0]


def _w(truth, song, **f):
    base = {"SM": -3.0, "r0": 0.3, "xcoff": 0.2, "lag": 10.0, "coh": 0.6, "cen": 0.3, "hard": 0.2, "ildcoh": 1.5}
    return {"truth": truth, "song": song, "sw": {**base, **f}}


def test_refit_moves_one_labels_thresholds_together():
    """Real doubles sit between the synthetic R2 box (hard < 0.1, cen > 0.4) and R3: hard 0.18, cen 0.33. Moving
    one bound at a time gains nothing; the R2 group search moves both."""
    from bandscribe.router import experiments as X

    dev = ([_w("R2", "s1", coh=0.6, hard=0.18, cen=0.33, ildcoh=1.5) for _ in range(6)]
           + [_w("R3", "s2", coh=0.45, hard=0.5, cen=0.08, ildcoh=12.0) for _ in range(6)]
           + [_w("R0", "s3", SM=-30.0, r0=1.0, coh=1.0, cen=1.0, hard=0.0, ildcoh=0.1, xcoff=0.7, lag=18.0)
              for _ in range(6)]
           + [_w("R1", "s4", SM=-3.0, r0=0.2, coh=0.95, cen=0.9, hard=0.0, ildcoh=0.7, xcoff=0.72, lag=15.0)
              for _ in range(6)])
    assert [RT.label(w["sw"])[0] for w in dev[:1]] == ["mixed"]
    thr, trace = X.refit(dev)
    assert trace["start"] == pytest.approx(0.75)
    assert trace["sweeps"][-1] == pytest.approx(1.0)
    assert thr["r2_hard"] >= 0.18 and thr["r2_cen"] <= 0.33
    assert all(RT.label(w["sw"], thr)[0] == w["truth"] for w in dev)
    assert thr["r0_sm_db"] == RT.THRESHOLDS["r0_sm_db"]  # nothing to gain there: the design value stays


def test_split_songs_alternates_and_is_deterministic():
    from bandscribe.router import experiments as X

    ids = [f"song{i}" for i in range(10)]
    can_a = {f"song{i}": i < 4 for i in range(10)}
    s1, s2 = X.split_songs(ids, can_a), X.split_songs(list(reversed(ids)), can_a)
    assert s1 == s2
    assert sorted(v for k, v in s1.items() if can_a[k]) == ["dev", "dev", "test", "test"]
    assert list(s1.values()).count("dev") == 5


def test_go_nogo_rule():
    from bandscribe.router import experiments as X

    assert X.go_nogo({"macro": 0.85, "r2": 0.75})["decision"] == "go"
    assert X.go_nogo({"macro": 0.85, "r2": 0.6})["decision"] == "partial"
    assert X.go_nogo({"macro": 0.85, "r2": None})["decision"] == "partial"
    assert X.go_nogo({"macro": 0.55, "r2": 0.9})["decision"] == "no-go"
    assert X.go_nogo({})["decision"] == "no-go"


def test_bootstrap_macro_point_and_interval():
    from bandscribe.router import experiments as X

    wins = ([{"song": "a", "truth": "R2", "p": "R2"}] * 4 + [{"song": "b", "truth": "R2", "p": "mixed"}] * 4
            + [{"song": "c", "truth": "R3", "p": "R3"}] * 4)
    r = X.bootstrap_macro(wins, "p", n=500)
    assert r["macro"] == pytest.approx((0.5 + 1.0) / 2) and r["r2"] == pytest.approx(0.5)
    assert r["macro_ci"][0] <= r["macro"] <= r["macro_ci"][1] and r["songs"] == 3 and r["windows"] == 12
