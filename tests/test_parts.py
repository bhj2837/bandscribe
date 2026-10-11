"""Guitar part diarizer S8 v0 (M7a): symbolic layers, the robust part count, per-note Lines, Part view names."""
from __future__ import annotations

import pytest

from bandscribe.parts import arbitration as A
from bandscribe.parts import diarize as D
from bandscribe.parts import layers as LY
from bandscribe.parts import reid as R
from bandscribe.parts import roles as RO

DIST = "distorted_electric_guitar"
CLEAN = "clean_electric_guitar"


def _n(t, p, d=0.4, cls=DIST):
    return {"onset_s": round(t, 4), "offset_s": round(t + d, 4), "pitch": p, "instrument": cls}


def _strum(t, root, d=0.45, cls=DIST, spread=0.01):
    return [_n(t + k * spread, root + iv, d, cls) for k, iv in enumerate((0, 7, 12))]


def _wins(n=4, w=8.0, label="R3", sections=("S1", "S1", "S2", "S2")):
    return [{"start_s": k * w, "end_s": (k + 1) * w, "label": label, "section": sections[k]} for k in range(n)]


# ------------------------------------------------------------------------------------------------ layers


def test_a_strum_is_a_chord_and_a_far_top_note_is_peeled():
    notes = [_n(1.0, 40), _n(1.01, 47), _n(1.02, 52), _n(1.015, 64)]  # E2 B2 E3 + E4 twelve above
    f = LY.note_features(notes)
    assert [x["chord"] for x in f] == [True, True, True, False]
    assert f[3]["peel"] and f[3]["lead"] and f[3]["r"] == 1.0
    assert {x["cluster"] for x in f} == {0} and f[0]["size"] == 4 and f[0]["r"] == -1.0


def test_power_chord_dyads_are_chords_but_thirds_are_not():
    f = LY.note_features([_n(0.0, 40), _n(0.0, 47), _n(2.0, 60), _n(2.0, 64)])
    assert [x["chord"] for x in f] == [True, True, False, False]


def test_lead_layer_is_a_note_over_a_sounding_chord_and_above_it():
    notes = [*_strum(0.0, 40, d=2.0),  # E5 chord ringing 0-2 s, top E3 (52)
             _n(0.5, 59),  # above the chord, while it sounds -> lead
             _n(0.7, 45),  # below its top -> not lead
             _n(0.05, 60),  # part of the strum's attack (within LEAD_AFTER_S... but its own cluster) -> not lead
             _n(3.0, 62)]  # after the chord ended -> not lead
    f = LY.note_features(notes)
    assert [x["lead"] for x in f[3:]] == [True, False, False, False]


def test_tabforge_split_sends_chordy_windows_to_rhythm_and_the_rest_to_lead():
    notes = [*_strum(0.1, 40), *_strum(0.5, 40), _n(0.8, 50),  # chord window: the single goes with it
             _n(2.1, 64), _n(2.4, 66), _n(2.7, 67)]  # a window of single notes
    assert LY.tabforge_split(notes) == ["rhythm"] * 7 + ["lead"] * 3


# ------------------------------------------------------------------------------------------- part count


def _p_song(ild_left=5.0, ild_right=-5.0, windows=4):
    """Left: power-chord strums every 0.5 s; right: a single-note line every 0.25 s; R3 windows."""
    notes, sigs = [], []
    for k in range(windows):
        for s in range(16):
            for x in _strum(k * 8 + s * 0.5, 40)[:2]:
                notes.append(x)
                sigs.append({"ild_db": ild_left})
        for s in range(32):
            notes.append(_n(k * 8 + s * 0.25 + 0.12, 64 + (s % 5), d=0.2))
            sigs.append({"ild_db": ild_right})
    return notes, sigs


def test_two_sides_in_enough_windows_and_sections_count_two_parts():
    notes, sigs = _p_song()
    est = R.estimate(notes, LY.note_features(notes), sigs, _wins())
    assert est["k"] == 2 and est["evidence"] == "spatial" and est["support"]["spatial"] == 4
    assert est["hypotheses"][0]["k"] == 2 and est["hypotheses"][0]["p"] > 0.5


def test_two_streams_in_too_few_windows_or_one_section_stay_one_part():
    notes, sigs = _p_song()
    labels = ["R3", "R3", "R0", "R0"]
    wins = [{**w, "label": lab} for w, lab in zip(_wins(), labels)]
    off = {**R.DEFAULT_PARAMS, "use_class": False, "use_symbolic": False}
    assert R.estimate(notes, LY.note_features(notes), sigs, wins, off)["k"] == 1  # 2 windows < 3
    one_sec = _wins(sections=("S1", "S1", "S1", "S2"))
    wins = [{**w, "label": lab} for w, lab in zip(one_sec, ["R3", "R3", "R3", "R0"])]
    assert R.estimate(notes, LY.note_features(notes), sigs, wins, off)["k"] == 1  # 3 windows, but one section
    same = _wins(sections=("S1",) * 4)
    wins = [{**w, "label": lab} for w, lab in zip(same, ["R3", "R3", "R3", "R0"])]
    assert R.estimate(notes, LY.note_features(notes), sigs, wins, off)["k"] == 2  # the input has one section


def test_one_strummed_guitar_is_one_part():
    notes = [x for k in range(64) for x in _strum(k * 0.5, 40 + (k % 3) * 2)]
    est = R.estimate(notes, LY.note_features(notes), [{} for _ in notes], _wins(label="R0"))
    assert est["k"] == 1 and est["windows_two"] == 0 and est["hypotheses"][0] == {"k": 1, "p": 1.0}


def test_two_classes_sounding_together_are_two_streams():
    notes = []
    for k in range(64):
        notes += _strum(k * 0.5, 40)
        notes.append(_n(k * 0.5 + 0.25, 70, cls=CLEAN))
    est = R.estimate(notes, LY.note_features(notes), [{} for _ in notes], _wins(label="R0"),
                     {**R.DEFAULT_PARAMS, "use_symbolic": False})
    assert est["k"] == 2 and est["evidence"] == "class"


def test_a_window_with_too_few_notes_has_no_stream():
    r = R.window_streams([0, 1], [_n(0, 40), _n(1, 41)], LY.note_features([_n(0, 40), _n(1, 41)]), [{}, {}],
                         "R3", R.DEFAULT_PARAMS)
    assert r["c"] == 0


# --------------------------------------------------------------------------------------- note decisions


def test_spatial_lines_follow_the_ild_sign_in_p_windows():
    notes, sigs = _p_song()
    feats = LY.note_features(notes)
    res = A.assign(notes, feats, sigs, ["R3"] * len(notes), 2, "spatial")
    keys = [res["lines"][j]["key"] for j in res["line"]]
    assert keys == ["L" if s["ild_db"] > 0 else "R" for s in sigs]
    assert res["lines"][0]["position"] > 0.9 and res["lines"][1]["position"] < -0.9


def test_symbolic_lines_split_chords_from_the_line_over_them():
    notes = []
    for k in range(32):
        notes += _strum(k * 1.0, 40, d=0.9)
        notes.append(_n(k * 1.0 + 0.5, 64 + k % 4, d=0.2))
    feats = LY.note_features(notes)
    res = A.assign(notes, feats, [{} for _ in notes], ["R0"] * len(notes), 2, "symbolic")
    keys = [res["lines"][j]["key"] for j in res["line"]]
    assert keys == ["lead" if f["lead"] else "rhythm" for f in feats]


def test_one_part_is_one_line_and_a_forced_split_without_evidence_says_why():
    notes = [_n(k * 0.5, 50 + k % 3) for k in range(20)]
    feats = LY.note_features(notes)
    res = A.assign(notes, feats, [{} for _ in notes], ["R0"] * len(notes), 1, None)
    assert len(res["lines"]) == 1 and set(res["line"]) == {0}
    res = A.assign(notes, feats, [{} for _ in notes], ["R0"] * len(notes), 2, None)
    assert len(res["lines"]) == 1 and res["reasons"]


def test_a_note_with_no_evidence_either_way_is_unassigned():
    notes, sigs = _p_song()
    notes.append(_n(100.0, 60, cls=CLEAN))  # no stereo signature, a class neither side has
    sigs.append({"ild_db": None})
    feats = LY.note_features(notes)
    res = A.assign(notes, feats, sigs, ["R3"] * len(notes), 2, "spatial", tau=0.6)
    assert res["line"][-1] is None and res["p"][-1] < 0.6 and res["amber"][-1]
    assert all(j is not None for j in res["line"][:-1])


# ------------------------------------------------------------------------------------------------- names


def test_gtr_1_is_the_line_in_most_bars_and_names_say_role_class_position():
    notes, sigs = _p_song()
    out = D.diarize(notes, sigs, _wins())
    assert [ln["number"] for ln in out["lines"]] == [1, 2]
    by = {ln["key"]: ln for ln in out["lines"]}
    assert by["L"]["name"] == "Gtr 1 – Rhythm (dist., L)"  # same bars and notes: the earlier first note
    assert by["R"]["name"] == "Gtr 2 – Lead (dist., R)"
    assert {n["line"] for n in out["note"]} == {1, 2}


def test_one_guitar_is_named_gtr_without_a_number():
    notes = [x for k in range(64) for x in _strum(k * 0.5, 40)]
    out = D.diarize(notes, [{} for _ in notes], _wins(label="R0"))
    assert [ln["name"] for ln in out["lines"]] == ["Gtr – Rhythm (dist.)"]
    assert out["count"]["chosen"] == 1


def test_forced_count_overrides_the_estimate():
    notes = []
    for k in range(32):
        notes += _strum(k * 1.0, 40, d=0.9)
        notes.append(_n(k * 1.0 + 0.5, 64, d=0.2))
    out = D.diarize(notes, [{} for _ in notes], _wins(label="R0"), k_force=2,
                    params={**R.DEFAULT_PARAMS, "n_min": 100})
    assert out["count"]["k"] == 1 and out["count"]["forced"] == 2 and out["count"]["chosen"] == 2


@pytest.mark.parametrize("pos, want", [(0.8, "L"), (-0.8, "R"), (0.1, "")])
def test_position_is_named_only_beyond_the_side_threshold(pos, want):
    notes = [_n(0, 40), _n(0, 47)]
    feats = LY.note_features(notes)
    rows = RO.order_and_names([{"key": "x", "position": pos}], [0, 0], notes, feats, [0, 0])
    assert rows[0]["position"] == want


# ---------------------------------------------------------------------------------------------- E25 helpers

from bandscribe.parts import eval as E  # noqa: E402


def test_macro_hungarian_matches_labels_to_lines_and_counts_unassigned_wrong():
    truth = [{"a"}] * 4 + [{"b"}] * 2 + [{"a", "b"}] + [set()]
    pred = ["2", "2", "2", None, "1", "1", "1", "1"]
    m, ex = E.macro_hungarian(truth, pred)
    assert m == pytest.approx((3 / 4 + 2 / 2) / 2) and ex["match"] == {"2": "a", "1": "b"}
    assert E.macro_hungarian(truth, ["x"] * 8)[0] == pytest.approx(0.5)  # the majority floor: one Line right
    assert E.unison_hit(truth, pred, ex["match"]) == 1.0  # a unison note is right in either of its Lines
    assert E.macro_hungarian([set(), set()], ["1", "2"])[0] is None


def test_line_groups_join_the_included_doubles():
    rel = {"links": [{"takes": [0, 1], "relation": "double"}, {"takes": [0, 2], "relation": "different"}]}
    assert E.line_groups(rel, [0, 1, 2]) == {0: "g0", 1: "g0", 2: "g2"}
    assert E.line_groups(rel, [1, 2]) == {1: "g1", 2: "g2"}


def test_coverage_accuracy_keeps_the_confident_notes():
    truth = [{"a"}, {"a"}, {"b"}, {"b"}]
    rows = E.coverage_accuracy(truth, ["1", "1", "2", "1"], [0.9, 0.55, 0.95, 0.5], {"1": "a", "2": "b"},
                               taus=(0.4, 0.6))
    assert rows == [{"tau": 0.4, "coverage": 1.0, "accuracy": 0.75}, {"tau": 0.6, "coverage": 0.5, "accuracy": 1.0}]


def _rows(song, sc, arm, **vals):
    return [{"item": f"{song}:{sc}", "group": song, "scenario": sc, "system": arm, "metric": m, "value": v,
             "section": "test"} for m, v in vals.items()]


def test_decide_applies_the_registered_rules():
    rows, detail = [], {}
    for s in ("s1", "s2", "s3"):
        rows += _rows(s, "P", "s8v0", k_correct=1.0, k_abs_err=0.0, assign_macro=0.95)
        rows += _rows(s, "P", "majority", assign_macro=0.5)
        rows += _rows(s, "P", "symbolic", assign_macro=0.6)
        rows += _rows(s, "N1", "s8v0", k_correct=1.0, k_abs_err=0.0, assign_macro=1.0)
        rows += _rows(s, "N1", "majority", assign_macro=1.0)
        detail[f"{s}:P"] = {"k_true": 2, "notes": 100, "arms": {"s8v0": {"k_pred": 2, "n_unassigned": 5}}}
        detail[f"{s}:N1"] = {"k_true": 1, "notes": 100, "arms": {"s8v0": {"k_pred": 1, "n_unassigned": 0}}}
    d = E.decide(rows, detail)
    assert d["count"]["pass"] and d["assign"]["P"]["pass"] and d["one_guitar"]["pass"]
    assert d["assign"]["P"]["vs_majority"]["delta"] == 45.0
    assert d["unassigned"] == {"share": 0.025, "pass": True}
    assert d["assign"]["A1"]["vs_majority"] is None and not d["assign"]["A1"]["pass"]
    assert d["no_regression"]["pass"]
    detail["s1:N1"]["arms"]["s8v0"]["k_pred"] = 2
    assert not E.decide(rows, detail)["one_guitar"]["pass"]


def test_dev_selection_keeps_the_first_of_tied_settings():
    notes, sigs = _p_song()
    p_item = {"scenario": "P", "sys": notes, "sigs": sigs, "windows": _wins(), "k_true": 2,
              "truth": [{"L"} if s["ild_db"] > 0 else {"R"} for s in sigs]}
    strums = [x for k in range(64) for x in _strum(k * 0.5, 40)]
    n1 = {"scenario": "N1", "sys": strums, "sigs": [{} for _ in strums], "windows": _wins(label="R0"), "k_true": 1,
          "truth": [{"g0"} for _ in strums]}
    chosen, trace = E.choose_params([p_item, n1])
    assert len(trace) == len(E.GRID) and all(t["one_guitar_k1"] == 1 and t["k_acc"] == 1.0 for t in trace)
    assert {k: chosen[k] for k in E.GRID[0]} == E.GRID[0]
