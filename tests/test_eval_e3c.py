"""E3c plumbing: explicit (held-out) windows and the condition subset of the distractor suite, and the E3c rows."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from bandscribe.amt import experiments as AX
from bandscribe.eval import distractors as D
from bandscribe.eval import experiments as X


def _power(n: int = 1000) -> dict[str, np.ndarray]:
    """10 fps curves: guitar always on, synth_pad only in the first 30 s, drums always on."""
    g = np.full(n, 1e-2)
    pad = np.zeros(n)
    pad[:300] = 1e-2
    return {"guitar_electric": g, "synth_pad": pad, "drums": np.full(n, 1e-2)}


def test_parse_windows_accepts_strings_and_lists():
    assert D.parse_windows("a@0-75, b@5.5-80") == {"a": [(0.0, 75.0)], "b": [(5.5, 80.0)]}
    assert D.parse_windows(["a@0-10", "a@20-30"]) == {"a": [(0.0, 10.0), (20.0, 30.0)]}
    assert D.parse_windows(None) == {} and D.parse_windows("") == {}


@pytest.mark.parametrize("bad", ["a0-75", "a@75", "a@80-75"])
def test_parse_windows_rejects_bad_specs(bad):
    with pytest.raises(D.DistractorError):
        D.parse_windows(bad)


def test_window_at_uses_the_selection_activity_rules():
    w = D.window_at(_power(), 0.0, 30.0, "h1")
    assert w is not None and w.wid == "h1" and w.guitar_share == 1.0
    assert w.present == ["synth_pad"] and w.shares["synth_pad"] == 1.0
    later = D.window_at(_power(), 40.0, 100.0, "h2")
    assert later is not None and later.present == []  # the pad is silent there


def test_window_at_rejects_ranges_outside_the_song_or_without_guitar():
    assert D.window_at(_power(), 90.0, 120.0, "h1") is None          # 100 s song
    assert D.window_at({"synth_pad": np.full(100, 1e-2)}, 0, 5, "h1") is None


def test_e3c_is_registered_with_a_noninferiority_rule():
    reg = X.load_registry()
    assert "E3c" in reg and "E3c" in X.ALL_IDS
    spec = reg["E3c"].spec
    assert spec["rule"] == "noninferiority" and spec["baseline"] == "production"
    assert float(spec["margin"]) == -1.0 and spec["metric"] == "note_f1_onset50"


def test_e3c_windows_are_the_registered_held_out_ones():
    wins = D.parse_windows(list(AX.E3C_WINDOWS))
    assert len(wins) == 5 and all(len(v) == 1 for v in wins.values())
    for track, [(a, b)] in wins.items():
        assert b - a == 75.0, track


def test_run_e3c_keeps_only_full_condition_rows(tmp_path, monkeypatch):
    seen = {}

    def fake_run(run_dir, cfg, args, progress=None):
        seen.update(args)
        cols = ["suite", "system", "dataset", "item", "group", "scenario", "section", "line", "rung", "metric",
                "value", "tp", "fp", "fn", "n", "ci_low", "ci_high", "note"]
        rows = [
            ["distractors", "production", "cmt", "cmt:a:h1", "a", "full", "h1", "guitar", "r", "note_f1_onset50",
             "0.8", "80", "20", "20", "100", "", "", ""],
            ["distractors", "guitar_only", "cmt", "cmt:a:h1", "a", "full", "h1", "guitar", "r", "note_f1_onset50",
             "0.85", "85", "15", "15", "100", "", "", ""],
            ["distractors", "production", "cmt", "cmt:a:h1", "a", "full", "h1", "guitar", "r", "mask_non_guitar",
             "1", "", "", "", "", "", "", ""],
            ["distractors", "production", "cmt", "cmt:a:h1", "a", "base", "h1", "guitar", "r", "note_f1_onset50",
             "0.9", "90", "10", "10", "100", "", "", ""],
            ["distractors", "production", "cmt", "*", "*", "full", "", "guitar", "r", "delta_f1",
             "-1.0", "", "", "", "1", "", "", ""],
        ]
        with open(run_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        return {}

    monkeypatch.setattr(D, "run", fake_run)
    rows = AX.run_e3c(tmp_path, cfg=None, args={})
    assert seen["conditions"] == "full" and seen["null"] is False
    assert seen["windows"] == list(AX.E3C_WINDOWS)
    assert {(r["system"], r["metric"]) for r in rows} == {("production", "note_f1_onset50"),
                                                           ("guitar_only", "note_f1_onset50"),
                                                           ("production", "mask_non_guitar")}
    assert all(r["scenario"] == "full" and "suite" not in r for r in rows)
    f1 = next(r for r in rows if r["system"] == "guitar_only" and r["metric"] == "note_f1_onset50")
    assert f1["tp"] == 85 and f1["value"] == 0.85


def test_noninferiority_with_mask_resource_decides_as_registered():
    spec = {"rule": "noninferiority", "metric": "note_f1_onset50", "baseline": "production", "margin": "-1.0",
            "resource": "mask_non_guitar:-100%"}

    def rows(cand_tp, mask_base):
        out = []
        for i in range(5):
            out += [{"system": "production", "item": f"s{i}", "group": f"s{i}", "scenario": "full",
                     "metric": "note_f1_onset50", "tp": 80, "fp": 20, "fn": 20, "value": 0.8},
                    {"system": "guitar_only", "item": f"s{i}", "group": f"s{i}", "scenario": "full",
                     "metric": "note_f1_onset50", "tp": cand_tp, "fp": 100 - cand_tp, "fn": 100 - cand_tp,
                     "value": cand_tp / 100},
                    {"system": "production", "item": f"s{i}", "group": f"s{i}", "scenario": "full",
                     "metric": "mask_non_guitar", "value": mask_base},
                    {"system": "guitar_only", "item": f"s{i}", "group": f"s{i}", "scenario": "full",
                     "metric": "mask_non_guitar", "value": 0}]
        return out

    assert X.evaluate(rows(82, 1), spec, n=500)["decision"] == "adopt"      # not worse, fewer mask classes
    assert X.evaluate(rows(75, 1), spec, n=500)["decision"] == "reject"     # clearly worse
    assert X.evaluate(rows(82, 0), spec, n=500)["decision"] == "hold"       # masks never differed: no information
