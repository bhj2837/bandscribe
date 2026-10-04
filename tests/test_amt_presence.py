"""Sampled piano+other presence windows (bandscribe.amt.presence): deterministic, bounded, spread, active-only."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from bandscribe.amt import presence as P

FPS = 10.0
BAR_S = 2.0


def bars(n: int, t0: float = 0.0) -> list[tuple[int, float, float]]:
    return [(b + 1, round(t0 + b * BAR_S, 6), round(t0 + (b + 1) * BAR_S, 6)) for b in range(n)]


def sections(labels: str, bars_per: int, t0: float = 0.0) -> list[dict]:
    out = []
    for i, lab in enumerate(labels):
        out.append({"id": f"S{i + 1}", "label": lab, "start_bar": 1 + i * bars_per, "end_bar": 1 + (i + 1) * bars_per,
                    "start_s": t0 + i * bars_per * BAR_S, "end_s": t0 + (i + 1) * bars_per * BAR_S})
    return out


def energy(song_s: float, po_rel_db: float | np.ndarray = -10.0,
           other_rel_db: float | np.ndarray = -180.0) -> dict[str, np.ndarray]:
    """Mix at -20 dB; 'piano' at mix + po_rel_db, 'other' at mix + other_rel_db (scalars or per frame)."""
    n = int(round(song_s * FPS))
    rel = np.broadcast_to(np.asarray(po_rel_db, dtype=np.float64), (n,)).copy()
    oth = np.broadcast_to(np.asarray(other_rel_db, dtype=np.float64), (n,)).copy()
    return {"mix": np.full(n, -20.0), "piano": -20.0 + rel, "other": -20.0 + oth}


def select(song_s: float, **kw):
    args = dict(bars=bars(int(song_s / BAR_S)), sections=None, energy=energy(song_s), fps=FPS, song_s=song_s)
    args.update(kw)
    return P.select_windows(**args)


# ------------------------------------------------------------------------------------------------ budget


@pytest.mark.parametrize("song_s,expected", [
    (240.0, (4, 10.0)),   # 20 % of 240 s = 48 s -> 4 windows
    (400.0, (6, 10.0)),   # 20 % = 80 s, capped by 60 s and 6 windows
    (30.0, (1, 10.0)),    # 20 % = 6 s < one window: still one
    (7.5, (1, 7.5)),      # shorter than a window: the whole song
    (0.0, (0, 0.0)),
])
def test_budget(song_s, expected):
    assert P.budget(song_s, window_s=10.0, max_windows=6, max_total_s=60.0, max_fraction=0.2) == expected


def test_budget_is_bounded_randomized():
    rng = random.Random(4)
    for _ in range(500):
        song = rng.uniform(1.0, 900.0)
        w, k, tot, fr = rng.choice([5.0, 10.0, 15.0]), rng.randint(1, 10), rng.uniform(5.0, 120.0), rng.uniform(.05, 1)
        n, win = P.budget(song, window_s=w, max_windows=k, max_total_s=tot, max_fraction=fr)
        assert 1 <= n <= k and win <= w and win <= song and win <= tot
        assert n * win <= max(win, min(tot, fr * song)) + 1e-9
        assert n * win <= tot + 1e-9  # never above the audio cap


# --------------------------------------------------------------------------------------------- selection


def test_windows_are_deterministic_sorted_disjoint_inside_and_within_budget():
    rng = np.random.default_rng(3)
    song = 240.0
    rel = rng.uniform(-60.0, -5.0, int(song * FPS))
    kw = dict(bars=bars(120), sections=sections("ABCABCABCABC", 10), energy=energy(song, rel))
    a, b = select(song, **kw), select(song, **kw)
    assert json.dumps(a) == json.dumps(b)
    w = a["windows"]
    assert 1 <= len(w) <= a["n_target"] == 4
    starts = [x["start_s"] for x in w]
    assert starts == sorted(starts)
    assert all(x["end_s"] <= y["start_s"] for x, y in zip(w, w[1:]))
    assert all(0.0 <= x["start_s"] and x["end_s"] <= song and x["end_s"] - x["start_s"] == pytest.approx(10.0)
               for x in w)
    assert a["transcribed_s"] <= min(60.0, 0.2 * song) and a["fraction"] <= 0.2
    assert all(x["active"] >= 0.5 for x in w)


def test_windows_start_on_bar_starts():
    t0 = 0.37
    sel = select(100.0, bars=bars(49, t0), energy=energy(100.0))
    assert sel["windows"]
    assert all(round((x["start_s"] - t0) / BAR_S, 6).is_integer() for x in sel["windows"])


def test_every_distinct_label_before_any_repeat_and_spread_over_repeats():
    song = 240.0
    secs = sections("ABABABABABAB", 10)  # 12 sections of 20 s, two labels
    sel = select(song, sections=secs)
    labels = [x["label"] for x in sel["windows"]]
    assert sorted(set(labels)) == ["A", "B"] and len(labels) == 4
    assert len({x["section"] for x in sel["windows"]}) == 4  # never two windows in one section while others wait
    # equal levels everywhere: the farthest-point rule spreads the windows over the song
    mids = sorted((x["start_s"] + x["end_s"]) / 2 for x in sel["windows"])
    assert min(b - a for a, b in zip(mids, mids[1:])) >= 40.0
    assert mids[0] < song / 4 and mids[-1] > 3 * song / 4


def test_louder_material_wins_among_new_labels():
    song = 120.0
    secs = sections("ABCDEF", 10)  # 6 labels, budget 2 windows (20 % of 120 s = 24 s)
    rel = np.full(int(song * FPS), -35.0)
    rel[int(60 * FPS):int(80 * FPS)] = -12.0  # section D is much louder
    sel = select(song, sections=secs, energy=energy(song, rel))
    assert sel["n_target"] == 2
    assert "D" in [x["label"] for x in sel["windows"]]


def test_only_active_regions_are_sampled():
    song = 200.0
    rel = np.full(int(song * FPS), -90.0)
    rel[int(60 * FPS):int(100 * FPS)] = -15.0  # piano/other plays only between 60 and 100 s
    sel = select(song, sections=sections("ABCDE", 20), energy=energy(song, rel))
    assert sel["windows"] and all(60.0 - 5.0 <= x["start_s"] and x["end_s"] <= 100.0 + 5.0 for x in sel["windows"])
    assert len(sel["windows"]) <= 4 and sel["reason"] in (None, "fewer_qualified_windows")


def test_quiet_piano_other_gives_no_window():
    sel = select(120.0, energy=energy(120.0, -80.0))
    assert sel["windows"] == [] and sel["reason"] == "piano_other_inactive" and sel["n_qualified"] == 0
    assert P.spans(sel) == []


def test_no_energy_and_empty_song():
    assert select(60.0, energy={})["reason"] == "no_energy"
    assert select(0.0, bars=[], energy=energy(1.0))["reason"] == "empty_song"


def test_without_grid_and_sections_time_zones_spread_the_windows():
    song = 300.0
    sel = select(song, bars=[], sections=None)
    assert len(sel["windows"]) == 6  # 20 % of 300 s = 60 s
    zones = {x["section"] for x in sel["windows"]}
    assert len(zones) == 6 and all(z.startswith("Z") for z in zones)


def test_short_song_is_one_window_covering_it():
    sel = select(8.0, bars=bars(4), sections=sections("A", 4))
    assert [(x["start_s"], x["end_s"]) for x in sel["windows"]] == [(0.0, 8.0)]
    assert sel["fraction"] == 1.0


def test_level_series_combines_piano_and_other():
    e = {"mix": np.array([-20.0]), "piano": np.array([-30.0]), "other": np.array([-30.0])}
    assert P.piano_other_rel_db(e)[0] == pytest.approx(-10.0 + 10 * np.log10(2))
    assert P.piano_other_rel_db({"mix": np.array([-20.0])}) is None


# ------------------------------------------------------------------------- review fixes (2026-10-04)


def test_window_longer_than_the_cap_is_clamped():
    """fast profile + a 60 s window: the window shrinks to the 30 s cap instead of sampling 60 s."""
    assert P.budget(300.0, window_s=60.0, max_windows=3, max_total_s=30.0, max_fraction=0.2) == (1, 30.0)
    sel = select(300.0, window_s=60.0, max_windows=3, max_total_s=30.0)
    assert sel["transcribed_s"] <= 30.0 and sel["window_s"] == 30.0


def test_min_gap_spreads_windows_over_many_labels():
    """青と夏 had 9 labels for 5 windows and got 4 windows in 152-253 s (two adjacent): with more labels than
    windows the gap rule still spreads them."""
    song = 240.0
    secs = sections("ABCDEFGHIJKL", 10)  # 12 distinct labels of 20 s each
    rel = np.full(int(song * FPS), -20.0)
    rel[int(100 * FPS):int(150 * FPS)] = -5.0  # a louder stretch that used to attract the windows
    sel = select(song, sections=secs, energy=energy(song, rel))
    mids = sorted((x["start_s"] + x["end_s"]) / 2 for x in sel["windows"])
    assert len(mids) == sel["n_target"] == 4 and sel["min_gap_s"] == 30.0
    assert min(b - a for a, b in zip(mids, mids[1:])) >= 30.0
    assert sum(1 for m in mids if 100.0 <= m <= 150.0) <= 2


def test_windows_stay_inside_their_section():
    """A quiet bridge next to a loud chorus: the bridge's window lies inside the bridge (no 40 % straddle)."""
    song = 150.0  # 3 windows: S1, S3, then the new label B
    secs = [{"id": "S1", "label": "A", "start_s": 0.0, "end_s": 50.0}, {"id": "S2", "label": "B", "start_s": 50.0,
                                                                       "end_s": 70.0},
            {"id": "S3", "label": "C", "start_s": 70.0, "end_s": 150.0}]
    rel = np.full(int(song * FPS), -10.0)
    rel[int(50 * FPS):int(70 * FPS)] = -35.0  # bridge S2 is quiet but active
    sel = select(song, sections=secs, energy=energy(song, rel), bars=bars(75))
    assert sel["n_target"] == 3
    w = next(x for x in sel["windows"] if x["section"] == "S2")
    assert 50.0 <= w["start_s"] and w["end_s"] <= 70.0 + 0.5


def test_every_active_stem_is_sampled_at_least_once():
    """Piano plays only in 150-170 s while the other stem is loud everywhere: one window must cover the piano."""
    song = 200.0
    piano = np.full(int(song * FPS), -180.0)
    piano[int(150 * FPS):int(170 * FPS)] = -25.0
    sel = select(song, sections=sections("ABCDE", 20), energy=energy(song, piano, other_rel_db=-8.0))
    assert sel["stems"] == {"piano": {"active": True, "covered": True}, "other": {"active": True, "covered": True}}
    assert any("piano" in x["covers"] and 150.0 <= x["start_s"] and x["end_s"] <= 170.0 for x in sel["windows"])


def test_staccato_part_qualifies_by_its_power_mean():
    """Active in 20 % of the frames but loud: the bar-level rule says active, so the window qualifies (the old
    frame-share rule skipped the pass and the stems alone decided)."""
    song = 120.0
    rel = np.full(int(song * FPS), -90.0)
    rel[::5] = -5.0  # every fifth frame
    sel = select(song, energy=energy(song, rel))
    assert sel["reason"] is None and sel["windows"]
    assert all(x["active"] == pytest.approx(0.2, abs=0.02) for x in sel["windows"])
