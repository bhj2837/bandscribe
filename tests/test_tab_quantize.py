"""M3 quantisation (DESIGN 6.2, 10 M3): synthetic performances on a known grid, triplet feel, shuffle vs 12/8.

The acceptance criterion "[판정] 격자를 아는 합성 연주(휴먼라이즈 ±20 ms)에서 음의 90 % 이상이 정확한 (마디, tick)에
온다" is ``test_humanised_performance_lands_on_exact_bar_tick``; "셋잇단 느낌 픽스처와 복합 박자 픽스처(6/8·12/8 vs
4/4 셔플) 판정이 모두 맞는다" is the ``feel`` / ``shuffle`` / ``compound`` tests.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bandscribe.schema.grid import Grid, TempoMap
from bandscribe.tab import quantize as Q

BIN_PATTERNS = [(0,), (0, .5), (0, .25, .5, .75), (0, .5, .75), (0, .25, .5), (.5,), (0, .75), (0, .25),
                (.25, .5, .75)]
TER_PATTERNS = [(0, 1 / 3, 2 / 3), (0, 2 / 3), (1 / 3, 2 / 3), (0, 1 / 3)]


def beat_times(n_beats: int, *, warp: bool = True, bpm: float = 120.0) -> list[float]:
    """Beat times: a 100 -> 130 BPM ramp with ±3 % drift (warp), or a constant tempo."""
    t, out = 0.5, []
    for i in range(n_beats):
        out.append(t)
        frac = i / max(1, n_beats - 1)
        b = (100 + 30 * frac) * (1 + 0.03 * np.sin(2 * np.pi * i / 11)) if warp else bpm
        t += 60.0 / b
    return out


def grid_doc(times: list[float], n_beats: int = 4, unit: str = "quarter", decision: str = "simple") -> dict:
    """A valid grid.json with whole bars of ``n_beats`` beats (the last listed bar may be partial)."""
    beats, bars = [], []
    n_bars = (len(times) + n_beats - 1) // n_beats
    for i, t in enumerate(times):
        beats.append({"t_s": round(t, 6), "t_raw_s": round(t, 6), "bar": 1 + i // n_beats, "beat": 1 + i % n_beats,
                      "is_downbeat": i % n_beats == 0, "shift_ms": 0.0, "activation": None})
    num, den = (3 * n_beats, 8) if unit == "dotted_quarter" else (n_beats, 4)
    for b in range(n_bars):
        s = times[b * n_beats]
        e = times[(b + 1) * n_beats] if (b + 1) * n_beats < len(times) else times[-1] + (times[-1] - times[-2])
        bars.append({"bar": b + 1, "start_s": round(s, 6), "end_s": round(e, 6), "n_beats": n_beats,
                     "numerator": num, "denominator": den, "beat_unit": unit, "pickup": False})
    doc = {"format": "bandscribe.grid/1", "tpb": 48, "duration_s": round(times[-1] + 1.0, 6), "beats": beats,
           "bars": bars, "tempo": [{"start_bar": 1, "end_bar": n_bars + 1, "bpm": 120.0}],
           "meter": [{"start_bar": 1, "end_bar": n_bars + 1, "numerator": num, "denominator": den, "beat_unit": unit}],
           "beat_unit": unit, "pickup": {"present": False, "beats": 0},
           "hypotheses": {"compound": {"decision": decision, "triple_ratio": None, "alternatives": []}},
           "confidence": {}, "params": {}}
    Grid.model_validate(doc)
    return doc


def at(tm: TempoMap, beat: float) -> float:
    return float(tm.beat_to_time(beat))


def performance(times: list[float], n_bars: int, *, seed: int, jitter_s: float = 0.02, p_ter: float = 0.12,
                n_beats: int = 4, chords: float = 0.3):
    """Notes from random per-beat patterns (binary, and runs of 1-2 triplet beats) humanised by
    U(-jitter, +jitter) per note; returns (notes, true (bar, tick) per note)."""
    tm = TempoMap(times, [1 + i // n_beats for i in range(len(times))], [1 + i % n_beats for i in range(len(times))],
                  {b: n_beats for b in range(1, len(times) // n_beats + 2)})
    rng = np.random.default_rng(seed)
    notes, truth = [], []
    i, total = 0, n_bars * n_beats
    while i < total:
        ter = rng.random() < p_ter
        for _ in range(int(rng.integers(1, 3)) if ter else 1):
            if i >= total:
                break
            pats = TER_PATTERNS if ter else BIN_PATTERNS
            pat = pats[int(rng.integers(len(pats)))]
            for f in pat:
                t_true = at(tm, i + f)
                for _c in range(1 + (int(rng.integers(1, 3)) if rng.random() < chords else 0)):
                    on = t_true + rng.uniform(-jitter_s, jitter_s)
                    notes.append({"onset_s": on, "offset_s": on + 0.09, "pitch": int(rng.integers(40, 80))})
                    truth.append((1 + i // n_beats, (i % n_beats) * 48 + round(f * 48)))
            i += 1
    order = np.argsort([n["onset_s"] for n in notes], kind="stable")
    return [notes[k] for k in order], [truth[k] for k in order]


# ------------------------------------------------------------------------------- [판정] exact (bar, tick)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("warp", [True, False])
def test_humanised_performance_lands_on_exact_bar_tick(seed, warp):
    n_bars = 24
    times = beat_times(n_bars * 4 + 8, warp=warp)
    grid = grid_doc(times)
    notes, truth = performance(times, n_bars, seed=seed)
    doc = Q.quantize({"g": notes}, grid)
    got = [(n["bar"], n["tick"]) for n in doc["tracks"]["g"]["notes"]]
    ratio = Q.exact_ratio(zip(got, truth))
    assert ratio >= 0.90, ratio
    assert doc["tracks"]["g"]["stats"]["n"] == len(notes)


def test_exact_ratio_pooled_over_seeds_is_well_above_the_bar():
    """Pooled over 10 seeds at ±20 ms the share is far above 90 % (a margin, so tuning noise cannot hide)."""
    got, truth = [], []
    for seed in range(10, 20):
        times = beat_times(24 * 4 + 8, warp=True)
        notes, tr = performance(times, 24, seed=seed)
        doc = Q.quantize({"g": notes}, grid_doc(times))
        got += [(n["bar"], n["tick"]) for n in doc["tracks"]["g"]["notes"]]
        truth += tr
    assert Q.exact_ratio(zip(got, truth)) >= 0.95


def test_triplet_beats_take_the_ternary_family():
    times = beat_times(40, warp=False, bpm=120.0)
    tm = TempoMap(times, [1 + i // 4 for i in range(40)], [1 + i % 4 for i in range(40)], {b: 4 for b in range(1, 12)})
    notes = []
    for i in range(32):
        fr = (0, 1 / 3, 2 / 3) if i in (5, 6, 17) else (0, 0.5)
        notes += [{"onset_s": at(tm, i + f), "offset_s": at(tm, i + f) + 0.1, "pitch": 60} for f in fr]
    doc = Q.quantize({"g": notes}, grid_doc(times))
    fam = dict(doc["beat_families"]["*"])
    assert [b for b, f in sorted(fam.items()) if f == "ter"] == [5, 6, 17]
    ticks = sorted({n["tick"] % 48 for n in doc["tracks"]["g"]["notes"]})
    assert ticks == [0, 16, 24, 32]


def test_joint_decision_puts_the_bass_note_of_a_triplet_beat_on_the_same_grid():
    times = beat_times(24, warp=False)
    tm = TempoMap(times, [1 + i // 4 for i in range(24)], [1 + i % 4 for i in range(24)], {b: 4 for b in range(1, 8)})
    gtr = [{"onset_s": at(tm, 9 + f), "offset_s": at(tm, 9 + f) + 0.1, "pitch": 64} for f in (0, 1 / 3, 2 / 3)]
    bass = [{"onset_s": at(tm, 9 + 2 / 3) + 0.025, "offset_s": at(tm, 10), "pitch": 40}]  # 25 ms late
    doc = Q.quantize({"guitar": gtr, "bass": bass}, grid_doc(times))
    assert doc["tracks"]["bass"]["notes"][0]["tick"] % 48 == 32  # with the guitar's triplets: the third triplet
    alone = Q.quantize({"bass": bass}, grid_doc(times))
    assert alone["tracks"]["bass"]["notes"][0]["tick"] % 48 == 36  # alone, nearer the 4th 16th


# ----------------------------------------------------------------------------------------- durations


def test_durations_follow_quantised_offsets_with_a_32nd_minimum():
    times = beat_times(24, warp=False)
    tm = TempoMap(times, [1 + i // 4 for i in range(24)], [1 + i % 4 for i in range(24)], {b: 4 for b in range(1, 8)})
    notes = [{"onset_s": at(tm, 4), "offset_s": at(tm, 5.5) + 0.01, "pitch": 60},      # 1.5 beats
             {"onset_s": at(tm, 6), "offset_s": at(tm, 6.5) - 0.012, "pitch": 62},     # an 8th
             {"onset_s": at(tm, 7), "offset_s": at(tm, 7) + 0.01, "pitch": 64}]        # a blip -> 32nd
    doc = Q.quantize({"g": notes}, grid_doc(times))
    assert [n["dur"] for n in doc["tracks"]["g"]["notes"]] == [72, 24, 6]


# ---------------------------------------------------------------------------------- feel and meter fixtures


def _swing_notes(tm: TempoMap, n_beats: int, fracs=(0, 2 / 3), seed: int = 3, jitter_s: float = 0.015):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_beats):
        for f in fracs:
            if f == 0 or rng.random() < 0.8:
                on = at(tm, i + f) + rng.uniform(-jitter_s, jitter_s)
                out.append({"onset_s": on, "offset_s": on + 0.08, "pitch": int(rng.integers(45, 70))})
    return sorted(out, key=lambda n: n["onset_s"])


def test_triplet_feel_fixture_swung_eighths_in_four_four():
    times = beat_times(16 * 4 + 4, warp=False, bpm=132.0)
    tm = TempoMap.from_dict(grid_doc(times))
    doc = Q.quantize({"g": _swing_notes(tm, 64)}, grid_doc(times))
    assert doc["meter"]["read_as"] is None  # the grid said simple 4/4: nothing to re-read
    assert [s["feel"] for s in doc["sections"]] == ["triplet8th"]
    assert all(b["feel"] == "triplet8th" for b in doc["bars"])
    off = [n for n in doc["tracks"]["g"]["notes"] if n["tick"] % 48]
    assert off and all(n["tick"] % 48 == 24 for n in off)  # the swung 8th is written as a straight 8th


def test_straight_eighths_get_no_feel():
    times = beat_times(16 * 4 + 4, warp=False, bpm=132.0)
    tm = TempoMap.from_dict(grid_doc(times))
    doc = Q.quantize({"g": _swing_notes(tm, 64, fracs=(0, 0.5))}, grid_doc(times))
    assert [s["feel"] for s in doc["sections"]] == [None]
    assert all(n["tick"] % 24 == 0 for n in doc["tracks"]["g"]["notes"])


def test_shuffle_fixture_twelve_eight_grid_read_as_four_four_shuffle():
    """The grid's triple ratio calls a 4/4 shuffle compound (onset energy at 2/3, none at 1/2); the notes never
    play 1/3, so the bars become 4/4 with triplet feel."""
    times = beat_times(16 * 4 + 4, warp=False, bpm=120.0)
    grid = grid_doc(times, unit="dotted_quarter", decision="compound")
    tm = TempoMap.from_dict(grid)
    doc = Q.quantize({"g": _swing_notes(tm, 64)}, grid)
    assert doc["meter"]["read_as"] == "shuffle"
    assert all((b["numerator"], b["denominator"], b["beat_unit"], b["feel"]) == (4, 4, "quarter", "triplet8th")
               for b in doc["bars"])
    assert doc["meter"]["alternatives"][0]["numerator"] == 12
    assert all(n["tick"] % 24 == 0 for n in doc["tracks"]["g"]["notes"])


def test_compound_fixture_twelve_eight_with_all_three_eighths_stays_compound():
    times = beat_times(16 * 4 + 4, warp=False, bpm=70.0)
    grid = grid_doc(times, unit="dotted_quarter", decision="compound")
    tm = TempoMap.from_dict(grid)
    doc = Q.quantize({"g": _swing_notes(tm, 64, fracs=(0, 1 / 3, 2 / 3))}, grid)
    assert doc["meter"]["read_as"] == "compound"
    assert all((b["numerator"], b["denominator"], b["beat_unit"], b["feel"]) == (12, 8, "dotted_quarter", None)
               for b in doc["bars"])
    assert sorted({n["tick"] % 48 for n in doc["tracks"]["g"]["notes"]}) == [0, 16, 32]
    assert doc["meter"]["alternatives"][0]["feel"] == "triplet8th"


def test_six_eight_long_short_stays_six_eight():
    times = beat_times(16 * 2 + 2, warp=False, bpm=60.0)
    grid = grid_doc(times, n_beats=2, unit="dotted_quarter", decision="compound")
    tm = TempoMap.from_dict(grid)
    doc = Q.quantize({"g": _swing_notes(tm, 32)}, grid)
    assert doc["meter"]["read_as"] is None
    assert all((b["numerator"], b["denominator"], b["beat_unit"]) == (6, 8, "dotted_quarter") for b in doc["bars"])
    assert sorted({n["tick"] % 48 for n in doc["tracks"]["g"]["notes"]}) == [0, 32]


def test_feel_is_decided_per_section():
    times = beat_times(16 * 4 + 4, warp=False, bpm=126.0)
    grid = grid_doc(times)
    tm = TempoMap.from_dict(grid)
    swing = [n for n in _swing_notes(tm, 64) if n["onset_s"] < at(tm, 32)]
    straight = [n for n in _swing_notes(tm, 64, fracs=(0, 0.5), seed=5) if n["onset_s"] >= at(tm, 32) - 0.03]
    secs = [{"id": "S1", "start_bar": 1, "end_bar": 9}, {"id": "S2", "start_bar": 9, "end_bar": 17}]
    doc = Q.quantize({"g": swing + straight}, grid, secs)
    assert [s["feel"] for s in doc["sections"]] == ["triplet8th", None]
    feel = {b["bar"]: b["feel"] for b in doc["bars"]}
    assert feel[4] == "triplet8th" and feel[12] is None


# ---------------------------------------------------------------------------------------------- misc


def test_deterministic_and_empty_tracks():
    times = beat_times(30, warp=True)
    notes, _ = performance(times, 6, seed=1)
    a = Q.quantize({"g": notes, "b": []}, grid_doc(times))
    b = Q.quantize({"g": notes, "b": []}, grid_doc(times))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["tracks"]["b"]["notes"] == [] and a["tracks"]["b"]["stats"]["n"] == 0
    assert Q.quantize({"g": []}, grid_doc(times))["tracks"]["g"]["notes"] == []


def test_notes_before_the_first_beat_get_bars_before_bar_one():
    times = beat_times(20, warp=False)  # 120 BPM, first beat at 0.5 s
    doc = Q.quantize({"g": [{"onset_s": 0.25, "offset_s": 0.4, "pitch": 50}]}, grid_doc(times))
    n = doc["tracks"]["g"]["notes"][0]
    assert n["bar"] == 0 and n["tick"] == 168  # half a beat before beat 1: the "and" of beat 4 of bar 0
