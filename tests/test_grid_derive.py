"""derive_grid on synthetic Beat This! output + percussive onsets (M1_M2_SPEC 9.3 item 3)."""
from __future__ import annotations

import numpy as np
import pytest

from bandscribe.analysis import grid as G

FPS_ENV = 22050 / 256


def q20(t):
    """Beat This! times are quantised to 20 ms (50 fps)."""
    return [round(float(x) * 50) / 50 for x in t]


def onsets_from(times, strengths=None, duration=None):
    times = np.asarray(times, dtype=float)
    strengths = np.ones(times.size) if strengths is None else np.asarray(strengths, dtype=float)
    duration = duration if duration is not None else (times.max() + 2.0 if times.size else 10.0)
    env = np.zeros(int(duration * FPS_ENV) + 2)
    for t, s in zip(times, strengths):
        i = int(round((t + G.ENV_LAG_S) * FPS_ENV))
        if 0 <= i < env.size:
            env[i] = max(env[i], s)
    order = np.argsort(times)
    return {"env": env.astype(np.float32), "fps": FPS_ENV, "times": times[order], "strength": strengths[order]}


def beats_raw(beats, downbeats, duration):
    return {"format": "bandscribe.beats/1", "tracker": "beat_this/final0", "checkpoint_sha256": None, "dbn": False,
            "fps": 50.0, "beats_s": q20(beats), "downbeats_s": q20(downbeats), "duration_s": duration}


def grid_of(beats, downbeats, onset_t, onset_s=None, duration=None, **params):
    duration = duration or float(max(beats)) + 1.0
    return G.derive_grid(beats_raw(beats, downbeats, duration), None,
                         onsets_from(onset_t, onset_s, duration + 1), params)


def full_bars(g):
    return [b for b in g["bars"] if not b["pickup"]]


def test_constant_tempo_4_4():
    bpm, nb = 120.0, 64
    t = 1.0 + np.arange(nb) * 60 / bpm
    hats = t[:-1] + 30 / bpm
    ons = np.concatenate([t, hats])
    strength = np.concatenate([np.ones(t.size), np.full(hats.size, 0.4)])
    g = grid_of(t, t[::4], ons, strength)
    assert g["hypotheses"]["tempo_octave"]["decision"] == "keep"
    assert g["hypotheses"]["compound"]["decision"] == "simple"
    assert not g["pickup"]["present"]
    bars = full_bars(g)
    assert len(bars) == nb // 4 and all(b["numerator"] == 4 and b["denominator"] == 4 for b in bars)
    assert bars[0]["bar"] == 1
    assert g["beat_unit"] == "quarter"
    # refinement put every beat back on the exact onset despite the 20 ms quantisation
    assert max(abs(b["t_s"] - x) for b, x in zip(g["beats"], t)) < 1e-6
    assert all(seg["bpm"] == 120.0 for seg in g["tempo"][:-1])
    assert g["meter"][0] == {"start_bar": 1, "end_bar": 1 + nb // 4, "numerator": 4, "denominator": 4,
                             "beat_unit": "quarter"}


def test_tempo_ramp():
    # 32 bars, 100 -> 130 BPM linearly per beat
    n = 128
    bpms = np.linspace(100, 130, n)
    t = 0.5 + np.concatenate([[0], np.cumsum(60 / bpms[:-1])])
    g = grid_of(t, t[::4], t)
    assert g["hypotheses"]["tempo_octave"]["decision"] == "keep"
    bars = full_bars(g)
    assert len(bars) == 32
    bpm_bars = [seg["bpm"] for seg in g["tempo"]]
    assert bpm_bars[0] == pytest.approx(101, abs=2) and bpm_bars[-2] == pytest.approx(129, abs=2)
    assert all(b2 >= b1 - 0.05 for b1, b2 in zip(bpm_bars[:-2], bpm_bars[1:-1]))


@pytest.mark.parametrize("pickup", [1, 2, 3])
def test_pickup_bar(pickup):
    t = 0.3 + np.arange(4 * 16 + pickup) * 0.5
    downs = t[pickup::4]
    g = grid_of(t, downs, t)
    assert g["pickup"] == {"present": True, "beats": pickup}
    b0 = g["bars"][0]
    assert b0["bar"] == 0 and b0["pickup"] and b0["n_beats"] == pickup
    assert b0["numerator"] == 4  # the pickup is a partial bar of the prevailing meter
    b1 = g["bars"][1]
    assert b1["bar"] == 1 and abs(b1["start_s"] - downs[0]) < 1e-6
    first_beats = [b for b in g["beats"] if b["bar"] == 0]
    assert [b["beat"] for b in first_beats] == list(range(1, pickup + 1))
    assert not any(b["is_downbeat"] for b in first_beats)


def test_leading_beats_longer_than_a_bar_become_full_bars():
    t = 0.3 + np.arange(4 * 10 + 6) * 0.5  # 6 beats before the first detected downbeat -> 1 bar + 2 pickup
    g = grid_of(t, t[6::4], t)
    assert g["pickup"]["beats"] == 2
    assert [b["n_beats"] for b in g["bars"][:3]] == [2, 4, 4]


def test_half_time_tracker_is_doubled():
    true = 1.0 + np.arange(64) * 0.5  # 120 BPM, a kick/snare on every beat
    tracked = true[::2]  # the tracker locked on at 60 BPM
    g = grid_of(tracked, tracked[::2], true)
    oct_ = g["hypotheses"]["tempo_octave"]
    assert oct_["decision"] == "double", oct_
    assert len(g["beats"]) == len(true) - 1 or len(g["beats"]) == len(true)
    assert all(b["n_beats"] == 4 for b in full_bars(g)[:-1])
    assert g["tempo"][0]["bpm"] == pytest.approx(120.0, abs=0.2)


def test_double_time_tracker_is_halved():
    true = 1.0 + np.arange(48) * 0.6  # 100 BPM quarter-note hits
    tracked = 1.0 + np.arange(96) * 0.3  # tracker at 200 BPM
    g = grid_of(tracked, tracked[::8], true)
    oct_ = g["hypotheses"]["tempo_octave"]
    assert oct_["decision"] == "half", oct_
    assert g["tempo"][0]["bpm"] == pytest.approx(100.0, abs=0.2)
    assert all(b["n_beats"] == 4 for b in full_bars(g)[:-1])


def _six_eight(n_bars=16, dq=0.9, t0=1.0):
    """6/8: dotted-quarter beats every dq s, eighth notes at thirds; accents on the dotted beats."""
    beats = t0 + np.arange(n_bars * 2) * dq
    eighths = np.concatenate([beats + dq / 3, beats + 2 * dq / 3])
    ons = np.concatenate([beats, eighths])
    s = np.concatenate([np.ones(beats.size), np.full(eighths.size, 0.5)])
    return beats, ons, s


def test_six_eight_dotted_quarter_tracked_is_compound():
    beats, ons, s = _six_eight()
    g = grid_of(beats, beats[::2], ons, s)
    comp = g["hypotheses"]["compound"]
    assert comp["decision"] == "compound", comp
    assert g["hypotheses"]["tempo_octave"]["decision"] == "keep"  # doubling is not offered for triple subdivision
    bars = full_bars(g)
    assert all((b["numerator"], b["denominator"], b["beat_unit"]) == (6, 8, "dotted_quarter") for b in bars)
    assert g["beat_unit"] == "dotted_quarter"
    assert g["tempo"][0]["bpm"] == pytest.approx(60 / 0.9, abs=0.2)  # BPM of the dotted-quarter beat


def test_six_eight_eighths_tracked_are_regrouped():
    beats, ons, s = _six_eight()
    eighths_all = np.sort(ons)
    g = grid_of(eighths_all, beats[::2], ons, s)  # the tracker counted 6 eighths per bar
    comp = g["hypotheses"]["compound"]
    assert comp["grouping_of_tracked_beats"] == "threes"
    assert comp["decision"] == "compound"
    bars = full_bars(g)
    assert all(b["n_beats"] == 2 and b["numerator"] == 6 for b in bars)
    assert max(abs(b["t_s"] - x) for b, x in zip(g["beats"], beats)) < 1e-6


def test_four_four_duple_subdivision_is_simple():
    t = 1.0 + np.arange(64) * 0.5
    eighths = t[:-1] + 0.25
    g = grid_of(t, t[::4], np.concatenate([t, eighths]),
                np.concatenate([np.ones(t.size), np.full(eighths.size, 0.5)]))
    comp = g["hypotheses"]["compound"]
    assert comp["decision"] == "simple" and comp["triple_ratio"] < 0.2


def test_ambiguous_band_keeps_both_hypotheses():
    # thirds and halves both present: ratio = e3 / (e3 + e2) = 0.6 exactly (energies 0.6 at thirds, 0.4 at 1/2)
    t = 1.0 + np.arange(64) * 0.6
    thirds = np.concatenate([t[:-1] + 0.2, t[:-1] + 0.4])
    halves = t[:-1] + 0.3
    ons = np.concatenate([t, thirds, halves])
    s = np.concatenate([np.ones(t.size), np.full(thirds.size, 0.6), np.full(halves.size, 0.4)])
    g = grid_of(t, t[::4], ons, s)
    comp = g["hypotheses"]["compound"]
    assert comp["decision"] == "ambiguous", comp
    units = {a["beat_unit"] for a in comp["alternatives"]}
    assert units == {"quarter", "dotted_quarter"}


def test_refinement_pulls_late_beat_onto_onset():
    t = 1.0 + np.arange(32) * 0.5
    tracked = t.copy()
    tracked[10] += 0.02  # one beat 20 ms late
    g = grid_of(tracked, tracked[::4], t)
    b = g["beats"][10]
    assert b["t_s"] == pytest.approx(t[10], abs=1e-6)
    assert b["t_raw_s"] == pytest.approx(round(tracked[10] * 50) / 50, abs=1e-6)
    assert b["shift_ms"] == pytest.approx(-20.0, abs=0.5)


def test_refinement_ignores_weak_onsets():
    t = 1.0 + np.arange(32) * 0.5
    ons = np.concatenate([t, t + 0.025])
    s = np.concatenate([np.ones(t.size), np.full(t.size, 1e-4)])
    env_heavy = onsets_from(ons, s, 20.0)
    env_heavy["env"][:] = np.maximum(env_heavy["env"], 0.5)  # a loud floor: weak onsets never pass 1.5 x median
    g = G.derive_grid(beats_raw(t + 0.02, (t + 0.02)[::4], 18.0), None, env_heavy, {})
    # the strong onsets (1.0 >= 1.5 x 0.5) still win; the weak ones do not pull beats
    assert all(abs(b["t_s"] - x) < 1e-6 for b, x in zip(g["beats"], t))


def test_missed_downbeat_splits_bar_and_odd_bar_policy():
    t = 1.0 + np.arange(40) * 0.5
    downs = list(t[::4])
    del downs[3]  # one missed downbeat -> an 8-beat bar
    g = grid_of(t, downs, t)
    assert all(b["n_beats"] == 4 for b in full_bars(g))
    assert len(full_bars(g)) == 10


def test_spurious_downbeat_run_is_recut():
    t = 1.0 + np.arange(40) * 0.5
    downs = list(t[::4])
    downs[3] = t[13]  # bars of 5 and 3 beats instead of 4 and 4
    g = grid_of(t, downs, t)
    assert [b["n_beats"] for b in full_bars(g)] == [4] * 10


def test_output_shape_matches_schema_fields():
    t = 1.0 + np.arange(16) * 0.5
    g = grid_of(t, t[::4], t)
    assert set(g) == {"format", "tpb", "duration_s", "beats", "bars", "tempo", "meter", "beat_unit", "pickup",
                      "hypotheses", "confidence", "params"}
    assert set(g["beats"][0]) == {"t_s", "t_raw_s", "bar", "beat", "is_downbeat", "shift_ms", "activation"}
    assert set(g["bars"][0]) == {"bar", "start_s", "end_s", "n_beats", "numerator", "denominator", "beat_unit",
                                 "pickup"}
    from bandscribe.schema.grid import Grid

    Grid.model_validate(g)


@pytest.mark.parametrize("pickup", [0, 1, 3])
def test_tempo_map_reads_the_grid(pickup):
    """The consumers' view (P0 TempoMap: MIDI export, evaluation) of a derived grid, pickup bar included."""
    from bandscribe.schema.grid import Grid, TempoMap

    t = 1.0 + np.arange(pickup + 32) * 0.5
    g = grid_of(t, t[pickup::4], t)
    tm = TempoMap.from_grid(Grid.model_validate(g))
    beats_t = np.asarray([b["t_s"] for b in g["beats"]])
    bar, tick = tm.time_to_bar_tick(beats_t)
    assert list(bar) == [b["bar"] for b in g["beats"]]
    assert np.allclose(tick, [(b["beat"] - 1) * 48 for b in g["beats"]])
    assert np.allclose(tm.bar_tick_to_time(bar, tick), beats_t, atol=1e-9)
    first_full = next(b for b in g["bars"] if not b["pickup"])
    assert first_full["bar"] == 1 and float(tm.bar_tick_to_time(1, 0)) == pytest.approx(first_full["start_s"])


def test_activations_feed_confidence_and_odd_bar():
    t = 1.0 + np.arange(40) * 0.5
    downs = list(t[:24:4]) + [t[24] - 0.0, t[26]] + list(t[30::4])  # bars 4 4 4 4 4 4 | 2 | 4 ...
    downs = sorted(set(round(x, 6) for x in downs))
    n = int(25 * 50)
    beat_logit = np.full(n, -5.0)
    down_logit = np.full(n, -5.0)
    for x in t:
        beat_logit[int(round(x * 50))] = 5.0
    for x in downs:
        down_logit[int(round(x * 50))] = 5.0
    acts = {"beat": beat_logit, "downbeat": down_logit, "fps": 50.0}
    g = G.derive_grid(beats_raw(t, downs, 22.0), acts, onsets_from(t, None, 23.0), {})
    counts = [b["n_beats"] for b in full_bars(g)]
    assert 2 in counts  # a confident single odd bar is kept
    assert g["confidence"]["beats"] > 0.9 and g["confidence"]["downbeats"] > 0.9


def test_no_beats_falls_back_to_constant_grid():
    g = G.derive_grid(beats_raw([], [], 10.0), None, None, {})
    assert g["confidence"]["beats"] == 0.0
    assert g["bars"] and all(b["n_beats"] <= 4 for b in g["bars"])
    assert g["hypotheses"]["warnings"]


def test_deterministic():
    beats, ons, s = _six_eight()
    a = grid_of(beats, beats[::2], ons, s)
    b = grid_of(beats, beats[::2], ons, s)
    assert a == b


def test_missed_beats_are_filled_but_breaks_are_not():
    t = 1.0 + np.arange(40) * 0.5
    tracked = np.delete(t, [9, 10])  # the tracker dropped two beats
    g = grid_of(tracked, tracked[::4], t)
    assert g["hypotheses"]["downbeat_check"]["beats_filled"] == 2
    assert len(g["beats"]) == 40
    brk = np.concatenate([t[:20], t[20:] + 0.75])  # a 1.25 s break: 2.5 beats, not a whole number
    g2 = grid_of(brk, brk[::4], brk)
    assert g2["hypotheses"]["downbeat_check"]["beats_filled"] == 0


def test_every_beat_a_downbeat_is_not_a_one_beat_meter():
    t = 1.0 + np.arange(40) * 0.5
    g = grid_of(t, t, t)
    assert all(b["n_beats"] == 4 for b in full_bars(g))


# ----------------------------------------------------------------- one metrical level (step 1b) and meter DP


def test_tracker_switching_to_double_time_mid_song_is_unified():
    """Real songs (AIZO, 絶対零度, 青と夏): Beat This! tracks some sections at twice the tempo of the rest."""
    true = 1.0 + np.arange(64) * 0.6  # 100 BPM, 16 bars of 4/4
    fast_part = 1.0 + 16 * 0.6 + np.arange(32) * 0.3  # bars 5-8 tracked at 200 BPM (8 beats per tracked bar)
    tracked = np.concatenate([true[:16], fast_part, true[32:]])
    downs = true[::4]
    g = grid_of(tracked, downs, true)
    lvl = g["hypotheses"]["beat_level"]
    assert lvl["fast_runs"] == 1 and lvl["dropped"] == 16 and lvl["inserted"] == 0
    assert lvl["reference_bpm"] == pytest.approx(100.0, abs=0.5)
    assert any("박 단위" in w for w in g["hypotheses"]["warnings"])
    assert [b["n_beats"] for b in full_bars(g)] == [4] * 16
    assert max(abs(b["t_s"] - x) for b, x in zip(g["beats"], true)) < 1e-6
    assert all(seg["bpm"] == pytest.approx(100.0, abs=0.2) for seg in g["tempo"][:-1])


def test_tracker_switching_to_half_time_mid_song_is_unified():
    true = 1.0 + np.arange(64) * 0.5  # 120 BPM
    tracked = np.concatenate([true[:24], true[24:40:2], true[40:]])  # bars 7-10 tracked at 60 BPM
    g = grid_of(tracked, true[::4], true)
    lvl = g["hypotheses"]["beat_level"]
    assert lvl["inserted"] == 8 and lvl["dropped"] == 0
    assert len(g["beats"]) == 64
    assert [b["n_beats"] for b in full_bars(g)] == [4] * 16


def test_real_tempo_change_is_not_a_level_switch():
    a = 1.0 + np.arange(32) * 0.6  # 100 BPM
    b = a[-1] + 0.46 + np.arange(32) * 0.46  # ~130 BPM: a 1.3x change is music, not a tracker octave error
    t = np.concatenate([a, b])
    g = grid_of(t, t[::4], t)
    lvl = g["hypotheses"]["beat_level"]
    assert lvl["dropped"] == 0 and lvl["inserted"] == 0
    assert len(g["beats"]) == 64 and [b_["n_beats"] for b_ in full_bars(g)] == [4] * 16


def test_single_half_interval_is_left_alone():
    """One half-length interval = the tracker jumped half a beat (syncopation), not a level switch."""
    t = 1.0 + np.arange(40) * 0.5
    t[20:] -= 0.25
    g = grid_of(t, t[::4], t)
    assert g["hypotheses"]["beat_level"]["dropped"] == 0


def test_noisy_downbeat_run_is_recut_into_regular_bars():
    """An intro where the tracker flags downbeats at random: regular bars, phase from the confident part."""
    t = 1.0 + np.arange(64) * 0.5
    rng = np.random.default_rng(5)
    noisy = sorted(rng.choice(np.arange(0, 20), size=7, replace=False).tolist())
    downs = [t[i] for i in noisy] + list(t[20::4])
    g = grid_of(t, downs, t)
    bars = full_bars(g)
    assert all(b["n_beats"] == 4 for b in bars[:-1]), [b["n_beats"] for b in bars]
    assert any(abs(b["start_s"] - t[20]) < 1e-6 for b in bars)  # the confident phase is kept
    assert g["hypotheses"]["meter_smoothing"]["odd_bars"] == 0


def test_real_two_four_bar_is_kept_when_later_bars_confirm_it():
    t = 1.0 + np.arange(42) * 0.5
    downs = list(t[0:25:4]) + list(t[26::4])  # 6 bars of 4, one bar of 2, then 4 bars of 4
    g = grid_of(t, downs, t)
    assert [b["n_beats"] for b in full_bars(g)] == [4] * 6 + [2] + [4] * 4
    two = full_bars(g)[6]
    assert (two["numerator"], two["denominator"]) == (2, 4)
    assert g["hypotheses"]["meter_smoothing"]["odd_bars"] == 1


def test_short_lived_phase_error_is_smoothed():
    """Downbeats shifted by one beat for only two bars, then back: no odd bars."""
    t = 1.0 + np.arange(48) * 0.5
    downs = list(t[0:16:4]) + [t[17], t[21]] + list(t[24::4])
    g = grid_of(t, downs, t)
    assert [b["n_beats"] for b in full_bars(g)] == [4] * 12


def test_phase_change_by_one_beat_becomes_a_long_bar_not_a_one_beat_bar():
    t = 1.0 + np.arange(49) * 0.5
    downs = list(t[0:20:4]) + list(t[21::4])  # from bar 6 on the downbeats come one beat later
    g = grid_of(t, downs, t)
    counts = [b["n_beats"] for b in full_bars(g)]
    assert counts == [4] * 4 + [5] + [4] * 7, counts


def test_halving_keeps_every_tracked_downbeat():
    """Tracker at double time with one 5/8 bar (5 tracked beats): halving counts from every downbeat, so the
    downbeats after the odd bar survive (a song-wide parity would drop all of them)."""
    fast = 0.3
    sec1 = 1.0 + np.arange(32) * fast  # 4 bars of 8 tracked beats (4/4 at 100 BPM)
    odd = sec1[-1] + fast + np.arange(5) * fast  # one bar of 5 tracked beats
    sec2 = odd[-1] + fast + np.arange(33) * fast  # 4 more bars of 8 (+1)
    tracked = np.concatenate([sec1, odd, sec2])
    downs = list(sec1[::8]) + [odd[0]] + list(sec2[::8])
    true = np.concatenate([sec1[::2], odd[::2], sec2[::2]])  # kicks on the real beats only
    g = grid_of(tracked, downs, true)
    assert g["hypotheses"]["tempo_octave"]["decision"] == "half"
    starts = [b["start_s"] for b in g["bars"]]
    for d in q20(downs):
        assert min(abs(s_ - d) for s_ in starts) < 1e-6
    assert [b["n_beats"] for b in full_bars(g)] == [4] * 4 + [3] + [4] * 4 + [1]


def test_meter_dp_unit_costs():
    """Direct checks of the segmentation: pickup, final partial bar, local mode."""
    p = np.full(14, G.P_DOWN_CLIP)
    p[[2, 6, 10]] = 1 - G.P_DOWN_CLIP
    pickup, bars, info = G._meter_dp(p, np.full(14, 4))
    assert pickup == 2 and bars == [(2, 4), (6, 4), (10, 4)] and info["odd_bars"] == 0
    p = np.full(12, G.P_DOWN_CLIP)
    p[[0, 3, 6, 9]] = 1 - G.P_DOWN_CLIP
    pickup, bars, _ = G._meter_dp(p, np.full(12, 3))  # 3/4 by the local mode
    assert pickup == 0 and bars == [(0, 3), (3, 3), (6, 3), (9, 3)]


def test_half_bar_tracking_in_a_double_time_section_is_merged():
    """Real songs: a section tracked at double tempo also had 4 tracked beats per bar there, i.e. half bars."""
    true = 1.0 + np.arange(64) * 0.6  # 100 BPM, 16 bars of 4/4
    fast_part = 1.0 + 24 * 0.6 + np.arange(32) * 0.3  # bars 7-10 tracked at 200 BPM, a "bar" every 4 beats
    tracked = np.concatenate([true[:24], fast_part, true[40:]])
    downs = list(true[:24:4]) + list(fast_part[::4]) + list(true[40::4])
    g = grid_of(tracked, downs, true)
    assert [b["n_beats"] for b in full_bars(g)] == [4] * 16
    assert g["hypotheses"]["meter_smoothing"]["tracked_mode_replaced"] is None


def test_song_wide_two_beat_bars_are_promoted_to_four():
    t = 1.0 + np.arange(64) * 0.55
    g = grid_of(t, t[::2], t)  # the tracker put a downbeat on every other beat all song long
    ms = g["hypotheses"]["meter_smoothing"]
    assert ms["tracked_mode_replaced"] == 2 and ms["mode"] == 4
    assert [b["n_beats"] for b in full_bars(g)] == [4] * 16
    assert any("반 마디" in w for w in g["hypotheses"]["warnings"])


def test_real_three_four_passage_in_four_four_is_kept():
    four = 1.0 + np.arange(32) * 0.5
    three = four[-1] + 0.5 + np.arange(24) * 0.5  # 8 bars of 3/4
    four2 = three[-1] + 0.5 + np.arange(33) * 0.5
    t = np.concatenate([four, three, four2])
    downs = list(four[::4]) + list(three[::3]) + list(four2[::4])
    g = grid_of(t, downs, t)
    assert [b["n_beats"] for b in full_bars(g)] == [4] * 8 + [3] * 8 + [4] * 8 + [1]


def test_half_tempo_tracker_with_four_tracked_beats_per_bar_gives_four_four():
    """Tracker at half tempo that also counted 4 (half-tempo) beats per bar: doubled -> 8 per tracked bar."""
    true = 1.0 + np.arange(64) * 0.5  # 120 BPM, a hit on every beat
    tracked = true[::2]
    g = grid_of(tracked, tracked[::4], true)
    assert g["hypotheses"]["tempo_octave"]["decision"] == "double"
    ms = g["hypotheses"]["meter_smoothing"]
    assert ms["tracked_mode_replaced"] == 8 and ms["mode"] == 4
    assert all(b["n_beats"] == 4 for b in full_bars(g)[:-1])
    assert any("8박" in w for w in g["hypotheses"]["warnings"])
