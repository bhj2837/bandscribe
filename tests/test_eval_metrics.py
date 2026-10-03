"""M1a metric identities (DESIGN §10 M1a, spec §11 a1–a7).

Run a single criterion with ``-k identity|deletion|octave|label_swap|unassigned_monotone|majority|grid_leak``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gtab.eval import baselines, metrics
from gtab.eval.metrics import UNASSIGNED, Event

pytest.importorskip("mir_eval")


# ------------------------------------------------------------------------------------------------ helpers


def random_notes(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n notes whose onsets are ≥ 120 ms apart (more than 2× the tolerance), so a shifted or deleted note can
    never be matched by a neighbour by coincidence."""
    rng = np.random.default_rng(seed)
    onsets = np.cumsum(rng.uniform(0.12, 0.4, n))
    pitches = rng.integers(40, 85, n).astype(float)
    durs = rng.uniform(0.05, 0.8, n)
    return np.stack([onsets, onsets + durs], axis=1), pitches


def prediction(notes: list[tuple[float, float, int, str]], *, item: str = "t", posteriors=None):
    from gtab.schema.evalio import PredNote, Prediction

    pn = [PredNote(onset_s=a, offset_s=b, pitch=p, line=ln, posterior=None if posteriors is None else posteriors[i])
          for i, (a, b, p, ln) in enumerate(notes)]
    return Prediction(format="gtab.prediction/1", system="test", item=item, lines=[], notes=pn)


def gt_events(notes: list[tuple[float, int, str]]) -> list[Event]:
    """Events from (onset, pitch, line); identical (onset, pitch) in two lines merge into a unison event."""
    rows = {}
    for t, p, ln in notes:
        rows.setdefault((round(t, 6), p), set()).add(ln)
    return sorted((Event(onset_s=t, pitch=p, lines=frozenset(ls)) for (t, p), ls in rows.items()),
                  key=lambda e: (e.onset_s, e.pitch))


def two_line_song(n: int = 400, seed: int = 1, unison_every: int = 0):
    """GT with lines L1 (≈ 2/3 of notes) and L2; optional unison events every k-th note."""
    rng = np.random.default_rng(seed)
    out = []
    t = 0.0
    for i in range(n):
        t += rng.uniform(0.12, 0.3)
        p = int(rng.integers(40, 80))
        ln = "L1" if rng.random() < 0.66 else "L2"
        out.append((round(t, 4), p, ln))
        if unison_every and i % unison_every == 0:
            out.append((round(t, 4), p, "L2" if ln == "L1" else "L1"))
    return out


# ------------------------------------------------------------------------------------------ a1 identity


def test_identity_note_f1_all_variants():
    ref = random_notes(2000, seed=3)
    for name, r in metrics.note_prf_variants(ref, ref).items():
        assert r.f1 == pytest.approx(1.0), name
        assert (r.tp, r.fp, r.fn) == (2000, 0, 0)


def test_identity_assignment_and_tick_f1():
    notes = two_line_song(300)
    ev = gt_events(notes)
    pred = prediction([(t, t + 0.1, p, ln) for t, p, ln in notes])
    res = metrics.assignment(ev, metrics.events_from_prediction(pred))
    assert res.macro == pytest.approx(1.0) and res.weighted == pytest.approx(1.0)
    assert res.unassigned_ratio == 0.0
    gt, tm = _grid_song(bars=8, warp=False)
    r = metrics.tick_prf(gt, _perf_arrays(gt, tm), tm)
    assert r.f1 == pytest.approx(1.0) and r.tp == len(gt.notes)


def test_prf_counts_consistent():
    ref = random_notes(1000, seed=5)
    rng = np.random.default_rng(9)
    keep = rng.random(1000) < 0.7
    est_iv = ref[0][keep]
    est_p = ref[1][keep].copy()
    est_p[::5] += 1  # some wrong pitches -> fp and fn
    extra_iv, extra_p = random_notes(100, seed=11)
    est = (np.concatenate([est_iv, extra_iv + 1000]), np.concatenate([est_p, extra_p]))
    r = metrics.note_prf(ref, est)
    assert r.tp + r.fp == len(est[1]) and r.tp + r.fn == len(ref[1])
    assert r.precision == pytest.approx(r.tp / (r.tp + r.fp))
    assert r.recall == pytest.approx(r.tp / (r.tp + r.fn))
    assert r.f1 == pytest.approx(2 * r.precision * r.recall / (r.precision + r.recall))


# ------------------------------------------------------------------------------------------ a2 deletion


def test_deletion_20_percent_gives_recall_080():
    ref = random_notes(5000, seed=21)
    rng = np.random.default_rng(22)
    keep = rng.random(len(ref[1])) >= 0.2
    r = metrics.note_prf(ref, (ref[0][keep], ref[1][keep]))
    assert r.recall == pytest.approx(0.80, abs=0.02)
    assert r.precision == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------- a3 octave


def test_octave_shift_pitch_f1_zero_chroma_one():
    ref = random_notes(2000, seed=31)
    est = (ref[0], ref[1] + 12)
    assert metrics.note_prf(ref, est).f1 == 0.0
    assert metrics.note_prf(ref, est, chroma=True).f1 == pytest.approx(1.0)
    assert metrics.octave_error_rate(ref, est) == pytest.approx(1.0)


def test_octave_semitone_shift_pitch_f1_zero():
    """+1 semitone must score 0: guards the MIDI -> Hz conversion (raw MIDI would look ≈ 29 cents apart)."""
    ref = random_notes(2000, seed=32)
    assert metrics.note_prf(ref, (ref[0], ref[1] + 1)).f1 == 0.0
    assert metrics.note_prf(ref, (ref[0], ref[1] - 1)).f1 == 0.0


# --------------------------------------------------------------------------------------- a4 label swap


def test_label_swap_zero_before_hungarian_one_after():
    notes = two_line_song(300, seed=41)
    swap = {"L1": "L2", "L2": "L1"}
    pred = prediction([(t, t + 0.1, p, swap[ln]) for t, p, ln in notes])
    res = metrics.assignment(gt_events(notes), metrics.events_from_prediction(pred))
    assert res.label_swap_raw == pytest.approx(0.0)
    assert res.macro == pytest.approx(1.0)
    assert res.mapping == {"L1": "L2", "L2": "L1"}


# ---------------------------------------------------------------------------------- a5 unassigned monotone


@pytest.mark.parametrize("unison_every", [0, 7])
def test_unassigned_monotone_property(unison_every):
    """Turning predicted notes into Unassigned (or adding unassigned notes) never raises macro accuracy."""
    for trial in range(25):
        rng = np.random.default_rng(1000 + trial)
        notes = two_line_song(120, seed=500 + trial, unison_every=unison_every)
        ev = gt_events(notes)
        lines = ["L1", "L2", "L3"]
        # noisy prediction: random labels for 40 %, some unassigned already
        pnotes = []
        for t, p, ln in notes:
            u = rng.random()
            lab = ln if u > 0.4 else lines[int(rng.integers(0, 3))]
            if rng.random() < 0.05:
                lab = UNASSIGNED
            pnotes.append([t, t + 0.1, p, lab])
        prev = metrics.assignment(ev, metrics.events_from_prediction(prediction([tuple(x) for x in pnotes]))).macro
        for _step in range(6):
            idx = [i for i, x in enumerate(pnotes) if x[3] != UNASSIGNED]
            if not idx:
                break
            for i in rng.choice(idx, size=max(1, len(idx) // 5), replace=False):
                pnotes[i][3] = UNASSIGNED
            cur = metrics.assignment(ev, metrics.events_from_prediction(prediction([tuple(x) for x in pnotes]))).macro
            assert cur <= prev + 1e-12, (trial, cur, prev)
            prev = cur
        extra = [(float(rng.uniform(0, 60)), float(rng.uniform(0, 60)) + 60.1, int(rng.integers(40, 80)), UNASSIGNED)
                 for _ in range(30)]
        added = [tuple(x) for x in pnotes] + [(a, a + 0.1, p, ln) for a, _b, p, ln in extra]
        cur = metrics.assignment(ev, metrics.events_from_prediction(prediction(added))).macro
        assert cur <= prev + 1e-12


# ------------------------------------------------------------------------------------------- a6 majority


@pytest.mark.parametrize("seed", [51, 52, 53])
def test_majority_all_in_one_line_equals_baseline(seed):
    notes = two_line_song(250, seed=seed)
    ev = gt_events(notes)
    pred = prediction([(t, t + 0.1, p, "X") for t, p, _ln in notes])
    one = metrics.assignment(ev, metrics.events_from_prediction(pred))
    maj_pred = baselines.majority(prediction([(t, t + 0.1, p, ln) for t, p, ln in notes]), ev)
    maj = metrics.assignment(ev, metrics.events_from_prediction(maj_pred))
    assert one.macro == pytest.approx(maj.macro)
    assert one.weighted == pytest.approx(maj.weighted)
    assert one.mapping == {"X": baselines.largest_line(ev)}
    assert one.macro == pytest.approx(0.5)  # two GT lines, one of them fully right


def test_guitar_classes_match_schema():
    from gtab.schema.instrumentation import GUITAR_CLASSES

    assert tuple(baselines.GUITAR_CLASSES) == tuple(GUITAR_CLASSES)


# ------------------------------------------------------------------------------------------- a7 grid leak


def _grid_song(bars: int = 16, *, warp: bool = True, seed: int = 71):
    """GT (bar/tick) on a warped 4/4 tempo map: 100 -> 130 BPM ramp with ±3 % sinusoidal drift."""
    from gtab.eval.refscore import tempo_map_dict, tempo_map_from_dict
    from gtab.schema.gt import GtNote, GtNotes

    n_beats = bars * 4 + 1
    t, times = 0.0, []
    for i in range(n_beats):
        times.append(t)
        frac = i / max(1, n_beats - 1)
        bpm = (100 + 30 * frac) * (1 + 0.03 * np.sin(2 * np.pi * i / 11)) if warp else 120.0
        t += 60.0 / bpm
    bar_of = [1 + i // 4 for i in range(n_beats)]
    beat_of = [1 + i % 4 for i in range(n_beats)]
    tm = tempo_map_from_dict(tempo_map_dict(times, bar_of, beat_of, {b: 4 for b in set(bar_of)}, source="synthetic"))
    rng = np.random.default_rng(seed)
    notes = []
    for b in range(1, bars + 1):
        for k in range(8):
            if rng.random() < 0.8:
                notes.append(GtNote(line="L1", ref_track=1, bar=b, tick=k * 24.0, dur_ticks=24.0,
                                    pitch=int(rng.integers(40, 76)), string=None, fret=None))
    gt = GtNotes(format="gtab.gtnotes/1", song_id="grid", tpb=48,
                 bars=[{"bar": b, "numerator": 4, "denominator": 4, "beat_unit": "quarter", "nominal_bpm": 120.0}
                       for b in range(1, bars + 1)], notes=notes)
    return gt, tm


def _perf_arrays(gt, tm, jitter_s: float = 0.0, seed: int = 0):
    t = np.asarray(tm.bar_tick_to_time(np.array([n.bar for n in gt.notes]), np.array([n.tick for n in gt.notes])))
    t = t + np.random.default_rng(seed).uniform(-jitter_s, jitter_s, len(t))
    return np.stack([t, t + 0.1], axis=1), np.array([n.pitch for n in gt.notes], dtype=float)


def test_grid_leak_system_grid_errors_do_not_reach_tick_f1():
    """Predictions at the true times, system grid off by half a beat and by one downbeat: tick F1 stays 1.0
    (it only uses the GT tempo map), while the system's own quantisation (notation match) is visibly wrong."""
    from gtab.eval.refscore import tempo_map_dict, tempo_map_from_dict

    gt, tm = _grid_song(bars=16, warp=True)
    perf = _perf_arrays(gt, tm, jitter_s=0.02, seed=5)
    assert metrics.tick_prf(gt, perf, tm).f1 == pytest.approx(1.0)
    # a system grid: half a beat late, and its downbeats one beat off (bar lines on GT beat 2)
    gt_beats = np.asarray(tm.beat_to_time(np.arange(0, 16 * 4 + 1)), dtype=float)
    half = np.diff(gt_beats) / 2
    sys_times = list(gt_beats[:-1] + half)
    sys_bar = [1 + (i + 1) // 4 for i in range(len(sys_times))]
    sys_beat = [1 + (i + 1) % 4 for i in range(len(sys_times))]
    sys_tm = tempo_map_from_dict(tempo_map_dict(sys_times, sys_bar, sys_beat, {b: 4 for b in set(sys_bar)}, source="x"))
    sb, st = sys_tm.time_to_bar_tick(perf[0][:, 0])
    downbeats = {int(b): float(sys_tm.bar_tick_to_time([b], [0.0])[0]) for b in sorted(set(sys_bar))}
    bmap = metrics.bar_map_by_downbeats(downbeats, tm, [n.bar for n in gt.notes])
    quant = [(int(b), float(np.round(t / 24) * 24), int(p)) for b, t, p in zip(sb, st, perf[1])]
    nm = metrics.notation_match(gt, quant, bmap)
    assert nm["exact_ratio"] < 0.2  # the system grid is wrong ...
    assert metrics.tick_prf(gt, perf, tm).f1 == pytest.approx(1.0)  # ... and tick F1 does not see it
    # sanity: with the GT grid itself the notation match is perfect
    gb, gtk = tm.time_to_bar_tick(_perf_arrays(gt, tm)[0][:, 0])
    exact = [(int(b), float(np.round(t / 24) * 24), int(p)) for b, t, p in zip(gb, gtk, perf[1])]
    ident = {b: b for b in range(0, 18)}
    assert metrics.notation_match(gt, exact, ident)["exact_ratio"] == pytest.approx(1.0)


def test_grid_leak_tick_f1_rejects_wrong_pitch_and_far_onsets():
    gt, tm = _grid_song(bars=8, warp=True)
    iv, p = _perf_arrays(gt, tm)
    assert metrics.tick_prf(gt, (iv, p + 1), tm).f1 == 0.0
    assert metrics.tick_prf(gt, (iv + 0.08, p), tm).f1 == 0.0  # 80 ms > ±50 ms
    assert metrics.tick_prf(gt, (iv + 0.04, p), tm).f1 == pytest.approx(1.0)


def test_tick_f1_sections_half_open():
    gt, tm = _grid_song(bars=8, warp=False)
    perf = _perf_arrays(gt, tm)
    r = metrics.tick_prf(gt, perf, tm, sections=[(3, 5)])
    n_in = sum(1 for n in gt.notes if 3 <= n.bar < 5)
    assert r.tp == n_in and r.fp == 0 and r.fn == 0


# ---------------------------------------------------------------------------------------- coverage / shared


def test_coverage_accuracy_curve_and_shared():
    notes = two_line_song(200, seed=81, unison_every=5)
    ev = gt_events(notes)
    rng = np.random.default_rng(3)
    post = rng.uniform(0, 1, len(notes)).tolist()
    pred = prediction([(t, t + 0.1, p, ln) for t, p, ln in notes], posteriors=post)
    curve = metrics.coverage_accuracy(ev, metrics.events_from_prediction(pred), taus=(0.0, 0.5, 0.9))
    covs = [c for _t, c, _a in curve]
    assert covs[0] == pytest.approx(1.0) and covs[0] >= covs[1] >= covs[2]
    assert all(a == pytest.approx(1.0) for _t, _c, a in curve if not np.isnan(a))
    single = metrics.coverage_accuracy(ev, metrics.events_from_prediction(
        prediction([(t, t + 0.1, p, ln) for t, p, ln in notes])))
    assert len(single) == 1
    r = metrics.shared_prf(ev, metrics.events_from_prediction(pred))
    assert r.recall == pytest.approx(1.0) and r.precision == pytest.approx(1.0)
