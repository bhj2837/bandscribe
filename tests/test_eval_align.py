"""a8: DTW alignment of a warped synthetic render recovers ≥ 95 % of downbeats within ±70 ms; with 4 taps the
anchored path passes through every tap within one feature frame; align.py never imports beat-tracking code."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from bandscribe import paths

ALIGN_PY = paths.ROOT / "bandscribe" / "eval" / "align.py"


def test_align_module_does_not_import_beat_code():
    tree = ast.parse(ALIGN_PY.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
            if node.module in ("bandscribe", "bandscribe.analysis", "bandscribe.workers"):
                modules += [f"{node.module}.{a.name}" for a in node.names]
    forbidden = ("bandscribe.analysis", "bandscribe.workers", "bandscribe.pipeline", "beat_this", "madmom")
    bad = [m for m in modules if m.startswith(forbidden)]
    assert modules and not bad, bad


def _score(bars: int = 32, seed: int = 7):
    """Distinct chords per bar (major/minor triads on random roots) + a random eighth-note melody."""
    from bandscribe.schema.gt import GtNote, GtNotes

    rng = np.random.default_rng(seed)
    notes = []
    for b in range(1, bars + 1):
        root = int(rng.integers(40, 52))
        chord = [0, 4, 7] if rng.random() < 0.5 else [0, 3, 7]
        for beat in range(4):
            for iv in chord:
                notes.append(GtNote(line="L1", ref_track=1, bar=b, tick=beat * 48.0, dur_ticks=40.0,
                                    pitch=root + iv + 12, string=None, fret=None))
        for e in range(8):
            if rng.random() < 0.7:
                notes.append(GtNote(line="L2", ref_track=2, bar=b, tick=e * 24.0, dur_ticks=22.0,
                                    pitch=int(rng.integers(64, 80)), string=None, fret=None))
    return GtNotes(format="bandscribe.gtnotes/1", song_id="a8", tpb=48, notes=notes,
                   bars=[{"bar": b, "numerator": 4, "denominator": 4, "beat_unit": "quarter", "nominal_bpm": 120.0}
                         for b in range(1, bars + 1)])


def _warped_map(bars: int = 32, lead_in_s: float = 1.0):
    """Known warp: tempo ramp 100 -> 130 BPM with ±3 % sinusoidal drift; 1 s of silence before bar 1."""
    from bandscribe.eval.refscore import tempo_map_dict, tempo_map_from_dict

    n = bars * 4 + 1
    t, times = lead_in_s, []
    for i in range(n):
        times.append(t)
        bpm = (100 + 30 * i / (n - 1)) * (1 + 0.03 * np.sin(2 * np.pi * i / 13))
        t += 60.0 / bpm
    b = [1 + i // 4 for i in range(n)]
    k = [1 + i % 4 for i in range(n)]
    return tempo_map_from_dict(tempo_map_dict(times, b, k, {x: 4 for x in set(b)}, source="synthetic"))


@pytest.mark.slow
def test_dtw_recovers_downbeats_of_warped_render():
    pytest.importorskip("synctoolbox")
    from bandscribe.eval import align, render_ref
    from bandscribe.eval.refscore import tempo_map_from_dict

    gt, true_tm = _score(), _warped_map()
    audio = align.to_mono_22k(render_ref.render_internal(gt, true_tm), 44100)
    res = align.align_audio(audio, gt, expected_bars=32)
    est = tempo_map_from_dict(res.tempo_map)
    bars = np.arange(1, 33)
    true_db = np.asarray(true_tm.bar_tick_to_time(bars, np.zeros(32)), dtype=float)
    est_db = np.asarray(est.bar_tick_to_time(bars, np.zeros(32)), dtype=float)
    err_ms = np.abs(true_db - est_db) * 1000
    ratio = float(np.mean(err_ms <= 70))
    print(f"downbeats within ±70 ms: {ratio:.3f} (median {np.median(err_ms):.1f} ms, max {err_ms.max():.1f} ms)")
    assert ratio >= 0.95
    assert res.report["bars_match"] is True and res.report["method"] == "dtw"
    # drift vs the nominal 120 BPM grid: the 100 -> 130 BPM ramp makes the performance fall far behind it
    true_t = np.asarray(true_tm.beat_times, dtype=float)
    true_drift = (true_t - true_t[0]) - 0.5 * np.arange(len(true_t))
    assert res.report["drift_s"]["max_abs"] == pytest.approx(float(np.abs(true_drift).max()), abs=0.1)
    assert res.report["drift_s"]["max_beat_to_beat"] < 0.2
    # with four tapped downbeats the anchored path goes through every tap within one feature frame (20 ms)
    taps = [(float(true_db[i]), int(bars[i])) for i in (4, 12, 20, 28)]
    res2 = align.align_audio(audio, gt, taps=taps)
    assert res2.report["method"] == "dtw+taps" and len(res2.report["anchors"]) == 4
    for row in res2.report["taps"]:
        assert abs(row["residual_ms"]) <= 1000 / align.FEATURE_RATE, row
    est2 = tempo_map_from_dict(res2.tempo_map)
    err2 = np.abs(true_db - np.asarray(est2.bar_tick_to_time(bars, np.zeros(32)), dtype=float)) * 1000
    assert float(np.mean(err2 <= 70)) >= 0.95


def test_render_internal_is_deterministic():
    from bandscribe.eval import render_ref
    from bandscribe.eval.refscore import nominal_tempo_map

    gt = _score(bars=4)
    tm = nominal_tempo_map(gt)
    a = render_ref.render_internal(gt, tm)
    b = render_ref.render_internal(gt, tm)
    assert a.dtype == np.float32 and a.tobytes() == b.tobytes()
    assert a.shape[1] == 2 and float(np.abs(a).max()) > 0.05


def test_map_times_extrapolates_at_nominal_speed():
    from bandscribe.eval import align

    path = np.array([[10.0, 20.0, 30.0], [15.0, 30.0, 45.0]])
    t = align.map_times(path, np.array([0.0, 0.3, 1.0]))  # frames 0, 15, 50 at 50 Hz
    assert np.allclose(t * align.FEATURE_RATE, [5.0, 22.5, 65.0])


def test_click_track_accents_downbeats():
    from bandscribe.eval import align
    from bandscribe.eval.refscore import nominal_tempo_map

    gt = _score(bars=2)
    tm = nominal_tempo_map(gt)
    clicks = align.click_track(tm, gt, 44100 * 5, 44100)
    down = int(round(float(np.asarray(tm.bar_tick_to_time([2], [0.0])).reshape(-1)[0]) * 44100))
    assert np.abs(clicks[down:down + 200]).max() > 0.5
