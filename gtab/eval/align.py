"""Ground-truth tempo map by score-to-audio DTW (``gtab gt align``; DESIGN 8.3, spec §9.4 item 6, A9).

Independent of Beat This! on purpose: a GT grid built from the tracker under test would grade the M3 beat metrics
against themselves. This module must not import ``gtab.analysis.grid`` or any beat worker (a test checks it).

Features (synctoolbox 1.4.2, feature rate 50 Hz): audio = mono 22050 Hz mix -> pitch features -> quantised chroma
+ pitch-onset features -> DLNCO; score = the expanded GT notes at nominal tempo -> ``df_to_pitch_features`` /
``df_to_pitch_onset_features`` -> the same chroma / DLNCO. A global chroma shift (tuning / transposition) is
estimated first. Without taps: ``sync_via_mrmsdtw(step_weights=[1.5, 1.5, 2.0], threshold_rec=10**6)``; with
tapped downbeats: ``sync_via_mrmsdtw_with_anchors(anchor_pairs=[(t_score, t_audio), …])`` (score time = nominal
time of that expanded bar's downbeat) — no re-anchoring after the fact. The path is made strictly monotonic and
every nominal beat time is mapped through it (linear between path points) -> ``tempo_map.json``.
synctoolbox / libfmp / pandas are imported inside functions (heavy; libfmp pulls IPython).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gtab import atomic

log = logging.getLogger(__name__)

FEATURE_RATE = 50
FS = 22050
STEP_WEIGHTS = (1.5, 1.5, 2.0)
_CROP_MARGIN = 5  # frames kept around the active regions (same on both sides)
THRESHOLD_REC = 10 ** 6


@dataclass
class AlignResult:
    tempo_map: dict[str, Any]          # tempo_map.json content
    report: dict[str, Any]             # align_report.json content
    path: np.ndarray                   # (2, L) strictly monotonic, frames (score, audio)


def score_dataframe(gt_notes: Any, beat_times: Sequence[float], bars: Sequence[int], beats: Sequence[int]):
    """GT notes at nominal tempo as the synctoolbox dataframe (start, duration, pitch, velocity, instrument)."""
    import pandas as pd

    from gtab.eval.refscore import tempo_map_dict, tempo_map_from_dict

    bpb: dict[int, int] = {}
    for b, k in zip(bars, beats):
        bpb[int(b)] = max(bpb.get(int(b), 0), int(k))
    tm = tempo_map_from_dict(tempo_map_dict(beat_times, bars, beats, bpb, source="nominal", tpb=gt_notes.tpb))
    notes = list(gt_notes.notes)
    on = np.asarray(tm.bar_tick_to_time(np.array([n.bar for n in notes]), np.array([n.tick for n in notes], dtype=float)))
    off = np.asarray(tm.bar_tick_to_time(np.array([n.bar for n in notes]),
                                         np.array([n.tick + n.dur_ticks for n in notes], dtype=float)))
    rows = [{"start": float(a), "duration": float(max(0.02, b - a)), "pitch": int(n.pitch), "velocity": 100,
             "instrument": "guitar"} for n, a, b in zip(notes, on, off)]
    return pd.DataFrame(rows, columns=["start", "duration", "pitch", "velocity", "instrument"])


def _chroma_dlnco_audio(x: np.ndarray):
    from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
    from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
    from synctoolbox.feature.pitch import audio_to_pitch_features
    from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features

    f_pitch = audio_to_pitch_features(f_audio=x, Fs=FS, feature_rate=FEATURE_RATE, verbose=False)
    chroma = quantize_chroma(pitch_to_chroma(f_pitch=f_pitch))
    onsets = audio_to_pitch_onset_features(f_audio=x, Fs=FS, verbose=False)
    dlnco = pitch_onset_features_to_DLNCO(f_peaks=onsets, feature_rate=FEATURE_RATE,
                                          feature_sequence_length=chroma.shape[1], visualize=False)
    return chroma, dlnco


def _chroma_dlnco_score(df, n_frames: int | None = None):
    from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
    from synctoolbox.feature.csv_tools import df_to_pitch_features, df_to_pitch_onset_features
    from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO

    f_pitch = df_to_pitch_features(df, feature_rate=FEATURE_RATE)
    chroma = quantize_chroma(pitch_to_chroma(f_pitch=f_pitch))
    onsets = df_to_pitch_onset_features(df)
    dlnco = pitch_onset_features_to_DLNCO(f_peaks=onsets, feature_rate=FEATURE_RATE,
                                          feature_sequence_length=chroma.shape[1], visualize=False)
    return chroma, dlnco


def _chroma_shift(chroma_audio: np.ndarray, chroma_score: np.ndarray) -> int:
    from synctoolbox.dtw.utils import compute_optimal_chroma_shift
    from synctoolbox.feature.chroma import quantized_chroma_to_CENS

    a = quantized_chroma_to_CENS(chroma_audio, 201, 50, FEATURE_RATE)[0]
    s = quantized_chroma_to_CENS(chroma_score, 201, 50, FEATURE_RATE)[0]
    return int(compute_optimal_chroma_shift(s, a))


def to_mono_22k(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if sr != FS:
        import soxr

        mono = soxr.resample(mono, sr, FS)
    return np.ascontiguousarray(mono, dtype=np.float64)


def map_times(path: np.ndarray, t_score: np.ndarray) -> np.ndarray:
    """Score seconds -> audio seconds through the (strictly monotonic) warping path; outside the path the
    mapping continues at nominal speed (slope 1) from its first / last point."""
    p = np.asarray(path, dtype=float)
    f = np.asarray(t_score, dtype=float) * FEATURE_RATE
    out = np.interp(f, p[0], p[1])
    out = np.where(f < p[0, 0], p[1, 0] + (f - p[0, 0]), out)
    out = np.where(f > p[0, -1], p[1, -1] + (f - p[0, -1]), out)
    return out / FEATURE_RATE


def _active_frames(n_frames: int, energy: np.ndarray, rel_db: float = -40.0, margin: int = 5) -> tuple[int, int]:
    """[first, last+1) frames whose energy is within ``rel_db`` of the maximum (plus a margin)."""
    if energy.size == 0 or not np.any(energy > 0):
        return 0, n_frames
    act = np.flatnonzero(energy >= energy.max() * 10 ** (rel_db / 10))
    return max(0, int(act[0]) - margin), min(n_frames, int(act[-1]) + 1 + margin)


def _audio_energy(x: np.ndarray, n_frames: int) -> np.ndarray:
    hop = FS // FEATURE_RATE
    e = np.array([float(np.mean(x[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(x) else 0.0 for i in range(n_frames)])
    return e


def align_audio(audio_22k: np.ndarray, gt_notes: Any, *, taps: Sequence[tuple[float, int]] | None = None,
                expected_bars: int | None = None) -> AlignResult:
    """Core alignment on arrays. ``taps`` = [(audio time s, expanded bar)] of tapped downbeats."""
    from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw, sync_via_mrmsdtw_with_anchors
    from synctoolbox.dtw.utils import make_path_strictly_monotonic, shift_chroma_vectors

    from gtab.eval.refscore import nominal_beat_times, tempo_map_dict

    import contextlib
    import io

    times, bars, beats, bpb = nominal_beat_times(gt_notes)
    df = score_dataframe(gt_notes, times, bars, beats)
    with contextlib.redirect_stdout(io.StringIO()):  # synctoolbox prints progress dots
        ch_s, on_s = _chroma_dlnco_score(df)
        ch_a, on_a = _chroma_dlnco_audio(audio_22k)
    # DTW pins both ends: crop silence (audio lead-in / fade, score rests) and extrapolate outside (map_times)
    a0, a1 = _active_frames(ch_a.shape[1], _audio_energy(audio_22k, ch_a.shape[1]), margin=0)
    s0 = int(np.floor(df["start"].min() * FEATURE_RATE)) if len(df) else 0
    s1 = min(ch_s.shape[1], int(np.ceil((df["start"] + df["duration"]).max() * FEATURE_RATE))) if len(df) else ch_s.shape[1]
    m0 = min(_CROP_MARGIN, s0, a0)                      # same margin on both sides keeps the ends paired
    m1 = min(_CROP_MARGIN, ch_s.shape[1] - s1, ch_a.shape[1] - a1)
    s0, a0, s1, a1 = s0 - m0, a0 - m0, s1 + m1, a1 + m1
    ch_s, on_s, ch_a, on_a = ch_s[:, s0:s1], on_s[:, s0:s1], ch_a[:, a0:a1], on_a[:, a0:a1]
    shift = _chroma_shift(ch_a, ch_s)
    if shift:
        ch_s = shift_chroma_vectors(ch_s, shift)
        on_s = shift_chroma_vectors(on_s, shift)
    kw = dict(f_chroma1=ch_s, f_onset1=on_s, f_chroma2=ch_a, f_onset2=on_a, input_feature_rate=FEATURE_RATE,
              step_weights=np.array(STEP_WEIGHTS), threshold_rec=THRESHOLD_REC, verbose=False)
    downbeat_nominal = {int(b): float(t) for t, b, k in zip(times, bars, beats) if int(k) == 1}
    anchors: list[tuple[float, float]] = []
    if taps:
        len_s, len_a = ch_s.shape[1] / FEATURE_RATE, ch_a.shape[1] / FEATURE_RATE
        for t_audio, bar in sorted(taps, key=lambda x: x[0]):
            if bar is None or int(bar) not in downbeat_nominal:
                continue
            ts = downbeat_nominal[int(bar)] - s0 / FEATURE_RATE
            ta = float(t_audio) - a0 / FEATURE_RATE
            if 0 < ts < len_s and 0 < ta < len_a and (not anchors or (ts > anchors[-1][0] and ta > anchors[-1][1])):
                anchors.append((round(ts, 6), round(ta, 6)))
    with contextlib.redirect_stdout(io.StringIO()):
        if anchors:
            wp = sync_via_mrmsdtw_with_anchors(anchor_pairs=list(anchors), **kw)
        else:
            wp = sync_via_mrmsdtw(**kw)
    wp = make_path_strictly_monotonic(wp).astype(float)
    wp[0] += s0
    wp[1] += a0
    t_audio = map_times(wp, np.asarray(times))
    t_audio = np.maximum.accumulate(t_audio + np.arange(len(t_audio)) * 1e-7)  # strictly increasing beats
    tm = tempo_map_dict(t_audio, bars, beats, bpb, source="dtw+taps" if anchors else "dtw", tpb=gt_notes.tpb)
    nominal_dur = np.diff(times)
    aligned_dur = np.diff(t_audio)
    ratio = nominal_dur / np.maximum(aligned_dur, 1e-6)
    # drift = how far the performance wandered from a strict nominal-tempo grid started at the first beat
    # (large values are normal for rubato / tempo changes the score does not write down; a sudden jump
    # between neighbouring beats points at a misalignment worth tapping)
    drift = (t_audio - t_audio[0]) - (np.asarray(times, dtype=float) - float(times[0]))
    step = np.abs(np.diff(drift)) if len(drift) > 1 else np.zeros(1)
    tap_rows = []
    for t_aud, bar in (taps or []):
        if bar is not None and int(bar) in downbeat_nominal:
            est = float(map_times(wp, [downbeat_nominal[int(bar)]])[0])
            tap_rows.append({"bar": int(bar), "t_tap_s": round(float(t_aud), 4), "t_aligned_s": round(est, 4),
                             "residual_ms": round((est - float(t_aud)) * 1000, 1)})
    n_bars = len([b for b in gt_notes.bars])
    report = {"format": "gtab.align_report/1", "method": "dtw+taps" if anchors else "dtw",
              "feature_rate": FEATURE_RATE, "step_weights": list(STEP_WEIGHTS), "chroma_shift": shift,
              "anchors": [[a[0] + s0 / FEATURE_RATE, a[1] + a0 / FEATURE_RATE] for a in anchors],
              "active_frames": {"score": [s0, s1], "audio": [a0, a1]}, "n_beats": len(times), "expanded_bars": n_bars,
              "reference_expanded_bars": expected_bars,
              "bars_match": None if expected_bars is None else int(expected_bars) == n_bars,
              "taps": tap_rows,
              "max_tap_residual_ms": max((abs(r["residual_ms"]) for r in tap_rows), default=None),
              "local_tempo_ratio": {"min": round(float(ratio.min()), 3), "max": round(float(ratio.max()), 3)},
              "drift_s": {"min": round(float(drift.min()), 4), "max": round(float(drift.max()), 4),
                          "max_abs": round(float(np.abs(drift).max()), 4),
                          "max_beat_to_beat": round(float(step.max()), 4)},
              "audio_s": round(len(audio_22k) / FS, 3), "score_s": round(float(times[-1]), 3)}
    return AlignResult(tm, report, wp)


def click_track(tempo_map: Any, gt_notes: Any, n: int, sr: int) -> np.ndarray:
    """Clicks at every GT note onset (2 kHz, short) plus accented downbeat clicks (1 kHz, louder)."""
    out = np.zeros(n)
    m = int(0.012 * sr)
    t = np.arange(m) / sr
    note_click = 0.3 * np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 400)
    down_click = 0.6 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 250)
    if gt_notes.notes:
        on = np.asarray(tempo_map.bar_tick_to_time(np.array([x.bar for x in gt_notes.notes]),
                                                   np.array([x.tick for x in gt_notes.notes], dtype=float)))
        for s in np.unique(np.round(on * sr).astype(int)):
            if 0 <= s < n:
                e = min(n, s + m)
                out[s:e] += note_click[: e - s]
    for b in sorted({int(x["bar"]) for x in gt_notes.bars}):
        s = int(round(float(np.asarray(tempo_map.bar_tick_to_time([b], [0.0])).reshape(-1)[0]) * sr))
        if 0 <= s < n:
            e = min(n, s + m)
            out[s:e] += down_click[: e - s]
    return out


def align_song(song_dir: Path, mix_wav: Path, *, taps: Sequence[tuple[float, int]] | None = None) -> AlignResult:
    """Align a Tier A song: writes tempo_map.json, align_report.json, align_check.wav; refreshes gt_notes times."""
    from gtab.audio import read_f32, write_f32
    from gtab.eval import gtstore
    from gtab.eval.refscore import fill_times, tempo_map_from_dict

    song = gtstore.load_song(song_dir)
    if song.gt_notes is None:
        raise gtstore.GtError("gt_notes.json 이 없습니다 (gtab gt import 먼저)")
    x, sr = read_f32(mix_wav)
    res = align_audio(to_mono_22k(x, sr), song.gt_notes, taps=taps,
                      expected_bars=(song.spec.reference or {}).get("expanded_bars"))
    d = Path(song_dir)
    atomic.write_json(d / "tempo_map.json", res.tempo_map)
    atomic.write_json(d / "align_report.json", res.report)
    tm = tempo_map_from_dict(res.tempo_map)
    gtn = fill_times(song.gt_notes, tm)
    atomic.write_text(d / "gt_notes.json", gtn.model_dump_json(indent=1) + "\n")
    x64 = np.asarray(x, dtype=np.float64)
    x64 = x64 if x64.ndim == 2 else x64[:, None].repeat(2, axis=1)
    clicks = click_track(tm, gtn, len(x64), sr)
    check = 10 ** (-6 / 20) * x64[:, :2] + clicks[:, None]
    write_f32(d / "align_check.wav", check.astype(np.float32), sr)
    gtstore.save_spec_fields(d / "gt.yaml", {"alignment": {
        **(song.spec.alignment or {}), "method": res.report["method"], "approved": False, "approved_utc": None}})
    return res
