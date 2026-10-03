"""MuScriptor latency constant (M1_M2_SPEC 9.2 item 8, DESIGN S6 "타이밍", M2 b10).

MuScriptor's onsets can all carry the same small lag (up to ~25 ms, model bias - upstream docstring). Its own
``transcribe_and_postprocess`` estimates that lag against a *constant-tempo* grid, so we measure it ourselves:

- ``estimate_latency(notes, ref_onsets)``: every transcribed onset is paired with the nearest reference onset
  (same pitch when both sides have pitches) within ``max_ms``; the constant is the **median** residual
  (transcribed - reference, positive = late), with the MAD as spread.
- The criterion must not grade itself (M1_M2_SPEC A10): the constant is estimated on a dev split (GuitarSet
  players 00-02, + EGDB dev if present) and the median **absolute** residual after correction is reported on
  held-out players 03-05 (``gtab eval exp latency``, rows with ``section`` = dev/test). A signed median would be
  ~0 on the estimation set by construction.
- ``spectral_flux_onsets`` gives an audio-only onset list for a cross-check (no GT needed).

The stage then writes ``onset_s = raw - amt.latency_s`` (clamped >= 0). Basic Pitch is not corrected (0).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEV_PLAYERS = ("00", "01", "02")
TEST_PLAYERS = ("03", "04", "05")

# 512-sample frames and mild log compression: on synthetic clicks and decaying tones the flux peak then sits
# ~2 ms before the true onset (2048 samples / log1p(100|X|) put it ~19 ms early: the log difference peaks when the
# onset first enters a long window). Measured 2026-09-30, data/scratch/amt/flux_bias.py.
FLUX_N_FFT = 512
FLUX_COMPRESSION = 10.0
FLUX_HOP = 128
FLUX_PEAK_S = 0.03   # a flux peak must be the maximum within +/- this window
FLUX_AVG_S = 0.10    # adaptive threshold: moving mean over +/- this window
FLUX_DELTA = 0.07    # ... plus this (flux is normalised to max 1)
FLUX_MIN_GAP_S = 0.03


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_onsets(x: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """(times, pitches|None) from an array, a (times, pitches) tuple, a NoteSet(-like) or a list of notes.

    The tuple may also be mir_eval-style ``(intervals (n, 2), pitches)`` (``notes_to_arrays`` / ``as_arrays``):
    the onset column is used.
    """
    if isinstance(x, tuple) and len(x) == 2:
        t = np.asarray(x[0], dtype=float)
        t = t[:, 0] if t.ndim == 2 else t.reshape(-1)
        p = None if x[1] is None else np.asarray(x[1], dtype=float).reshape(-1)
        if p is not None and len(p) != len(t):
            raise ValueError(f"{len(t)} onsets but {len(p)} pitches")
        return t, p
    if isinstance(x, np.ndarray):
        return x.astype(float).reshape(-1), None
    notes = _get(x, "notes", x) if not isinstance(x, (list, tuple)) else x
    notes = list(notes or [])
    if notes and not isinstance(notes[0], (int, float, np.floating, np.integer)):
        t = np.array([float(_get(n, "onset_s")) for n in notes], dtype=float)
        p = np.array([float(_get(n, "pitch")) for n in notes], dtype=float)
        return t, p
    return np.asarray(notes, dtype=float).reshape(-1), None


def match_residuals(est: Any, ref: Any, *, max_ms: float = 100.0) -> np.ndarray:
    """Residuals (s) of every estimated onset against its nearest reference onset within ``max_ms``."""
    et, ep = _as_onsets(est)
    rt, rp = _as_onsets(ref)
    if len(et) == 0 or len(rt) == 0:
        return np.zeros(0)
    tol = max_ms / 1000.0
    use_pitch = ep is not None and rp is not None
    out: list[float] = []
    if use_pitch:
        for pitch in np.unique(ep):
            r = np.sort(rt[rp == pitch])
            if len(r) == 0:
                continue
            e = et[ep == pitch]
            out.extend(_nearest(e, r, tol))
    else:
        out.extend(_nearest(et, np.sort(rt), tol))
    return np.asarray(sorted(out), dtype=float)


def _nearest(e: np.ndarray, r_sorted: np.ndarray, tol: float) -> list[float]:
    idx = np.searchsorted(r_sorted, e)
    res = []
    for x, i in zip(e, idx):
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(r_sorted):
                d = x - r_sorted[j]
                if best is None or abs(d) < abs(best):
                    best = d
        if best is not None and abs(best) <= tol:
            res.append(float(best))
    return res


def estimate_latency(notes: Any, ref_onsets: Any, max_ms: float = 100.0) -> dict[str, Any]:
    """{"median_ms", "mad_ms", "n"}: the constant lag of ``notes`` against ``ref_onsets`` (positive = late)."""
    r = match_residuals(notes, ref_onsets, max_ms=max_ms)
    if len(r) == 0:
        return {"median_ms": None, "mad_ms": None, "n": 0}
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    return {"median_ms": round(med * 1000.0, 3), "mad_ms": round(mad * 1000.0, 3), "n": int(len(r))}


def median_abs_residual_ms(notes: Any, ref_onsets: Any, latency_s: float, *, max_ms: float = 100.0) -> dict[str, Any]:
    """After subtracting ``latency_s``: median |residual| in ms (the b10 quantity), with n."""
    et, ep = _as_onsets(notes)
    corrected = (np.maximum(0.0, et - float(latency_s)), ep)
    r = match_residuals(corrected, ref_onsets, max_ms=max_ms)
    if len(r) == 0:
        return {"median_abs_ms": None, "n": 0}
    return {"median_abs_ms": round(float(np.median(np.abs(r))) * 1000.0, 3), "n": int(len(r))}


def split_of_player(player: str) -> str:
    """Pre-registered GuitarSet split of the latency experiment (never overlapping)."""
    if player in DEV_PLAYERS:
        return "dev"
    if player in TEST_PLAYERS:
        return "test"
    raise ValueError(f"unknown GuitarSet player {player!r}")


# ------------------------------------------------------------------------------------ spectral flux


def spectral_flux_onsets(wav: Path | np.ndarray, sr: int | None = None) -> np.ndarray:
    """Onset times (s) from half-wave-rectified log-magnitude spectral flux with adaptive peak picking.

    Frames are centred (a frame's time is its centre), so the onset estimate is where the flux rises.
    A cross-check only: its own bias depends on window length and signal, unlike GT onsets.
    """
    if isinstance(wav, np.ndarray):
        x = wav.astype(np.float64)
        if sr is None:
            raise ValueError("sr is required with an array")
    else:
        import soundfile as sf

        x, sr = sf.read(str(wav), dtype="float64", always_2d=True)
    if x.ndim == 2:
        x = x.mean(axis=1)
    n_fft, hop = FLUX_N_FFT, FLUX_HOP
    if len(x) < n_fft:
        return np.zeros(0)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad))
    n_frames = 1 + (len(xp) - n_fft) // hop
    win = np.hanning(n_fft)
    prev = None
    flux = np.zeros(n_frames)
    for i in range(n_frames):  # frame loop keeps RAM flat for long files
        seg = xp[i * hop:i * hop + n_fft] * win
        mag = np.log1p(FLUX_COMPRESSION * np.abs(np.fft.rfft(seg)))
        if prev is not None:
            d = mag - prev
            flux[i] = float(np.sum(d[d > 0]))
        prev = mag
    if flux.max() <= 0:
        return np.zeros(0)
    flux /= flux.max()
    fps = sr / hop
    w_peak = max(1, int(round(FLUX_PEAK_S * fps)))
    w_avg = max(1, int(round(FLUX_AVG_S * fps)))
    kernel = np.ones(2 * w_avg + 1) / (2 * w_avg + 1)
    avg = np.convolve(flux, kernel, mode="same")
    onsets: list[float] = []
    last = -1e9
    for i in range(1, n_frames - 1):
        lo, hi = max(0, i - w_peak), min(n_frames, i + w_peak + 1)
        if flux[i] < flux[lo:hi].max() or flux[i] < avg[i] + FLUX_DELTA:
            continue
        t = i * hop / sr
        if t - last >= FLUX_MIN_GAP_S:
            onsets.append(t)
            last = t
    return np.asarray(onsets)
