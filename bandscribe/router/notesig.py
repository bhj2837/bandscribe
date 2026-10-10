"""Per-note stereo signature (DESIGN 4.3 (d), S7; the hypothesis E22 tests, M6).

For each transcribed note, where in the stereo image its own energy sits, read from the stereo stem at the note's
harmonics during its first ``HOLD_S`` (the attack and early sustain, where the note dominates its bins):

- ``ild_db``: 10 log10(E_L / E_R) over the note's time-frequency points;
- ``coh``: |sum X_L X_R*| / sqrt(E_L E_R) over the same points (1 = one coherent source, panned or centred; low =
  two takes of the same note, a double);
- ``ms_db``: 10 log10(E_mid / E_side), mid = (L+R)/2, side = (L-R)/2 (high = centred).

Bins (n_fft ``N_FFT`` = 46 ms at 44.1 kHz, 10.8 Hz apart): the bin nearest each harmonic h * f0 below ``F_MAX``,
from the 3rd harmonic for notes under ``LOW_HZ`` (DESIGN S7: E2-A2 fundamentals and 2nd harmonics are not resolved
and are shared with chords and bass bleed). A bin within ``SHARE_BINS`` of a harmonic of another note sounding at
the same time is left out (it belongs to two notes); a note left with no bins of its own uses all of them and is
marked ``shared_bins``. Notes shorter than ``MIN_FRAMES`` STFT frames use those they have.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import signal

N_FFT = 2048
HOP = 512
HOLD_S = 0.15
F_MAX = 6000.0
LOW_HZ = 110.0
N_HARM = 10
SHARE_BINS = 1
MIN_FRAMES = 1


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((float(midi) - 69) / 12)


def _as_nx2(stereo: np.ndarray) -> np.ndarray:
    x = np.asarray(stereo, dtype=np.float32)
    if x.ndim != 2 or 2 not in x.shape:
        raise ValueError(f"stereo array must be (n, 2) or (2, n), got {x.shape}")
    return x if x.shape[1] == 2 else x.T


_WINDOW = signal.get_window("hann", N_FFT).astype(np.float32)  # periodic Hann, as scipy's stft uses


def frames_lr(x: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Spectra (bins, frames) of frames a..b-1 of a (n, 2) signal, frame k centred on sample k * HOP (zeros outside
    the signal): only the frames a note needs, so a whole song never sits in memory as an STFT."""
    n = x.shape[0]
    idx = (np.arange(a, b)[:, None] * HOP - N_FFT // 2) + np.arange(N_FFT)[None, :]
    ok = (idx >= 0) & (idx < n)
    seg = np.where(ok[..., None], x[np.clip(idx, 0, n - 1)], 0.0) * _WINDOW[None, :, None]
    spec = np.fft.rfft(seg, axis=1)  # (frames, bins, 2)
    return spec[..., 0].T, spec[..., 1].T


def harmonic_bins(pitch: float, sr: int) -> list[int]:
    f0 = _hz(pitch)
    first = 3 if f0 < LOW_HZ else 1
    df = sr / N_FFT
    out = []
    for h in range(first, N_HARM + 1):
        f = f0 * h
        if f > F_MAX:
            break
        out.append(int(round(f / df)))
    return sorted(set(out))


def signatures(stereo: np.ndarray, sr: int, notes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Signature per note (module doc), in the notes' order: {ild_db, coh, ms_db, n_bins, shared_bins}.
    ``notes`` need ``onset_s``, ``offset_s``, ``pitch``."""
    x = _as_nx2(stereo)
    n_frames = x.shape[0] // HOP + 1
    n_bins = N_FFT // 2 + 1
    starts = np.array([float(n["onset_s"]) for n in notes])
    offs = np.array([float(n["offset_s"]) for n in notes])
    ends = np.minimum(offs, starts + HOLD_S)
    bins = [harmonic_bins(float(n["pitch"]), sr) for n in notes]
    order = np.argsort(starts, kind="stable")
    s_sorted = starts[order]
    max_dur = float(np.max(offs - starts)) if len(notes) else 0.0
    out: list[dict[str, Any]] = [{} for _ in notes]
    for i in range(len(notes)):
        a = max(0, int(math.floor(starts[i] * sr / HOP)))
        b = min(n_frames, max(a + MIN_FRAMES, int(math.ceil(ends[i] * sr / HOP)) + 1))
        # notes sounding at the same time (any overlap with [onset, offset]); onsets sorted, durations bounded
        on, off = starts[i], offs[i]
        lo = int(np.searchsorted(s_sorted, on - max_dur, side="left"))
        hi = int(np.searchsorted(s_sorted, off, side="left"))
        others: set[int] = set()
        for j in order[lo:hi]:
            if j != i and offs[j] > on:
                others.update(bins[j])
        own = [k for k in bins[i] if all(abs(k - o) > SHARE_BINS for o in others)]
        shared = not own
        use = np.array(own or bins[i], dtype=int)
        use = use[(use >= 0) & (use < n_bins)]
        if use.size == 0 or b <= a:
            out[i] = {"ild_db": None, "coh": None, "ms_db": None, "n_bins": 0, "shared_bins": shared}
            continue
        xl, xr = frames_lr(x, a, b)
        L = xl[use].astype(np.complex128)
        R = xr[use].astype(np.complex128)
        el, er = float(np.sum(np.abs(L) ** 2)), float(np.sum(np.abs(R) ** 2))
        em = float(np.sum(np.abs((L + R) / 2) ** 2))
        es = float(np.sum(np.abs((L - R) / 2) ** 2))
        coh = float(np.abs(np.sum(L * np.conj(R))) / math.sqrt(el * er)) if el > 0 and er > 0 else None
        out[i] = {"ild_db": round(10 * math.log10((el + 1e-20) / (er + 1e-20)), 3),
                  "coh": None if coh is None else round(coh, 4),
                  "ms_db": round(10 * math.log10((em + 1e-20) / (es + 1e-20)), 3),
                  "n_bins": int(use.size), "shared_bins": shared}
    return out
