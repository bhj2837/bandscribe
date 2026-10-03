"""Stereo-image features of a guitar stem (router evidence, DESIGN §4.3 (a); used by the M2 stereo check).

``window_features`` is an **exact port** of ``docs/research/experiments/router_exp.py::features`` (same
STFT, smoothing constant, thresholds and weighting), so the synthetic evidence in DESIGN §4.5 carries over;
``tests/test_eval_remix_parity.py`` checks it against the original script (±0.02).

Arrays are channel-first ``(2, n)`` like the research code; ``(n, 2)`` is accepted and transposed.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

NFFT = 4096
HOP = 1024


def as_channel_first(stereo: np.ndarray) -> np.ndarray:
    x = np.asarray(stereo, dtype=float)
    if x.ndim != 2 or 2 not in x.shape:
        raise ValueError(f"stereo array must be (2, n) or (n, 2), got {x.shape}")
    if x.shape[0] != 2:
        x = x.T
    return x


def _stft(x: np.ndarray, sr: int, n_fft: int = NFFT, hop: int = HOP) -> np.ndarray:
    return signal.stft(x, sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")[2]


def _smooth(a: np.ndarray, lam: float) -> np.ndarray:
    """One-pole recursive smoothing along time (last axis)."""
    return signal.lfilter([1 - lam], [1, -lam], a, axis=-1)


def coherence_ild_maps(stereo: np.ndarray, sr: int, n_fft: int = NFFT, hop: int = HOP,
                       tau_s: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (freq, frame) maps: time-smoothed L/R coherence, ILD in dB and power weight |L|²+|R|²."""
    st = as_channel_first(stereo)
    xl, xr = _stft(st[0], sr, n_fft, hop), _stft(st[1], sr, n_fft, hop)
    lam = np.exp(-hop / sr / tau_s)
    cross = _smooth(xl * np.conj(xr), lam)
    pl = _smooth(np.abs(xl) ** 2, lam)
    pr = _smooth(np.abs(xr) ** 2, lam)
    coh = np.abs(cross) / np.sqrt(pl * pr + 1e-20)
    ild = 20 * np.log10((np.abs(xl) + 1e-9) / (np.abs(xr) + 1e-9))
    w = np.abs(xl) ** 2 + np.abs(xr) ** 2
    return coh, ild, w


def window_features(stereo: np.ndarray, sr: int) -> dict[str, float]:
    """Router features of one window: keys ``SM, r0, xcoff, lag, coh, cen, hard, fcoh, ildcoh``.

    SM = side/mid energy (dB); r0 = zero-lag correlation; xcoff/lag = strongest normalised cross-correlation
    within ±40 ms excluding |lag| < 1 ms (ADT signature); coh = power-weighted coherence (150 ms smoothing);
    cen/hard = energy fraction with |ILD| < 3 dB / > 15 dB; fcoh = energy fraction of bins with coherence > 0.9;
    ildcoh = energy-weighted mean |ILD| of those coherent bins (NaN if none).
    """
    st = as_channel_first(stereo)
    left, right = st[0], st[1]
    mid, side = (left + right) / 2, (left - right) / 2
    em, es = np.mean(mid ** 2), np.mean(side ** 2)
    side_mid_db = 10 * np.log10(es / (em + 1e-12) + 1e-12)
    corr0 = np.sum(left * right) / np.sqrt(np.sum(left ** 2) * np.sum(right ** 2) + 1e-12)
    maxlag = int(0.04 * sr)
    c = signal.correlate(left, right, mode="full", method="fft")
    lags = np.arange(-len(left) + 1, len(left))
    sel = (np.abs(lags) <= maxlag) & (np.abs(lags) > int(0.001 * sr))
    norm = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)) + 1e-12
    k = np.argmax(np.abs(c[sel]))
    xc_off = c[sel][k] / norm
    xc_lag_ms = lags[sel][k] / sr * 1000
    coh, ild, w = coherence_ild_maps(st, sr)
    coh_w = np.sum(coh * w) / np.sum(w)
    f_centre = np.sum(w[np.abs(ild) < 3]) / np.sum(w)
    f_hard = np.sum(w[np.abs(ild) > 15]) / np.sum(w)
    cm = coh > 0.9
    ild_coh = np.sum((np.abs(ild) * w)[cm]) / (np.sum(w[cm]) + 1e-20) if cm.any() else np.nan
    f_coh = np.sum(w[cm]) / np.sum(w)
    return {"SM": float(side_mid_db), "r0": float(corr0), "xcoff": float(xc_off), "lag": float(xc_lag_ms),
            "coh": float(coh_w), "cen": float(f_centre), "hard": float(f_hard), "fcoh": float(f_coh),
            "ildcoh": float(ild_coh)}


def weighted_mean_abs_delta(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Energy-weighted mean |a − b| (used for Δcoherence / ΔILD between a true and a separated stem)."""
    w = np.asarray(w, dtype=float)
    tot = float(np.sum(w))
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b)) * w) / tot) if tot > 0 else float("nan")
