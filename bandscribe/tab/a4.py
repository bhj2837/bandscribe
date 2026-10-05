"""A4 reference (DESIGN 6.1 T1, M3): how far a recording's tuning sits from A440, in cents within [-50, 50).

Spectral peaks of the pitched stems (``librosa.piptrack`` with parabolic interpolation on the guitar, bass and
piano+other views; drums carry no pitch and vocals scoop and vibrate) are folded onto the semitone grid:
``d = (1200 log2(f / 440) + 50) mod 100 - 50``. The estimate is the magnitude-weighted circular mean of d
around the mode of its 1-cent histogram (peaks within ±``trim_cents`` of the mode only, so vibrato, bends and
out-of-tune notes cannot drag it), and the resultant length R of all peaks (0-1) is its confidence.

Only [-50, 50) is knowable: E standard tuned 30 cents flat and Eb standard 70 cents sharp look the same here;
the whole-semitone part is the tuning search's job (DESIGN 6.1 T2). Whether to varispeed the analysis copy when
the offset is large is E16 (pre-registered, not run): M3 reports the offset and warns from ``warn_cents``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SR = 22050
DEFAULT_PARAMS: dict[str, Any] = {
    "n_fft": 8192,          # 2.7 Hz bins at 22.05 kHz: low strings resolved, parabolic interpolation below that
    "hop": 2048,
    "fmin_hz": 70.0,
    "fmax_hz": 2000.0,
    "peak_rel_db": -30.0,   # a frame's peaks within 30 dB of its strongest
    "trim_cents": 20.0,
    "max_seconds": 240.0,   # per view, evenly spread 10 s windows beyond this
    "warn_cents": 25.0,
}


def _fold(cents: np.ndarray) -> np.ndarray:
    return np.mod(cents + 50.0, 100.0) - 50.0


def peaks(y: np.ndarray, sr: int, params: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies Hz, magnitudes) of the spectral peaks used for the estimate."""
    import librosa

    p = {**DEFAULT_PARAMS, **(params or {})}
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if y.size < int(p["n_fft"]):
        return np.zeros(0), np.zeros(0)
    pitches, mags = librosa.piptrack(y=y, sr=SR, n_fft=int(p["n_fft"]), hop_length=int(p["hop"]),
                                     fmin=float(p["fmin_hz"]), fmax=float(p["fmax_hz"]), threshold=0.05)
    top = mags.max(axis=0, keepdims=True)
    keep = (pitches > 0) & (mags > 0) & (mags >= top * 10 ** (float(p["peak_rel_db"]) / 20.0))
    return pitches[keep].astype(float), mags[keep].astype(float)


def estimate_from_peaks(freqs: np.ndarray, mags: np.ndarray, trim_cents: float = 20.0) -> dict[str, Any]:
    """{"cents", "confidence", "n_peaks"} from spectral peaks (cents None without peaks)."""
    f = np.asarray(freqs, dtype=float)
    w = np.asarray(mags, dtype=float)
    ok = (f > 0) & (w > 0) & np.isfinite(f) & np.isfinite(w)
    f, w = f[ok], w[ok]
    if f.size == 0:
        return {"cents": None, "confidence": 0.0, "n_peaks": 0}
    d = _fold(1200.0 * np.log2(f / 440.0))
    hist, _edges = np.histogram(d, bins=100, range=(-50.0, 50.0), weights=w)
    smooth = np.convolve(np.concatenate([hist[-3:], hist, hist[:3]]), np.ones(7) / 7.0, mode="valid")
    mode = -50.0 + float(np.argmax(smooth)) + 0.5
    near = np.abs(_fold(d - mode)) <= trim_cents
    ang = 2 * np.pi * d / 100.0
    c, s = float(np.sum(w[near] * np.cos(ang[near]))), float(np.sum(w[near] * np.sin(ang[near])))
    cents = float(_fold(np.array([np.arctan2(s, c) * 100.0 / (2 * np.pi)]))[0])
    r_all = float(np.hypot(np.sum(w * np.cos(ang)), np.sum(w * np.sin(ang))) / np.sum(w))
    return {"cents": round(cents, 2), "confidence": round(r_all, 4), "n_peaks": int(f.size)}


def _windows(n: int, sr: int, max_s: float, win_s: float = 10.0) -> list[tuple[int, int]]:
    if n <= max_s * sr:
        return [(0, n)]
    k = max(1, int(max_s // win_s))
    starts = np.linspace(0, n - int(win_s * sr), k).astype(int)
    return [(int(a), int(a + win_s * sr)) for a in starts]


def estimate(views: Mapping[str, tuple[np.ndarray, int]] | Iterable[tuple[str, np.ndarray, int]],
             params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``a4.json`` document from {view id: (mono samples, sample rate)}."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    items = list(views.items()) if isinstance(views, Mapping) else [(v, (y, sr)) for v, y, sr in views]
    all_f, all_w, per_view = [], [], {}
    for vid, (y, sr) in items:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        fs, ws = [], []
        for a, b in _windows(y.size, sr, float(p["max_seconds"])):
            f, w = peaks(y[a:b], sr, p)
            fs.append(f)
            ws.append(w)
        f = np.concatenate(fs) if fs else np.zeros(0)
        w = np.concatenate(ws) if ws else np.zeros(0)
        if w.size:
            w = w / max(float(np.sum(w)), 1e-12)  # every view weighs the same in the pooled estimate
        per_view[vid] = estimate_from_peaks(f, w, float(p["trim_cents"]))
        all_f.append(f)
        all_w.append(w)
    pooled = estimate_from_peaks(np.concatenate(all_f) if all_f else np.zeros(0),
                                 np.concatenate(all_w) if all_w else np.zeros(0), float(p["trim_cents"]))
    cents = pooled["cents"]
    warn = None
    if cents is not None and abs(cents) >= float(p["warn_cents"]):
        warn = (f"녹음의 기준음이 A4 = {440.0 * 2 ** (cents / 1200.0):.1f} Hz ({cents:+.0f} cents)로 A440 에서 어긋나 "
                f"있습니다. 채보 음높이가 반음씩 틀린 곳이 있을 수 있습니다 (자동 보정은 E16 결정 뒤).")
    return {"format": "bandscribe.a4/1", "cents": cents,
            "a4_hz": None if cents is None else round(440.0 * 2 ** (cents / 1200.0), 3),
            "confidence": pooled["confidence"], "n_peaks": pooled["n_peaks"], "per_view": per_view,
            "warning": warn, "params": dict(p)}


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.mean(axis=1), int(sr)
