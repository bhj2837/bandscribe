"""Repeat residual, pass 1 (DESIGN S3.5 step 5; M1_M2_SPEC 9.3 item 6, §6.6).

For every repeat group (``repeat_map.json``) with at least ``s35.resid_min_occurrences`` occurrences, the
guitar-stem mono view is split into its occurrences, their STFT magnitudes (n_fft 4096, hop 1024, Hann; the
same transform as ``docs/research/experiments/repet_exp.py``) are warped onto the reference occurrence's
beat grid (per-beat linear frame resampling, the occurrence's ``offset_s`` applied) and stacked, and the
background mask is **exactly repet3's** (the evidence behind DESIGN's 0 -> 17 dB lead SIR gain)::

    W  = min(median over occurrences of V, V)
    Mb = W^2 / (W^2 + max(V - W, 0)^2 + 1e-12)
    residual (foreground) = X * (1 - Mb)

The mask is warped back to each occurrence's own frames, applied to its complex STFT and inverted; the
residual is written back into the original timeline -> ``repeat_resid_pass1.wav`` (mono float32, zeros
outside processed occurrences) plus per-bar residual energy relative to the guitar stem. No solo gate here
(pass 1): the median is assumed robust to a lead that appears in a minority of occurrences (DESIGN: 미검증).
S5 recomputes this with S4's gate.

``background_mask`` is a pure function so it can be parity-tested against repet3 itself
(``tests/test_s35_repet3_parity.py``). Occurrences shorter than the group's common length are processed over
the common bars only; bars left out get ``null``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

NFFT = 4096
HOP = 1024
METHOD = "beat-aligned median background, repet3 soft mask (W=min(median,V))"


def background_mask(V_stack: np.ndarray) -> np.ndarray:
    """Background soft mask per occurrence; ``V_stack`` = magnitudes (occ, freq, frames) on a common grid."""
    med = np.median(V_stack, axis=0)
    W = np.minimum(med[None, ...], V_stack)
    return W ** 2 / (W ** 2 + np.maximum(V_stack - W, 0) ** 2 + 1e-12)


def _stft(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy import signal

    return signal.stft(x, sr, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")[2]


def _istft(X: np.ndarray, n: int, sr: int) -> np.ndarray:
    from scipy import signal

    y = signal.istft(X, sr, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")[1]
    out = np.zeros(n, dtype=np.float32)
    m = min(n, len(y))
    out[:m] = y[:m]
    return out


def _pwl(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Piecewise-linear map through (xp, fp), extrapolated linearly with the first/last segment."""
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    if lo.any():
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    if hi.any():
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def _interp_frames(A: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Linear interpolation of A (freq, frames) at fractional frame positions (clamped to the edges)."""
    F = A.shape[1]
    if F == 1:
        return np.repeat(A, pos.size, axis=1)
    p = np.clip(pos, 0.0, F - 1.0)
    i0 = np.minimum(np.floor(p).astype(np.int64), F - 2)
    w = (p - i0).astype(A.dtype)
    return A[:, i0] * (1 - w) + A[:, i0 + 1] * w


class _Occ:
    def __init__(self, anchors: np.ndarray, offset: float, x: np.ndarray, sr: int) -> None:
        self.anchors = anchors + offset  # where the occurrence's beats really are in the audio
        n = len(x)
        s = int(round(self.anchors[0] * sr))
        e = int(round(self.anchors[-1] * sr))
        self.s, self.e = max(0, s), min(n, e)
        self.seg0 = max(0, self.s - NFFT)
        seg1 = min(n, self.e + NFFT)
        self.X = _stft(x[self.seg0:seg1], sr)
        self.n_seg = seg1 - self.seg0
        self.sr = sr

    def frame_times(self) -> np.ndarray:
        return (self.seg0 + np.arange(self.X.shape[1]) * HOP) / self.sr

    def frame_pos(self, t: np.ndarray) -> np.ndarray:
        return (t * self.sr - self.seg0) / HOP


def _anchors(grid: dict[str, Any], start_bar: int, n_bars: int) -> tuple[list[float], list[float]]:
    """(beat anchors, bar anchors) for bars [start_bar, start_bar + n_bars), each closed by the end time."""
    bars = {int(b["bar"]): b for b in grid["bars"]}
    beats: list[float] = []
    for b in grid["beats"]:
        if start_bar <= int(b["bar"]) < start_bar + n_bars:
            beats.append(float(b["t_s"]))
    last = bars[start_bar + n_bars - 1]
    bar_anchors = [float(bars[start_bar + k]["start_s"]) for k in range(n_bars)] + [float(last["end_s"])]
    return sorted(beats) + [float(last["end_s"])], bar_anchors


def residual(x: np.ndarray, sr: int, grid: dict[str, Any], sections: dict[str, Any], repeat_map: dict[str, Any],
             min_occurrences: int) -> tuple[np.ndarray, dict[str, Any]]:
    """(residual signal float32 (n,), resid1.json document) for a mono guitar-stem signal ``x``."""
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros(len(x), dtype=np.float32)
    bars = {int(b["bar"]): b for b in grid.get("bars", [])}
    per_bar: dict[int, dict[str, Any]] = {b: {"bar": b, "resid_rel_db": None, "group": None} for b in sorted(bars)}
    used: list[str] = []
    skipped: list[dict[str, str]] = []

    for g in repeat_map.get("groups", []):
        occs = list(g.get("occurrences", []))
        label = g.get("label", "?")
        if len(occs) < int(min_occurrences):
            skipped.append({"label": label, "reason": f"반복 {len(occs)}회 < {int(min_occurrences)}회"})
            continue
        n_common = min(int(o["n_bars"]) for o in occs)
        if n_common < 1 or any(int(o["start_bar"]) + k not in bars for o in occs for k in range(n_common)):
            skipped.append({"label": label, "reason": "마디 정보가 격자와 맞지 않음"})
            continue
        ref_i = next((i for i, o in enumerate(occs) if o.get("section") == g.get("reference")), 0)
        anchors = [_anchors(grid, int(o["start_bar"]), n_common) for o in occs]
        use_beats = len({len(a[0]) for a in anchors}) == 1
        pts = [np.asarray(a[0] if use_beats else a[1], dtype=np.float64) for a in anchors]
        offs = [float(o.get("offset_s", 0.0)) - float(occs[ref_i].get("offset_s", 0.0)) for o in occs]
        oc = [_Occ(p, off, x, sr) for p, off in zip(pts, offs)]
        ref = oc[ref_i]
        ref_t = ref.frame_times()
        V_stack = []
        for o in oc:
            if o is ref:
                V_stack.append(np.abs(o.X))
                continue
            t_occ = _pwl(ref_t, ref.anchors, o.anchors)
            V_stack.append(_interp_frames(np.abs(o.X), o.frame_pos(t_occ)))
        Mb = background_mask(np.stack(V_stack, axis=0))
        del V_stack
        for i, o in enumerate(oc):
            if o is ref:
                mask = Mb[i]
            else:
                t_ref = _pwl(o.frame_times(), o.anchors, ref.anchors)
                mask = _interp_frames(Mb[i], ref.frame_pos(t_ref))
            fg = _istft(o.X * (1 - mask), o.n_seg, sr)
            out[o.s:o.e] = fg[o.s - o.seg0:o.e - o.seg0]
            for k in range(n_common):
                bar_no = int(occs[i]["start_bar"]) + k
                b = bars[bar_no]
                a = max(0, int(round((float(b["start_s"]) + offs[i]) * sr)))
                z = min(len(x), int(round((float(b["end_s"]) + offs[i]) * sr)))
                e_g = float(np.sum(x[a:z].astype(np.float64) ** 2))
                e_r = float(np.sum(out[a:z].astype(np.float64) ** 2))
                rel = None if e_g <= 1e-12 else round(float(10 * np.log10(max(e_r, 1e-30) / e_g)), 2)
                per_bar[bar_no] = {"bar": bar_no, "resid_rel_db": rel, "group": label}
        used.append(label)
        del Mb, oc

    doc = {"format": "bandscribe.resid1/1", "per_bar": [per_bar[b] for b in sorted(per_bar)], "groups_used": used,
           "skipped": skipped, "method": METHOD}
    log.info("repeat residual pass 1: groups used %s, skipped %s", used, [s["label"] for s in skipped])
    return out, doc


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1, dtype=np.float32) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(x), int(sr)
