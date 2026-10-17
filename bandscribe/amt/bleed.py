"""Bleed flags on guitar notes (DESIGN S7 "감점", M8; E30 in docs/decisions.md).

A flag never deletes a note: the notes stage moves a flagged note out of ``guitar_all`` into ``bleed.json`` and the
viewer offers it back as a candidate (``tab/viewer_media.candidates``).

- ``vocal``: the vocal f0 (pyin on the vocal stem) sits on the note's pitch for ``head_frac`` of its first
  ``head_s`` and ``full_frac`` of its length, **and the voice arrives at that pitch with the note** (on it within
  ``arrive_s`` after the onset, not on it ``pre_s`` before). The arrival keeps out a guitar chord tone that merely
  equals the sung note (E30's dev search: 0.70 precision without it, 0.84 with it).
- ``bass``: a note under E2 whose pitch class the bass f0 holds for ``frac`` of the voiced frames of its first
  ``span_s`` (DESIGN: "E2 미만이면서 같은 pitch class의 베이스 음과 겹치는 음").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

VOCAL_DEFAULTS: dict[str, Any] = {"head_s": 0.3, "head_frac": 0.9, "full_frac": 0.8, "cents": 50.0, "vprob_min": 0.5,
                                  "arrive_s": 0.06, "pre_s": [0.15, 0.06]}
BASS_DEFAULTS: dict[str, Any] = {"below_midi": 40, "span_s": 0.2, "frac": 0.5, "conf_min": 0.5}
VOCAL_F0 = {"sr": 16000, "hop": 160, "frame_length": 1024, "fmin_hz": 80.0, "fmax_hz": 1000.0}


def vocal_f0(y: np.ndarray, sr: int) -> dict[str, Any]:
    """pyin on a mono vocal signal -> {"hz", "vprob" (float32 arrays, 0 where unvoiced), "hop_s"}."""
    import librosa

    p = VOCAL_F0
    x = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=int(sr), target_sr=p["sr"])
    f, _v, vp = librosa.pyin(x, fmin=p["fmin_hz"], fmax=p["fmax_hz"], sr=p["sr"], frame_length=p["frame_length"],
                             hop_length=p["hop"], center=True)
    return {"hz": np.nan_to_num(f).astype(np.float32), "vprob": np.nan_to_num(vp).astype(np.float32),
            "hop_s": p["hop"] / p["sr"]}


def _midi(hz: np.ndarray, ok: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    return np.where(ok & (hz > 0), 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0), np.nan)


def vocal_flags(notes: Sequence[Mapping[str, Any]], hz: np.ndarray, vprob: np.ndarray, hop_s: float,
                params: Mapping[str, Any] | None = None) -> list[int]:
    """Indices of ``notes`` the ``vocal`` rule flags."""
    p = {**VOCAL_DEFAULTS, **(params or {})}
    m = _midi(hz, np.asarray(vprob) > float(p["vprob_min"]))
    tol = float(p["cents"]) / 100.0
    pre_a, pre_b = (float(x) for x in p["pre_s"])
    out = []
    for i, n in enumerate(notes):
        on = float(n["onset_s"])
        off = max(on, float(n.get("offset_s") or on))
        a = int(on / hop_s)
        if a >= len(m):
            continue
        b = max(a + 1, int(min(off, on + float(p["head_s"])) / hop_s))
        pitch = float(n["pitch"])
        head = np.abs(m[a:b] - pitch) <= tol
        if not len(head) or float(np.mean(head)) < float(p["head_frac"]):
            continue
        full = np.abs(m[a:max(a + 1, int(off / hop_s))] - pitch) <= tol
        if float(np.mean(full)) < float(p["full_frac"]):
            continue
        post = m[a:a + int(float(p["arrive_s"]) / hop_s) + 1]
        pre = m[max(0, a - int(pre_a / hop_s)):max(0, a - int(pre_b / hop_s))]
        if np.any(np.abs(post - pitch) <= tol) and not np.any(np.abs(pre - pitch) <= tol):
            out.append(i)
    return out


def bass_flags(notes: Sequence[Mapping[str, Any]], hz: np.ndarray, conf: np.ndarray, hop_s: float,
               params: Mapping[str, Any] | None = None) -> list[int]:
    """Indices of ``notes`` the ``bass`` rule flags."""
    p = {**BASS_DEFAULTS, **(params or {})}
    m = _midi(hz, np.asarray(conf) > float(p["conf_min"]))
    out = []
    for i, n in enumerate(notes):
        if int(n["pitch"]) >= int(p["below_midi"]):
            continue
        on = float(n["onset_s"])
        a = int(on / hop_s)
        seg = m[a:max(a + 1, int((on + float(p["span_s"])) / hop_s))]
        seg = seg[~np.isnan(seg)]
        if len(seg) and float(np.mean(np.mod(np.round(seg), 12) == int(n["pitch"]) % 12)) >= float(p["frac"]):
            out.append(i)
    return out
