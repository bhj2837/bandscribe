"""Bends by partial tracking - a research record (M9 follow-up, 2026-10-18), not part of the package.

Explored on the dev songs only (seisyun complex, ギターと孤独と蒼い惑星, あのバンド: 116 marked bends) and stopped
there: recall 0.28-0.44 (the pyin rule of E32/E32b found 1 of 116), but 4-5 false glides per true bend - precision
0.09-0.12, F1 0.15-0.16 against DESIGN's 0.6. True and false detections alike sit at onsets with 3+ notes (the mix
transcription puts the rhythm guitar's chord there), so dropping chord notes removes the bends too. No test run.

Original description:

E32/E32b showed that a channel f0 (pyin) is unvoiced where the bends are: the lead shares each stem channel with a
distorted rhythm. Here each transcribed guitar note is followed on its own: in a high-resolution CQT of the stem
channel(s) (``BPO`` bins per octave), the strongest bin of a narrow band around the note - the fundamental
[p - 0.6, p + 3.2] semitones plus the same band an octave up (2nd harmonic), summed - is tracked frame by frame with
a jump cost (a ridge, not a flicker between partials). A **bend** is a ridge that

- starts at the note's pitch (or ends at it: the transcriber may write the target), within ``pitch_tol`` semitones,
- rises at least ``min_rise`` semitones within ``window_s`` of the onset,
- glides: at least ``min_glide`` frames lie strictly between start and top (a hammer-on or a new note jumps),
- and is salient: the band's peak over the frame's median CQT magnitude reaches ``min_salience``.

Depth classes as DESIGN (½ < 1.5 semitones, full < 2.5, 1½ above).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

BPO = 60
FMIN_MIDI = 40
N_OCT = 5
SR = 22050
HOP = 256
DEFAULTS: dict[str, Any] = {"min_rise": 0.8, "max_rise": 3.2, "pitch_tol": 0.5, "window_s": 0.35, "min_glide": 2,
                            "min_salience": 4.0, "jump_cost": 0.6, "lead_s": 0.02, "merge_s": 0.25}


def cqt(x: np.ndarray, sr: int) -> np.ndarray:
    """|CQT| of a mono signal: ``BPO`` bins per octave from MIDI ``FMIN_MIDI`` over ``N_OCT`` octaves, hop ``HOP``
    at ``SR``."""
    import librosa

    y = librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=int(sr), target_sr=SR)
    return np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(FMIN_MIDI), n_bins=BPO * N_OCT,
                              bins_per_octave=BPO)).astype(np.float32)


def _bin(midi: float) -> int:
    return int(round((midi - FMIN_MIDI) * BPO / 12))


def ridge(M: np.ndarray, p: float, a: int, b: int, jump_cost: float) -> tuple[np.ndarray, np.ndarray]:
    """(ridge in MIDI per frame, salience per frame) of the note band between frames a and b (Viterbi, jump cost
    per semitone)."""
    lo, hi = _bin(p - 0.6), _bin(p + 3.2) + 1
    if lo < 0 or hi > M.shape[0] or b <= a:
        return np.zeros(0), np.zeros(0)
    band = M[lo:hi, a:b].astype(np.float64)
    h2lo, h2hi = lo + BPO, hi + BPO
    if h2hi <= M.shape[0]:
        band = band + M[h2lo:h2hi, a:b]
    med = np.median(M[:, a:b], axis=0) + 1e-9
    sal = band.max(axis=0) / med
    lg = np.log(band + 1e-9)
    nb, nf = lg.shape
    steps = np.abs(np.arange(nb)[:, None] - np.arange(nb)[None, :]) * (12 / BPO) * jump_cost
    acc = lg[:, 0].copy()
    back = np.zeros((nf, nb), dtype=np.int32)
    for t in range(1, nf):
        cand = acc[None, :] - steps  # [to, from]
        back[t] = np.argmax(cand, axis=1)
        acc = cand[np.arange(nb), back[t]] + lg[:, t]
    path = [int(np.argmax(acc))]
    for t in range(nf - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    path.reverse()
    return p - 0.6 + np.array(path) * 12 / BPO, sal


def bends(notes: Sequence[Mapping[str, Any]], spectra: Sequence[np.ndarray], params: Mapping[str, Any] | None = None
          ) -> list[dict[str, Any]]:
    """[{"i", "rise", "class", "onset_s", "channel"}]: the bent notes (one per ``merge_s``, the strongest rise)."""
    pr = {**DEFAULTS, **(params or {})}
    fps = SR / HOP
    out = []
    for i, n in enumerate(notes):
        on = float(n["onset_s"])
        p = float(n["pitch"])
        best = None
        for ci, M in enumerate(spectra):
            a = int((on - float(pr["lead_s"])) * fps)
            b = int((on + float(pr["window_s"])) * fps)
            r, sal = ridge(M, p, max(0, a), min(M.shape[1], b), float(pr["jump_cost"]))
            if len(r) < 4:
                continue
            start = float(np.median(r[:3]))
            k_top = int(np.argmax(r))
            top = float(r[k_top])
            rise = top - start
            if not float(pr["min_rise"]) <= rise <= float(pr["max_rise"]):
                continue
            if abs(start - p) > float(pr["pitch_tol"]) and abs(top - p) > float(pr["pitch_tol"]):
                continue
            between = int(np.sum((r[:k_top + 1] > start + 0.25) & (r[:k_top + 1] < top - 0.25)))
            if between < int(pr["min_glide"]):
                continue
            if float(np.median(sal[:k_top + 1])) < float(pr["min_salience"]):
                continue
            if best is None or rise > best["rise"]:
                cls = "half" if rise < 1.5 else ("full" if rise < 2.5 else "one_half")
                best = {"i": i, "rise": round(rise, 2), "class": cls, "onset_s": on, "channel": ci}
        if best is not None:
            out.append(best)
    out.sort(key=lambda d: d["onset_s"])
    merged: list[dict[str, Any]] = []
    for d in out:
        if merged and d["onset_s"] - merged[-1]["onset_s"] < float(pr["merge_s"]):
            if d["rise"] > merged[-1]["rise"]:
                merged[-1] = d
            continue
        merged.append(d)
    return merged
