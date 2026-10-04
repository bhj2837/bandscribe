"""Vocal activity from the vocals stem (DESIGN S3.5 "보컬 활동"; M1_M2_SPEC 9.3 item 5, §6.6).

Vocals-stem RMS at 10 Hz relative to the mix RMS in the same frame, median-smoothed over 0.5 s; a frame is
active above ``s35.vocal_active_rel_db`` (default -30 dB). Silence does not read as "vocals": the mix RMS is
floored at 50 dB below the song's overall mix RMS before dividing, so a near-silent intro gives a very low
ratio instead of 0 dB. ``per_bar`` = share of active frames whose centre lies inside the bar.

Streams both files in blocks (M0 RAM rule); pure numpy + soundfile; deterministic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

FPS = 10
SMOOTH_FRAMES = 5  # 0.5 s median
FLOOR_REL_DB = -50.0
CLAMP_DB = (-100.0, 20.0)
BLOCK_FRAMES = 200  # 20 s per read


def frame_ms(path: Path, fps: int = FPS) -> tuple[np.ndarray, int]:
    """Mean-square per 1/fps s frame (all channels), streamed. Returns (ms (n_frames,), samplerate)."""
    import soundfile as sf

    out: list[np.ndarray] = []
    with sf.SoundFile(str(path)) as f:
        sr = int(f.samplerate)
        hop = sr // fps
        carry = np.zeros((0, f.channels), dtype=np.float64)
        for block in f.blocks(blocksize=hop * BLOCK_FRAMES, dtype="float32", always_2d=True):
            x = np.concatenate([carry, block.astype(np.float64)]) if carry.size else block.astype(np.float64)
            n = (x.shape[0] // hop) * hop
            if n:
                sq = (x[:n] ** 2).mean(axis=1)
                out.append(sq.reshape(-1, hop).mean(axis=1))
            carry = x[n:]
        if carry.shape[0]:
            out.append(np.asarray([(carry ** 2).mean()]))
    return (np.concatenate(out) if out else np.zeros(0)), sr


def vocal_activity(vocals_wav: Path, mix_wav: Path, bars: list[dict[str, Any]], threshold_rel_db: float
                   ) -> dict[str, Any]:
    import scipy.ndimage

    voc, _ = frame_ms(Path(vocals_wav))
    mix, _ = frame_ms(Path(mix_wav))
    n = min(voc.size, mix.size)
    voc, mix = voc[:n], mix[:n]
    overall = float(mix.mean()) if n else 0.0
    floor = max(overall * 10 ** (FLOOR_REL_DB / 10), 1e-20)
    rel = 10 * np.log10(np.maximum(voc, 1e-20) / np.maximum(mix, floor))
    rel = np.clip(rel, *CLAMP_DB)
    if n:
        rel = scipy.ndimage.median_filter(rel, size=SMOOTH_FRAMES, mode="nearest")
    active = rel > float(threshold_rel_db)
    centres = (np.arange(n) + 0.5) / FPS
    per_bar = []
    for b in bars:
        sel = (centres >= float(b["start_s"])) & (centres < float(b["end_s"]))
        ratio = float(active[sel].mean()) if sel.any() else 0.0
        per_bar.append({"bar": int(b["bar"]), "active_ratio": round(ratio, 4)})
    log.info("vocal activity: %.0f %% of %d frames active", 100.0 * float(active.mean()) if n else 0.0, n)
    return {"format": "bandscribe.vocal/1", "fps": FPS, "rel_db": [round(float(v), 1) for v in rel],
            "active": [int(a) for a in active], "per_bar": per_bar, "threshold_rel_db": float(threshold_rel_db)}
