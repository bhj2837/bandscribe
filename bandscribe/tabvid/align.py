"""A tab video's sound against our song (docs/TAB_VIDEO.md gate 3-3): is it the same version, a cut of it, or another
song, and which song time each video time plays.

The video's audio track (``audio/<id>.*``, the YouTube download's audio format) and the job's mix are reduced to
chroma (CENS, 5 frames/s) and aligned by DTW (cosine). The path gives video time -> song time; along it:
- ``similarity``: the mean cosine similarity of aligned frames (the same recording or a cover over its backing track
  scores high; another song low).
- ``rate``: the median slope of the path (song seconds per video second) - not 1 when the video plays at another tempo
  (a re-recorded version, a sped-up upload).
- ``cuts``: song spans of 8 s or more the video skips (the path runs along the song axis), and ``extra``: video spans
  the song does not have (the path runs along the video axis: an intro talk, a section the song lacks).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

FPS = 5.0
SR = 22050
HOP = int(SR / FPS)


def load(path: Path) -> np.ndarray:
    """Mono float32 at ``SR`` (ffmpeg)."""
    from bandscribe.ingest.decode import tool_path

    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        subprocess.run([tool_path("ffmpeg"), "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                        "-f", "wav", str(wav)], check=True, timeout=600)
        import soundfile as sf

        y, _sr = sf.read(str(wav), dtype="float32")
    return y


def chroma(y: np.ndarray) -> np.ndarray:
    import librosa

    return librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP)


def _runs(steps: np.ndarray, axis: int, min_s: float) -> list[tuple[float, float]]:
    """Spans (in the other axis' seconds) where the path moves only along ``axis`` for at least ``min_s``."""
    out, start = [], None
    for i, (dv, ds) in enumerate(steps):
        flat = (dv == 0) if axis == 1 else (ds == 0)
        if flat and start is None:
            start = i
        elif not flat and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(steps)))
    return [(a, b) for a, b in out if (b - a) / FPS >= min_s]


def align(video: np.ndarray, song: np.ndarray, span: tuple[float, float] | None = None) -> dict[str, Any]:
    """DTW of two mono signals (module doc): {similarity, rate, cuts, extra, path: [(video s, song s)] every second}.
    ``span``: the video seconds to align (where a track is on screen: a video can play the song twice, once per
    part); path times stay video times."""
    import librosa

    t0 = 0.0
    if span is not None:
        t0 = max(0.0, span[0])
        video = video[int(t0 * SR):int(span[1] * SR)]
    cv, cs = chroma(video), chroma(song)
    _d, wp = librosa.sequence.dtw(X=cv, Y=cs, metric="cosine")
    wp = wp[::-1]
    sim = float(np.mean(np.sum(cv[:, wp[:, 0]] * cs[:, wp[:, 1]], axis=0)
                        / (np.linalg.norm(cv[:, wp[:, 0]], axis=0) * np.linalg.norm(cs[:, wp[:, 1]], axis=0) + 1e-9)))
    steps = np.diff(wp, axis=0)
    cuts = [(round(float(wp[a, 1] / FPS), 1), round(float(wp[b, 1] / FPS), 1))
            for a, b in _runs(steps, 1, 8.0)]
    extra = [(round(float(wp[a, 0] / FPS + t0), 1), round(float(wp[b, 0] / FPS + t0), 1))
             for a, b in _runs(steps, 0, 8.0)]
    # the rate over 10 s windows of the path that move along both axes
    rates = []
    for i in range(0, len(wp) - int(10 * FPS), int(5 * FPS)):
        a, b = wp[i], wp[i + int(10 * FPS)]
        if b[0] - a[0] > 5 * FPS and b[1] - a[1] > 5 * FPS:
            rates.append((b[1] - a[1]) / (b[0] - a[0]))
    path = []
    last = -1
    for v, s in wp:
        sec = int(v / FPS)
        if sec != last:
            path.append((round(v / FPS + t0, 2), round(s / FPS, 2)))
            last = sec
    return {"similarity": round(sim, 3), "rate": round(float(np.median(rates)), 3) if rates else None,
            "cuts": cuts, "extra": extra, "video_s": round(len(video) / SR, 1), "song_s": round(len(song) / SR, 1),
            "span": [round(t0, 1), round(t0 + len(video) / SR, 1)], "path": path}


def song_time(path: list[tuple[float, float]], t: float) -> float:
    """The song second the video plays at video second ``t`` (the path, interpolated)."""
    v = np.array([p[0] for p in path])
    s = np.array([p[1] for p in path])
    return float(np.interp(t, v, s))


TITLE_FLAGS = ("replica", "remix", "live", "acoustic", "short", "tv size", "tv ver", "arrange", "piano", "slow",
               "speed up", "sped up", "nightcore", "instrumental", "version", "ver.")


def title_flags(title: str) -> list[str]:
    """Words in a video title that mark another version than the record, for a person to check: ``-replica-`` (a
    re-recording) sounded the same to the alignment - same tempo, same chords. A full version is the record."""
    t = title.lower().replace("full ver", "")
    return [w for w in TITLE_FLAGS if w in t]


def verdict(res: dict[str, Any]) -> tuple[str, list[str]]:
    """('same' | 'cut' | 'other', reasons). Thresholds from the measured videos (docs/TAB_VIDEO.md)."""
    why = []
    sim, rate = res["similarity"], res["rate"]
    if sim < SAME_MIN_SIM:
        why.append(f"소리 유사도 {sim} < {SAME_MIN_SIM}")
    if rate is None or abs(rate - 1) > RATE_TOL:
        why.append(f"빠르기 비 {rate} (같은 판이면 1 ± {RATE_TOL})")
    if why:
        return "other", why
    if res["cuts"]:
        return "cut", [f"원곡 {a}–{b} s 가 영상에 없다" for a, b in res["cuts"]]
    return "same", []


SAME_MIN_SIM = 0.8
RATE_TOL = 0.015
