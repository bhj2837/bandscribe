"""Render the reference score to audio for listening / A-B (``gtab gt render``; DESIGN 8.3 step 1).

- ``internal`` (default): deterministic numpy synth — harmonic plucked tones per GT note, one timbre and pan
  position per line, timed by ``tempo_map.json`` if the song is aligned, else by the nominal tempo. Byte-stable
  (fixed phases, float32 via ``gtab.audio.write_f32``).
- ``alphatab``: ``node/gtab_score.mjs render`` (AlphaSynth + sonivox soundfont), rewritten with ``write_f32``.
  Its timeline is the file's own nominal tempo with alphaTab's repeat expansion.
Alignment itself uses symbolic score features, not this audio (A9).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SR = 44100


def _tone(freq: float, n: int, sr: int, bright: float) -> np.ndarray:
    t = np.arange(n) / sr
    y = np.zeros(n)
    for k in range(1, 9):
        if k * freq > 0.45 * sr:
            break
        y += (bright ** (k - 1)) / k * np.sin(2 * np.pi * k * freq * t) * np.exp(-t * (3.0 + 1.5 * k))
    att = min(n, int(0.003 * sr))
    y[:att] *= np.linspace(0, 1, att, endpoint=False) if att else 1
    return y


def render_internal(gt_notes: Any, tempo_map: Any, *, sr: int = SR, tail_s: float = 1.0) -> np.ndarray:
    """(n, 2) float32. Lines are panned from left to right in id order; bass lines centred and darker."""
    notes = list(gt_notes.notes)
    if not notes:
        return np.zeros((sr, 2), dtype=np.float32)
    bars = np.array([n.bar for n in notes])
    on = np.asarray(tempo_map.bar_tick_to_time(bars, np.array([n.tick for n in notes], dtype=float)), dtype=float)
    off = np.asarray(tempo_map.bar_tick_to_time(bars, np.array([n.tick + n.dur_ticks for n in notes], dtype=float)),
                     dtype=float)
    total = int((max(off.max(), on.max()) + tail_s) * sr) + 1
    out = np.zeros((total, 2))
    lines = sorted({n.line for n in notes})
    guitar_lines = [ln for ln in lines if not ln.startswith("B")]
    for n, a, b in zip(notes, on, off):
        start = int(round(max(0.0, a) * sr))
        length = int(max(0.08, min(4.0, b - a + 0.15)) * sr)
        length = min(length, total - start)
        if length <= 0:
            continue
        is_bass = n.line.startswith("B")
        freq = 440.0 * 2 ** ((n.pitch - 69) / 12)
        y = _tone(freq, length, sr, 0.35 if is_bass else 0.6) * (0.35 if is_bass else 0.25)
        if is_bass:
            pan = 0.0
        else:
            k = guitar_lines.index(n.line)
            pan = 0.0 if len(guitar_lines) == 1 else -0.6 + 1.2 * k / (len(guitar_lines) - 1)
        th = (pan + 1) * np.pi / 4
        out[start:start + length, 0] += np.cos(th) * y
        out[start:start + length, 1] += np.sin(th) * y
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out *= 0.99 / peak
    return out.astype(np.float32)


def render_song(song_dir: Path, *, engine: str = "internal") -> Path:
    """Write ``render.wav`` for a Tier A song folder."""
    from gtab.audio import read_f32, write_f32
    from gtab.eval import gtstore
    from gtab.eval.refscore import nominal_tempo_map

    song = gtstore.load_song(song_dir)
    out = Path(song_dir) / "render.wav"
    if engine == "internal":
        if song.gt_notes is None:
            raise gtstore.GtError("gt_notes.json 이 없습니다 (gtab gt import 먼저)")
        tm = song.tempo_map or nominal_tempo_map(song.gt_notes)
        write_f32(out, render_internal(song.gt_notes, tm), SR)
        return out
    if engine == "alphatab":
        from gtab.eval.node import alphatab_render

        ref = next(iter(sorted(Path(song_dir).glob("ref.*"))), None)
        if ref is None:
            raise gtstore.GtError("ref.<확장자> 파일이 없습니다 (gtab gt import 먼저)")
        tmp = Path(song_dir) / ".render_alphatab.tmp.wav"
        alphatab_render(ref, tmp, sample_rate=SR)
        x, sr = read_f32(tmp)
        write_f32(out, x, sr)
        tmp.unlink(missing_ok=True)
        return out
    raise ValueError(f"unknown engine {engine!r} (internal | alphatab)")
