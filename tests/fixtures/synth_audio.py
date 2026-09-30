"""Synthetic band-ish audio for ingest tests, encoded with the same ffmpeg the app uses.

- ``normal``: 12 s, chord tones split left/right + independent white noise (real stereo, full band),
  sample peak -6 dBFS.
- ``loud``: the same mix saturated like a loud commercial master (RMS about -8 dBFS, sample peak
  -0.1 dBFS). Its Opus/AAC encodes overshoot 1.0 on decode (DESIGN S0, codec_exp3.py).
- ``mono`` (1 channel), ``quasi`` (stereo with side 40 dB under mid), ``sr32k`` (non-standard rate).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
SECONDS = 12.0
CHORDS = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]  # Am F C G

CODECS: dict[str, list[str]] = {
    "flac": ["-c:a", "flac"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "128k"],
    "m4a": ["-c:a", "aac", "-b:a", "128k"],
    "opus": ["-c:a", "libopus", "-b:a", "128k"],
    "webm": ["-c:a", "libopus", "-b:a", "128k", "-f", "webm"],
}


def _tone(freq: float, t: np.ndarray) -> np.ndarray:
    return sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, 8))


def chords(seconds: float = SECONDS, sr: int = SR, split: bool = True) -> np.ndarray:
    """(n, 2) chord progression; with ``split`` alternate chord tones go left/right."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    out = np.zeros((n, 2))
    seg = n // 6
    for i in range(6):
        sl = slice(i * seg, (i + 1) * seg if i < 5 else n)
        tt = t[sl] - t[sl][0]
        # Sustained part keeps every 400 ms block well above the -10 LU relative gate: with sharp decays a
        # 12 s clip has blocks right at the gate, and one block flipping moves the result ~0.08 LU.
        env = 0.5 + 0.5 * np.exp(-tt * 1.5)
        for k, midi in enumerate(CHORDS[i % 4]):
            tone = _tone(440.0 * 2 ** ((midi - 69) / 12), tt) * env
            if split:
                out[sl, k % 2] += tone
            else:
                out[sl, :] += tone[:, None]
    return out


def normal_mix(seconds: float = SECONDS, sr: int = SR, seed: int = 0, peak: float = 0.5) -> np.ndarray:
    """Moderately mastered: -6 dBFS sample peak, so even lossy encodes stay under the -1 dBTP ceiling."""
    rng = np.random.default_rng(seed)
    x = chords(seconds, sr) + rng.standard_normal((int(seconds * sr), 2)) * 0.05
    return x / np.max(np.abs(x)) * peak


def loud_master(x: np.ndarray, drive: float = 2.0, peak_dbfs: float = -0.1) -> np.ndarray:
    y = np.tanh(x / np.max(np.abs(x)) * drive)
    return y / np.max(np.abs(y)) * 10 ** (peak_dbfs / 20)


def quasi_mono(seconds: float = SECONDS, sr: int = SR, side_db: float = -40.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mid = chords(seconds, sr, split=False)[:, 0]
    d = rng.standard_normal(mid.shape)
    d *= np.sqrt(np.mean(mid**2) / np.mean(d**2)) * 10 ** (side_db / 20)
    x = np.stack([mid + d, mid - d], axis=1)
    return x / np.max(np.abs(x)) * 0.9


def rms_dbfs(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def ffmpeg_encode(src: Path, dst: Path, codec_args: list[str]) -> Path:
    exe = shutil.which("ffmpeg")
    assert exe, "ffmpeg not on PATH"
    proc = subprocess.run(
        [exe, "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", str(src), *codec_args, str(dst)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return dst


def make_fixture_set(d: Path) -> dict[str, Path]:
    """Write every fixture into ``d``; keys like "normal.wav", "loud.opus", "mono.wav"."""
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    base = normal_mix()
    for name, x, sr in (
        ("normal", base, SR),
        ("loud", loud_master(base), SR),
        ("quasi", quasi_mono(), SR),
        ("sr32k", normal_mix(sr=32000), 32000),
    ):
        wav = d / f"{name}.wav"
        sf.write(str(wav), x.astype(np.float32), sr, subtype="PCM_16")
        out[wav.name] = wav
    mono = d / "mono.wav"
    sf.write(str(mono), (chords(split=False)[:, 0] * 0.1).astype(np.float32), SR, subtype="PCM_16")
    out[mono.name] = mono

    for ext in ("flac", "mp3", "m4a", "opus", "webm"):
        out[f"normal.{ext}"] = ffmpeg_encode(out["normal.wav"], d / f"normal.{ext}", CODECS[ext])
    for ext in ("opus", "m4a"):
        out[f"loud.{ext}"] = ffmpeg_encode(out["loud.wav"], d / f"loud.{ext}", CODECS[ext])
    return out
