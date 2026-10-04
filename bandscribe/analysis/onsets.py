"""Percussive onsets of the full mix (DESIGN S2; M1_M2_SPEC 9.3 item 2).

The beat grid is refined and its tempo octave checked against *percussive* onsets of the **full mix**: the
drums stem does not exist yet at S2 (it comes from S3), and the drums and bass in the mix are what carry the
beat anyway. Pipeline: mix -> mono (L+R)/2 -> 22050 Hz (soxr) -> STFT (n_fft 2048, hop 256) -> HPSS
(``librosa.decompose.hpss``, margin 2.0 [잠정]) -> mel power of the percussive part -> ``onset_strength``
(spectral flux) -> peak picking -> each onset time refined on the waveform (steepest log-energy rise, ~1 ms).

Memory: the STFT/HPSS runs in 30 s windows with 3 s of context on each side (DESIGN 9.5 "STFT는 창 단위로"),
so a 10-minute song never holds its whole spectrogram. Every step is frame-local (STFT frame, 31-frame HPSS
median, lag-1 flux), so the kept centre of each window does not depend on where the windows fall; decibels
use a fixed reference (ref=1, no top_db) for the same reason.

Determinism: numpy FFT and scipy median filters are single-threaded; the mel projection is a BLAS product,
so callers run this under ``threadpoolctl.threadpool_limits(1)`` (the grid stage does). Outputs are written
with :func:`write_npz`, a zip writer with fixed timestamps: ``np.savez`` stamps the current local time into
every member header, so two identical runs would differ byte-wise (M1_M2_SPEC 0 "data outputs").

librosa, soxr and scipy are imported inside functions (CLI start-up stays light, M1_M2_SPEC 0).
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic

log = logging.getLogger(__name__)

SR = 22050
N_FFT = 2048
HOP = 256
FPS = SR / HOP  # 86.13 frames per second
N_MELS = 128
HPSS_MARGIN = 2.0  # [잠정] DESIGN S2 / M1_M2_SPEC 9.3
HPSS_KERNEL = 31
WINDOW_S = 30.0
CONTEXT_S = 3.0
READ_BLOCK = 1 << 18
# Peak picking (librosa's onset_detect defaults, expressed in seconds so they survive a hop change).
PEAK_PRE_MAX_S = 0.03
PEAK_POST_MAX_S = 0.0
PEAK_PRE_AVG_S = 0.10
PEAK_POST_AVG_S = 0.10
PEAK_WAIT_S = 0.03
PEAK_DELTA = 0.07  # on the envelope normalised to its maximum
# Waveform refinement of each peak (see refine_times)
REFINE_BACK_S = 0.025
REFINE_FWD_S = 0.006
REFINE_SMOOTH = 32  # samples at 22050 Hz = 1.45 ms
REFINE_DLAG = 16


def load_mono(path: Path, sr: int = SR) -> np.ndarray:
    """Read a WAV (any channel count) in blocks, average the channels, resample to ``sr`` with soxr (float32)."""
    import soundfile as sf
    import soxr

    parts: list[np.ndarray] = []
    with sf.SoundFile(str(path)) as f:
        in_sr = int(f.samplerate)
        for block in f.blocks(blocksize=READ_BLOCK, dtype="float32", always_2d=True):
            parts.append(block.mean(axis=1, dtype=np.float32) if block.shape[1] > 1 else block[:, 0].copy())
    x = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if in_sr != sr and x.size:
        x = soxr.resample(x, in_sr, sr).astype(np.float32, copy=False)
    return np.ascontiguousarray(x, dtype=np.float32)


def _envelope_window(y: np.ndarray, mel_basis: np.ndarray) -> np.ndarray:
    import librosa

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True)
    mag = np.abs(stft)
    del stft
    _harm, perc = librosa.decompose.hpss(mag, kernel_size=HPSS_KERNEL, margin=HPSS_MARGIN)
    del mag, _harm
    mel = mel_basis @ (perc.astype(np.float32) ** 2)
    mel_db = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    env = librosa.onset.onset_strength(S=mel_db, sr=SR, n_fft=N_FFT, hop_length=HOP, lag=1, max_size=1,
                                       center=True, detrend=False, aggregate=np.mean)
    return env.astype(np.float32)


def percussive_envelope(y: np.ndarray) -> np.ndarray:
    """Percussive spectral-flux envelope at ``FPS`` (frame i <-> time i * HOP / SR), computed in windows."""
    import librosa

    n_frames = 1 + len(y) // HOP
    env = np.zeros(n_frames, dtype=np.float32)
    if len(y) == 0:
        return env
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
    win = max(1, int(round(WINDOW_S * FPS)))
    ctx = int(round(CONTEXT_S * FPS))
    for f0 in range(0, n_frames, win):
        f1 = min(n_frames, f0 + win)
        a = max(0, f0 - ctx)
        b = min(n_frames, f1 + ctx)
        seg = y[a * HOP: min(len(y), b * HOP)]
        part = _envelope_window(seg, mel_basis)
        # frame j of the segment is frame a + j of the whole signal (both centred, segment starts on a hop)
        lo, hi = f0 - a, f1 - a
        take = part[lo:min(hi, len(part))]
        env[f0:f0 + len(take)] = take
    return env


def pick_onsets(env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Peak-pick the envelope. Returns (peak frames, strengths = envelope value at the peak)."""
    import librosa

    if env.size < 3 or float(np.max(env)) <= 0.0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    norm = env.astype(np.float64) / float(np.max(env))
    fr = lambda s: max(1, int(round(s * FPS)))  # noqa: E731
    peaks = librosa.util.peak_pick(norm, pre_max=fr(PEAK_PRE_MAX_S), post_max=int(round(PEAK_POST_MAX_S * FPS)) + 1,
                                   pre_avg=fr(PEAK_PRE_AVG_S), post_avg=int(round(PEAK_POST_AVG_S * FPS)) + 1,
                                   delta=PEAK_DELTA, wait=fr(PEAK_WAIT_S))
    peaks = np.asarray(peaks, dtype=np.int64)
    return peaks, env[peaks].astype(np.float32)


def refine_times(y: np.ndarray, peak_frames: np.ndarray) -> np.ndarray:
    """Onset time of each envelope peak with ~1 ms precision, from the waveform ``y`` (mono, ``SR``).

    The flux peak is frame-quantised (11.6 ms) and lands ~8 ms after the attack (measured on synthetic clicks,
    kicks and snares over sustained chords: +8 ms mean, 0.6-14 ms spread; a parabola through the peak does not
    help because the peak is one frame wide). So within [frame time - 25 ms, frame time + 6 ms] we take the
    steepest rise of the log short-time energy (1.45 ms window, +-0.73 ms difference) of the first-differenced
    signal, which emphasises transients over sustained tones. Measured error on the same signals: -0.9..+0.2 ms
    mean, sd <= 0.35 ms.
    """
    out = np.empty(len(peak_frames), dtype=np.float64)
    smooth, dlag = REFINE_SMOOTH, REFINE_DLAG
    kernel = np.ones(smooth) / smooth
    for k, f in enumerate(peak_frames):
        t = float(f) / FPS
        lo_s, hi_s = int((t - REFINE_BACK_S) * SR), int((t + REFINE_FWD_S) * SR)
        a = max(0, lo_s - smooth - dlag - 1)
        b = min(len(y), hi_s + smooth + dlag + 1)
        if b - a <= 2 * dlag + 2:
            out[k] = t
            continue
        seg = y[a:b].astype(np.float64)
        d = np.diff(seg, prepend=seg[:1])
        le = np.log(np.convolve(d * d, kernel, mode="same") + 1e-12)
        rise = np.full(le.size, -np.inf)
        rise[dlag:-dlag] = le[2 * dlag:] - le[:-2 * dlag]
        lo, hi = max(lo_s - a, dlag), min(hi_s - a, le.size - dlag)
        out[k] = (a + lo + int(np.argmax(rise[lo:hi]))) / SR if hi > lo else t
    return out


def compute(mix_wav: Path) -> dict[str, Any]:
    """Percussive onset analysis of a mix file: ``{"env", "fps", "times", "strength"}``."""
    y = load_mono(Path(mix_wav))
    env = percussive_envelope(y)
    frames, strength = pick_onsets(env)
    times = refine_times(y, frames)
    log.info("percussive onsets: %d peaks over %.1f s", len(times), len(y) / SR)
    return {"env": env, "fps": float(FPS), "times": times, "strength": strength, "duration_s": len(y) / SR}


def to_arrays(result: dict[str, Any]) -> dict[str, np.ndarray]:
    """The ``onsets.npz`` members (fixed order, fixed dtypes)."""
    return {
        "env": np.asarray(result["env"], dtype=np.float32),
        "fps": np.asarray(float(result["fps"]), dtype=np.float64),
        "times": np.round(np.asarray(result["times"], dtype=np.float64), 6),
        "strength": np.asarray(result["strength"], dtype=np.float32),
        "sr": np.asarray(SR, dtype=np.int64),
        "hop": np.asarray(HOP, dtype=np.int64),
    }


def load(path: Path) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=False) as z:
        return {"env": z["env"], "fps": float(z["fps"]), "times": z["times"], "strength": z["strength"]}


# Fixed member timestamp (the zip epoch) so identical arrays give identical bytes.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Byte-deterministic ``.npz`` (readable by ``np.load``), written atomically.

    ``np.savez`` puts the current local time into each zip member header; this writer uses a fixed
    timestamp, fixed permissions, no compression and the caller's member order.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asanyarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = 0
            zf.writestr(info, member.getvalue())
    atomic.write_bytes(Path(path), buf.getvalue())
