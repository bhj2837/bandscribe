"""Float32 WAV I/O shared by CPU stages and workers (M1_M2_SPEC 1.4).

Why a hand-written writer: libsndfile's float WAVs are not byte-deterministic. ``soundfile.write(...,
subtype="FLOAT")`` adds a ``PEAK`` chunk carrying a timestamp, so two writes of the same array a second apart
differ (verified on this PC). CPU stage outputs must be byte-identical between runs (DESIGN 8.7), so every
stage writes WAVs through :func:`write_f32` / :class:`F32Writer`: a plain RIFF/WAVE IEEE-float file with exactly
three chunks, ``fmt `` (format tag 3, 18-byte WAVEFORMATEX with cbSize 0), ``fact`` (frame count) and ``data``.
soundfile reads these back bit-exactly; values above 1.0 survive (loud masters overshoot full scale).

Reading goes through soundfile (the stdlib ``wave`` module cannot read float32). Arrays are always float32
``(n, channels)``; mono helpers return ``(n,)``.

Importable in all three envs (core, gpu, bp310): stdlib + numpy + soundfile only, Python-3.10-safe
(``tests/test_py310_compat.py``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic

log = logging.getLogger(__name__)

SR = 44100
BLOCK_FRAMES = 1 << 18

_FMT_IEEE_FLOAT = 3
_FMT_CHUNK_SIZE = 18  # WAVEFORMATEX incl. cbSize = 0 (what non-PCM formats should carry)
_HEADER_BYTES = 12 + (8 + _FMT_CHUNK_SIZE) + (8 + 4) + 8  # RIFF + fmt + fact + data header = 58
_MAX_DATA_BYTES = 0xFFFFFFFF - _HEADER_BYTES  # RIFF sizes are 32-bit (a 4-min stereo song is ~85 MB)


def _header(frames: int, sr: int, channels: int) -> bytes:
    data_bytes = frames * channels * 4
    if data_bytes > _MAX_DATA_BYTES:
        raise ValueError(f"WAV too large for RIFF: {data_bytes} data bytes")
    fmt = struct.pack("<HHIIHHH", _FMT_IEEE_FLOAT, channels, sr, sr * channels * 4, channels * 4, 32, 0)
    return b"".join([
        b"RIFF", struct.pack("<I", _HEADER_BYTES - 8 + data_bytes), b"WAVE",
        b"fmt ", struct.pack("<I", _FMT_CHUNK_SIZE), fmt,
        b"fact", struct.pack("<I", 4), struct.pack("<I", frames),
        b"data", struct.pack("<I", data_bytes),
    ])


def _as_frames(data: Any, channels: int | None = None) -> np.ndarray:
    """``(n,)`` or ``(n, ch)`` -> C-contiguous little-endian float32 ``(n, ch)``; rejects NaN/inf."""
    x = np.asarray(data)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"audio must be (n,) or (n, channels), got shape {x.shape}")
    if channels is not None and x.shape[1] != channels:
        raise ValueError(f"expected {channels} channel(s), got {x.shape[1]}")
    if x.shape[1] < 1:
        raise ValueError("audio needs at least one channel")
    x = np.ascontiguousarray(x, dtype="<f4")
    # A NaN in a stem poisons every later sum (leftover, views, metrics); fail where it is written.
    if x.size and not np.isfinite(x).all():
        raise ValueError("audio contains NaN or inf")
    return x


def info(path: str | os.PathLike[str]) -> tuple[int, int, int]:
    """(frames, samplerate, channels) without decoding."""
    import soundfile as sf

    i = sf.info(str(path))
    return int(i.frames), int(i.samplerate), int(i.channels)


def iter_blocks(path: str | os.PathLike[str], block_frames: int = BLOCK_FRAMES) -> Iterator[np.ndarray]:
    """Stream a file as float32 ``(n, ch)`` blocks (M0 RAM rule: never six whole stems in memory)."""
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        for block in f.blocks(blocksize=int(block_frames), dtype="float32", always_2d=True):
            yield block


def read_f32(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    """Whole file as float32 ``(n, ch)`` and its sample rate. Only for small files (views, clips, tests)."""
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, int(sr)


def write_f32(path: str | os.PathLike[str], data: Any, sr: int = SR) -> None:
    """Deterministic IEEE-float WAV (no PEAK chunk); temp file in the same dir, then os.replace."""
    x = _as_frames(data)
    atomic.write_bytes(Path(path), _header(x.shape[0], int(sr), x.shape[1]) + x.tobytes())


class F32Writer:
    """Streaming :func:`write_f32`: same bytes for the same samples; the header is patched on close.

    The file appears at ``path`` only on :meth:`close` (atomic replace). :meth:`abort` (or leaving a ``with``
    block through an exception) deletes the temp file and publishes nothing.
    """

    def __init__(self, path: str | os.PathLike[str], sr: int, channels: int) -> None:
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self.path = Path(path)
        self.sr = int(sr)
        self.channels = int(channels)
        self.frames = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, self._tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        self._f: Any = os.fdopen(fd, "wb")
        self._f.write(_header(0, self.sr, self.channels))
        self._done = False

    def write(self, block: Any) -> None:
        if self._done:
            raise ValueError(f"F32Writer for {self.path} is closed")
        x = _as_frames(block, self.channels)
        if (self.frames + x.shape[0]) * self.channels * 4 > _MAX_DATA_BYTES:
            raise ValueError(f"WAV too large for RIFF: {self.path}")
        self._f.write(x.tobytes())
        self.frames += x.shape[0]

    def close(self) -> None:
        """Patch the header, fsync and atomically publish. Idempotent."""
        if self._done:
            return
        self._done = True
        try:
            self._f.seek(0)
            self._f.write(_header(self.frames, self.sr, self.channels))
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()
            atomic._replace(self._tmp, self.path)
        except BaseException:
            self._discard()
            raise

    def abort(self) -> None:
        """Drop the temp file; nothing is published. Idempotent."""
        if self._done:
            return
        self._done = True
        self._discard()

    def _discard(self) -> None:
        try:
            self._f.close()
        except OSError:
            pass
        try:
            os.unlink(self._tmp)
        except OSError:
            pass

    def __enter__(self) -> F32Writer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def to_mono(x: Any) -> np.ndarray:
    """(L+R)/2 as float32 ``(n,)`` (mono input is returned as float32; >2 channels are averaged)."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        return np.ascontiguousarray(a)
    if a.ndim != 2:
        raise ValueError(f"audio must be (n,) or (n, channels), got shape {a.shape}")
    if a.shape[1] == 1:
        return np.ascontiguousarray(a[:, 0])
    if a.shape[1] == 2:
        # The exact expression every view uses (float32 add, then * 0.5): views must be bit-identical
        # however they are computed (block-wise in the stems stage, whole-file elsewhere).
        return ((a[:, 0] + a[:, 1]) * np.float32(0.5)).astype(np.float32)
    return a.mean(axis=1, dtype=np.float32)


def pcm_sha256(path: str | os.PathLike[str], block_frames: int = BLOCK_FRAMES) -> str:
    """sha256 over the interleaved little-endian float32 samples (headers ignored).

    Same definition as ``bandscribe.ingest.decode.pcm_sha256`` (the job key), so a view's hash does not depend on
    which writer produced the file.
    """
    h = hashlib.sha256()
    for block in iter_blocks(path, block_frames):
        h.update(np.ascontiguousarray(block, dtype="<f4").tobytes())
    return h.hexdigest()


def rms_db(x: Any) -> float:
    """10*log10(mean(x^2) + 1e-20) in float64 (empty input -> -200 dB)."""
    a = np.asarray(x, dtype=np.float64)
    ms = float(np.mean(a * a)) if a.size else 0.0
    return float(10.0 * np.log10(ms + 1e-20))
