"""Float stem archiving to 24-bit FLAC and restore (DESIGN 9.3 ``bandscribe clean --archive``; M2 b7, M1_M2_SPEC 9.1 item 7).

``archive`` peak-normalises the stem to -0.1 dBFS, quantises it to 24-bit integers itself (``round(x * 2**23)``,
clipped) and stores the gain in a ``<name>.gain.json`` sidecar; ``restore`` reads the integers back exactly and
divides by the gain. Quantising here instead of letting libsndfile convert float -> int keeps the round trip
exact apart from the rounding itself (libsndfile scales by 2**23 - 1 on write and 2**23 on read).

Error measure (b7): error RMS after un-gaining **relative to the mix RMS** (``ref_rms_db``). 24-bit quantisation
noise after peak normalisation is about -149 dBFS RMS, so relative to the stem's *own* RMS a sparse stem with a
crest factor above ~49 dB would fail -100 dB although nothing audible is lost. Stems quieter than -60 dB
relative to the mix are reported ``skipped_sparse`` rather than judged.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic

log = logging.getLogger(__name__)

TARGET_PEAK_DBFS = -0.1
FULL = float(1 << 23)
INT24_MAX = (1 << 23) - 1
INT24_MIN = -(1 << 23)
BLOCK_FRAMES = 1 << 18
PASS_DB = -100.0
SPARSE_DB = -60.0
_EPS = 1e-30


def gain_path(flac: Path) -> Path:
    flac = Path(flac)
    return flac.with_name(flac.stem + ".gain.json")


def _peak(wav: Path) -> tuple[float, int, int, int]:
    import soundfile as sf

    peak = 0.0
    with sf.SoundFile(str(wav)) as f:
        sr, ch, n = f.samplerate, f.channels, f.frames
        for block in f.blocks(blocksize=BLOCK_FRAMES, dtype="float64", always_2d=True):
            if block.size:
                peak = max(peak, float(np.max(np.abs(block))))
    return peak, sr, ch, n


def archive(wav: Path, flac: Path) -> float:
    """Write ``flac`` (24-bit) + ``<stem>.gain.json``; returns the linear gain applied before quantising."""
    import soundfile as sf

    wav, flac = Path(wav), Path(flac)
    peak, sr, ch, n = _peak(wav)
    target = 10.0 ** (TARGET_PEAK_DBFS / 20.0)
    gain = target / peak if peak > 0 else 1.0
    flac.parent.mkdir(parents=True, exist_ok=True)
    tmp = flac.with_name(f".{flac.name}.{os.getpid()}.tmp")
    try:
        with sf.SoundFile(str(wav)) as src, sf.SoundFile(str(tmp), "w", samplerate=sr, channels=ch,
                                                         subtype="PCM_24", format="FLAC") as dst:
            for block in src.blocks(blocksize=BLOCK_FRAMES, dtype="float64", always_2d=True):
                q = np.clip(np.rint(block * gain * FULL), INT24_MIN, INT24_MAX).astype(np.int32)
                dst.write(q << 8)  # libsndfile maps int32 to 24-bit by dropping the low 8 bits
        os.replace(tmp, flac)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    atomic.write_json(gain_path(flac), {"format": "bandscribe.archive/1", "gain": repr(gain), "gain_db":
                                        round(20.0 * math.log10(gain), 6), "source_peak": peak, "frames": n,
                                        "samplerate": sr, "channels": ch, "bits": 24})
    return gain


def read_gain(flac: Path) -> float:
    doc = atomic.read_json(gain_path(Path(flac)))
    return float(doc["gain"])


def restore(flac: Path) -> np.ndarray:
    """Float32 (frames, channels) array of the original stem (whole file: stems are ~85 MB for 4 min)."""
    import soundfile as sf

    gain = read_gain(flac)
    q, _sr = sf.read(str(flac), dtype="int32", always_2d=True)
    return ((q >> 8).astype(np.float64) / FULL / gain).astype(np.float32)


def roundtrip_error_db(wav: Path, *, ref_rms_db: float, work_dir: Path | None = None) -> float:
    """Archive + restore ``wav``; error RMS (dB) after un-gaining, relative to ``ref_rms_db`` (the mix RMS)."""
    import tempfile

    import soundfile as sf

    wav = Path(wav)
    with tempfile.TemporaryDirectory(dir=work_dir) as td:
        flac = Path(td) / "a.flac"
        archive(wav, flac)
        rest = restore(flac)
        err_sq = 0.0
        count = 0
        pos = 0
        with sf.SoundFile(str(wav)) as f:
            for block in f.blocks(blocksize=BLOCK_FRAMES, dtype="float32", always_2d=True):
                d = rest[pos:pos + len(block)].astype(np.float64) - block.astype(np.float64)
                err_sq += float(np.sum(d * d))
                count += d.size
                pos += len(block)
        if pos != len(rest):
            raise RuntimeError(f"restored length {len(rest)} != original {pos}")
    err_db = 10.0 * math.log10(err_sq / max(count, 1) + _EPS)
    return err_db - float(ref_rms_db)


def rms_db(wav: Path) -> float:
    import soundfile as sf

    s, c = 0.0, 0
    with sf.SoundFile(str(wav)) as f:
        for block in f.blocks(blocksize=BLOCK_FRAMES, dtype="float64", always_2d=True):
            s += float(np.sum(block * block))
            c += block.size
    return 10.0 * math.log10(s / max(c, 1) + 1e-20)


def roundtrip_check(wav: Path, *, mix_rms_db: float, pass_db: float = PASS_DB, sparse_db: float = SPARSE_DB,
                    work_dir: Path | None = None) -> dict[str, Any]:
    """b7 verdict for one stem: ``ok`` / ``fail`` (error vs the mix RMS) or ``skipped_sparse`` (< -60 dB rel. mix)."""
    rel = rms_db(Path(wav)) - float(mix_rms_db)
    if rel < sparse_db:
        return {"status": "skipped_sparse", "rel_rms_db": round(rel, 3), "error_db": None, "pass_db": pass_db}
    err = roundtrip_error_db(Path(wav), ref_rms_db=mix_rms_db, work_dir=work_dir)
    return {"status": "ok" if err < pass_db else "fail", "rel_rms_db": round(rel, 3), "error_db": round(err, 3),
            "pass_db": pass_db}
