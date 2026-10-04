"""Loudness and true-peak measurement, and the float gain that keeps the true peak under a ceiling.

Integrated loudness: pyloudnorm (ITU-R BS.1770-4 gating) with the "DeMan" K-weighting filter. That
filter reproduces the standard's published 48 kHz coefficients and re-derives them for other rates the
same way libebur128/ffmpeg do; pyloudnorm's default "K-weighting" is a generic shelf approximation
that read ~0.05 LU lower than ffmpeg on our fixtures, eating half of the ±0.1 LU agreement budget.

True peak: ffmpeg's ebur128 filter (peak=true, 4x oversampled). Its printed summary has only 0.1 dB
resolution, so we also ask the filter for its frame metadata (linear value, 3 decimals) and keep an
explicit upper bound on the true peak. apply_ceiling uses that bound, so the ceiling holds even when
the displayed value is rounded down.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from bandscribe.ingest.decode import FFMPEG_TIMEOUT_S, IngestError, run_tool, stderr_tail, tool_path

log = logging.getLogger(__name__)

PYLOUDNORM_FILTER = "DeMan"
# ffmpeg prints "I: -70.0 LUFS" when everything is below the absolute gate; pyloudnorm says -inf.
_FFMPEG_LUFS_FLOOR = -70.0
# Resolution of the two ffmpeg outputs: summary "%5.1f" dB, metadata "%.3f" linear.
_SUMMARY_HALF_STEP_DB = 0.05
_META_HALF_STEP_LIN = 0.0005
# Below this linear level the 3-decimal metadata is coarser than the 0.1 dB summary.
_META_PRECISE_ABOVE_LIN = 0.1
_BLOCK_FRAMES = 1 << 18

_SUMMARY_I_RE = re.compile(r"^\s*I:\s*(\S+)\s+LUFS", re.MULTILINE)
_SUMMARY_TP_RE = re.compile(r"True peak:\s*\n\s*Peak:\s*(\S+)\s+dBFS")
_META_I_RE = re.compile(r"lavfi\.r128\.I=(\S+)")
_META_TP_RE = re.compile(r"lavfi\.r128\.true_peak=(\S+)")


def _finite(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _parse_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)  # also accepts "-inf" / "inf" / "nan"
    except ValueError:
        return None


def _db(lin: float | None) -> float | None:
    if lin is None or lin <= 0.0:
        return None
    return 20.0 * math.log10(lin)


def ffmpeg_ebur128(wav: Path) -> dict[str, float | None]:
    """Run ffmpeg ebur128 once and return integrated loudness and true peak (with an upper bound).

    Keys: integrated_lufs, true_peak_dbfs (best estimate), true_peak_bound_dbfs (guaranteed >= the
    filter's exact value, given the print resolution). None means silence / not measurable.
    """
    wav = Path(wav)
    # framelog=verbose keeps the 10-per-second frame lines out of the info log; metadata=1 plus
    # ametadata=print exposes the precise running values (the last frame carries the final ones).
    af = "ebur128=peak=true:framelog=verbose:metadata=1,ametadata=mode=print"
    args = [tool_path("ffmpeg"), "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-i", str(wav)]
    args += ["-map", "0:a:0", "-af", af, "-f", "null", "-"]
    proc = run_tool(args, timeout=FFMPEG_TIMEOUT_S)
    err = proc.stderr or ""
    pos = err.rfind("Summary:")
    if proc.returncode != 0 or pos < 0:
        raise IngestError(f"ffmpeg ebur128 측정에 실패했습니다: {wav.name}\n{stderr_tail(err)}")
    summary = err[pos:]

    m_i = _SUMMARY_I_RE.search(summary)
    m_tp = _SUMMARY_TP_RE.search(summary)
    sum_i = _finite(_parse_float(m_i.group(1) if m_i else None))
    sum_tp = _finite(_parse_float(m_tp.group(1) if m_tp else None))
    meta_i_all = _META_I_RE.findall(err)
    meta_tp_all = _META_TP_RE.findall(err)
    meta_i = _finite(_parse_float(meta_i_all[-1])) if meta_i_all else None
    meta_tp_lin = _finite(_parse_float(meta_tp_all[-1])) if meta_tp_all else None

    # Integrated: prefer the 3-decimal metadata unless it disagrees with the summary (it should not).
    integrated = sum_i
    if meta_i is not None and (sum_i is None or abs(meta_i - sum_i) <= _SUMMARY_HALF_STEP_DB + 1e-3):
        integrated = meta_i
    if integrated is not None and integrated <= _FFMPEG_LUFS_FLOOR + 1e-6:
        integrated = None

    true_peak = sum_tp
    if meta_tp_lin is not None and meta_tp_lin >= _META_PRECISE_ABOVE_LIN:
        true_peak = _db(meta_tp_lin)
    bounds = []
    if sum_tp is not None:
        bounds.append(sum_tp + _SUMMARY_HALF_STEP_DB)
    if meta_tp_lin is not None:
        bounds.append(20.0 * math.log10(meta_tp_lin + _META_HALF_STEP_LIN))
    bound = min(bounds) if bounds else None
    if true_peak is None and meta_tp_lin is not None and meta_tp_lin > 0:
        true_peak = _db(meta_tp_lin)
    return {"integrated_lufs": integrated, "true_peak_dbfs": true_peak, "true_peak_bound_dbfs": bound}


def measure(wav: Path) -> dict[str, float | None]:
    """Loudness summary of a WAV.

    Returns integrated_lufs (pyloudnorm), true_peak_dbfs and true_peak_bound_dbfs (ffmpeg),
    sample_peak_dbfs (numpy) and ffmpeg_integrated_lufs (ffmpeg, cross-check). None = silence.
    """
    wav = Path(wav)
    data, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    sample_peak = float(np.max(np.abs(data))) if data.size else 0.0
    # Imported here: pyloudnorm pulls in scipy.signal (~1 s), which a cache-hit ingest must not pay.
    import pyloudnorm

    integrated: float | None
    try:
        with np.errstate(divide="ignore"):
            integrated = _finite(pyloudnorm.Meter(rate, filter_class=PYLOUDNORM_FILTER).integrated_loudness(data))
    except ValueError as e:  # shorter than one 400 ms gating block
        log.warning("integrated loudness not measurable for %s: %s", wav.name, e)
        integrated = None
    del data
    ff = ffmpeg_ebur128(wav)
    return {
        "integrated_lufs": integrated,
        "true_peak_dbfs": ff["true_peak_dbfs"],
        "true_peak_bound_dbfs": ff["true_peak_bound_dbfs"],
        "sample_peak_dbfs": _db(sample_peak),
        "ffmpeg_integrated_lufs": ff["integrated_lufs"],
    }


def apply_ceiling(
    wav_in: Path, wav_out: Path, ceiling_dbfs: float, *, measured: dict[str, Any] | None = None
) -> float:
    """Write ``wav_out`` = ``wav_in`` * gain (float32) so the true peak is <= ``ceiling_dbfs``.

    Returns the gain in dB: 0.0 when the input is already under the ceiling, negative otherwise.
    Never boosts. ``measured`` may pass a measure()/ffmpeg_ebur128() result for ``wav_in`` to skip
    a second ffmpeg pass. ``wav_out`` may equal ``wav_in`` (written via a temp file + os.replace).
    """
    wav_in, wav_out = Path(wav_in), Path(wav_out)
    if measured is None or "true_peak_bound_dbfs" not in measured:
        measured = ffmpeg_ebur128(wav_in)
    bound = measured.get("true_peak_bound_dbfs")
    gain_db = 0.0 if bound is None or bound <= ceiling_dbfs else float(ceiling_dbfs - bound)
    factor = np.float32(10.0 ** (gain_db / 20.0))

    tmp = wav_out.with_name(f".{wav_out.name}.{os.getpid()}.tmp")
    try:
        with sf.SoundFile(str(wav_in)) as src:
            with sf.SoundFile(
                str(tmp), "w", samplerate=src.samplerate, channels=src.channels, subtype="FLOAT", format="WAV"
            ) as dst:
                for block in src.blocks(blocksize=_BLOCK_FRAMES, dtype="float32", always_2d=True):
                    dst.write(block * factor if gain_db != 0.0 else block)
        os.replace(tmp, wav_out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    log.debug("apply_ceiling %s: bound %s dBTP -> gain %.3f dB", wav_in.name, bound, gain_db)
    return gain_db
