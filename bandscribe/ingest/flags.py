"""Input flags (DESIGN S0): properties of the source that later stages must know about.

- quasi_mono / side_mid_db: 10*log10(E_side / E_mid). Near-mono mixes carry no stereo evidence for the
  guitar router, so they go to the R0 path.
- lowpass_hz_est / lossy_cutoff: a brick-wall spectral edge well below Nyquist means the audio went
  through a low-bitrate lossy codec before (e.g. an MP3 upload re-encoded by YouTube).
- nonstandard_sr, drc (YouTube's dynamic-range-compressed stream), mono_source.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

STANDARD_SAMPLE_RATES = frozenset({44100, 48000, 88200, 96000})
# cfg.ingest.lowpass_detect_hz is the nominal "~16 kHz" class of DESIGN S0. Encoders put the actual
# edge a little above the nominal value (measured on this PC: LAME 128k ~16.7 kHz, ffmpeg AAC 128k
# ~17.5 kHz, Opus/YouTube ~20 kHz), so the flag accepts edges up to 10 % above it.
LOSSY_CUTOFF_TOLERANCE = 1.10
# An edge must drop at least this much (1 kHz median below vs. everything above) to count as a
# codec lowpass; natural high-frequency roll-off is a few dB per kHz.
MIN_EDGE_DROP_DB = 30.0
# An edge this close to the *source* Nyquist is the sample rate's own band limit (anti-alias filter).
SR_EDGE_FRACTION = 0.9
EDGE_SEARCH_FROM_HZ = 8000.0
_BAND_HZ = 100.0
_REF_BANDS = 10  # 1 kHz reference just below the candidate edge
_TRANSITION_BANDS = 3  # 300 Hz allowed for the filter skirt above it
_NPERSEG = 8192
_BLOCK_FRAMES = 1 << 20
# Exactly mono (L == R) has zero side energy; clamp instead of storing -inf in JSON.
SIDE_MID_FLOOR_DB = -120.0


def _spectrum_and_stereo(wav: Path) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """One pass over the file: channel-averaged Welch PSD and the side/mid energy ratio in dB."""
    # Imported here: scipy.signal costs ~0.9 s to import, which would blow the < 1 s budget of a
    # cache-hit `bandscribe ingest` (that path never computes flags).
    from scipy.signal import welch

    e_mid = e_side = 0.0
    psd_sum: np.ndarray | None = None
    freqs: np.ndarray | None = None
    weight = 0
    with sf.SoundFile(str(wav)) as f:
        rate = f.samplerate
        for block in f.blocks(blocksize=_BLOCK_FRAMES, dtype="float32", always_2d=True):
            x = block.astype(np.float64)
            left = x[:, 0]
            right = x[:, 1] if x.shape[1] > 1 else x[:, 0]
            e_mid += float(np.sum(((left + right) * 0.5) ** 2))
            e_side += float(np.sum(((left - right) * 0.5) ** 2))
            if len(x) < _NPERSEG:
                continue  # a short tail adds nothing to a long-term spectrum
            fr, p = welch(x, fs=rate, nperseg=_NPERSEG, axis=0)
            p = p.mean(axis=1)
            psd_sum = p * len(x) if psd_sum is None else psd_sum + p * len(x)
            freqs = fr
            weight += len(x)
    if e_mid <= 0.0:
        side_mid_db = 0.0 if e_side > 0.0 else SIDE_MID_FLOOR_DB  # pure anti-phase vs. digital silence
    elif e_side <= 0.0:
        side_mid_db = SIDE_MID_FLOOR_DB
    else:
        side_mid_db = max(SIDE_MID_FLOOR_DB, 10.0 * math.log10(e_side / e_mid))
    psd = psd_sum / weight if psd_sum is not None and weight else None
    return freqs, psd, side_mid_db


def estimate_lowpass(freqs: np.ndarray, psd: np.ndarray) -> tuple[float | None, float]:
    """Find a brick-wall edge in a long-term PSD.

    Returns (edge_hz or None, largest drop in dB). A candidate edge needs every 100 Hz band above it
    (after a 300 Hz filter skirt) to stay >= MIN_EDGE_DROP_DB below the median of the 1 kHz just
    under it; the reported edge is where the level falls through the middle of that step.
    """
    nyq = float(freqs[-1])
    nb = int(nyq // _BAND_HZ)
    if nb <= _REF_BANDS + _TRANSITION_BANDS + 1:
        return None, 0.0
    idx = np.minimum((freqs // _BAND_HZ).astype(int), nb - 1)
    counts = np.bincount(idx, minlength=nb)
    power = np.bincount(idx, weights=psd, minlength=nb) / np.maximum(counts, 1)
    level = 10.0 * np.log10(power + 1e-30)
    # max over everything above each band, computed once from the top down
    above_max = np.maximum.accumulate(level[::-1])[::-1]

    best_edge, best_drop = None, 0.0
    start = max(int(EDGE_SEARCH_FROM_HZ // _BAND_HZ), _REF_BANDS)
    for e in range(start, nb - _TRANSITION_BANDS - 1):
        ref = float(np.median(level[e - _REF_BANDS : e]))
        drop = ref - float(above_max[e + _TRANSITION_BANDS])
        if drop > best_drop:
            best_edge, best_drop = e, drop
    if best_edge is None or best_drop < MIN_EDGE_DROP_DB:
        return None, round(best_drop, 1)
    # Several neighbouring candidates share the best drop; report where the level actually falls
    # through the middle of the step (first band more than half the drop below the reference).
    lo = best_edge - _REF_BANDS
    ref = float(np.median(level[lo:best_edge]))
    below = np.nonzero(level[lo:] < ref - best_drop / 2)[0]
    return (lo + int(below[0])) * _BAND_HZ, round(best_drop, 1)


def compute_flags(
    wav: Path, probe_info: dict[str, Any], cfg: Any, *, format_id: str | None = None
) -> dict[str, Any]:
    """Flags for meta.json. ``wav`` is the decoded stereo float WAV, ``probe_info`` the source's probe().

    ``format_id`` is the YouTube format (e.g. "251-drc"); None for local files.
    """
    icfg = cfg.ingest
    freqs, psd, side_mid_db = _spectrum_and_stereo(Path(wav))
    if freqs is not None and psd is not None:
        lowpass_hz, drop_db = estimate_lowpass(freqs, psd)
    else:
        lowpass_hz, drop_db = None, 0.0
    src_sr = probe_info.get("sample_rate")
    channels = probe_info.get("channels")
    lossy = lowpass_hz is not None and lowpass_hz <= float(icfg.lowpass_detect_hz) * LOSSY_CUTOFF_TOLERANCE
    if lossy and src_sr and lowpass_hz >= SR_EDGE_FRACTION * src_sr / 2:
        lossy = False  # e.g. a 32 kHz source ends at 16 kHz because of its rate, not a codec
    flags = {
        "quasi_mono": bool(side_mid_db < float(icfg.quasi_mono_side_mid_db)),
        "side_mid_db": round(side_mid_db, 2),
        "lowpass_hz_est": lowpass_hz,
        "lowpass_drop_db": drop_db,
        "lossy_cutoff": bool(lossy),
        "nonstandard_sr": bool(src_sr is not None and int(src_sr) not in STANDARD_SAMPLE_RATES),
        "source_sample_rate": src_sr,
        "drc": bool(format_id and "drc" in str(format_id).lower()),
        "mono_source": channels == 1,
    }
    log.debug("flags for %s: %s", Path(wav).name, flags)
    return flags
