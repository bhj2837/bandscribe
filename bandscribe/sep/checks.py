"""Separation quality checks (M1_M2_SPEC 9.1 item 8; DESIGN S3 risk "bin별 coherence·ILD 보존 미검증").

``stereo_preservation`` compares the true guitar-only stereo stem with SW's guitar stem through the router's
own stereo features (``bandscribe.analysis.stereo``, EVAL): per-bin coherence and ILD maps, energy-weighted by the
**true** stem (the image the router should see), plus the router ``window_features`` of both. It is the M6
go/no-go's early look (b8): if SW flattens coherence or ILD, the stereo router cannot work on SW stems.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _channel_first(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or 2 not in a.shape:
        raise ValueError(f"stereo array must be (2, n) or (n, 2), got {a.shape}")
    return a if a.shape[0] == 2 else a.T


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def stereo_preservation(true_stereo: np.ndarray, est_stereo: np.ndarray, sr: int) -> dict[str, Any]:
    """Energy-weighted mean |Δcoherence| and |ΔILD| (dB) per bin, and router window-feature deltas.

    Inputs are (2, n) or (n, 2); the longer one is trimmed to the shorter. Weights = power of the true stem
    (bins where the true guitar is silent do not count).
    """
    from bandscribe.analysis import stereo

    t, e = _channel_first(true_stereo), _channel_first(est_stereo)
    n = min(t.shape[1], e.shape[1])
    t, e = t[:, :n], e[:, :n]
    coh_t, ild_t, w_t = stereo.coherence_ild_maps(t, sr)
    coh_e, ild_e, _w_e = stereo.coherence_ild_maps(e, sr)
    w = np.asarray(w_t, dtype=np.float64)
    wsum = float(np.sum(w))
    if wsum <= 0:
        raise ValueError("true stem is silent: nothing to compare")
    d_coh = float(np.sum(w * np.abs(coh_e - coh_t)) / wsum)
    d_ild = float(np.sum(w * np.abs(ild_e - ild_t)) / wsum)
    wf_t = {k: _finite(v) for k, v in stereo.window_features(t, sr).items()}
    wf_e = {k: _finite(v) for k, v in stereo.window_features(e, sr).items()}
    deltas = {k: (None if wf_t.get(k) is None or wf_e.get(k) is None else wf_e[k] - wf_t[k]) for k in wf_t}
    energy_ratio_db = 10.0 * math.log10((float(np.sum(e * e)) + 1e-20) / (float(np.sum(t * t)) + 1e-20))
    return {
        "delta_coh": d_coh,
        "delta_ild_db": d_ild,
        "energy_ratio_db": energy_ratio_db,
        "window_features_true": wf_t,
        "window_features_est": wf_e,
        "window_feature_deltas": deltas,
        "samples": int(n),
        "sr": int(sr),
    }
