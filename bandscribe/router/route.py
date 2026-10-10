"""Router windows, features, labels and the rule-based solo gate (DESIGN S4, 4.3 (a); M6).

**Windows** (DESIGN 4.3): 4 bars of the grid, cut at section boundaries so a window never spans two sections; a
section's remainder of 2-3 bars is its own window and a single remaining bar joins the window before it in the
same section. Bar 0 (a pickup) joins the first window. Whether 2-bar windows do better is E9's question (M7b).

**Features**: ``bandscribe.analysis.stereo.window_features`` of the stereo guitar stem in the window (the exact
port of the research script whose synthetic scenes set the starting thresholds) and the window's level: dBFS of
the mid signal, and relative to the song's loudest window.

**Labels** (DESIGN S4 table; ``THRESHOLDS`` = the synthetic starting values, re-fit on separated stems by E6):

- ``R1`` ADT: a delayed copy, |xcoff| > ``adt_xcorr`` at a lag of ``adt_min_lag_ms``-40 ms, and that delayed peak
  beats the zero-lag correlation r0 by more than ``adt_over_r0`` (a centred mono part is periodic: its off-lag peak
  reaches 0.5-0.75 while r0 = 1).
- ``R0`` narrow (B/F): S/M < ``r0_sm_db`` and coherence > ``r0_coh``.
- ``R3`` different positions (P, twin): mean |ILD| of the coherent bins > ``r3_ild_db``, or hard share >
  ``r3_hard`` and coherence < ``r3_coh``.
- ``R2`` wide same-part double (A): coherence < ``r2_coh``, hard share < ``r2_hard``, centre share > ``r2_cen``.

A condition's score is its signed distance to the threshold in units of ``SCALES`` (positive = met); a label's
score is the minimum over its conditions (R3: the maximum of its two alternatives). The label is the best label
with a positive score; when none is met the window is ``mixed`` (DESIGN S4 대안: every plausible path runs and S8
decides). ``p`` = softmax(score / ``P_TEMPERATURE``) over R0-R3 and ``margin`` = best - second score: a measure
of how far the window is from a rule boundary, not a calibrated probability (E6 reports how often it is right).
A window more than ``silence_db`` under the song's loudest window is ``silent`` (no label).

**Solo gate v0** (DESIGN S4, rule version; the MERT classifier is E9, M7b). Evidence by path, then a vocal veto:

- A (``R2`` windows): the coherent-energy share ``fcoh`` rises over the song's own rhythm level, the median
  ``fcoh`` of its ``R2`` windows (a centred lead over a wide double adds coherent energy; synthetic 0.18 -> 0.42,
  0.23 for a lead at -6 dB). On when the rise is >= ``solo_coh_rise``.
- N (``R0`` windows): the repetition residual of pass 1 (``resid1``: the stem minus its section-aligned median,
  per bar, dB relative to the stem) is >= ``solo_resid_db``: much of the guitar does not repeat.
- Other labels have no gate evidence here (P/twin roles are S8's).
- Vocal veto: a window whose bars are vocally active more than ``solo_vocal_max`` of the time is not a solo (a weak
  prior: guitars fill between vocal lines too, so it only vetoes, it never turns the gate on).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bandscribe.analysis import stereo
from bandscribe.router.thresholds import THRESHOLDS

LABELS = ("R0", "R1", "R2", "R3")
SCALES: dict[str, float] = {"sm_db": 3.0, "coh": 0.05, "hard": 0.05, "cen": 0.1, "ild_db": 2.0, "xcorr": 0.1,
                            "lag_ms": 2.0}
P_TEMPERATURE = 1.0
NAN_SCORE = -10.0  # an undefined feature (no coherent bins) never meets a condition
WINDOW_BARS = 4
DEFAULT_PARAMS: dict[str, Any] = {
    "window_bars": WINDOW_BARS,
    "silence_db": 40.0,
    "thresholds": dict(THRESHOLDS),
    "solo_coh_rise": 0.10,
    "solo_resid_db": -6.0,
    "solo_vocal_max": 0.5,
}


# ------------------------------------------------------------------------------------------------ windows


def windows(bars: Sequence[Mapping[str, Any]], sections: Sequence[Mapping[str, Any]] | None,
            window_bars: int = WINDOW_BARS) -> list[dict[str, Any]]:
    """Windows of ``window_bars`` bars (module doc) -> [{start_bar, end_bar, start_s, end_s, section}].
    ``bars`` are ``grid.json`` bars (``bar``, ``start_s``, ``end_s``); ``sections`` are ``sections.json`` sections
    (``id``, ``start_bar``, ``end_bar``); without sections the song is one section."""
    if not bars:
        return []
    by_bar = {int(b["bar"]): b for b in bars}
    first, last = min(by_bar), max(by_bar)
    secs = [(str(s.get("id")), int(s["start_bar"]), int(s["end_bar"])) for s in (sections or [])]
    if not secs:
        secs = [("S1", first, last + 1)]
    secs.sort(key=lambda s: s[1])
    # bars outside the listed sections (before the first / after the last) join the nearest section
    secs[0] = (secs[0][0], min(secs[0][1], first), secs[0][2])
    secs[-1] = (secs[-1][0], secs[-1][1], max(secs[-1][2], last + 1))
    out: list[dict[str, Any]] = []
    for sid, a, b in secs:
        cuts = list(range(a, b, window_bars))
        spans = [(c, min(c + window_bars, b)) for c in cuts]
        if len(spans) > 1 and spans[-1][1] - spans[-1][0] == 1:
            spans[-2] = (spans[-2][0], spans[-1][1])
            spans.pop()
        for s0, e0 in spans:
            have = [by_bar[k] for k in range(s0, e0) if k in by_bar]
            if not have:
                continue
            out.append({"start_bar": s0, "end_bar": e0, "start_s": float(have[0]["start_s"]),
                        "end_s": float(have[-1]["end_s"]), "section": sid})
    if len(out) > 1 and out[0]["start_bar"] == 0 and out[0]["end_bar"] == 1:  # a lone pickup bar joins window 1
        nxt = out.pop(1)
        out[0] = {**nxt, "start_bar": 0, "start_s": out[0]["start_s"]}
    return out


# ------------------------------------------------------------------------------------------------ features


def level_db(stereo_seg: np.ndarray) -> float:
    st = stereo.as_channel_first(stereo_seg)
    mid = (st[0] + st[1]) / 2
    return float(10 * np.log10(np.mean(mid ** 2) + 1e-20))


def features(stereo_seg: np.ndarray, sr: int) -> dict[str, float]:
    """``window_features`` (keys SM, r0, xcoff, lag, coh, cen, hard, fcoh, ildcoh) rounded to 1e-4."""
    f = stereo.window_features(stereo_seg, sr)
    return {k: (round(float(v), 4) if math.isfinite(float(v)) else None) for k, v in f.items()}


# ------------------------------------------------------------------------------------------------ labels


def _cond(value: float | None, thr: float, scale: float, above: bool) -> float:
    if value is None or not math.isfinite(value):
        return NAN_SCORE
    d = (value - thr) if above else (thr - value)
    return d / scale


def scores(feat: Mapping[str, float | None], thr: Mapping[str, float] = THRESHOLDS) -> dict[str, float]:
    """Signed rule scores per label (module doc): > 0 = the label's conditions are met."""
    t = {**THRESHOLDS, **thr}
    s = SCALES
    xc = feat.get("xcoff")
    lag = feat.get("lag")
    r0c = feat.get("r0")
    r1 = min(_cond(None if xc is None else abs(xc), t["adt_xcorr"], s["xcorr"], True),
             _cond(None if lag is None else abs(lag), t["adt_min_lag_ms"], s["lag_ms"], True),
             _cond(None if xc is None or r0c is None else abs(xc) - r0c, t["adt_over_r0"], s["xcorr"], True))
    r0 = min(_cond(feat.get("SM"), t["r0_sm_db"], s["sm_db"], False),
             _cond(feat.get("coh"), t["r0_coh"], s["coh"], True))
    r3 = max(_cond(feat.get("ildcoh"), t["r3_ild_db"], s["ild_db"], True),
             min(_cond(feat.get("hard"), t["r3_hard"], s["hard"], True),
                 _cond(feat.get("coh"), t["r3_coh"], s["coh"], False)))
    r2 = min(_cond(feat.get("coh"), t["r2_coh"], s["coh"], False),
             _cond(feat.get("hard"), t["r2_hard"], s["hard"], False),
             _cond(feat.get("cen"), t["r2_cen"], s["cen"], True))
    return {"R0": round(r0, 4), "R1": round(r1, 4), "R2": round(r2, 4), "R3": round(r3, 4)}


FEATURE_KEYS = ("SM", "r0", "xcoff", "lag", "coh", "cen", "hard", "ildcoh")


def feature_matrix(feats: Sequence[Mapping[str, float | None]]) -> dict[str, np.ndarray]:
    """Windows' features as arrays (NaN where missing) for ``score_matrix``."""
    return {k: np.array([np.nan if (f.get(k) is None) else float(f[k]) for f in feats], dtype=np.float64)
            for k in FEATURE_KEYS}


def _vcond(value: np.ndarray, thr: float, scale: float, above: bool) -> np.ndarray:
    d = (value - thr) if above else (thr - value)
    out = d / scale
    return np.where(np.isfinite(out), out, NAN_SCORE)


def score_matrix(fm: Mapping[str, np.ndarray], thr: Mapping[str, float] = THRESHOLDS) -> dict[str, np.ndarray]:
    """``scores`` for many windows at once (same rules, same numbers; E6's threshold search)."""
    t = {**THRESHOLDS, **thr}
    s = SCALES
    axc = np.abs(fm["xcoff"])
    r1 = np.minimum(np.minimum(_vcond(axc, t["adt_xcorr"], s["xcorr"], True),
                               _vcond(np.abs(fm["lag"]), t["adt_min_lag_ms"], s["lag_ms"], True)),
                    _vcond(axc - fm["r0"], t["adt_over_r0"], s["xcorr"], True))
    r0 = np.minimum(_vcond(fm["SM"], t["r0_sm_db"], s["sm_db"], False), _vcond(fm["coh"], t["r0_coh"], s["coh"], True))
    r3 = np.maximum(_vcond(fm["ildcoh"], t["r3_ild_db"], s["ild_db"], True),
                    np.minimum(_vcond(fm["hard"], t["r3_hard"], s["hard"], True),
                               _vcond(fm["coh"], t["r3_coh"], s["coh"], False)))
    r2 = np.minimum(np.minimum(_vcond(fm["coh"], t["r2_coh"], s["coh"], False),
                               _vcond(fm["hard"], t["r2_hard"], s["hard"], False)),
                    _vcond(fm["cen"], t["r2_cen"], s["cen"], True))
    return {"R0": r0, "R1": r1, "R2": r2, "R3": r3}


def labels_from_scores(sm: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per window: the best label with a positive score (``label``'s order on ties), else ``mixed``."""
    stack = np.stack([np.round(sm[k], 4) for k in LABELS])  # rounded like ``scores``
    best = np.argmax(stack, axis=0)  # first maximum = LABELS order, as in ``label``
    top = stack[best, np.arange(stack.shape[1])]
    return np.where(top > 0, np.array(LABELS, dtype=object)[best], "mixed")


def label(feat: Mapping[str, float | None], thr: Mapping[str, float] = THRESHOLDS
          ) -> tuple[str, dict[str, float], dict[str, float], float]:
    """-> (label, scores, p, margin). ``mixed`` when no label's conditions are met."""
    sc = scores(feat, thr)
    order = sorted(LABELS, key=lambda k: (-sc[k], LABELS.index(k)))
    best = order[0]
    lab = best if sc[best] > 0 else "mixed"
    z = np.array([sc[k] for k in LABELS]) / P_TEMPERATURE
    z = np.exp(z - z.max())
    p = {k: round(float(v), 4) for k, v in zip(LABELS, z / z.sum())}
    margin = round(sc[order[0]] - sc[order[1]], 4)
    return lab, sc, p, margin


# ------------------------------------------------------------------------------------------------ solo gate


def _bar_mean(per_bar: Mapping[int, float | None], a: int, b: int) -> float | None:
    vals = [per_bar[k] for k in range(a, b) if per_bar.get(k) is not None]
    return float(np.mean(vals)) if vals else None


def solo_gate(wins: Sequence[Mapping[str, Any]], vocal_per_bar: Mapping[int, float | None],
              resid_per_bar: Mapping[int, float | None], params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Solo gate v0 per window (module doc) -> [{on, path, evidence..}] in window order."""
    r2 = [w["features"]["fcoh"] for w in wins if w.get("label") == "R2" and w["features"].get("fcoh") is not None]
    base = float(np.median(r2)) if r2 else None
    out = []
    for w in wins:
        a, b = int(w["start_bar"]), int(w["end_bar"])
        voc = _bar_mean(vocal_per_bar, a, b)
        res = _bar_mean(resid_per_bar, a, b)
        g: dict[str, Any] = {"on": False, "path": None, "vocal": None if voc is None else round(voc, 3),
                             "resid_rel_db": None if res is None else round(res, 2), "coh_rise": None}
        lab = w.get("label")
        if lab == "R2" and base is not None and w["features"].get("fcoh") is not None:
            rise = float(w["features"]["fcoh"]) - base
            g["path"], g["coh_rise"] = "A", round(rise, 4)
            g["on"] = rise >= float(params["solo_coh_rise"])
        elif lab == "R0" and res is not None:
            g["path"] = "N"
            g["on"] = res >= float(params["solo_resid_db"])
        if g["on"] and voc is not None and voc > float(params["solo_vocal_max"]):
            g["on"], g["vetoed_by_vocal"] = False, True
        out.append(g)
    return out


# ------------------------------------------------------------------------------------------------ song


def route_song(guitar: np.ndarray, sr: int, bars: Sequence[Mapping[str, Any]],
               sections: Sequence[Mapping[str, Any]] | None, params: Mapping[str, Any] | None = None, *,
               vocal_per_bar: Mapping[int, float | None] | None = None,
               resid_per_bar: Mapping[int, float | None] | None = None) -> dict[str, Any]:
    """Route one song: stereo guitar stem ``(n, 2)`` or ``(2, n)`` + grid bars + sections -> the ``router.json``
    body (``windows``, ``summary``). Deterministic, CPU."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    thr = {**THRESHOLDS, **(p.get("thresholds") or {})}
    st = stereo.as_channel_first(guitar)
    n = st.shape[1]
    wins = windows(bars, sections, int(p["window_bars"]))
    for w in wins:
        a, b = max(0, int(round(w["start_s"] * sr))), min(n, int(round(w["end_s"] * sr)))
        seg = st[:, a:b]
        w["level_db"] = round(level_db(seg), 2) if seg.shape[1] > 0 else -200.0
    top = max((w["level_db"] for w in wins), default=-200.0)
    for w in wins:
        w["rel_db"] = round(w["level_db"] - top, 2)
        a, b = max(0, int(round(w["start_s"] * sr))), min(n, int(round(w["end_s"] * sr)))
        if w["rel_db"] < -float(p["silence_db"]) or b - a < int(0.5 * sr):
            w.update(label="silent", features={}, scores={}, p={}, margin=None)
            continue
        f = features(st[:, a:b], sr)
        lab, sc, pr, margin = label(f, thr)
        w.update(label=lab, features=f, scores=sc, p=pr, margin=margin)
    gates = solo_gate(wins, vocal_per_bar or {}, resid_per_bar or {}, p)
    for w, g in zip(wins, gates):
        w["solo"] = g
    for i, w in enumerate(wins):
        w["i"] = i
    counts: dict[str, int] = {}
    secs: dict[str, float] = {}
    for w in wins:
        counts[w["label"]] = counts.get(w["label"], 0) + 1
        secs[w["label"]] = round(secs.get(w["label"], 0.0) + (w["end_s"] - w["start_s"]), 3)
    return {"windows": [{k: w[k] for k in ("i", "start_bar", "end_bar", "start_s", "end_s", "section", "level_db",
                                           "rel_db", "label", "margin", "p", "scores", "features", "solo")}
                        for w in wins],
            "summary": {"windows": counts, "seconds": secs,
                        "solo_windows": sum(1 for w in wins if w["solo"]["on"])},
            "thresholds": thr, "scales": dict(SCALES)}
