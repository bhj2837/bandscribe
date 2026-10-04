"""Sampled presence pass for the piano+other view (S3.5 in every profile; ``eval`` adds a full-length pass).

Why: the full-length ``piano_other_mono`` MuScriptor pass dominated the pipeline (measured 2-864 s per song; pad-heavy
songs decode at ~3 s per audio second on the RTX 2060, e.g. 864 s for 274 s on 青と夏), yet instrumentation only has
to know *which* keys/synth/strings classes play in the song. A bounded set of excerpts answers that question.

The excerpts are transcribed by the worker as **segments of the full view** (one ``transcribe`` call per window,
fresh decoder state, times shifted back to song time), so the presence NoteSet lives on the song's time axis and its
notes can be compared with every other transcription. The full-length pass stays available (``eval`` runs it as an
extra output; ``amt.presence_pass = "full"`` makes it the presence source): S7's piano/other bleed penalty (M6) needs
notes for the whole song, so M6 must either request the full pass or run an on-demand pass over the bars it
penalises.

Window selection (deterministic, numpy only):

- level: piano+other power per 10 Hz frame relative to the mix, from the ``stems`` stage's ``energy.npz``
  (``10 log10(10^(piano/10) + 10^(other/10)) - mix``); a frame is *active* above ``active_rel_db``.
- candidates: a window of ``window_s`` starting at every bar start (pseudo-bars every ``window_s / 2`` without a
  grid) whose window fits in the song (only if none fits, one window shifted inside the song). A candidate
  **qualifies** if at least ``min_active`` of its frames are active **or** its power-mean level is above
  ``active_rel_db`` (the bar-level rule of the instrumentation estimator: a staccato part active in few frames
  but most bars still qualifies). Its score is the mean of
  ``clip((rel_db - active_rel_db) / 30, 0, 1)`` over its frames (louder = clearer content). It *covers* the piano
  (other) stem if that stem's power-mean level in the window is above ``active_rel_db``.
- home section: a candidate belongs to the section that holds the whole window (0.5 s slack); a section shorter
  than a window takes the candidates that start in it; a window wholly before the first or after the last section
  belongs to that section; a candidate that straddles out of a long section belongs to none (its level would be
  partly the neighbour's). Equal time zones stand in for missing sections.
- budget: window length ``min(window_s, song_s, max_total_s)``; ``n = min(max_windows, floor(min(max_total_s,
  max(window, max_fraction * song_s)) / window))``, at least one window (a song shorter than one window is
  transcribed whole). So ``n * window <= max_total_s`` always.
- greedy: each round takes the free candidate (not overlapping a chosen window) with the largest
  ``(share of its frames where a not yet covered stem is active, to 0.1, label not sampled yet, section not sampled yet, at least min_gap from every chosen
  window, score to 0.1, distance to the nearest chosen window, score, earlier)``, ``min_gap = max(window,
  song_s / (2 n))`` between window centres. So every active stem (piano, other) is sampled at least once, then
  distinct section labels, never two neighbouring windows while a far one is free, the louder piano/other material
  among comparable spreads (clearer notes: on AZ the organ plays only in one loud verse), then the part of the
  song farthest from what is already sampled. Review 2026-10-04: without the gap rule, 4 of 5 windows of 青と夏
  fell in 152-253 s (two adjacent) and nothing between 55 and 152 s was sampled.

Windows come back sorted by start time; times are rounded to 1 ms. Classes that play only outside the windows are
missed: the estimator treats the stem activity outside the windows as unknown (``gtab.amt.instrumentation``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PRESENCE_VIEW = "piano_other_presence"  # the sampled NoteSet's view id (raw/piano_other_presence__muscriptor.json)
SOURCE_VIEW = "piano_other_mono"  # the view the windows are cut from
FULL_VIEW = "piano_other_mono"
LEVEL_SPAN_DB = 30.0  # score: 0 at active_rel_db, 1 at active_rel_db + 30 dB
SECTION_SLACK_S = 0.5  # a window may end this far into the next section and still belong to its own
COVER_STEMS = ("piano", "other")  # stems whose activity the windows must sample at least once
CLEAR_SCORE = 0.1  # a window this loud on average holds decodable piano/other material (ranked before labels)

DEFAULTS: dict[str, Any] = {
    "window_s": 10.0, "max_windows": 6, "max_total_s": 60.0, "max_fraction": 0.2, "min_active": 0.5,
    "active_rel_db": -40.0,
}


def _r3(x: float) -> float:
    return round(float(x), 3)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def piano_other_rel_db(energy: Mapping[str, np.ndarray]) -> np.ndarray | None:
    """piano+other level relative to the mix per energy frame (dB), or None if a series is missing."""
    if not all(k in energy for k in ("mix", "piano", "other")):
        return None
    mix = np.asarray(energy["mix"], dtype=np.float64)
    po = 10.0 ** (np.asarray(energy["piano"], dtype=np.float64) / 10.0) + \
        10.0 ** (np.asarray(energy["other"], dtype=np.float64) / 10.0)
    n = min(len(mix), len(po))
    return 10.0 * np.log10(po[:n] + 1e-20) - mix[:n]


def _power_db(x: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(10.0 ** (np.asarray(x, dtype=np.float64) / 10.0)) + 1e-20))


def stem_rel_db(energy: Mapping[str, np.ndarray], stem: str) -> np.ndarray | None:
    """One stem's level relative to the mix per energy frame (dB), or None if a series is missing."""
    if "mix" not in energy or stem not in energy:
        return None
    mix = np.asarray(energy["mix"], dtype=np.float64)
    x = np.asarray(energy[stem], dtype=np.float64)
    n = min(len(mix), len(x))
    return x[:n] - mix[:n]


def budget(song_s: float, *, window_s: float, max_windows: int, max_total_s: float,
           max_fraction: float) -> tuple[int, float]:
    """(number of windows, window length) for a song of ``song_s`` seconds; at least one window.

    The window is clamped to the song and to ``max_total_s``, so ``n * window <= max_total_s`` holds even for a
    window longer than the audio cap (review 2026-10-04: fast + a 60 s window used to sample 60 s of a 30 s cap).
    """
    if song_s <= 0:
        return 0, 0.0
    win = min(float(window_s), float(song_s), float(max_total_s))
    total = min(float(max_total_s), max(win, float(max_fraction) * float(song_s)))
    n = int(math.floor(total / win + 1e-9))
    return max(1, min(int(max_windows), n)), win


def _sections(sections: Sequence[Any] | None, song_s: float, n_target: int) -> list[dict[str, Any]]:
    out = []
    for s in sections or []:
        st, en = _get(s, "start_s"), _get(s, "end_s")
        if st is None or en is None or float(en) <= float(st):
            continue
        out.append({"id": str(_get(s, "id")), "label": str(_get(s, "label") or _get(s, "id")),
                    "start_s": float(st), "end_s": float(en)})
    out.sort(key=lambda s: (s["start_s"], s["id"]))
    if out:
        return out
    k = max(1, n_target)  # no sections: equal time zones, each its own "label"
    return [{"id": f"Z{i + 1}", "label": f"Z{i + 1}", "start_s": song_s * i / k, "end_s": song_s * (i + 1) / k}
            for i in range(k)]


def _home(c: Mapping[str, Any], secs: list[dict[str, Any]], win: float) -> int | None:
    """Index of the candidate's home section (see the module docstring), or None for a straddling window."""
    for i, sec in enumerate(secs):
        if sec["start_s"] - SECTION_SLACK_S <= c["start_s"] and c["end_s"] <= sec["end_s"] + SECTION_SLACK_S:
            return i
    for i, sec in enumerate(secs):
        if sec["start_s"] <= c["start_s"] < sec["end_s"]:
            return i if sec["end_s"] - sec["start_s"] < win else None
    if c["end_s"] <= secs[0]["start_s"] + SECTION_SLACK_S:  # intro / outro outside every section
        return 0
    if c["start_s"] >= secs[-1]["end_s"] - SECTION_SLACK_S:
        return len(secs) - 1
    return None


def select_windows(*, bars: Sequence[tuple[int, float, float]], sections: Sequence[Any] | None,
                   energy: Mapping[str, np.ndarray], fps: float, song_s: float, window_s: float = 10.0,
                   max_windows: int = 6, max_total_s: float = 60.0, max_fraction: float = 0.2,
                   min_active: float = 0.5, active_rel_db: float = -40.0) -> dict[str, Any]:
    """The presence windows of one song (see the module docstring). Pure and deterministic."""
    n_target, win = budget(song_s, window_s=window_s, max_windows=max_windows, max_total_s=max_total_s,
                           max_fraction=max_fraction)
    rel = piano_other_rel_db(energy)
    info: dict[str, Any] = {
        "song_s": _r3(song_s), "window_s": _r3(win), "n_target": n_target,
        "budget_s": _r3(n_target * win), "min_gap_s": 0.0, "n_candidates": 0, "n_qualified": 0, "windows": [],
        "transcribed_s": 0.0, "fraction": 0.0, "stems": {}, "reason": None,
    }
    if n_target == 0:
        info["reason"] = "empty_song"
        return info
    if rel is None or len(rel) == 0:
        info["reason"] = "no_energy"
        return info
    mix = np.asarray(energy["mix"], dtype=np.float64)[: len(rel)]
    stem_rel = {st: stem_rel_db(energy, st) for st in COVER_STEMS}

    def level(x: np.ndarray, a: int, z: int) -> float:
        """Power-mean level of a relative-dB series over frames [a, z) relative to the mix's power mean."""
        seg = x[a:z]
        m = mix[a:a + len(seg)]
        return _power_db(seg + m) - _power_db(m)

    # candidate starts: bar starts (else every half window) whose window fits in the song; only when none fits
    # (a song barely longer than a window) the window is shifted inside the song
    last = max(0.0, song_s - win)
    starts = [float(b[1]) for b in bars] if bars else list(np.arange(0.0, last + 1e-9, win / 2))
    fit = [s for s in starts if -1e-9 <= s <= last + 1e-6]
    starts = fit if fit else [min(max(0.0, s), last) for s in starts[:1]] or [0.0]
    cands: dict[float, dict[str, Any]] = {}
    for s in starts:
        s = _r3(min(max(0.0, s), last))
        if s in cands:
            continue
        a = int(math.floor(s * fps))
        z = min(len(rel), max(a + 1, int(math.ceil((s + win) * fps))))
        seg = rel[a:z]
        if len(seg) == 0:
            continue
        active = float(np.mean(seg > active_rel_db))
        lev = level(rel, a, z)
        score = float(np.mean(np.clip((seg - active_rel_db) / LEVEL_SPAN_DB, 0.0, 1.0)))
        covers = [st for st, sr in stem_rel.items() if sr is not None and len(sr[a:z]) and
                  level(sr, a, z) > active_rel_db]
        shares = {st: round(float(np.mean(sr[a:z] > active_rel_db)), 4) for st, sr in stem_rel.items()
                  if sr is not None and len(sr[a:z])}
        cands[s] = {"start_s": s, "end_s": _r3(s + win), "active": round(active, 4), "score": round(score, 4),
                    "level_db": round(lev, 2), "covers": covers, "shares": shares,
                    "qualified": active >= min_active or lev > active_rel_db}
    info["n_candidates"] = len(cands)
    secs = _sections(sections, song_s, n_target)
    qualified = [c for c in cands.values() if c["qualified"]]
    info["n_qualified"] = len(qualified)
    needed = {st for c in qualified for st in c["covers"]}
    info["stems"] = {st: {"active": st in needed, "covered": False} for st in COVER_STEMS}
    if not qualified:
        info["reason"] = "piano_other_inactive"
        return info
    pool = [(h, c) for h, c in ((_home(c, secs, win), c) for c in qualified) if h is not None]
    min_gap = max(win, song_s / (2.0 * n_target))
    info["min_gap_s"] = _r3(min_gap)

    chosen: list[dict[str, Any]] = []

    def free(c: Mapping[str, Any]) -> bool:
        return all(c["end_s"] <= w["start_s"] + 1e-9 or c["start_s"] >= w["end_s"] - 1e-9 for w in chosen)

    def dist(c: Mapping[str, Any]) -> float:
        if not chosen:
            return math.inf
        mid = (c["start_s"] + c["end_s"]) / 2
        return round(min(abs(mid - (w["start_s"] + w["end_s"]) / 2) for w in chosen), 2)

    while len(chosen) < n_target:
        used_labels = {w["label"] for w in chosen}
        used_secs = {w["section"] for w in chosen}
        covered = {st for w in chosen for st in w["covers"]}
        best_rank: tuple | None = None
        best: tuple[int, dict[str, Any]] | None = None
        for si, c in pool:
            if not free(c):
                continue
            sec = secs[si]
            d = dist(c)
            gain = sum(c["shares"].get(st, 0.0) for st in needed if st not in covered)
            rank = (round(gain, 1), d >= min_gap - 1e-9,
                    c["score"] >= CLEAR_SCORE, sec["label"] not in used_labels, sec["id"] not in used_secs,
                    round(c["score"], 1), d, c["score"], -c["start_s"])
            if best_rank is None or rank > best_rank:
                best_rank, best = rank, (si, c)
        if best is None:
            break
        si, c = best
        chosen.append({"start_s": c["start_s"], "end_s": c["end_s"], "section": secs[si]["id"],
                       "label": secs[si]["label"], "score": c["score"], "active": c["active"],
                       "level_db": c["level_db"], "covers": list(c["covers"])})
    chosen.sort(key=lambda w: w["start_s"])
    total = sum(w["end_s"] - w["start_s"] for w in chosen)
    for st in COVER_STEMS:
        info["stems"][st]["covered"] = any(st in w["covers"] for w in chosen)
    info.update(windows=chosen, transcribed_s=_r3(total), fraction=round(total / song_s, 4) if song_s else 0.0)
    if len(chosen) < n_target:
        info["reason"] = "fewer_qualified_windows"
    return info


def spans(selection: Mapping[str, Any]) -> list[list[float]]:
    """``[[start_s, end_s], ...]`` of a selection (the worker's ``segments``)."""
    return [[float(w["start_s"]), float(w["end_s"])] for w in selection.get("windows") or []]
