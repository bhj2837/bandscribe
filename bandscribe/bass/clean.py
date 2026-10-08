"""S11 clean-up of MuScriptor's bass with the F0 tracks, and technique suggestions (DESIGN 5.2-5.5, M5).

Input: the pass-1 bass after the silence gate (``notes/bass_raw.json``), dicts with ``onset_s``, ``offset_s``,
``pitch``, ``instrument``. Every step can be switched off (``params``); E13 (docs/decisions.md) decides the defaults.

1. ``anchor``: the 2-of-3 tracker vote per note (``bandscribe.bass.f0.vote``) moves a note by octaves or one
   semitone. Then notes starting together (within ``group_s``): a note whose pitch, after the moves, repeats an
   F0-confirmed note of the group, or sits an octave, a twelfth or two octaves above it (a harmonic MuScriptor
   wrote as a second note), is dropped (``dupes``).
2. ``kick``: a note starting within ``kick_s`` of a kick (low-band onsets of the drums stem) with no F0 estimate
   from any tracker is dropped (DESIGN 5.4 "킥 타격과 겹치면서 안정된 F0가 없는 onset").
3. ``fragments``: a note that continues a same-pitch note (gap <= ``join_gap_s``) without a new attack is merged
   into it when it is shorter than ``fragment_s`` or has no F0 estimate (a split or a decaying tail written twice).
   A re-plucked repeat of the same pitch has an attack and stays.
4. ``slides``: glides of the F0 contour, taken off the recording's tuning offset (``tuning_offset``). Two notes
   are joined by a glide when the contour runs from one into the other without a jump (every 10 ms step <=
   ``glide_step_st``) or a long gap, taking >= ``glide_min_s`` to cross the middle half of the interval from ``a``'s
   side to ``b``'s (a plucked change crosses it in a frame or two). A chain of
   glides in one direction whose inner notes have no plateau of ``hold_s`` is one slide: the first note keeps its
   onset and pitch, the passing notes go, and the first note slides (``tech.slide = "legato"``) into the last
   (the note the glide stops on), or slides out (``out_down`` / ``out_up``) when the glide fades out instead.
   A glide between two held notes keeps both notes and marks the slide. Slide partners must share a string
   (the fretter's ``links``).
5. ``gap_fill`` (DESIGN 5.2 two-voter policy): a stable two-tracker F0 run of ``fill_min_s`` where no note sounds
   becomes a low-confidence note (``src = "f0"``, ``conf`` = ``fill_conf``).
6. ``policy = "and"`` (E13's comparison arm only): keep only notes whose final pitch the F0 vote names.

Techniques (``tech``, suggestions that never change a note): ``vibrato`` (the contour inside a note of
>= ``vib_min_s`` oscillates at 3-9 Hz with >= 15 cents depth over >= 2 cycles) and ``dead`` (a note with an attack
but no F0 from any tracker). Output notes carry ``src`` ("ms" / "f0"), ``f0`` (the vote's action), ``from_pitch``
(when moved) and ``merged`` (how many notes it absorbed).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bandscribe.bass import f0 as F0

ARMS: tuple[str, ...] = ("raw", "anchor", "anchor_merge", "anchor_merge_kick", "two_voter", "and")
STEPS: tuple[str, ...] = ("anchor", "kick", "fragments", "slides", "gap_fill", "policy")

DEFAULT_PARAMS: dict[str, Any] = {
    "anchor": False,  # E13 (2026-10-08): on hold, so off; the merges were adopted
    "kick": False,
    "fragments": True,
    "slides": True,
    "gap_fill": False,
    "policy": "muscriptor",  # muscriptor (MuScriptor primary, DESIGN 5.2) | and (comparison only)
    "techniques": True,
    "group_s": 0.04,
    "kick_s": 0.03,
    "join_gap_s": 0.03,
    "fragment_s": 0.08,
    "tail_attack_db": 3.0,
    "dead_attack_db": 12.0,
    "glide_step_st": 0.8,
    "glide_min_s": 0.04,
    "cross_step_st": 0.4,  # review 2026-10-08: no pluck-sized step inside the crossing
    "glide_gap_frames": 5,
    "glide_edge_s": 0.06,
    "hold_s": 0.25,
    "hold_slope_st_s": 1.5,
    "hold_std_st": 0.15,
    "fill_min_s": 0.10,
    "fill_tol_st": 0.4,
    "fill_conf": 0.3,
    "vib_min_s": 0.25,
    "vib_hz": [3.0, 9.0],
    "vib_depth_st": 0.15,
}


def arm_params(arm: str) -> dict[str, Any]:
    """The step switches of an E13 arm (thresholds stay the defaults)."""
    base = {**DEFAULT_PARAMS, "anchor": False, "kick": False, "fragments": False, "slides": False,
            "gap_fill": False, "policy": "muscriptor", "techniques": False}
    if arm == "raw":
        return base
    if arm == "anchor":
        return {**base, "anchor": True}
    merge = {**base, "anchor": True, "fragments": True, "slides": True}
    if arm == "anchor_merge":
        return merge
    if arm == "anchor_merge_kick":
        return {**merge, "kick": True}
    if arm == "two_voter":
        return {**merge, "gap_fill": True}
    if arm == "and":
        return {**merge, "policy": "and"}
    raise ValueError(f"unknown E13 arm {arm!r} ({', '.join(ARMS)})")


# ---------------------------------------------------------------------------------------------- audio cues


class Attack:
    """Rise of the high-passed (> 400 Hz: string and pick noise) RMS at a time: the louder of the 25 ms after
    ``t + d`` over the 25 ms ending 10 ms before it, for ``d`` within ±30 ms, in dB."""

    STEP_S = 0.005
    WIN_S = 0.025

    def __init__(self, mono: np.ndarray, sr: int) -> None:
        from scipy.signal import butter, sosfiltfilt

        x = np.asarray(mono, dtype=np.float64).reshape(-1)
        hp = sosfiltfilt(butter(4, 400.0, "highpass", fs=sr, output="sos"), x) if len(x) > 30 else x
        step = max(1, int(round(self.STEP_S * sr)))
        win = max(1, int(round(self.WIN_S * sr)))
        c = np.concatenate([[0.0], np.cumsum(hp * hp)])
        starts = np.arange(0, max(1, len(hp) - win), step)
        self.env = np.sqrt((c[starts + win] - c[starts]) / win + 1e-20)
        self.step_s = step / float(sr)
        self.lag = int(round(0.035 / self.step_s))  # pre window starts 35 ms earlier

    def rise_db(self, t: float) -> float:
        n = len(self.env)
        best = -math.inf
        for d in range(-6, 7):  # ±30 ms in 5 ms steps
            k = int(round(float(t) / self.step_s)) + d
            if 0 <= k - self.lag and k < n:
                best = max(best, 20.0 * math.log10(self.env[k] / self.env[k - self.lag]))
        return best if math.isfinite(best) else 0.0


def kick_onsets(drums_mono: np.ndarray, sr: int) -> np.ndarray:
    """Kick candidates: peaks of the 40-120 Hz spectral flux of the drums stem (librosa's peak picking)."""
    import librosa

    y = np.asarray(drums_mono, dtype=np.float32).reshape(-1)
    if len(y) < 4096:
        return np.zeros(0)
    hop = int(round(0.01 * sr))
    spec = np.abs(librosa.stft(y, n_fft=4096, hop_length=hop, center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    band = (freqs >= 40.0) & (freqs <= 120.0)
    logp = np.log1p(100.0 * spec[band] ** 2)
    flux = np.maximum(0.0, np.diff(logp, axis=1, prepend=logp[:, :1])).sum(axis=0)
    peaks = librosa.onset.onset_detect(onset_envelope=flux, sr=sr, hop_length=hop, units="time", backtrack=False)
    return np.asarray(peaks, dtype=np.float64)


# ------------------------------------------------------------------------------------------------- helpers


@dataclass
class Context:
    track: F0.F0Track
    f0_params: Mapping[str, Any]
    attack: Attack | None = None
    kicks: np.ndarray | None = None
    offset: float = 0.0  # the recording's tuning offset in semitones (``tuning_offset``), taken off every contour


def tuning_offset(notes: Sequence[Mapping[str, Any]], track: F0.F0Track, f0_params: Mapping[str, Any],
                  min_frames: int = 50) -> float:
    """Median of contour - note pitch over the steady frames of the notes where the two are within half a semitone:
    how far the recording sits from equal temperament at A440 as the bass plays it (0 with fewer than
    ``min_frames``). A detuned recording otherwise puts every held note in the middle of an interval (review
    2026-10-08); glides are left out so a slide does not read as detuning."""
    ref = ref_contour(notes, track.n, track.hop_s)
    c = F0.consensus(track, f0_params, ref_midi=ref, min_trackers=1)
    dev = c - ref
    steady = np.zeros(len(c), dtype=bool)  # held pitch: under 0.1 semitone of change over 40 ms (not a glide)
    steady[2:-2] = np.abs(c[4:] - c[:-4]) <= 0.1
    ok = np.isfinite(dev) & (np.abs(dev) <= 0.5) & steady
    return float(np.median(dev[ok])) if int(ok.sum()) >= min_frames else 0.0


def _copy(n: Mapping[str, Any]) -> dict[str, Any]:
    out = {"onset_s": float(n["onset_s"]), "offset_s": float(n["offset_s"]), "pitch": int(n["pitch"]),
           "instrument": n.get("instrument"), "src": n.get("src", "ms")}
    for k in ("f0", "from_pitch", "merged", "tech", "conf"):
        if k in n:
            out[k] = n[k]
    return out


def _order(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(notes, key=lambda n: (n["onset_s"], n["pitch"], n["offset_s"]))


def _tech(n: dict[str, Any]) -> dict[str, Any]:
    t = n.get("tech")
    if not isinstance(t, dict):
        t = {}
        n["tech"] = t
    return t


def ref_contour(notes: Sequence[Mapping[str, Any]], n_frames: int, hop_s: float) -> np.ndarray:
    """Per frame the pitch of the lowest note sounding (NaN where none): the octave reference for the contour."""
    ref = np.full(n_frames, np.nan)
    for x in sorted(notes, key=lambda x: -int(x["pitch"])):  # lower notes written last win
        a = max(0, int(math.floor(float(x["onset_s"]) / hop_s)))
        b = min(n_frames, int(math.ceil(float(x["offset_s"]) / hop_s)) + 1)
        if b > a:
            ref[a:b] = float(x["pitch"])
    return ref


def _hold(c: np.ndarray, hop_s: float, p: Mapping[str, Any]) -> bool:
    """A plateau of ``hold_s`` inside ``c``: a window whose fitted slope and spread are both small."""
    w = max(3, int(round(float(p["hold_s"]) / hop_s)))
    if len(c) < w:
        return False
    t = np.arange(w) * hop_s
    for i in range(0, len(c) - w + 1, 2):
        seg = c[i:i + w]
        if np.sum(np.isfinite(seg)) < 0.8 * w:
            continue
        ok = np.isfinite(seg)
        slope = np.polyfit(t[ok], seg[ok], 1)[0]
        if abs(slope) <= float(p["hold_slope_st_s"]) and float(np.std(seg[ok])) <= float(p["hold_std_st"]):
            return True
    return False


def _span(c: np.ndarray, a: float, b: float, hop_s: float) -> np.ndarray:
    i0 = max(0, int(math.floor(a / hop_s)))
    i1 = min(len(c), int(math.ceil(b / hop_s)) + 1)
    return c[i0:max(i0, i1)]


def _filled(seg: np.ndarray, max_gap: int, max_step: float) -> np.ndarray | None:
    """``seg`` with NaN runs of <= ``max_gap`` frames interpolated; None when an end is missing, a run is longer, or
    the two sides of a run of ``r`` frames differ by more than ``max_step * (1 + r / 2)`` (a jump in a dropout)."""
    ok = np.isfinite(seg)
    if len(seg) < 4 or not ok[0] or not ok[-1]:
        return None
    idx = np.flatnonzero(ok)
    gaps = np.diff(idx) - 1
    if np.any(gaps > max_gap) or np.any(np.abs(np.diff(seg[idx])) > max_step * (1.0 + gaps / 2.0)):
        return None
    return np.interp(np.arange(len(seg)), idx, seg[idx])


def _glide(c: np.ndarray, a: Mapping[str, Any], b: Mapping[str, Any], hop_s: float, p: Mapping[str, Any]) -> bool:
    """The contour (tuning offset removed) slides from note ``a`` into note ``b``: no long gap or jump on the way
    (every 10 ms step <= ``glide_step_st``), and between the last frame on ``a``'s side of the interval (its first
    quarter) and the first frame on ``b``'s side (its last quarter) it spends >= ``glide_min_s`` of measured frames
    crossing the middle half, leaving ``a``'s side and reaching ``b``'s in steps of <= ``cross_step_st``. A plucked or
    hammered change crosses in a frame or two
    (or inside a dropout, where nothing is measured), so a fast chromatic run is not a slide; a held note sitting off
    its pitch (vibrato) never crosses."""
    d = int(b["pitch"]) - int(a["pitch"])
    if d == 0 or float(b["onset_s"]) - float(a["offset_s"]) > float(p["join_gap_s"]):
        return False
    if float(b["onset_s"]) - float(a["onset_s"]) < 0.04:  # starts together: a chord, not a line
        return False
    start = float(a["onset_s"]) + 0.03  # from where a's pitch is measured: a passing note is gliding already
    # to the end of b: a slow slide reaches b's side of the interval well after b's (rounded) onset
    stop = max(float(b["offset_s"]) - 0.02, float(b["onset_s"]) + float(p["glide_edge_s"]))
    raw = _span(c, start, stop, hop_s)
    last = np.flatnonzero(np.isfinite(raw))
    raw = raw[:int(last[-1]) + 1] if len(last) else raw  # b may fade out before its written end
    seg = _filled(raw, int(p["glide_gap_frames"]), float(p["glide_step_st"]))
    if seg is None:
        return False
    pos = (seg - int(a["pitch"])) / d  # 0 = a, 1 = b
    reach = np.flatnonzero(pos >= 0.75)
    if not len(reach):
        return False
    i1 = int(reach[0])
    left = np.flatnonzero(pos[:i1] <= 0.25)
    if not len(left):
        return False
    i0 = int(left[-1])
    # leaving a's side and arriving on b's side are glides too, not a pluck from a wobble near the middle (one noisy
    # frame inside a long crossing is no jump: seisyun complex's slide has a 0.61-semitone dropout step at 5.1 s)
    if max(abs(float(seg[i0 + 1] - seg[i0])), abs(float(seg[i1] - seg[i1 - 1]))) > float(p["cross_step_st"]):
        return False
    measured = int(np.sum(np.isfinite(raw[i0 + 1:i1])))
    return measured * hop_s >= float(p["glide_min_s"])


def _still_moving(c: np.ndarray, n: Mapping[str, Any], direction: float, hop_s: float) -> bool:
    """The contour inside ``n`` keeps going in ``direction`` (>= 2 semitones per second) or fades out."""
    seg = _span(c, float(n["onset_s"]) + 0.01, float(n["offset_s"]), hop_s)
    ok = np.isfinite(seg)
    if ok.sum() < 4:
        return True
    t = np.arange(len(seg))[ok] * hop_s
    return direction * float(np.polyfit(t, seg[ok], 1)[0]) >= 2.0


def _inner(c: np.ndarray, n: Mapping[str, Any], hop_s: float) -> np.ndarray:
    a = int(math.ceil((float(n["onset_s"]) + 0.02) / hop_s))
    b = int(math.floor((float(n["offset_s"]) - 0.02) / hop_s)) + 1
    return c[max(0, a):max(max(0, a), min(len(c), b))]


# --------------------------------------------------------------------------------------------------- steps


def _anchor(notes: list[dict[str, Any]], ctx: Context, p: Mapping[str, Any], rep: dict[str, Any],
            *, move: bool) -> list[dict[str, Any]]:
    counts = {a: 0 for a in F0.VOTE_ACTIONS}
    for n in notes:
        ests = F0.note_estimates(ctx.track, n["onset_s"], n["offset_s"], ctx.f0_params)
        v = F0.vote(n["pitch"], {k: e - ctx.offset for k, e in ests.items()}, ctx.f0_params)
        n["_ests"] = ests
        n["f0"] = v.action
        n["_target"] = v.target
        counts[v.action] += 1
        if move and v.action in ("octave", "semitone"):
            n.setdefault("from_pitch", n["pitch"])
            n["pitch"] = v.pitch
    rep["vote"] = counts
    if not move:
        return notes
    # simultaneous duplicates / harmonics of an F0-confirmed note
    out: list[dict[str, Any]] = []
    dropped = 0
    notes = _order(notes)
    i = 0
    while i < len(notes):
        j = i + 1
        while j < len(notes) and notes[j]["onset_s"] - notes[i]["onset_s"] <= float(p["group_s"]):
            j += 1
        grp = notes[i:j]
        sure = [x for x in grp if x.get("_target") is not None and x["_target"] == x["pitch"]]
        if len(grp) > 1 and sure:
            base = min(x["pitch"] for x in sure)
            keep_one = min(sure, key=lambda x: (x["pitch"], -(x["offset_s"] - x["onset_s"])))
            for x in grp:
                if x is keep_one or (x.get("_target") == x["pitch"] and x["pitch"] != base):
                    out.append(x)
                elif x["pitch"] - base in (0, 12, 19, 24):
                    dropped += 1
                    if x["pitch"] == keep_one["pitch"]:  # the same note written twice: keep its full length
                        keep_one["offset_s"] = max(keep_one["offset_s"], x["offset_s"])
                else:
                    out.append(x)
        else:
            out.extend(grp)
        i = j
    rep["dupes_dropped"] = dropped
    return out


def _kick(notes: list[dict[str, Any]], ctx: Context, p: Mapping[str, Any], rep: dict[str, Any]) -> list[dict[str, Any]]:
    kicks = ctx.kicks if ctx.kicks is not None else np.zeros(0)
    rep["kicks"] = int(len(kicks))
    if not len(kicks):
        rep["kick_dropped"] = 0
        return notes
    out, dropped = [], 0
    for n in notes:
        k = int(np.searchsorted(kicks, n["onset_s"]))
        near = min((abs(kicks[i] - n["onset_s"]) for i in (k - 1, k) if 0 <= i < len(kicks)), default=math.inf)
        if near <= float(p["kick_s"]) and not n.get("_ests"):
            dropped += 1
            continue
        out.append(n)
    rep["kick_dropped"] = dropped
    return out


def _fragments(notes: list[dict[str, Any]], ctx: Context, p: Mapping[str, Any], rep: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    merged = 0
    for n in _order(notes):
        prev = next((x for x in reversed(out) if x["pitch"] == n["pitch"]), None)
        if prev is not None and -0.005 <= n["onset_s"] - prev["offset_s"] <= float(p["join_gap_s"]):
            # a re-plucked repeat has an attack of its own: only a split without one is joined (review 2026-10-08:
            # fast 16ths on one pitch were joined into one note)
            no_attack = ctx.attack is None or ctx.attack.rise_db(n["onset_s"]) < float(p["tail_attack_db"])
            short = n["offset_s"] - n["onset_s"] < float(p["fragment_s"]) - 1e-6
            if no_attack and (short or not n.get("_ests")):
                prev["offset_s"] = max(prev["offset_s"], n["offset_s"])
                prev["merged"] = int(prev.get("merged", 0)) + 1 + int(n.get("merged", 0))
                merged += 1
                continue
        out.append(n)
    rep["fragments_merged"] = merged
    return out


def _slides(notes: list[dict[str, Any]], ctx: Context, p: Mapping[str, Any], rep: dict[str, Any]) -> list[dict[str, Any]]:
    tr = ctx.track
    notes = _order(notes)
    c = F0.running_median(F0.consensus(tr, ctx.f0_params, ref_midi=ref_contour(notes, tr.n, tr.hop_s),
                                       min_trackers=1)) - ctx.offset
    # a single tracker is enough for the shape of a glide the notes already outline (pyin / PESTO are often silent
    # or an octave off on bass stems; the reference octave comes from the notes); the 50 ms running median keeps the
    # switching between trackers that answer a little apart from looking like jumps
    mono = notes
    joins = [_glide(c, a, b, tr.hop_s, p) for a, b in zip(mono, mono[1:])]
    holds = [_hold(_inner(c, n, tr.hop_s), tr.hop_s, p) for n in mono]
    out: list[dict[str, Any]] = []
    chains = passing = marked = outs = 0
    i = 0
    while i < len(mono):
        a = mono[i]
        if i == len(mono) - 1 or not joins[i]:
            out.append(a)
            i += 1
            continue
        direction = math.copysign(1.0, mono[i + 1]["pitch"] - a["pitch"])
        j = i + 1
        # extend through notes that only pass: no plateau, the glide goes on in the same direction
        while (j < len(mono) - 1 and not holds[j] and joins[j]
               and math.copysign(1.0, mono[j + 1]["pitch"] - mono[j]["pitch"]) == direction):
            j += 1
        dest = mono[j]
        inner = mono[i + 1:j]
        nxt = mono[j + 1] if j + 1 < len(mono) else None
        fades = (not holds[j] and dest["offset_s"] - dest["onset_s"] < float(p["hold_s"])
                 and (nxt is None or nxt["onset_s"] - dest["offset_s"] >= 0.1)
                 and _still_moving(c, dest, direction, tr.hop_s))
        if fades:  # the glide never settles: the first note slides out
            _tech(a)["slide"] = "out_down" if direction < 0 else "out_up"
            a["offset_s"] = max(a["offset_s"], dest["offset_s"])
            a["merged"] = int(a.get("merged", 0)) + len(inner) + 1
            passing += len(inner) + 1
            outs += 1
            out.append(a)
            i = j + 1
            continue
        if inner:
            chains += 1
            passing += len(inner)
            a["merged"] = int(a.get("merged", 0)) + len(inner)
            a["offset_s"] = max(a["offset_s"], dest["onset_s"])
        else:
            marked += 1
        _tech(a)["slide"] = "legato"
        out.append(a)
        i = j  # the destination stays and may start the next glide
    rep["slides"] = {"chains": chains, "passing_removed": passing, "pairs_marked": marked, "slide_outs": outs}
    return out


def _gap_fill(notes: list[dict[str, Any]], ctx: Context, p: Mapping[str, Any], rep: dict[str, Any]) -> list[dict[str, Any]]:
    tr = ctx.track
    c = F0.consensus(tr, ctx.f0_params, min_trackers=2) - ctx.offset
    busy = np.zeros(tr.n, dtype=bool)
    for n in notes:
        a = max(0, int(math.floor(n["onset_s"] / tr.hop_s)) - 2)
        b = min(tr.n, int(math.ceil(n["offset_s"] / tr.hop_s)) + 3)
        busy[a:b] = True
    need = max(2, int(round(float(p["fill_min_s"]) / tr.hop_s)))
    tol = float(p["fill_tol_st"])
    added: list[dict[str, Any]] = []
    i = 0
    while i < tr.n:
        if busy[i] or not np.isfinite(c[i]):
            i += 1
            continue
        j = i
        while j < tr.n and not busy[j] and np.isfinite(c[j]) and abs(c[j] - c[i]) <= 2 * tol:
            j += 1
        seg = c[i:j]
        if j - i >= need:
            med = float(np.median(seg))
            if float(np.mean(np.abs(seg - med) <= tol)) >= 0.9:
                added.append({"onset_s": round(max(0.0, i * tr.hop_s - 0.02), 6), "offset_s": round(j * tr.hop_s, 6),
                              "pitch": int(round(med)), "instrument": "electric_bass", "src": "f0",
                              "conf": float(p["fill_conf"]), "f0": "confirmed", "_target": int(round(med)),
                              "_ests": {"consensus": med}})
        i = max(j, i + 1)
    rep["gap_filled"] = len(added)
    return _order(notes + added)


def _and_policy(notes: list[dict[str, Any]], rep: dict[str, Any]) -> list[dict[str, Any]]:
    out = [n for n in notes if n.get("_target") is not None and n["_target"] == n["pitch"]]
    rep["and_dropped"] = len(notes) - len(out)
    return out


def _techniques(notes: list[dict[str, Any]], ctx: Context, p: Mapping[str, Any], rep: dict[str, Any]) -> None:
    tr = ctx.track
    c = F0.running_median(F0.consensus(tr, ctx.f0_params, ref_midi=ref_contour(notes, tr.n, tr.hop_s),
                                       min_trackers=1), width=3, min_count=2)
    lo, hi = (float(x) for x in p["vib_hz"])
    vib = dead = 0
    ordered = _order(list(notes))
    # a slide's end settles onto its pitch: that wobble is not vibrato
    landed = {id(b) for a, b in zip(ordered, ordered[1:]) if (a.get("tech") or {}).get("slide") == "legato"}
    for n in ordered:
        if n.get("src") == "f0":
            continue
        if (n["offset_s"] - n["onset_s"] >= float(p["vib_min_s"]) and "slide" not in (n.get("tech") or {})
                and id(n) not in landed):
            seg = _inner(c, {**n, "onset_s": n["onset_s"] + 0.06}, tr.hop_s)
            if len(seg) >= 10 and np.mean(np.isfinite(seg)) >= 0.9:
                s = seg[np.isfinite(seg)]
                t = np.arange(len(s)) * tr.hop_s
                resid = s - np.polyval(np.polyfit(t, s, 2), t)
                sign = np.sign(resid)
                sign = sign[sign != 0]
                crossings = int(np.sum(sign[1:] != sign[:-1]))
                dur = len(s) * tr.hop_s
                rate = crossings / 2.0 / dur if dur > 0 else 0.0
                depth = float(np.percentile(resid, 95) - np.percentile(resid, 5)) / 2.0
                if crossings >= 4 and lo <= rate <= hi and depth >= float(p["vib_depth_st"]):
                    _tech(n)["vibrato"] = True
                    vib += 1
        if not n.get("_ests") and ctx.attack is not None and ctx.attack.rise_db(n["onset_s"]) >= float(p["dead_attack_db"]):
            _tech(n)["dead"] = True
            dead += 1
    rep["techniques"] = {"vibrato": vib, "dead": dead}


# ----------------------------------------------------------------------------------------------------- run


def clean(notes: Sequence[Mapping[str, Any]], track: F0.F0Track, params: Mapping[str, Any] | None = None, *,
          f0_params: Mapping[str, Any] | None = None, attack: Attack | None = None,
          kicks: np.ndarray | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(cleaned notes in canonical order, report)."""
    p = {**DEFAULT_PARAMS, **dict(params or {})}
    ctx = Context(track=track, f0_params=f0_params or F0.DEFAULT_PARAMS, attack=attack, kicks=kicks)
    work = _order([_copy(n) for n in notes])
    ctx.offset = tuning_offset(work, track, ctx.f0_params)
    rep: dict[str, Any] = {"in": len(work), "params": {k: p[k] for k in STEPS + ("techniques",)},
                           "tuning_offset_st": round(ctx.offset, 3)}
    work = _anchor(work, ctx, p, rep, move=bool(p["anchor"]))
    rep["moved"] = sum(1 for n in work if "from_pitch" in n)
    if p["kick"]:
        work = _kick(work, ctx, p, rep)
    if p["fragments"]:
        work = _fragments(work, ctx, p, rep)
    if p["slides"]:
        work = _slides(work, ctx, p, rep)
    if p["gap_fill"]:
        work = _gap_fill(work, ctx, p, rep)
    if p["policy"] == "and":
        work = _and_policy(work, rep)
    elif p["policy"] != "muscriptor":
        raise ValueError(f"unknown bass policy {p['policy']!r}")
    if p["techniques"]:
        _techniques(work, ctx, p, rep)
    out = []
    for n in _order(work):
        row = {k: v for k, v in n.items() if not k.startswith("_")}
        row["onset_s"], row["offset_s"] = round(float(row["onset_s"]), 6), round(float(row["offset_s"]), 6)
        if not row.get("tech"):
            row.pop("tech", None)
        out.append(row)
    rep["out"] = len(out)
    return out, rep
