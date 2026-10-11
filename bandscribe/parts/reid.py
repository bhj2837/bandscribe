"""How many guitar parts play at once: streams per window and the robust part count (DESIGN 4.4; M7a v0).

A window's streams (DESIGN 4.4 step 2, "음이 8개 이상이면서 서로 일치하지 않는 동시 스트림 수"), each kind of evidence
counted on its own and the window taking the largest:

- ``spatial``: in a window the router labels ``R3`` (parts at different stereo positions), the notes on each side
  by their own ILD (``bandscribe.router.notesig``; E22b split them by the sign): two streams when both sides have
  ``n_min`` notes at ``|ILD| >= side_db``;
- ``class``: two MuScriptor guitar classes with ``n_min`` notes each, half of the second one's notes in
  ``cooccur_s`` cells that hold the first one too (the two sound together, not one after the other);
- ``symbolic``: ``n_min`` chord notes and ``n_min`` lead-layer notes (``bandscribe.parts.layers``);
- otherwise one stream when the window has ``n_min`` notes, else none.

A double (router ``R2``) and an ADT (``R1``) are one part by R1 / R2 of the output contract and add no spatial
stream. The count of a song (``k_low``) is 2 only when two streams hold in ``min_windows`` windows that span
``min_sections`` sections (all of them when the song has fewer; DESIGN: "3개 이상의 창과 2개 이상의 구간"), else 1
when there is any guitar note. v0 counts at most two parts by itself; more come only from the user's ``--parts``.
The two hypotheses shown with it get a logistic of the support margin (not a calibrated probability).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_PARAMS: dict[str, Any] = {
    "n_min": 8,
    "min_windows": 3,
    "min_sections": 2,
    "side_db": 0.0,
    "use_class": True,
    "use_symbolic": True,
    "cooccur_s": 1.0,
}
KINDS = ("spatial", "class", "symbolic")


def _ild(sig: Mapping[str, Any] | None) -> float | None:
    v = (sig or {}).get("ild_db")
    return None if v is None or not math.isfinite(float(v)) else float(v)


def window_streams(idx: Sequence[int], notes: Sequence[Mapping[str, Any]], feats: Sequence[Mapping[str, Any]],
                   sigs: Sequence[Mapping[str, Any]], label: str | None, params: Mapping[str, Any]) -> dict[str, Any]:
    """Streams of one window (module doc) -> {notes, spatial, class, symbolic, c}."""
    p = {**DEFAULT_PARAMS, **params}
    nmin = int(p["n_min"])
    out: dict[str, Any] = {"notes": len(idx), "spatial": 0, "class": 0, "symbolic": 0, "c": 0}
    if len(idx) < nmin:
        return out
    out["c"] = 1
    if label == "R3":
        side = float(p["side_db"])
        ilds = [v for v in (_ild(sigs[i]) for i in idx) if v is not None]
        nl = sum(1 for v in ilds if (v >= side if side > 0 else v >= 0))
        nr = sum(1 for v in ilds if (v <= -side if side > 0 else v < 0))
        out["spatial"] = 2 if min(nl, nr) >= nmin else 1
    if p["use_class"]:
        counts: dict[str, int] = {}
        for i in idx:
            c = str(notes[i].get("instrument"))
            counts[c] = counts.get(c, 0) + 1
        top = sorted(counts, key=lambda c: (-counts[c], c))
        out["class"] = 1
        if len(top) >= 2 and counts[top[1]] >= nmin:
            cell = float(p["cooccur_s"])
            cells1 = {int(math.floor(float(notes[i]["onset_s"]) / cell)) for i in idx
                      if str(notes[i].get("instrument")) == top[0]}
            together = sum(1 for i in idx if str(notes[i].get("instrument")) == top[1]
                           and int(math.floor(float(notes[i]["onset_s"]) / cell)) in cells1)
            out["class"] = 2 if together >= counts[top[1]] / 2 else 1
    if p["use_symbolic"]:
        n_lead = sum(1 for i in idx if feats[i]["lead"])
        n_chord = sum(1 for i in idx if feats[i]["chord"])
        out["symbolic"] = 2 if min(n_lead, n_chord) >= nmin else 1
    out["c"] = max(1, out["spatial"], out["class"], out["symbolic"])
    return out


def window_index(notes: Sequence[Mapping[str, Any]], windows: Sequence[Mapping[str, Any]]) -> list[int | None]:
    """The window (index into ``windows``, sorted by start) each note's onset falls in, or None."""
    starts = [float(w["start_s"]) for w in windows]
    out: list[int | None] = []
    for n in notes:
        t = float(n["onset_s"])
        k = bisect.bisect_right(starts, t) - 1
        out.append(k if k >= 0 and t < float(windows[k]["end_s"]) else None)
    return out


def estimate(notes: Sequence[Mapping[str, Any]], feats: Sequence[Mapping[str, Any]],
             sigs: Sequence[Mapping[str, Any]], windows: Sequence[Mapping[str, Any]],
             params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The song's part count (module doc) -> {k, k_low, evidence, support, hypotheses, windows}. ``windows`` are
    sorted by start and carry ``label`` (router) and ``section``."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    wins = sorted(windows, key=lambda w: float(w["start_s"]))
    of = window_index(notes, wins)
    members: list[list[int]] = [[] for _ in wins]
    for i, k in enumerate(of):
        if k is not None:
            members[k].append(i)
    rows = []
    for w, idx in zip(wins, members):
        r = window_streams(idx, notes, feats, sigs, w.get("label"), p)
        rows.append({"start_s": float(w["start_s"]), "end_s": float(w["end_s"]), "label": w.get("label"),
                     "section": w.get("section"), **r})
    sections = {r["section"] for r in rows}
    need_sec = min(int(p["min_sections"]), len(sections)) if sections else 0
    two = [r for r in rows if r["c"] >= 2]
    sec_two = len({r["section"] for r in two})
    support = {kind: sum(1 for r in rows if r[kind] >= 2) for kind in KINDS}
    margin = len(two) - int(p["min_windows"]) + 0.5
    if sec_two < need_sec:
        margin = min(margin, -0.5)
    p2 = 1.0 / (1.0 + math.exp(-margin)) if two else 0.0
    k_low = 2 if p2 > 0.5 else (1 if notes else 0)
    evidence = None
    if any(support.values()):
        evidence = max(KINDS, key=lambda kd: (support[kd], -KINDS.index(kd)))
    hyps = []
    if notes:
        hyps = sorted([{"k": 2, "p": round(p2, 4)}, {"k": 1, "p": round(1.0 - p2, 4)}], key=lambda h: -h["p"])
    return {"k": k_low, "k_low": k_low, "evidence": evidence, "support": support, "windows_two": len(two),
            "sections_two": sec_two, "hypotheses": hyps, "windows": rows}
