"""Symbolic layers of the merged guitar notes (DESIGN 4.3 (e), 4.7 "기호 역할 사전"; M7a).

One transcription of the guitar stem holds every part. What the notes themselves say about the hands that play
them:

- onset clusters: notes whose onsets lie within ``CLUSTER_S`` of the cluster's first note are one strum (DESIGN 4.6
  must-link: "한 스트럼 안의 음, onset 약 30 ms 이내");
- a cluster's top note at least ``PEEL_SEMITONES`` above the next note is peeled off as a separate line (DESIGN 4.7
  "코드보다 10반음 이상 높은 윗음은 떼어 낸다", as TabForge ``split_lead_rhythm`` does);
- ``chord``: the rest of a cluster when it has 3+ notes, or a dyad a fourth or a fifth apart (a power chord);
- ``lead`` layer: peeled tops, and non-chord notes that start while a chord is still sounding (``LEAD_AFTER_S`` after
  its strum or later, before its end) and above its top note - two things at once that one strum does not make;
- role score ``r``: chord -1, lead layer +1, any other note 0.

``tabforge_split`` is the symbolic-only baseline of DESIGN 8.5 ("기호 분할만: TabForge ``split_lead_rhythm`` 방식"),
our implementation of its published description (the code was not read): chord notes go to the rhythm line, peeled
tops to the lead line, every other note by its ``window_s`` window - rhythm when the window's chord share, chord
notes weighted by ``chord_weight``, reaches ``rhythm_threshold``, else lead.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from typing import Any

CLUSTER_S = 0.035
PEEL_SEMITONES = 10
POWER_INTERVALS = frozenset({5, 7})
LEAD_AFTER_S = 0.08  # a note this soon after a strum belongs to the strum's own attack (a late string), not a line
MAX_CHORD_S = 8.0  # how far back a still-sounding chord is looked for
TABFORGE = {"window_s": 1.0, "rhythm_threshold": 0.6, "chord_weight": 2.0}


def clusters(notes: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Note indices grouped by onset (module doc), in time order."""
    order = sorted(range(len(notes)), key=lambda i: (float(notes[i]["onset_s"]), int(notes[i]["pitch"])))
    out: list[list[int]] = []
    cur: list[int] = []
    t0 = 0.0
    for i in order:
        t = float(notes[i]["onset_s"])
        if cur and t - t0 > CLUSTER_S:
            out.append(cur)
            cur = []
        if not cur:
            t0 = t
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def note_features(notes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per note: ``cluster`` (index into ``clusters``), ``size``, ``chord``, ``peel``, ``lead`` and the role score
    ``r`` (module doc)."""
    feats: list[dict[str, Any]] = [{"cluster": -1, "size": 1, "chord": False, "peel": False, "lead": False, "r": 0.0}
                                   for _ in notes]
    chords: list[tuple[float, float, int]] = []  # (strum onset, end, top pitch) of the chord clusters
    for cid, c in enumerate(clusters(notes)):
        by_p = sorted(c, key=lambda i: (int(notes[i]["pitch"]), i))
        body = by_p
        if len(by_p) >= 2 and int(notes[by_p[-1]]["pitch"]) - int(notes[by_p[-2]]["pitch"]) >= PEEL_SEMITONES:
            feats[by_p[-1]]["peel"] = True
            body = by_p[:-1]
        is_chord = len(body) >= 3 or (
            len(body) == 2 and int(notes[body[1]]["pitch"]) - int(notes[body[0]]["pitch"]) in POWER_INTERVALS)
        for i in c:
            feats[i]["cluster"] = cid
            feats[i]["size"] = len(c)
        if is_chord:
            for i in body:
                feats[i]["chord"] = True
            chords.append((min(float(notes[i]["onset_s"]) for i in body),
                           max(float(notes[i]["offset_s"]) for i in body),
                           max(int(notes[i]["pitch"]) for i in body)))
    chords.sort()
    starts = [c[0] for c in chords]
    for i, n in enumerate(notes):
        f = feats[i]
        if f["chord"]:
            f["r"] = -1.0
            continue
        if f["peel"]:
            f["lead"] = True
        else:
            t, pitch = float(n["onset_s"]), int(n["pitch"])
            k = bisect.bisect_right(starts, t - LEAD_AFTER_S)
            j = bisect.bisect_left(starts, t - MAX_CHORD_S)
            f["lead"] = any(chords[m][1] > t and pitch > chords[m][2] for m in range(j, k))
        f["r"] = 1.0 if f["lead"] else 0.0
    return feats


def tabforge_split(notes: Sequence[Mapping[str, Any]], feats: Sequence[Mapping[str, Any]] | None = None,
                   params: Mapping[str, float] | None = None) -> list[str]:
    """The symbolic-only baseline (module doc) -> ``"rhythm"`` / ``"lead"`` per note."""
    p = {**TABFORGE, **(params or {})}
    feats = feats if feats is not None else note_features(notes)
    win = float(p["window_s"])
    n_chord: dict[int, int] = {}
    n_other: dict[int, int] = {}
    for n, f in zip(notes, feats):
        w = int(math.floor(float(n["onset_s"]) / win))
        if f["chord"]:
            n_chord[w] = n_chord.get(w, 0) + 1
        else:
            n_other[w] = n_other.get(w, 0) + 1
    cw = float(p["chord_weight"])
    out = []
    for n, f in zip(notes, feats):
        if f["chord"]:
            out.append("rhythm")
        elif f["peel"]:
            out.append("lead")
        else:
            w = int(math.floor(float(n["onset_s"]) / win))
            c, o = n_chord.get(w, 0), n_other.get(w, 0)
            score = cw * c / (cw * c + o) if (c or o) else 0.0
            out.append("rhythm" if score >= float(p["rhythm_threshold"]) else "lead")
    return out
