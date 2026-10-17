"""S8 identity tracking v1 (M7b, DESIGN 4.6) on top of the v0 per-note posteriors (``diarize``).

v0 decides every note on its own: a part's notes in a window without stereo evidence go by role and class only,
so a window can come out relabelled as a whole, or two windows of one stream can disagree. The tracker re-decides
whole windows:

1. **Layers:** in each router window, the notes v0 gave Line j (posterior >= tau) form layer j, with a signature -
   median pitch, chord / lead-layer shares, note density, mean ILD in R3 windows, class shares - when it has
   ``min_layer`` notes or more.
2. **Tracklet links:** the layers of a window and of the previous window with layers are matched by signature
   distance (Hungarian); a pair closer than ``link_max`` is a link of strength 1 - distance / ``link_max``.
3. **Re-identification + smoothing:** each window chooses a relabelling of its layers, a permutation ``pi`` of the K
   Lines (K <= 4: at most 24). Emission: ``w_post`` x the sum over its layers of the mean log posterior of the
   relabelled Line (the identity is v0's choice), posteriors clipped to [``p_floor``, 1 - ``p_floor``] (v0's are
   over-confident, M7a k3: unclipped, a relabelled layer costs ~14 and nothing can outweigh it). Transition: ``w_link`` x the strength of every link whose two
   layers end in different Lines, times ``boundary`` across a section boundary (a role change belongs at one,
   DESIGN 4.6 step 4). Viterbi over the windows.
4. **Repeated sections:** windows paired by the repeat map (most of a window's bars that have a match match bars of
   the other; the repeat map matches only some bars) add
   ``w_rep`` x the strength of every disagreeing layer pair; ``sweeps`` passes of iterated conditional modes after
   the Viterbi pass.

A note keeps v0's Line through its window's permutation; Unassigned notes and notes outside every window stay
as v0 left them. With K < 2 there is nothing to track and v0's result comes back unchanged.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_PARAMS: dict[str, Any] = {"min_layer": 4, "w_post": 1.0, "w_link": 2.0, "boundary": 0.5, "w_rep": 1.0,
                                  "link_max": 3.0, "sweeps": 2, "p_floor": 0.0}
EPS = 1e-6


def window_of(notes: Sequence[Mapping[str, Any]], wins: Sequence[Mapping[str, Any]]) -> list[int | None]:
    """Index into ``wins`` (sorted, non-overlapping) of the window each note starts in, or None."""
    starts = [float(w["start_s"]) for w in wins]
    out: list[int | None] = []
    for n in notes:
        t = float(n["onset_s"])
        j = int(np.searchsorted(starts, t, side="right")) - 1
        out.append(j if 0 <= j < len(wins) and t < float(wins[j]["end_s"]) else None)
    return out


def signature(idx: Sequence[int], notes: Sequence[Mapping[str, Any]], feats: Sequence[Mapping[str, Any]],
              sigs: Sequence[Mapping[str, Any]], r3: bool, span_s: float) -> dict[str, Any]:
    pitch = [int(notes[i]["pitch"]) for i in idx]
    ild = [float(sigs[i]["ild_db"]) for i in idx if r3 and sigs[i].get("ild_db") is not None]
    cls = Counter(str(notes[i].get("instrument")) for i in idx)
    n = len(idx)
    return {"pitch": float(np.median(pitch)), "chord": sum(1 for i in idx if feats[i].get("chord")) / n,
            "lead": sum(1 for i in idx if feats[i].get("lead")) / n, "density": n / max(span_s, 1.0),
            "ild": float(np.mean(ild)) if ild else None, "cls": {c: v / n for c, v in cls.items()}}


def distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Signature distance: a fifth of register ~ 1, a role flip ~ 2, double density ~ 0.7, opposite sides ~ 2."""
    d = abs(a["pitch"] - b["pitch"]) / 7.0 + 2.0 * abs(a["chord"] - b["chord"]) + 2.0 * abs(a["lead"] - b["lead"])
    d += abs(math.log((a["density"] + EPS) / (b["density"] + EPS)))
    if a["ild"] is not None and b["ild"] is not None and a["ild"] * b["ild"] < 0:
        d += min(2.0, abs(a["ild"] - b["ild"]) / 3.0)
    keys = set(a["cls"]) | set(b["cls"])
    d += 0.5 * sum(abs(a["cls"].get(k, 0.0) - b["cls"].get(k, 0.0)) for k in keys)
    return float(d)


def links(la: Mapping[int, Mapping[str, Any]], lb: Mapping[int, Mapping[str, Any]], link_max: float
          ) -> list[tuple[int, int, float]]:
    """(layer of a, layer of b, strength) pairs of the Hungarian matching closer than ``link_max``."""
    from scipy.optimize import linear_sum_assignment

    ka, kb = sorted(la), sorted(lb)
    if not ka or not kb:
        return []
    cost = np.array([[distance(la[x], lb[y]) for y in kb] for x in ka])
    rows, cols = linear_sum_assignment(cost)
    return [(ka[r], kb[c], 1.0 - float(cost[r, c]) / link_max) for r, c in zip(rows, cols) if cost[r, c] < link_max]


def repeat_pairs(wins: Sequence[Mapping[str, Any]], repeat: Mapping[str, Any] | None, min_score: float = 0.5
                 ) -> list[tuple[int, int]]:
    """Window pairs (i < j) where most of window i's matched bars have their best repeat-map match in window j
    (bars without a match do not vote)."""
    if not repeat:
        return []
    win_of_bar: dict[int, int] = {}
    for wi, w in enumerate(wins):
        if w.get("start_bar") is None:
            continue
        nb = int(w["end_bar"]) - int(w["start_bar"]) if w.get("end_bar") is not None else int(w.get("n_bars", 4))
        for b in range(int(w["start_bar"]), int(w["start_bar"]) + nb):
            win_of_bar.setdefault(b, wi)
    votes: Counter = Counter()
    nbars: Counter = Counter()
    for bm in repeat.get("bars") or []:
        wi = win_of_bar.get(int(bm["bar"]))
        best = next((m for m in bm.get("matches") or [] if float(m.get("score", 0)) >= min_score), None)
        if wi is None or best is None:
            continue
        nbars[wi] += 1
        wj = win_of_bar.get(int(best["bar"]))
        if wj is not None and wj != wi:
            votes[(wi, wj)] += 1
    out = set()
    for (wi, wj), v in votes.items():
        if v * 2 > nbars[wi]:
            out.add((min(wi, wj), max(wi, wj)))
    return sorted(out)


def track(notes: Sequence[Mapping[str, Any]], feats: Sequence[Mapping[str, Any]], sigs: Sequence[Mapping[str, Any]],
          windows: Sequence[Mapping[str, Any]], post: np.ndarray, line0: Sequence[int | None], *,
          repeat: Mapping[str, Any] | None = None, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """-> {line (1..K or None per note), p, perms (per window, as v0 Line -> Line), changed (notes relabelled),
    pairs (repeat window pairs), links (per window: links to the previous layered window)}."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    k = int(post.shape[1]) if post.ndim == 2 else 0
    line = list(line0)
    base = {"line": line, "p": [float(post[i, l - 1]) if l else 0.0 for i, l in enumerate(line)] if k else [],
            "perms": [], "changed": 0, "pairs": [], "links": []}
    if k < 2 or not notes:
        return base
    wins = sorted(windows, key=lambda w: float(w["start_s"]))
    of = window_of(notes, wins)
    members: list[dict[int, list[int]]] = [{} for _ in wins]
    for i, (w, l) in enumerate(zip(of, line0)):
        if w is not None and l:
            members[w].setdefault(int(l) - 1, []).append(i)
    lay: list[dict[int, dict[str, Any]]] = []
    for wi, w in enumerate(wins):
        span = float(w["end_s"]) - float(w["start_s"])
        r3 = w.get("label") == "R3"
        lay.append({j: signature(ix, notes, feats, sigs, r3, span) for j, ix in members[wi].items()
                    if len(ix) >= int(p["min_layer"])})
    perms = list(itertools.permutations(range(k)))
    fl = max(float(p["p_floor"]), EPS)
    logp = np.log(np.clip(post, fl, 1.0 - fl if fl < 0.5 else 1.0))

    def emit(wi: int, pi: tuple[int, ...]) -> float:
        return float(p["w_post"]) * sum(float(np.mean(logp[members[wi][j], pi[j]])) for j in lay[wi])

    layered = [wi for wi in range(len(wins)) if lay[wi]]
    prev_links: dict[int, tuple[int, list[tuple[int, int, float]]]] = {}
    for a, b in zip(layered, layered[1:]):
        bd = float(p["boundary"]) if wins[a].get("section") != wins[b].get("section") else 1.0
        prev_links[b] = (a, [(x, y, s * bd) for x, y, s in links(lay[a], lay[b], float(p["link_max"]))])

    def fixed(pi: tuple[int, ...]) -> int:  # ties: the permutation that moves the fewest Lines
        return sum(1 for i, x in enumerate(pi) if i == x)

    def trans(lk: Sequence[tuple[int, int, float]], pa: tuple[int, ...], pb: tuple[int, ...]) -> float:
        return float(p["w_link"]) * sum(s for x, y, s in lk if pa[x] != pb[y])

    # Viterbi over the layered windows
    choice: dict[int, tuple[int, ...]] = {}
    if layered:
        score = {pi: emit(layered[0], pi) for pi in perms}
        back: list[dict[tuple[int, ...], tuple[int, ...]]] = []
        for wb in layered[1:]:
            _wa, lk = prev_links[wb]
            new, bk = {}, {}
            for pb in perms:
                best = max(perms, key=lambda pa: (score[pa] - trans(lk, pa, pb), fixed(pa)))
                new[pb] = score[best] - trans(lk, best, pb) + emit(wb, pb)
                bk[pb] = best
            score, back = new, back + [bk]
        last = max(perms, key=lambda pi: (score[pi], fixed(pi)))
        path = [last]
        for bk in reversed(back):
            path.append(bk[path[-1]])
        path.reverse()
        choice = dict(zip(layered, path))
    # repeated sections: iterated conditional modes over the paired windows
    pairs = [(a, b) for a, b in repeat_pairs(wins, repeat) if lay[a] and lay[b]]
    rep_links = {(a, b): links(lay[a], lay[b], float(p["link_max"])) for a, b in pairs}
    nxt = {a: b for b, (a, _lk) in prev_links.items()}
    for _ in range(int(p["sweeps"])) if pairs else ():
        for wi in sorted({x for pr in pairs for x in pr}):
            def total(pi: tuple[int, ...], wi: int = wi) -> float:
                s = emit(wi, pi)
                if wi in prev_links:
                    a, lk = prev_links[wi]
                    s -= trans(lk, choice[a], pi)
                if wi in nxt:
                    b = nxt[wi]
                    s -= trans(prev_links[b][1], pi, choice[b])
                for (a, b), lk in rep_links.items():
                    if a == wi:
                        s -= float(p["w_rep"]) * sum(st for x, y, st in lk if pi[x] != choice[b][y])
                    elif b == wi:
                        s -= float(p["w_rep"]) * sum(st for x, y, st in lk if choice[a][x] != pi[y])
                return s
            choice[wi] = max(perms, key=lambda pi: (total(pi), pi == choice[wi], fixed(pi)))
    changed = 0
    for i, (w, l) in enumerate(zip(of, line0)):
        if w is None or not l or w not in choice:
            continue
        nl = choice[w][int(l) - 1] + 1
        if nl != l:
            changed += 1
        line[i] = nl
    return {"line": line, "p": [float(post[i, l - 1]) if l else 0.0 for i, l in enumerate(line)],
            "perms": [[int(x) + 1 for x in choice.get(wi, tuple(range(k)))] for wi in range(len(wins))],
            "changed": changed, "pairs": pairs,
            "links": {b: [(x + 1, y + 1, round(s, 3)) for x, y, s in lk] for b, (_a, lk) in prev_links.items()}}


def repeat_links(notes: Sequence[Mapping[str, Any]], bars: Sequence[Mapping[str, Any]],
                 repeat: Mapping[str, Any] | None, *, min_score: float = 0.5, tol_s: float = 0.06
                 ) -> list[list[int]]:
    """Per note, the notes it repeats in the repeat map's matched bars (DESIGN 4.10: an edit in verse 1 reaches
    verse 2): a bar's best match (score >= ``min_score``) maps the note's place in its bar onto the matched bar, and
    the nearest note of the same pitch within ``tol_s`` there is linked (both ways)."""
    out: list[set[int]] = [set() for _ in notes]
    if not repeat or not bars:
        return [[] for _ in notes]
    start = {int(b["bar"]): float(b["start_s"]) for b in bars}
    starts = sorted(start.values())
    bar_nos = [b for b, _s in sorted(start.items(), key=lambda kv: kv[1])]
    by_bar: dict[int, list[int]] = {}
    for i, n in enumerate(notes):
        j = int(np.searchsorted(starts, float(n["onset_s"]), side="right")) - 1
        if j >= 0:
            by_bar.setdefault(bar_nos[j], []).append(i)
    for bm in repeat.get("bars") or []:
        b = int(bm["bar"])
        best = next((m for m in bm.get("matches") or [] if float(m.get("score", 0)) >= min_score), None)
        if best is None or b not in start or int(best["bar"]) not in start:
            continue
        b2 = int(best["bar"])
        cands = by_bar.get(b2, []) + by_bar.get(b2 + 1, []) + by_bar.get(b2 - 1, [])
        for i in by_bar.get(b, []):
            t2 = float(notes[i]["onset_s"]) - start[b] + start[b2]
            near = [(abs(float(notes[j]["onset_s"]) - t2), j) for j in cands
                    if j != i and int(notes[j]["pitch"]) == int(notes[i]["pitch"])]
            if near:
                d, j = min(near)
                if d <= tol_s:
                    out[i].add(j)
                    out[j].add(i)
    return [sorted(s) for s in out]
