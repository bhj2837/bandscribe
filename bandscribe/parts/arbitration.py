"""Per-note Line posteriors and Unassigned (DESIGN 4.7 "음표 단위 판정", hand weights as v1 says; M7a v0).

The Lines come from the evidence ``reid.estimate`` found, or from the user's ``--parts``. Each Line is seeded by the
notes that evidence assigns outright:

- ``spatial``: the notes of ``R3`` windows by the sign of their ILD (E22b) - ``L`` / ``R``;
- ``class``: the notes of the song's two most frequent guitar classes;
- ``symbolic``: chord notes (``rhythm``) and lead-layer notes (``lead``, ``bandscribe.parts.layers``);
- three Lines (user): the two spatial (or class) Lines and a ``lead`` Line seeded by the lead layer; four: the two
  spatial (or class) Lines each split into its chord and lead-layer notes.

Each Line gets a profile from its seeds: position (mean ``pan`` over its seeds in ``R3`` windows), role (mean role
score), class shares (add-one) and its share of the seeds. A note's score for a Line is

    s = w_pan * pan * position + w_role * r * role + w_cls * log P(class | Line) + w_prior * log share

with ``pan = tanh(ILD / ild_scale_db)`` for a note in an ``R3`` window and 0 elsewhere (the ILD of a centred or
doubled note says nothing about its part), and its posterior is the softmax over the Lines. Notes of one strum share
one decision, their scores summed (DESIGN 4.6 must-link: "한 스트럼 안의 음") - only outside ``R3`` windows, and only
notes of one class and not a peeled top: two parts strumming on the same beat at different positions (the usual P
case) or in different classes are two strums. A note whose best posterior is below ``tau`` is Unassigned (DESIGN
4.2: τ 0.4 to start); below ``amber`` it is flagged low-confidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_WEIGHTS: dict[str, float] = {"pan": 6.0, "ild_scale_db": 1.0, "role": 1.0, "cls": 1.0, "prior": 1.0}
TAU = 0.4
AMBER = 0.6


def _ild(sig: Mapping[str, Any] | None) -> float | None:
    v = (sig or {}).get("ild_db")
    return None if v is None or not math.isfinite(float(v)) else float(v)


def _classes(notes: Sequence[Mapping[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for n in notes:
        c = str(n.get("instrument"))
        counts[c] = counts.get(c, 0) + 1
    return sorted(counts, key=lambda c: (-counts[c], c))


def seed_lines(k: int, evidence: str | None, notes: Sequence[Mapping[str, Any]],
               feats: Sequence[Mapping[str, Any]], sigs: Sequence[Mapping[str, Any]],
               labels: Sequence[str | None]) -> tuple[list[dict[str, Any]], list[str]]:
    """Seeded Lines for ``k`` parts (module doc) -> ([{key, kind, seeds}], reasons when fewer than ``k`` came out)."""
    n = len(notes)
    if k <= 1 or n == 0:
        return ([{"key": "all", "kind": "all", "seeds": list(range(n))}] if n else []), []
    left = [i for i in range(n) if labels[i] == "R3" and _ild(sigs[i]) is not None and _ild(sigs[i]) >= 0]
    right = [i for i in range(n) if labels[i] == "R3" and _ild(sigs[i]) is not None and _ild(sigs[i]) < 0]
    cls = _classes(notes)
    by_cls = {c: [i for i in range(n) if str(notes[i].get("instrument")) == c] for c in cls[:2]}
    chord = [i for i in range(n) if feats[i]["chord"]]
    lead = [i for i in range(n) if feats[i]["lead"]]
    pairs = {
        "spatial": ([("L", "spatial", left), ("R", "spatial", right)] if left and right else None),
        "class": ([(c, "class", by_cls[c]) for c in cls[:2]] if len(cls) >= 2 else None),
        "symbolic": ([("rhythm", "symbolic", chord), ("lead", "symbolic", lead)] if chord and lead else None),
    }
    order = [evidence] if evidence else []
    order += [kd for kd in ("symbolic", "class", "spatial") if kd not in order]
    primary = next((kd for kd in order if kd and pairs.get(kd)), None)
    reasons: list[str] = []
    if primary is None:
        return [{"key": "all", "kind": "all", "seeds": list(range(n))}], ["파트를 가를 근거(좌우·음색·코드/리드 층)가 없습니다"]
    two = [{"key": key, "kind": kind, "seeds": list(s)} for key, kind, s in pairs[primary]]
    if k == 2 or primary == "symbolic":
        if k > 2:
            reasons.append("좌우·음색 근거가 없어 코드/리드 두 파트까지만 나눕니다")
        return two, reasons
    lead_set = set(lead)
    if k == 3:
        for ln in two:  # a side made only of lead-layer notes is the lead itself: it keeps them (review 2026-10-11)
            rest = [i for i in ln["seeds"] if i not in lead_set]
            if rest:
                ln["seeds"] = rest
            else:
                lead_set -= set(ln["seeds"])
        third = [i for i in lead if i in lead_set]
        if not third:
            return two, ["두 파트 말고 따로 둘 리드 층 음이 없어 두 파트까지만 나눕니다"]
        return [*two, {"key": "lead", "kind": "symbolic", "seeds": third}], reasons
    out = []
    for ln in two:
        r = [i for i in ln["seeds"] if feats[i]["chord"]]
        ld = [i for i in ln["seeds"] if i in lead_set]
        for key, s in ((f"{ln['key']}.rhythm", r), (f"{ln['key']}.lead", ld)):
            if s:
                out.append({"key": key, "kind": f"{ln['kind']}+symbolic", "seeds": s})
    if len(out) < 4:
        reasons.append(f"근거로 만들 수 있는 파트가 {len(out)}개입니다")
    return (out if len(out) >= 2 else two), reasons


def _pan(sig: Mapping[str, Any] | None, label: str | None, scale: float) -> float:
    v = _ild(sig)
    return math.tanh(v / scale) if (label == "R3" and v is not None) else 0.0


def posteriors(lines: Sequence[Mapping[str, Any]], notes: Sequence[Mapping[str, Any]],
               feats: Sequence[Mapping[str, Any]], sigs: Sequence[Mapping[str, Any]],
               labels: Sequence[str | None], weights: Mapping[str, float] | None = None
               ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """(notes x Lines) posterior matrix and the Line profiles (module doc)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    n, k = len(notes), len(lines)
    if n == 0 or k == 0:
        return np.zeros((n, k)), []
    scale = float(w["ild_scale_db"])
    pan = np.array([_pan(sigs[i], labels[i], scale) for i in range(n)])
    r = np.array([float(f["r"]) for f in feats])
    cls_all = _classes(notes)
    ci = {c: j for j, c in enumerate(cls_all)}
    cls = np.array([ci[str(nt.get("instrument"))] for nt in notes])
    total = sum(len(ln["seeds"]) for ln in lines) or 1
    prof = []
    score = np.zeros((n, k))
    for j, ln in enumerate(lines):
        s = np.array(ln["seeds"], dtype=int)
        in_p = [i for i in ln["seeds"] if labels[i] == "R3" and _ild(sigs[i]) is not None]
        position = float(np.mean(pan[in_p])) if in_p else 0.0
        role = float(np.mean(r[s])) if s.size else 0.0
        counts = np.bincount(cls[s], minlength=len(cls_all)) if s.size else np.zeros(len(cls_all))
        pc = (counts + 1.0) / (counts.sum() + len(cls_all))
        share = max(len(ln["seeds"]), 1) / total
        prof.append({"key": ln["key"], "kind": ln["kind"], "seeds": int(s.size), "position": round(position, 4),
                     "role": round(role, 4), "classes": {c: round(float(pc[ci[c]]), 4) for c in cls_all},
                     "share": round(share, 4)})
        score[:, j] = (w["pan"] * pan * position + w["role"] * r * role + w["cls"] * np.log(pc[cls])
                       + w["prior"] * math.log(share))
    groups: dict[tuple[int, int], list[int]] = {}
    for i, f in enumerate(feats):
        if labels[i] != "R3" and not f["peel"] and f["size"] > 1:
            groups.setdefault((int(f["cluster"]), int(cls[i])), []).append(i)
    for idx in groups.values():
        if len(idx) > 1:
            score[idx] = score[idx].sum(axis=0)
    z = score - score.max(axis=1, keepdims=True)
    post = np.exp(z)
    post /= post.sum(axis=1, keepdims=True)
    return post, prof


def assign(notes: Sequence[Mapping[str, Any]], feats: Sequence[Mapping[str, Any]],
           sigs: Sequence[Mapping[str, Any]], labels: Sequence[str | None], k: int, evidence: str | None, *,
           weights: Mapping[str, float] | None = None, tau: float = TAU, amber: float = AMBER) -> dict[str, Any]:
    """Lines and per-note decisions (module doc) -> {lines (profiles), line (index or None per note), p, amber,
    reasons}."""
    lines, reasons = seed_lines(k, evidence, notes, feats, sigs, labels)
    post, prof = posteriors(lines, notes, feats, sigs, labels, weights)
    line: list[int | None] = []
    pmax: list[float] = []
    for i in range(len(notes)):
        if post.shape[1] == 0:
            line.append(None)
            pmax.append(0.0)
            continue
        j = int(np.argmax(post[i]))
        p = float(post[i, j])
        line.append(j if p >= tau else None)
        pmax.append(round(p, 4))
    return {"lines": prof, "line": line, "p": pmax, "amber": [p < amber for p in pmax], "reasons": reasons}
