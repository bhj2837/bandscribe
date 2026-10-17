"""S8 v0 end to end (M7a): symbolic features -> part count -> per-note Lines -> Part view order and names.

``diarize`` is what the ``parts`` stage and E25 run. ``k_force`` (the user's ``--parts N``) replaces the estimated
count; Lines that end up with no notes are dropped and the rest renumbered (S9). ``tracking`` (M7b: the tracker's
params, ``{}`` for its defaults; None = off) re-decides whole windows on the v0 posteriors before the Lines are
named (``bandscribe.parts.track``), with the repeat map's window pairs when ``repeat`` is given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bandscribe.parts import arbitration as A
from bandscribe.parts import layers as LY
from bandscribe.parts import reid as R
from bandscribe.parts import roles as RO
from bandscribe.parts import track as TR


def diarize(notes: Sequence[Mapping[str, Any]], sigs: Sequence[Mapping[str, Any]],
            windows: Sequence[Mapping[str, Any]], *, bar_of: Sequence[int] | None = None,
            params: Mapping[str, Any] | None = None, weights: Mapping[str, float] | None = None,
            k_force: int | None = None, tau: float = A.TAU, tracking: Mapping[str, Any] | None = None,
            repeat: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """-> {count: reid estimate + chosen / forced / reasons, lines: [{number, name, short, role, cls, position,
    notes, bars, key, kind}] in number order, note: [{line: number or None, p, amber}] per note}. ``windows`` carry
    ``start_s``, ``end_s``, ``label`` (router) and ``section``; ``bar_of`` defaults to 2 s cells."""
    feats = LY.note_features(notes)
    est = R.estimate(notes, feats, sigs, windows, params)
    k = int(k_force) if k_force else int(est["k"])
    wins = sorted(windows, key=lambda w: float(w["start_s"]))
    of = R.window_index(notes, wins)
    labels = [wins[j].get("label") if j is not None else None for j in of]
    res = A.assign(notes, feats, sigs, labels, k, est["evidence"], weights=weights, tau=tau, count_params=params)
    tracked = None
    if tracking is not None and res["post"].shape[1] >= 2:
        line0 = [None if j is None else j + 1 for j in res["line"]]
        tracked = TR.track(notes, feats, sigs, wins, res["post"], line0, repeat=repeat, params=tracking)
        res["line"] = [None if x is None else int(x) - 1 for x in tracked["line"]]
        res["p"] = [round(x, 4) for x in tracked["p"]]
        res["amber"] = [x < A.AMBER for x in res["p"]]
    bars = list(bar_of) if bar_of is not None else [int(float(n["onset_s"]) // 2.0) for n in notes]
    named = RO.order_and_names(res["lines"], res["line"], notes, feats, bars)
    keep = sorted((r for r in named if r["notes"] > 0), key=lambda r: r["number"])
    renum = {r["j"]: num for num, r in enumerate(keep, start=1)}
    lines = []
    for num, r in enumerate(keep, start=1):
        short = "Gtr" if len(keep) == 1 else f"Gtr {num}"
        name = short + r["name"][len(r["short"]):]
        prof = res["lines"][r["j"]]
        lines.append({"number": num, "name": name, "short": short, "role": r["role"], "cls": r["cls"],
                      "position": r["position"], "notes": r["notes"], "bars": r["bars"], "key": prof["key"],
                      "kind": prof["kind"], "profile": prof})
    note = [{"line": renum.get(j) if j is not None else None, "p": p, "amber": a}
            for j, p, a in zip(res["line"], res["p"], res["amber"])]
    reasons = list(res["reasons"])
    if len(keep) < len(named):
        empty = [res["lines"][r["j"]]["key"] for r in named if r["notes"] == 0]
        reasons.append(f"근거로 만든 파트 {len(named)}개 중 음이 하나도 가지 않은 파트({', '.join(empty)})가 있어 "
                       f"{len(keep)}파트입니다")
    count = {**{k_: v for k_, v in est.items() if k_ != "windows"}, "chosen": len(lines),
             "forced": int(k_force) if k_force else None, "reasons": reasons}
    if tracked is not None:
        count["track"] = {"changed": tracked["changed"], "repeat_pairs": len(tracked["pairs"])}
    # the posteriors in the renumbered Lines' order (column number - 1), for the M7b tracker
    post = res["post"][:, [r["j"] for r in keep]] if keep else res["post"][:, :0]
    return {"count": count, "windows": est["windows"], "lines": lines, "note": note, "feats": feats, "post": post}
