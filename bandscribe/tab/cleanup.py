"""Note clean-up before quantisation (M10 follow-up; E39 / E40 in docs/decisions.md).

- ``drop_short`` (E40): notes shorter than ``min_dur_s``. MuScriptor sometimes degenerates over a stretch of a song
  and emits bursts of 10 ms notes (lulu.'s bass after 2:30: 10,965 of them; Tier B moosmusic: 53-83 % of its guitar
  notes); outside such bursts notes this short are rare.
- ``dedupe_within_line`` (E39): two notes of the same pitch whose onsets are within ``window_s`` in the same part
  (``line``; notes without one form their own group) are one note - a player cannot sound the same pitch twice at
  once, and MuScriptor often emits one guitar note under two classes (acoustic + clean electric). The earlier note is
  kept and stretched to the later offset.
- ``lowest_of_bundle`` (E46, bass): notes starting within ``window_s`` of a bundle's first note are one event of which
  only the lowest stays, stretched to the bundle's latest offset. A bass plays one note at a time; the upper ones are
  mostly harmonics and a degenerating decoder's stacks (lulu. 48 %, 狭心症 43 % of its bass events against under 1 % on
  most sheet songs).

All return ``(kept, dropped)`` and keep the input order of the kept notes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _dur(n: Mapping[str, Any]) -> float:
    on = float(n["onset_s"])
    off = n.get("offset_s")
    return (float(off) if off is not None else on) - on


def drop_short(notes: Sequence[Mapping[str, Any]], min_dur_s: float) -> tuple[list[dict], list[dict]]:
    if min_dur_s <= 0:
        return [dict(n) for n in notes], []
    kept, dropped = [], []
    for n in notes:
        (kept if _dur(n) >= min_dur_s else dropped).append(dict(n))
    return kept, dropped


def lowest_of_bundle(notes: Sequence[Mapping[str, Any]], window_s: float) -> tuple[list[dict], list[dict]]:
    out = [dict(n) for n in notes]
    if window_s <= 0 or len(out) < 2:
        return out, []
    order = sorted(range(len(out)), key=lambda i: (float(out[i]["onset_s"]), i))
    drop: set[int] = set()
    k = 0
    while k < len(order):
        t0 = float(out[order[k]]["onset_s"])
        j = k + 1
        while j < len(order) and float(out[order[j]]["onset_s"]) - t0 <= window_s:
            j += 1
        bundle = order[k:j]
        if len(bundle) > 1:
            keep = min(bundle, key=lambda i: (float(out[i]["pitch"]), float(out[i]["onset_s"]), i))
            ends = [float(out[i]["offset_s"]) for i in bundle if out[i].get("offset_s") is not None]
            if ends:
                out[keep]["offset_s"] = max(ends)
            drop.update(i for i in bundle if i != keep)
        k = j
    return [n for i, n in enumerate(out) if i not in drop], [n for i, n in enumerate(out) if i in drop]


def dedupe_within_line(notes: Sequence[Mapping[str, Any]], window_s: float) -> tuple[list[dict], list[dict]]:
    out = [dict(n) for n in notes]
    if window_s <= 0 or len(out) < 2:
        return out, []
    groups: dict[tuple[Any, int], list[int]] = {}
    for i, n in enumerate(out):
        groups.setdefault((n.get("line"), int(round(float(n["pitch"])))), []).append(i)
    drop: set[int] = set()
    for idx in groups.values():
        idx.sort(key=lambda i: (float(out[i]["onset_s"]), i))
        keep = idx[0]
        for i in idx[1:]:
            if float(out[i]["onset_s"]) - float(out[keep]["onset_s"]) <= window_s:
                drop.add(i)
                ko, io = out[keep].get("offset_s"), out[i].get("offset_s")
                if io is not None and (ko is None or float(io) > float(ko)):
                    out[keep]["offset_s"] = float(io)
            else:
                keep = i
    return [n for i, n in enumerate(out) if i not in drop], [n for i, n in enumerate(out) if i in drop]
