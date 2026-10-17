"""S9 Role and Player views (DESIGN 4.8, 4.9; M7c) on an edited or plain quantisation's guitar notes.

Every function works on the merged guitar's quantised notes ``g`` (``quantize.notes_of(q, "guitar_all")``: each
with ``bar``, ``gtick``, ``dur``, ``pitch``, ``onset_s``, ``offset_s``, ``line`` = "1".."4" or "u", ``shared``) and
the sections (``start_bar``, ``end_bar``), and returns note indices into ``g`` - the tab stage frets and writes them.

- ``section_roles``: per part and section one of ``ROLES`` (DESIGN 4.8): ``Tacet`` under ``TACET_MIN`` notes;
  ``Rhythm`` with ``RHYTHM_CHORD`` chord notes; ``Arpeggio`` when mostly single notes that ring into the next
  (``ARP_SINGLE``, ``ARP_RING``); ``Lead`` when lead-layer or high single notes (``bandscribe.parts.roles``) cover
  ``LEAD_COVER`` of the section's bars, ``Fill`` when they cover less; else ``Riff``.
- ``role_view``: per section the rhythm-like parts (Rhythm, Riff, Arpeggio) go to ``Rhythm``, ``Rhythm 2``, ..., the
  lead-like ones (Lead, Fill) to ``Lead``, ``Lead 2``, ... (most notes first). A ``shared`` (unison) note is in every
  role track that plays in its section.
- ``player_view``: ``players`` guitarists (DESIGN default 2). Per section the parts with notes (a "Tacet" part's
  last few notes too - no note is left out) are given to
  players by a minimum-cost assignment over the sections (dynamic programming): two parts that sound in one bar
  never share a player (DESIGN: "동시에 울리는 Line 은 같은 트랙에 넣을 수 없다"); a part moving to another player
  costs ``MOVE``; a player changing role group (rhythm <-> lead) ``ROLE_SWITCH``, register (per octave of median
  pitch) ``REGISTER``, side (L <-> R) ``PAN``. A part no player can take goes to ``Gtr extra (overdub)``, at
  ``EXTRA`` x its priority (unique / lead lines 3 > rhythm 2 > a part that is mostly unison 1 > a section's few
  stray notes 0) plus ``EXTRA_PER_NOTE`` per note: the least important goes first, of equals the one with fewer
  notes. When the parts fit the players each part gets a player of its own; only when they do not may two parts
  that never sound in one bar share a player. ``idle_policy``: ``rest`` (an idle player rests) or ``double`` (an idle player doubles the section's
  main rhythm part). ``unison_policy``: ``duplicate`` (a shared note goes to every player playing in its section) or
  ``primary_only`` (only its own part's player; the others get a text marker). Text markers name what changes.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bandscribe.parts import layers as LY
from bandscribe.parts import roles as RO

ROLES = ("Rhythm", "Riff", "Lead", "Fill", "Arpeggio", "Tacet")
RHYTHM_LIKE = ("Rhythm", "Riff", "Arpeggio", "Tacet")  # a "Tacet" part's few notes are still played (as rhythm)
LEAD_LIKE = ("Lead", "Fill")
PART_LINES = ("1", "2", "3", "4")
TACET_MIN = 4
RHYTHM_CHORD = 0.4
ARP_SINGLE = 0.7
ARP_RING = 0.5
LEAD_COVER = 0.5
MOVE = 1.0
ROLE_SWITCH = 0.3
REGISTER = 0.3
PAN = 0.3
EXTRA = 10.0
EXTRA_PER_NOTE = 0.001
IDLE_POLICIES = ("rest", "double")
UNISON_POLICIES = ("duplicate", "primary_only")
EXTRA_NAME = "Gtr extra (overdub)"


def _lines(g: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(n.get("line")) for n in g} & set(PART_LINES))


def _in(n: Mapping[str, Any], s: Mapping[str, Any]) -> bool:
    return int(s.get("_lo", s["start_bar"])) <= int(n["bar"]) < int(s.get("_hi", s["end_bar"]))


def _edges(sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The sections in order, the first reaching back and the last forward to every bar (a pickup before the first
    section or a tail after the last is played too)."""
    out = [dict(s) for s in sorted(sections, key=lambda s: int(s["start_bar"]))]
    if out:
        out[0]["_lo"], out[-1]["_hi"] = -10**9, 10**9
    return out


def role_of(idx: Sequence[int], g: Sequence[Mapping[str, Any]], feats: Sequence[Mapping[str, Any]],
            n_bars: int) -> str:
    if len(idx) < TACET_MIN:
        return "Tacet"
    n = len(idx)
    chord = sum(1 for i in idx if feats[i]["chord"]) / n
    single = [i for i in idx if feats[i]["size"] == 1]
    if chord >= RHYTHM_CHORD:
        return "Rhythm"
    if len(single) / n >= ARP_SINGLE:
        s = sorted(single, key=lambda i: float(g[i]["onset_s"]))
        ring = sum(1 for a, b in zip(s, s[1:]) if float(g[a]["offset_s"]) > float(g[b]["onset_s"]) + 0.02)
        if len(s) > 1 and ring / (len(s) - 1) >= ARP_RING:
            return "Arpeggio"
    lead = sum(1 for i in idx if feats[i]["lead"]) / n
    med = float(np.median([int(g[i]["pitch"]) for i in idx]))
    if lead >= RO.LEAD_SHARE or (len(single) / n >= 0.5 and med >= RO.LEAD_PITCH):
        cover = len({int(g[i]["bar"]) for i in idx}) / max(1, n_bars)
        return "Lead" if cover >= LEAD_COVER else "Fill"
    return "Riff"


def section_roles(g: Sequence[Mapping[str, Any]], sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """[{line, section, start_bar, end_bar, role, notes (indices into g, the part's own), bars}] per part x section."""
    feats = LY.note_features(g) if g else []
    out = []
    for s in _edges(sections):
        nb = int(s["end_bar"]) - int(s["start_bar"])
        for ln in _lines(g):
            idx = [k for k, n in enumerate(g) if str(n.get("line")) == ln and _in(n, s)]
            out.append({"line": ln, "section": s.get("id"), "start_bar": int(s["start_bar"]),
                        "end_bar": int(s["end_bar"]), "role": role_of(idx, g, feats, nb), "notes": idx,
                        "bars": sorted({int(g[k]["bar"]) for k in idx})})
    return out


def _shared_in(g: Sequence[Mapping[str, Any]], s: Mapping[str, Any]) -> list[int]:
    return [k for k, n in enumerate(g) if n.get("shared") and _in(n, s)]


def role_view(g: Sequence[Mapping[str, Any]], sections: Sequence[Mapping[str, Any]],
              roles: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """[{id, name, notes}] in the order Rhythm, Rhythm 2, ..., Lead, Lead 2, ... (non-empty only)."""
    roles = list(roles) if roles is not None else section_roles(g, sections)
    tracks: dict[str, set[int]] = {}
    for s in _edges(sections):
        rows = [r for r in roles if r["start_bar"] == int(s["start_bar"]) and r["notes"]]  # no note is left out
        named = []
        for group, base in ((RHYTHM_LIKE, "Rhythm"), (LEAD_LIKE, "Lead")):
            rs = sorted((r for r in rows if r["role"] in group), key=lambda r: (-len(r["notes"]), r["line"]))
            for k, r in enumerate(rs):
                name = base if k == 0 else f"{base} {k + 1}"
                tracks.setdefault(name, set()).update(r["notes"])
                named.append(name)
        for name in named:  # unison notes are in every role track playing here
            tracks[name].update(_shared_in(g, s))
    order = sorted(tracks, key=lambda t: (0 if t.startswith("Rhythm") else 1, t))
    return [{"id": "role_" + t.lower().replace(" ", "_"), "name": t, "notes": sorted(tracks[t])} for t in order
            if tracks[t]]


def _priority(r: Mapping[str, Any], g: Sequence[Mapping[str, Any]]) -> int:
    if r["role"] == "Tacet":  # a few stray notes in the section
        return 0
    if r["notes"] and sum(1 for k in r["notes"] if g[k].get("shared")) / len(r["notes"]) > 0.5:
        return 1
    return 2 if r["role"] == "Rhythm" else 3


def _group(role: str) -> str:
    return "lead" if role in LEAD_LIKE else "rhythm"


def player_view(g: Sequence[Mapping[str, Any]], sections: Sequence[Mapping[str, Any]], players: int = 2, *,
                idle_policy: str = "rest", unison_policy: str = "duplicate",
                positions: Mapping[str, str] | None = None,
                roles: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """-> {tracks: [{id, name, notes, lines}] (players, then the extra track when used), assign: [{section,
    start_bar, map: {line: player index or "extra"}}], markers: [{bar, track, text}], violations, copies}.
    ``positions``: a part's side ("L" / "R" / "") for the pan cost."""
    if idle_policy not in IDLE_POLICIES or unison_policy not in UNISON_POLICIES:
        raise ValueError(f"idle_policy {idle_policy!r} / unison_policy {unison_policy!r}")
    n_p = max(1, int(players))
    roles = list(roles) if roles is not None else section_roles(g, sections)
    pos = dict(positions or {})
    secs = _edges(sections)
    per_sec: list[list[dict[str, Any]]] = []
    for s in secs:
        rows = [dict(r) for r in roles if r["start_bar"] == int(s["start_bar"]) and r["notes"]]
        for r in rows:  # once per row (the transitions read them many times)
            r["med"] = float(np.median([int(g[k]["pitch"]) for k in r["notes"]]))
            r["bar_set"] = set(r["bars"])
        per_sec.append(rows)

    def options(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        """The assignments (tuple aligned with ``rows``: player index or "x") to search: one player per part when
        they fit; else parts that never sound in one bar may share a player, and a part goes to "x" only when every
        player already has a part sounding with it there."""
        if len(rows) <= n_p:
            return list(itertools.permutations(range(n_p), len(rows)))
        clash = [[bool(a["bar_set"] & b["bar_set"]) for b in rows] for a in rows]
        out = []
        for combo in itertools.product([*range(n_p), "x"], repeat=len(rows)):
            ok = all(not (combo[a] != "x" and combo[a] == combo[b] and clash[a][b])
                     for a, b in itertools.combinations(range(len(rows)), 2))
            if ok:
                for a, p in enumerate(combo):
                    if p == "x" and any(all(not (combo[b] == q and clash[a][b]) for b in range(len(rows)))
                                        for q in range(n_p)):
                        ok = False  # a player is free for it: never left out
                        break
            if ok:
                out.append(combo)
        return out

    def static(rows: list[dict[str, Any]], combo: tuple[Any, ...]) -> float:
        c = sum(EXTRA * _priority(r, g) + EXTRA_PER_NOTE * len(r["notes"]) for r, p in zip(rows, combo) if p == "x")
        # first appearance: Gtr 1 -> player 1 ... (a tiny tie-break that keeps numbers aligned)
        return c + sum(0.01 * abs(p - (int(r["line"]) - 1) % n_p) for r, p in zip(rows, combo) if p != "x")

    def trans(prev_rows: list[dict[str, Any]], pc: tuple[Any, ...], rows: list[dict[str, Any]],
              cc: tuple[Any, ...]) -> float:
        c = 0.0
        prev_of = {r["line"]: p for r, p in zip(prev_rows, pc)}
        for r, p in zip(rows, cc):
            if r["line"] in prev_of and prev_of[r["line"]] != p:
                c += MOVE
        by_p_prev = {p: r for r, p in zip(prev_rows, pc) if p != "x"}
        for r, p in zip(rows, cc):
            q = by_p_prev.get(p) if p != "x" else None
            if q is None or q["line"] == r["line"]:
                continue
            c += ROLE_SWITCH * (_group(q["role"]) != _group(r["role"]))
            c += REGISTER * abs(q["med"] - r["med"]) / 12.0
            c += PAN * (bool(pos.get(q["line"])) and bool(pos.get(r["line"])) and pos[q["line"]] != pos[r["line"]])
        return c

    def tk(c: tuple[Any, ...]) -> tuple[int, ...]:  # ties: a sortable form ("x" = after every player)
        return tuple(n_p if x == "x" else int(x) for x in c)

    # dynamic programming over the sections (one state per assignment of that section's active parts)
    choice: list[tuple[Any, ...]] = []
    if secs:
        opts = [options(rows) for rows in per_sec]
        score = {c: static(per_sec[0], c) for c in opts[0]}
        back: list[dict[tuple[Any, ...], tuple[Any, ...]]] = []
        for si in range(1, len(secs)):
            new, bk = {}, {}
            for c in opts[si]:
                best = min(score, key=lambda pc: (score[pc] + trans(per_sec[si - 1], pc, per_sec[si], c), tk(pc)))
                new[c] = score[best] + trans(per_sec[si - 1], best, per_sec[si], c) + static(per_sec[si], c)
                bk[c] = best
            score = new
            back.append(bk)
        last = min(score, key=lambda c: (score[c], tk(c)))
        choice = [last]
        for bk in reversed(back):
            choice.append(bk[choice[-1]])
        choice.reverse()
    names = [f"Player {p + 1}" for p in range(n_p)]
    notes: list[set[int]] = [set() for _ in range(n_p)]
    lines: list[set[str]] = [set() for _ in range(n_p)]
    extra: set[int] = set()
    copies = 0
    markers: list[dict[str, Any]] = []
    assign = []
    prev_line: list[str | None] = [None] * n_p  # the part each player played last (a rest keeps it)
    for s, rows, combo in zip(secs, per_sec, choice or [()] * len(secs)):
        m = {r["line"]: ("extra" if p == "x" else int(p)) for r, p in zip(rows, combo)}
        assign.append({"section": s.get("id"), "start_bar": int(s["start_bar"]), "map": m})
        shared = _shared_in(g, s)
        playing = sorted({int(p) for p in combo if p != "x"})
        for r, p in zip(rows, combo):
            own = [k for k in r["notes"] if not g[k].get("shared")]
            if p == "x":
                extra.update(r["notes"])
                markers.append({"bar": int(s["start_bar"]), "track": EXTRA_NAME,
                                "text": f"Gtr {r['line']} {r['role']} (no free player)"})
                continue
            notes[p].update(own)
            lines[p].add(r["line"])
        for k in shared:  # unison notes
            own_p = m.get(str(g[k].get("line")))
            if unison_policy == "duplicate":
                for p in playing:
                    notes[p].add(k)
                copies += max(0, len(playing) - 1)
            elif isinstance(own_p, int):
                notes[own_p].add(k)
        if unison_policy == "primary_only" and shared:
            for p in playing:
                owner = {m.get(str(g[k].get("line"))) for k in shared}
                if p not in owner:
                    markers.append({"bar": int(s["start_bar"]), "track": names[p], "text": "unison with " + ", ".join(
                        names[o] for o in sorted(o for o in owner if isinstance(o, int)))})
        idle = [p for p in range(n_p) if p not in playing]
        main = max((r for r, p in zip(rows, combo) if r["role"] in RHYTHM_LIKE and p != "x"),
                   key=lambda r: (len(r["notes"]), r["line"]), default=None)
        for p in idle:
            if idle_policy == "double" and main is not None:
                notes[p].update(main["notes"])
                copies += len(main["notes"])
                markers.append({"bar": int(s["start_bar"]), "track": names[p],
                                "text": f"doubles Gtr {main['line']} ({main['role']})"})
            elif rows:
                markers.append({"bar": int(s["start_bar"]), "track": names[p], "text": "rest"})
        for r, p in zip(rows, combo):
            if p != "x" and prev_line[int(p)] not in (None, r["line"]):
                markers.append({"bar": int(s["start_bar"]), "track": names[int(p)],
                                "text": f"plays Gtr {r['line']} ({r['role']})"})
            if p != "x":
                prev_line[int(p)] = r["line"]
    tracks = [{"id": f"player_{p + 1}", "name": names[p], "notes": sorted(notes[p]), "lines": sorted(lines[p])}
              for p in range(n_p)]
    if extra:
        tracks.append({"id": "player_extra", "name": EXTRA_NAME, "notes": sorted(extra), "lines": []})
    return {"tracks": tracks, "assign": assign, "markers": markers, "copies": copies,
            "violations": violations(g, tracks[:n_p])}


def violations(g: Sequence[Mapping[str, Any]], tracks: Sequence[Mapping[str, Any]]) -> int:
    """Bars where one player track holds own (not unison) notes of two parts - the M7c criterion (0 expected; an
    ``idle_policy = double`` copy is the player's only part there, a unison note belongs to both)."""
    bad = 0
    for t in tracks:
        by_bar: dict[int, set[str]] = {}
        for k in t["notes"]:
            n = g[k]
            if n.get("shared"):
                continue
            by_bar.setdefault(int(n["bar"]), set()).add(str(n.get("line")))
        bad += sum(1 for v in by_bar.values() if len(v) > 1)
    return bad
