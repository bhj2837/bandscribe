"""Tier A sheet evaluation (mid-term plan 1-2; ``bandscribe eval sheets``): our score against the user's band-score
PDFs, extracted once into JSON under ``data/`` (never tracked; this module reads that JSON only, never a PDF).

Songs: ``data/eval/sheets/songs.yaml`` (local, see ``load_songs``). A song's stage outputs are the ones behind its
current export (``tab.health.stage_dirs``: export.json -> score -> quant / grid), never simply the newest folder.

Definitions (docs/MIDTERM_REVIEW.md 5.1-5.2). Pooled = TP / FP / FN summed over songs (and parts): the headline;
the song mean is for reference only.
- **sheet bar bag:** per sheet bar, the multiset of sounding pitches - fret + the string's open pitch (the sheet
  header's tuning, retuned strings and capo), dead and tied notes left out. The two guitars are united (a pitch
  both play counts once per bar, as many times as the busier one plays it).
- **our bar span:** the sheet bar's beats ``beat0 - offset .. + numerator`` through our grid's tempo map, both ends
  60 ms early (``EARLY_S``); our notes whose onset falls in it form our bag. Bar F1 = 2 TP / (2 TP + FP + FN).
- **offset:** sheet beat 0 is our beat ``offset``; the one in -8..8 with the best merged guitar F1 (ties: smaller |o|).
- **drifting offset:** per 8 sheet bars holding at least 40 sheet guitar notes, the best offset in ``offset ± 4``
  replaces the current one when it beats it by 0.03 F1 (a spurious beat shifts every later beat index).
  The bass is scored at the guitar's offsets.
- **per part:** each sheet guitar against one of our parts (``line``), the assignment with the best mean F1.
- **downbeats:** share of sheet bar lines (in our beat range) that fall on one of our bar lines.
- **string / fret agreement:** among the notes of our fretted parts (export ``tab.json``, onsets from the quant note
  their ``uid`` names) whose pitch matches the sheet bar's bag, the share also on the sheet's string, and on the
  sheet's string and written fret (capo and tuning differences show up only in the second).
- **chord texture** (rhythm part = the sheet guitar with fewer single-note columns): each sheet column of 3 or more
  sounding pitches, placed by its graphical x phase (``col_beat``) at the drifting offset, against our same-part tab
  event (one tab event = the part's notes at one gtick): the column onset snaps to the nearest event of either part
  within 0.375 beat that shares a pitch, then the part's event within 0.125 beat of it. Shares of chords whose event
  has exactly the sheet's pitches, the same pitch-class set, the same shape class (``shape``). Lead part (songs with
  ``lead: true``): of the sheet's single-note columns that have such an event, the share where it has 3+ notes.
- **practisable phrases** (``phrases``): 4-bar phrases of each part, from each rehearsal mark, at the drifting
  offset; the share with phrase F1 >= 0.7 (the plan's headline) and the strict light-edit share.
"""

from __future__ import annotations

import bisect
import itertools
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths

FORMAT = "bandscribe.sheet_eval/1"
GUITAR = (40, 45, 50, 55, 59, 64)   # open strings, low to high
BASS = (28, 33, 38, 43)
EARLY_S = 0.06
OFFSET_RANGE = range(-8, 9)
DRIFT_WINDOW_BARS = 8
DRIFT_MIN_NOTES = 40
DRIFT_SPAN = 4
DRIFT_GAIN = 0.03
GUITAR_PARTS = ("gtr1", "gtr2")
NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
CIRCLED = {chr(0xE834 + k): k + 1 for k in range(7)}  # Guitar Pro's string-number glyphs: string 1, 2, ...
FLAT, SHARP = "", ""


class SheetEvalError(RuntimeError):
    """User-facing (Korean) problem of the sheet evaluation."""


def songs_file() -> Path:
    return Path(paths.DATA) / "eval" / "sheets" / "songs.yaml"


@dataclass
class Song:
    id: str
    title: str
    job: str                     # job key under paths.JOBS
    sheets: Path                 # folder of the extracted JSON (gtr1.json, gtr2.json, bass.json, headers.json, ...)
    sheet_bpm: float | None = None
    tempo_hint: bool = False     # the job was run with the sheet tempo as its hint
    lead: bool = True            # the sheet has a lead part (HBT: both guitars play rhythm)


def load_songs(path: Path | None = None) -> tuple[list[Song], dict[str, str]]:
    """(the scored songs, {job key: why it is never scored}) from ``songs.yaml``::

        format: bandscribe.sheet_songs/1
        songs:
          - {id: sc, title: "...", job: f-..., sheets: tmp/sheets_sc, sheet_bpm: 190, tempo_hint: false}
        invalid:
          - {job: f-..., why: "기타를 뺀 반주"}

    ``sheets`` is relative to ``data/``."""
    import yaml

    p = Path(path or songs_file())
    if not p.is_file():
        raise SheetEvalError(f"악보 곡 목록이 없습니다: {p} (data/eval/sheets/songs.yaml, 로컬 전용)")
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    songs = [Song(id=str(s["id"]), title=str(s.get("title") or s["id"]), job=str(s["job"]),
                  sheets=Path(paths.DATA) / str(s["sheets"]), sheet_bpm=s.get("sheet_bpm"),
                  tempo_hint=bool(s.get("tempo_hint", False)), lead=bool(s.get("lead", True)))
             for s in doc.get("songs") or []]
    invalid = {str(x["job"]): str(x.get("why") or "") for x in doc.get("invalid") or []}
    bad = [s.id for s in songs if s.job in invalid]
    if bad:
        raise SheetEvalError(f"무효로 표시된 작업을 채점하려 합니다: {', '.join(bad)}")
    return songs, invalid


# ------------------------------------------------------------------------------------------------- the sheet


def _load(song: Song, name: str) -> Any:
    return atomic.read_json(song.sheets / name)


def beat0(tab: Mapping[str, Any]) -> dict[int, int]:
    """Sheet bar -> its first beat (numerators summed from bar 1)."""
    out, acc = {}, 0
    for b in tab["bars"]:
        out[b["bar"]] = acc
        acc += b["numerator"]
    return out


def sheet_tuning(header: str, standard: Sequence[int]) -> tuple[int, ...]:
    """A part's open-string pitches (low to high) from its PDF header: 'Tune down 1/2 step' / '1 step', per-string
    retunes ('<string glyph> = Db', nearest the standard), 'Capo. fret N' (added to every string)."""
    tun = list(standard)
    n = len(tun)
    if "Tune down 1/2 step" in header:
        tun = [x - 1 for x in tun]
    if "Tune down 1 step" in header or "Tune down whole step" in header:
        tun = [x - 2 for x in tun]
    for line in header.splitlines():
        line = line.strip()
        if len(line) >= 4 and line[0] in CIRCLED and "=" in line:
            k = CIRCLED[line[0]]
            name = line.split("=", 1)[1].strip().replace(FLAT, "b").replace(SHARP, "#")
            if not name or name[0] not in NOTE_PC or k > n:
                continue
            pc = (NOTE_PC[name[0]] + (-1 if name[1:2] == "b" else 1 if name[1:2] == "#" else 0)) % 12
            std = standard[n - k]
            tun[n - k] = min((std + d for d in range(-6, 7) if (std + d) % 12 == pc), key=lambda v: abs(v - std))
    m = re.search(r"Capo\.?\s*fret\s*(\d+)", header)
    if m:
        tun = [x + int(m.group(1)) for x in tun]
    return tuple(tun)


@dataclass
class SheetBar:
    bar: int
    beat0: int
    beats: int
    pitches: list[int]
    notes: list[tuple[int, int, int]] = field(default_factory=list)  # (pitch, string, written fret)


def sheet_part(song: Song, part: str, standard: Sequence[int]) -> list[SheetBar]:
    """Every bar of one sheet part with its sounding pitches (module doc)."""
    doc = _load(song, f"{part}.json")
    try:
        header = str((_load(song, "headers.json") or {}).get(part) or "")
    except (OSError, ValueError):
        header = ""
    tun = sheet_tuning(header, standard)
    n = len(tun)
    b0 = beat0(doc)
    out = []
    for b in doc["bars"]:
        trip = [(tun[n - x["string"]] + x["fret"], int(x["string"]), int(x["fret"])) for c in b["columns"]
                for x in c["notes"] if not x["dead"] and not x["tied"] and x["fret"] is not None]
        out.append(SheetBar(b["bar"], b0[b["bar"]], b["numerator"], [t[0] for t in trip], trip))
    return out


# ------------------------------------------------------------------------------------------------- our side


@dataclass
class Ours:
    job_dir: Path
    quant_key: str
    grid_key: str
    tempo: Any                              # schema.grid.TempoMap
    bar_starts: set[int]                    # our bar lines, in beats
    n_beats: int
    guitar: list[tuple[float, int, str]]   # (onset_s, pitch, line), by onset
    bass: list[tuple[float, int, str]]
    quant_guitar: list[tuple[float, int]]  # (onset_s, pitch) in quant.json order: a tab note's uid indexes it


def ours(job_dir: Path) -> Ours:
    from bandscribe.schema.grid import TempoMap
    from bandscribe.tab import health as HL
    from bandscribe.tab import quantize as Q

    dirs = HL.stage_dirs(job_dir)
    if "quant" not in dirs or "grid" not in dirs:
        raise SheetEvalError(f"{Path(job_dir).name}: export 뒤의 quant·grid 단계를 찾지 못했습니다(먼저 bandscribe run)")
    q = atomic.read_json(dirs["quant"] / "quant.json")
    g = atomic.read_json(dirs["grid"] / "grid.json")
    starts, acc = set(), 0
    for b in g["bars"]:
        starts.add(acc)
        acc += int(b["n_beats"])

    def notes(track: str) -> list[tuple[float, int, str]]:
        return sorted((float(n["onset_s"]), int(n["pitch"]), str(n.get("line"))) for n in Q.notes_of(q, track))

    return Ours(Path(job_dir), dirs["quant"].name, dirs["grid"].name, TempoMap.from_dict(g), starts, acc,
                notes("guitar_all"), notes("bass"),
                [(float(n["onset_s"]), int(n["pitch"])) for n in Q.notes_of(q, "guitar_all")])


# ------------------------------------------------------------------------------------------------- scoring


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __iadd__(self, o: Counts) -> Counts:
        self.tp, self.fp, self.fn = self.tp + o.tp, self.fp + o.fp, self.fn + o.fn
        return self

    @property
    def f1(self) -> float:
        return 100.0 * 2 * self.tp / max(1, 2 * self.tp + self.fp + self.fn)

    def doc(self) -> dict[str, Any]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "F1": round(self.f1, 2),
                "P": round(100.0 * self.tp / max(1, self.tp + self.fp), 2),
                "R": round(100.0 * self.tp / max(1, self.tp + self.fn), 2)}


def bag_counts(tempo: Any, sheet: Sequence[Sequence[SheetBar]], est: Sequence[tuple[float, int, str]],
               offset_of: Callable[[int], int], bars: set[int] | None = None) -> Counts:
    """Bar-bag counts of ``est`` against the union of the ``sheet`` parts (module doc). ``offset_of(sheet bar)``;
    ``bars`` limits the sheet bars scored."""
    t_on = [t for t, _p, _l in est]
    c = Counts()
    for row in zip(*sheet):
        b = row[0]
        if bars is not None and b.bar not in bars:
            continue
        off = offset_of(b.bar)
        if b.beat0 - off < 0:
            continue
        a = float(tempo.beat_to_time(b.beat0 - off)) - EARLY_S
        e = float(tempo.beat_to_time(b.beat0 - off + b.beats)) - EARLY_S
        gt: Counter = Counter()
        for x in row:
            gt |= Counter(x.pitches)
        got = Counter(p for _t, p, _l in est[bisect.bisect_left(t_on, a):bisect.bisect_left(t_on, e)])
        k = sum((gt & got).values())
        c += Counts(k, sum(got.values()) - k, sum(gt.values()) - k)
    return c


def best_offset(tempo: Any, guitars: Sequence[Sequence[SheetBar]], est: Sequence[tuple[float, int, str]]) -> int:
    scores = {o: bag_counts(tempo, guitars, est, lambda _b, o=o: o).f1 for o in OFFSET_RANGE}
    return max(scores, key=lambda o: (scores[o], -abs(o)))


def drifting_offsets(tempo: Any, guitars: Sequence[Sequence[SheetBar]], est: Sequence[tuple[float, int, str]],
                     offset: int) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """({sheet bar: offset}, [changes]) - module doc."""
    bars = [b.bar for b in guitars[0]]
    n_notes = Counter()
    for part in guitars:
        for b in part:
            n_notes[b.bar] += len(b.pitches)
    cur, local, changes = offset, {}, []
    for i in range(0, len(bars), DRIFT_WINDOW_BARS):
        sel = set(bars[i:i + DRIFT_WINDOW_BARS])
        if sum(n_notes[b] for b in sel) >= DRIFT_MIN_NOTES:
            sc = {o: bag_counts(tempo, guitars, est, lambda _b, o=o: o, sel).f1 / 100.0
                  for o in range(offset - DRIFT_SPAN, offset + DRIFT_SPAN + 1)}
            best = max(sc, key=lambda o: (sc[o], -abs(o - cur)))
            if best != cur and sc[best] >= sc[cur] + DRIFT_GAIN:
                changes.append({"sheet_bar": bars[i], "from": cur, "to": best,
                                "f1": [round(sc[cur], 3), round(sc[best], 3)]})
                cur = best
        for b in sel:
            local[b] = cur
    return local, changes


def downbeats(o: Ours, sheet_bars: Sequence[SheetBar], offset_of: Callable[[int], int]) -> dict[str, Any]:
    hit = tot = 0
    for b in sheet_bars:
        gb = b.beat0 - offset_of(b.bar)
        if 0 <= gb < o.n_beats:
            tot += 1
            hit += gb in o.bar_starts
    return {"sheet_bars": tot, "on_our_bar_line": hit, "share": round(hit / max(1, tot), 4)}


def per_part(o: Ours, guitars: Sequence[Sequence[SheetBar]], offset_of: Callable[[int], int]
             ) -> tuple[dict[str, str | None], dict[str, Counts]]:
    """({sheet guitar: our line}, {sheet guitar: counts}) of the best-mean assignment."""
    lines = sorted({ln for _t, _p, ln in o.guitar if ln not in ("u", "None")})
    cand = list(lines) + [None] * max(0, len(GUITAR_PARTS) - len(lines))
    by_line = {ln: [n for n in o.guitar if n[2] == ln] for ln in lines}
    best: tuple[float, dict[str, str | None], dict[str, Counts]] | None = None
    for perm in itertools.permutations(cand, len(GUITAR_PARTS)):
        res = {gk: bag_counts(o.tempo, [part], by_line.get(ln, []) if ln else [], offset_of)
               for gk, part, ln in zip(GUITAR_PARTS, guitars, perm)}
        mean = sum(c.f1 for c in res.values()) / len(res)
        if best is None or mean > best[0]:
            best = (mean, dict(zip(GUITAR_PARTS, perm)), res)
    assert best is not None
    return best[1], best[2]


def fretting(o: Ours, guitars: Sequence[Sequence[SheetBar]], offset_of: Callable[[int], int]) -> dict[str, Any]:
    """String / fret agreement (module doc) of the fretted part tracks ``gtr_1`` .. ``gtr_4``."""
    tab = atomic.read_json(o.job_dir / "export" / "tab" / "tab.json")
    ours_f: list[tuple[float, int, int, int]] = []
    mismatch = 0
    for part in ("gtr_1", "gtr_2", "gtr_3", "gtr_4"):
        for n in ((tab.get("tracks") or {}).get(part) or {}).get("notes") or []:
            uid = str(n.get("uid") or "")
            if not uid.startswith("guitar_all:") or n.get("string") is None:
                continue
            k = int(uid.split(":", 1)[1])
            if not 0 <= k < len(o.quant_guitar):
                mismatch += 1
                continue
            t, qp = o.quant_guitar[k]
            mismatch += qp != int(n["pitch"])
            ours_f.append((t, int(n["pitch"]), int(n["string"]), int(n["fret"])))
    ours_f.sort()
    t_on = [x[0] for x in ours_f]
    pm = sm = fm = 0
    for row in zip(*guitars):
        b = row[0]
        off = offset_of(b.bar)
        if b.beat0 - off < 0:
            continue
        a = float(o.tempo.beat_to_time(b.beat0 - off)) - EARLY_S
        e = float(o.tempo.beat_to_time(b.beat0 - off + b.beats)) - EARLY_S
        gp: Counter = Counter()
        gps: Counter = Counter()
        gpf: Counter = Counter()
        for x in row:
            gp |= Counter(x.pitches)
            gps |= Counter((p, s) for p, s, _f in x.notes)
            gpf |= Counter(x.notes)
        seg = ours_f[bisect.bisect_left(t_on, a):bisect.bisect_left(t_on, e)]
        pm += sum((gp & Counter(x[1] for x in seg)).values())
        sm += sum((gps & Counter((x[1], x[2]) for x in seg)).values())
        fm += sum((gpf & Counter((x[1], x[2], x[3]) for x in seg)).values())
    return {"pitch_matched": pm, "string_matched": sm, "fret_matched": fm, "string": round(sm / max(1, pm), 4),
            "fret": round(fm / max(1, pm), 4), "uid_pitch_mismatch": mismatch,
            "tab_tuning": {p: (tab["tracks"][p].get("label"), tab["tracks"][p].get("capo"))
                           for p in ("gtr_1", "gtr_2") if p in (tab.get("tracks") or {})}}


TPB = 48
SNAP_BEATS, SAME_BEATS, MAP_EARLY_BEATS = 0.375, 0.125, 0.125


def shape(pitches: Sequence[int]) -> str:
    """Shape class: none, single, octave (one pitch class), power (two pitch classes a fourth / fifth apart), dyad,
    triad34 (3-4 notes), full56 (5-6 notes)."""
    n = len(pitches)
    if n == 0:
        return "none"
    if n == 1:
        return "single"
    pcs = sorted({p % 12 for p in pitches})
    if len(pcs) == 1:
        return "octave"
    if len(pcs) == 2 and (pcs[1] - pcs[0]) in (5, 7):
        return "power"
    if n == 2:
        return "dyad"
    return "triad34" if n <= 4 else "full56"


def col_beat(bar: Mapping[str, Any], phase: float) -> float:
    """Beat inside a sheet bar of a column at x fraction ``phase``: linear from the bar's first column (its left pad;
    0.085 when the first column starts at 0.12 or later) to the bar end."""
    ph = [c["phase"] for c in bar["columns"]]
    p0 = min(ph) if ph else 0.0
    p0 = p0 if p0 < 0.12 else 0.085
    return min(max((phase - p0) / max(1e-6, 1.0 - p0), 0.0), 0.999) * bar["numerator"]


def sheet_columns(song: Song, part: str, offset_of: Callable[[int], int]) -> list[dict[str, Any]]:
    """[{bar, obeat (our beat), pitches (sorted unique sounding)}] of every column with a sounding note."""
    doc = _load(song, f"{part}.json")
    try:
        header = str((_load(song, "headers.json") or {}).get(part) or "")
    except (OSError, ValueError):
        header = ""
    tun = sheet_tuning(header, GUITAR)
    n = len(tun)
    b0 = beat0(doc)
    out = []
    for b in doc["bars"]:
        for c in b["columns"]:
            ns = [x for x in c["notes"] if not x["dead"] and not x["tied"] and x["fret"] is not None]
            if ns:
                out.append({"bar": b["bar"], "obeat": b0[b["bar"]] + col_beat(b, c["phase"]) - offset_of(b["bar"]),
                            "pitches": sorted({tun[n - x["string"]] + x["fret"] for x in ns})})
    return out


def tab_events(job_dir: Path) -> dict[str, list[tuple[float, frozenset[int]]]]:
    """{gtr_1 / gtr_2: [(our beat, pitches)]}: what the tab prints as one stacked event (a note without a string,
    dropped by the playability check, is not printed)."""
    tab = atomic.read_json(Path(job_dir) / "export" / "tab" / "tab.json")
    out = {}
    for tid in ("gtr_1", "gtr_2"):
        by: dict[int, set[int]] = {}
        for nt in ((tab.get("tracks") or {}).get(tid) or {}).get("notes") or []:
            if nt.get("string") is not None:
                by.setdefault(int(nt["gtick"]), set()).add(int(nt["pitch"]))
        out[tid] = [(g / TPB, frozenset(ps)) for g, ps in sorted(by.items())]
    return out


def _nearest(beats: Sequence[float], t: float, tol: float) -> int | None:
    lo, hi = bisect.bisect_left(beats, t - tol), bisect.bisect_right(beats, t + tol)
    return min(range(lo, hi), key=lambda k: abs(beats[k] - t)) if lo < hi else None


def texture(song: Song, job_dir: Path, offset_of: Callable[[int], int]) -> dict[str, Any]:
    """Chord texture of the rhythm part and the lead part's stacked events (module doc)."""
    ev = tab_events(job_dir)
    cols = {p: sheet_columns(song, p, offset_of) for p in GUITAR_PARTS}
    sheet_doc = _load(song, "gtr1.json")
    b0 = beat0(sheet_doc)
    num = {b["bar"]: b["numerator"] for b in sheet_doc["bars"]}

    def part_f1(part: str, tid: str) -> float:
        want_by: dict[int, Counter] = {}
        for c in cols[part]:
            want_by.setdefault(c["bar"], Counter()).update(c["pitches"])
        beats = [e[0] for e in ev[tid]]
        c = Counts()
        for bar, n in num.items():
            a = b0[bar] - offset_of(bar) - MAP_EARLY_BEATS
            if a < 0:
                continue
            got = Counter(p for k in range(bisect.bisect_left(beats, a), bisect.bisect_left(beats, a + n))
                          for p in ev[tid][k][1])
            want = want_by.get(bar, Counter())
            k = sum((want & got).values())
            c += Counts(k, sum(got.values()) - k, sum(want.values()) - k)
        return c.f1

    mapping = max(({"gtr1": a, "gtr2": b} for a, b in itertools.permutations(("gtr_1", "gtr_2"))),
                  key=lambda m: sum(part_f1(p, m[p]) for p in m))
    single = {p: sum(1 for c in cols[p] if len(c["pitches"]) == 1) / max(1, len(cols[p])) for p in cols}
    lead_part = max(single, key=lambda p: single[p])
    rhythm_part = next(p for p in cols if p != lead_part)
    allev = sorted(ev["gtr_1"] + ev["gtr_2"], key=lambda e: e[0])
    allb = [e[0] for e in allev]
    last = allb[-1] if allb else 0.0
    beats = {tid: [e[0] for e in ev[tid]] for tid in ev}
    res: dict[str, Counter] = {"rhythm": Counter(), "lead": Counter()}
    for part, kind in ((rhythm_part, "rhythm"), (lead_part, "lead")):
        tid = mapping[part]
        for col in cols[part]:
            o, want = col["obeat"], set(col["pitches"])
            if o < 0 or o > last or (kind == "rhythm" and len(want) < 3) or (kind == "lead" and len(want) != 1):
                continue
            lo, hi = bisect.bisect_left(allb, o - SNAP_BEATS), bisect.bisect_right(allb, o + SNAP_BEATS)
            cand = [allev[k][0] for k in range(lo, hi) if want & allev[k][1]]
            t = min(cand, key=lambda x: abs(x - o)) if cand else o
            k = _nearest(beats[tid], t, SAME_BEATS)
            got = set(ev[tid][k][1]) if k is not None else set()
            r = res[kind]
            if kind == "lead":
                r["single_cols"] += 1
                if k is not None:
                    r["with_event"] += 1
                    r["event_ge3"] += len(got) >= 3
                continue
            r["chords"] += 1
            r["exact"] += got == want
            r["pc_same"] += {p % 12 for p in got} == {p % 12 for p in want}
            r["shape_same"] += shape(sorted(want)) == shape(sorted(got))
    return {"mapping": mapping, "lead_part": lead_part, "rhythm": dict(res["rhythm"]),
            "lead": dict(res["lead"]) if song.lead else None}


SECTION_RE = re.compile(r"^(Intro|Verse|Pre-chorus|Chorus|Inter|Solo|Bridge|Outro)\.?$")
PHRASE_BARS = 4
PRACTISABLE_F1 = 0.7


def sheet_sections(song: Song) -> dict[int, str]:
    """{sheet bar: rehearsal mark} from the marks of gtr1 / gtr2 / bass (the most common label of a bar)."""
    lab: dict[int, Counter] = {}
    for part in ("gtr1", "gtr2", "bass"):
        try:
            words = (_load(song, f"{part}_marks.json") or {}).get("words") or []
        except (OSError, ValueError):
            continue
        for w in words:
            m = SECTION_RE.match(str(w.get("text", "")).strip())
            if m:
                lab.setdefault(int(w["bar"]), Counter())[m.group(1)] += 1
    return {b: c.most_common(1)[0][0] for b, c in sorted(lab.items())}


def tab_notes(job_dir: Path) -> dict[str, list[tuple[float, int]]]:
    """{gtr_1 / gtr_2: [(our beat, pitch)]} of the printed (fretted) notes, by beat."""
    tab = atomic.read_json(Path(job_dir) / "export" / "tab" / "tab.json")
    return {tid: sorted((int(n["gtick"]) / TPB, int(n["pitch"]))
                        for n in ((tab.get("tracks") or {}).get(tid) or {}).get("notes") or []
                        if n.get("string") is not None)
            for tid in ("gtr_1", "gtr_2")}


def phrases(song: Song, o: Ours, guitars: Sequence[Sequence[SheetBar]], offset_of: Callable[[int], int]
            ) -> dict[str, Any]:
    """Phrase quality: each sheet guitar's bars against our mapped part's printed notes whose beat falls in
    ``beat0 - offset .. + numerator`` (the tab's beat axis, no early shift), grouped in 4-bar phrases from each
    rehearsal mark. A phrase with at least 2 played bars counts; its F1 pools its bars. ``practisable`` = F1 >= 0.7;
    ``light`` (strict) = at most 2 wrong + missing notes per bar on average and F1 >= 0.5."""
    notes = tab_notes(o.job_dir)
    beats = {tid: [t for t, _p in ns] for tid, ns in notes.items()}
    marks = sheet_sections(song)

    def bar_counts(part_bars: Sequence[SheetBar], tid: str) -> dict[int, Counts]:
        out = {}
        for b in part_bars:
            a = b.beat0 - offset_of(b.bar)
            if a < 0 or a >= o.n_beats:
                continue
            got = Counter(p for _t, p in notes[tid][bisect.bisect_left(beats[tid], a):
                                                   bisect.bisect_left(beats[tid], a + b.beats)])
            want = Counter(b.pitches)
            k = sum((want & got).values())
            out[b.bar] = Counts(k, sum(got.values()) - k, sum(want.values()) - k)
        return out

    def pooled_f1(mp: Mapping[str, str]) -> float:
        c = Counts()
        for part, bars in zip(GUITAR_PARTS, guitars):
            for x in bar_counts(bars, mp[part]).values():
                c += x
        return c.f1

    maps = [{"gtr1": "gtr_1", "gtr2": "gtr_2"}, {"gtr1": "gtr_2", "gtr2": "gtr_1"}]
    mapping = maps[0] if pooled_f1(maps[0]) >= pooled_f1(maps[1]) else maps[1]
    phrase_of, cur, start = {}, "(pre)", min(b.bar for b in guitars[0])
    for b in sorted(x.bar for x in guitars[0]):
        if b in marks:
            cur, start = marks[b], b
        phrase_of[b] = f"{cur}@{start}#{(b - start) // PHRASE_BARS}"
    out = Counter()
    for part, bars in zip(GUITAR_PARTS, guitars):
        per_bar = bar_counts(bars, mapping[part])
        groups: dict[str, list[Counts]] = {}
        for bar, c in per_bar.items():
            groups.setdefault(phrase_of[bar], []).append(c)
        for cs in groups.values():
            if sum(1 for c in cs if c.tp + c.fn > 0) < 2:
                continue
            tot = Counts()
            for c in cs:
                tot += c
            err = sum(c.fp + c.fn for c in cs) / len(cs)
            out["phrases"] += 1
            out[f"phrases_{part}"] += 1
            out["practisable"] += tot.f1 >= 100 * PRACTISABLE_F1
            out[f"practisable_{part}"] += tot.f1 >= 100 * PRACTISABLE_F1
            out["light"] += err <= 2 and tot.f1 >= 50
    return {"mapping": mapping, **dict(out)}


# ------------------------------------------------------------------------------------------------- the table


@dataclass
class SongResult:
    song: Song
    doc: dict[str, Any]
    counts: dict[str, Counts] = field(default_factory=dict)


def evaluate_song(song: Song) -> SongResult:
    o = ours(Path(paths.JOBS) / song.job)
    guitars = [sheet_part(song, p, GUITAR) for p in GUITAR_PARTS]
    bass = [sheet_part(song, "bass", BASS)]
    off = best_offset(o.tempo, guitars, o.guitar)
    local, changes = drifting_offsets(o.tempo, guitars, o.guitar, off)

    def glob_(_b: int) -> int:
        return off

    def loc(b: int) -> int:
        return local.get(b, off)

    counts = {"merged": bag_counts(o.tempo, guitars, o.guitar, glob_),
              "merged_drift": bag_counts(o.tempo, guitars, o.guitar, loc),
              "bass": bag_counts(o.tempo, bass, o.bass, glob_),
              "bass_drift": bag_counts(o.tempo, bass, o.bass, loc)}
    mapping, parts = per_part(o, guitars, glob_)
    counts.update({f"part_{gk}": c for gk, c in parts.items()})
    doc = {"song": song.id, "title": song.title, "job": song.job, "quant": o.quant_key, "grid": o.grid_key,
           "offset": off, "offset_changes": changes, "part_map": mapping,
           "downbeats": downbeats(o, guitars[0], glob_), "downbeats_drift": downbeats(o, guitars[0], loc),
           "fretting": fretting(o, guitars, glob_), "texture": texture(song, o.job_dir, loc),
           "phrases": phrases(song, o, guitars, loc),
           **{k: c.doc() for k, c in counts.items()}}
    doc["part_mean_F1"] = round(sum(parts[gk].f1 for gk in GUITAR_PARTS) / len(GUITAR_PARTS), 2)
    return SongResult(song, doc, counts)


def evaluate(songs: Sequence[Song] | None = None) -> dict[str, Any]:
    """The whole table: per song and pooled (module doc)."""
    if songs is None:
        songs, _ = load_songs()
    results = [evaluate_song(s) for s in songs]
    pooled: dict[str, Any] = {}
    keys = ("merged", "merged_drift", "bass", "bass_drift", "part_gtr1", "part_gtr2")
    for k in keys:
        c = Counts()
        for r in results:
            c += r.counts[k]
        pooled[k] = c.doc()
    both = Counts()
    for r in results:
        both += r.counts["part_gtr1"]
        both += r.counts["part_gtr2"]
    pooled["parts"] = both.doc()
    fr = {k: sum(r.doc["fretting"][k] for r in results) for k in ("pitch_matched", "string_matched", "fret_matched")}
    pooled["fretting"] = {**fr, "string": round(fr["string_matched"] / max(1, fr["pitch_matched"]), 4),
                          "fret": round(fr["fret_matched"] / max(1, fr["pitch_matched"]), 4)}
    rh: Counter = Counter()
    ld: Counter = Counter()
    for r in results:
        rh.update(r.doc["texture"]["rhythm"])
        ld.update(r.doc["texture"]["lead"] or {})
    n = max(1, rh["chords"])
    pooled["texture"] = {"rhythm_chords": rh["chords"], "exact": round(rh["exact"] / n, 4),
                         "pc_same": round(rh["pc_same"] / n, 4), "shape_same": round(rh["shape_same"] / n, 4),
                         "lead_single_with_event": ld["with_event"],
                         "lead_ge3": round(ld["event_ge3"] / max(1, ld["with_event"]), 4)}
    ph: Counter = Counter()
    for r in results:
        ph.update({k: v for k, v in r.doc["phrases"].items() if k != "mapping"})
    pooled["phrases"] = {**dict(ph), "practisable_share": round(ph["practisable"] / max(1, ph["phrases"]), 4),
                         "light_share": round(ph["light"] / max(1, ph["phrases"]), 4),
                         **{f"practisable_share_{p}": round(ph[f"practisable_{p}"] / max(1, ph[f"phrases_{p}"]), 4)
                            for p in GUITAR_PARTS}}
    for k in ("downbeats", "downbeats_drift"):
        hit = sum(r.doc[k]["on_our_bar_line"] for r in results)
        tot = sum(r.doc[k]["sheet_bars"] for r in results)
        pooled[k] = {"sheet_bars": tot, "on_our_bar_line": hit, "share": round(hit / max(1, tot), 4)}
    mean = {k: round(sum(r.counts[k].f1 for r in results) / max(1, len(results)), 2) for k in keys}
    mean["parts"] = round(sum(r.doc["part_mean_F1"] for r in results) / max(1, len(results)), 2)
    return {"format": FORMAT, "songs": [r.doc for r in results], "pooled": pooled, "song_mean": mean}
