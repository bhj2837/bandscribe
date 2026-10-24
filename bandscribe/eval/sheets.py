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
                  tempo_hint=bool(s.get("tempo_hint", False))) for s in doc.get("songs") or []]
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
    return [SheetBar(b["bar"], b0[b["bar"]], b["numerator"],
                     [tun[n - x["string"]] + x["fret"] for c in b["columns"] for x in c["notes"]
                      if not x["dead"] and not x["tied"] and x["fret"] is not None])
            for b in doc["bars"]]


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
                notes("guitar_all"), notes("bass"))


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
    for k in ("downbeats", "downbeats_drift"):
        hit = sum(r.doc[k]["on_our_bar_line"] for r in results)
        tot = sum(r.doc[k]["sheet_bars"] for r in results)
        pooled[k] = {"sheet_bars": tot, "on_our_bar_line": hit, "share": round(hit / max(1, tot), 4)}
    mean = {k: round(sum(r.counts[k].f1 for r in results) / max(1, len(results)), 2) for k in keys}
    mean["parts"] = round(sum(r.doc["part_mean_F1"] for r in results) / max(1, len(results)), 2)
    return {"format": FORMAT, "songs": [r.doc for r in results], "pooled": pooled, "song_mean": mean}
