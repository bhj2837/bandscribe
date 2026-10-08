"""M4a edits (DESIGN 7.2 "편집", 10 M4a): a job's ``edits.jsonl`` replayed on its pipeline result.

What an edit changes, and what is recomputed from it (never a GPU stage; the tuning stays the run's):
- ``barline`` (a bar line moves, every later one with it): the grid's bars -> quantisation, fretting, score;
- ``delete`` / ``pitch`` / ``assign`` (to the other track) / ``add`` (notes): the quantised notes -> fretting, score;
- ``pin`` (string/fret): fretting keeps the note there and fingers its neighbours around it.

Notes are anchored by (track, the transcription's own onset, pitch): neither quantisation nor a bar-line edit
moves a raw onset. A note an edit added is anchored by that edit's id; a bar line by the times of its beats.
Every edit is stored absolute (the pitch, position or beat it set), so replaying the log on the same pipeline
result is deterministic (``state_hash``). After a re-run changed the transcription or the beats, an anchor takes
the nearest note of its pitch within ``MATCH_TOL_S`` (a beat within a third of a beat); otherwise the edit is
reported as unmatched, never guessed.

The log is append-only (one JSON object per line); ``undo`` / ``redo`` entries switch an earlier edit off / on. An
edit is written only after it replayed and rendered. One viewer at a time edits a job (``edits.lock``); a viewer
that sees a newer run in ``export/`` reloads it instead of writing the old one over it.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.schema.grid import TempoMap
from bandscribe.tab import alphatex as ATX
from bandscribe.tab import fretting as F
from bandscribe.tab import quantize as Q
from bandscribe.tab import stage as T

log = logging.getLogger(__name__)

EDITS_FILE = "edits.jsonl"
LOCK_FILE = "edits.lock"
VERSION = 1
TRACKS = ("guitar_all", "bass")
# track ids an older log may name: the bass was "bass_raw" until M5 (2026-10-08)
OLD_TRACK_IDS = {"bass_raw": "bass"}
NOTE_OPS = ("delete", "pitch", "assign", "add", "pin")
GRID_OPS = ("barline",)
# what the page sends (Session.command): relative commands become the absolute edits above
COMMANDS = ("delete", "pitch", "string", "assign", "add", "barline", "tap", "undo", "redo")
MATCH_TOL_S = 0.03
EXACT_TOL_S = 1e-4
BEAT_TOL = 0.3  # a bar-line anchor matches a beat within this share of a beat
ADD_STEP = Q.TPB // 4  # an added note starts on the 16th grid and lasts a 16th
MANIFEST = "manifest.json"
_ID = re.compile(r"e(\d+)$")


class EditError(RuntimeError):
    """User-facing (Korean) problem of an edit or of the edit log."""


class StaleVersion(EditError):
    """The page edited an older version of the score (reload it)."""


# -------------------------------------------------------------------------------------------------- base


@dataclass
class Base:
    """The pipeline result an edit log is replayed on (the stage dirs behind the job's current export)."""

    job_dir: Path
    song_key: str
    meta: dict[str, Any]
    title: str
    grid: dict[str, Any]
    sections: list[dict[str, Any]]
    raw: dict[str, list[dict[str, Any]]]  # track -> notes the quantiser reads (notes stage)
    quant_params: dict[str, Any]
    tab: dict[str, Any]                    # the run's tab.json: tunings, guitar_parts
    tab_params: dict[str, Any]
    score_params: dict[str, Any]
    keys: dict[str, str] = field(default_factory=dict)  # stage -> key12


def _manifest(d: Path) -> dict[str, Any]:
    return atomic.read_json(d / MANIFEST)


def export_score_key(job_dir: Path) -> str | None:
    """key12 of the score stage the job's ``export/`` comes from (None: no export)."""
    try:
        return str(atomic.read_json(Path(job_dir) / "export" / "export.json")["files"]["score.json"]["key12"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_base(job_dir: Path) -> Base:
    """The run behind ``export/``: score stage -> its tab, quant, grid, sections -> quant's notes stage."""
    job_dir = Path(job_dir)
    stages = job_dir / "stages"
    try:
        key12 = export_score_key(job_dir)
        if key12 is None:
            raise KeyError("export/export.json")
        sm = _manifest(stages / "score" / key12)
        dep = {k: str(v)[:12] for k, v in sm["deps"].items()}
        tm = _manifest(stages / "tab" / dep["tab"])
        qm = _manifest(stages / "quant" / dep["quant"])
        notes_dir = stages / "notes" / str(qm["deps"]["notes"])[:12]
        raw_dirs = {"notes": notes_dir}
        if qm["deps"].get("bass"):  # M5: the cleaned bass (a run from before M5 reads the notes stage's)
            raw_dirs["bass"] = stages / "bass" / str(qm["deps"]["bass"])[:12]
        grid = atomic.read_json(stages / "grid" / dep["grid"] / "grid.json")
        sections = atomic.read_json(stages / "sections" / dep["sections"] / "sections.json").get("sections") or []
        tab = atomic.read_json(stages / "tab" / dep["tab"] / "tab.json")
        raw = T.quant_tracks(raw_dirs)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise EditError(f"편집할 악보를 찾지 못했습니다({job_dir.name}): 먼저 `bandscribe run` 으로 악보를 만드세요. ({e})") from e
    try:
        meta = atomic.read_json(job_dir / "input" / "meta.json")
    except (OSError, ValueError):
        meta = {}
    score_params = dict(sm.get("params") or {})
    return Base(job_dir=job_dir, song_key=str(sm.get("song_key") or job_dir.name), meta=meta,
                title=T.title_for(str(score_params.get("title") or ""), meta, job_dir.name), grid=grid,
                sections=sections, raw=raw, quant_params=dict(qm.get("params") or {}), tab=tab,
                tab_params=dict(tm.get("params") or {}), score_params=score_params,
                keys={"score": key12, "tab": dep["tab"], "quant": dep["quant"], "grid": dep["grid"],
                      "notes": str(qm["deps"]["notes"])[:12]})


# ---------------------------------------------------------------------------------------------- the log


def log_path(job_dir: Path) -> Path:
    return Path(job_dir) / EDITS_FILE


def _load(p: Path) -> tuple[list[dict[str, Any]], bool]:
    """(entries, torn): a last line that is not JSON (a crash mid-append) is left out and reported as torn."""
    lines = [(i, ln) for i, ln in enumerate(p.read_bytes().decode("utf-8-sig").split("\n"), 1) if ln.strip()]
    out: list[dict[str, Any]] = []
    for j, (i, ln) in enumerate(lines):
        try:
            e = json.loads(ln)
        except ValueError as err:
            if j == len(lines) - 1:
                log.warning("edits: line %d of %s is incomplete and was skipped", i, p)
                return out, True
            raise EditError(f"편집 기록 {i}번째 줄을 읽지 못했습니다: {p}") from err
        if not isinstance(e, dict) or "op" not in e or "id" not in e:
            raise EditError(f"편집 기록 {i}번째 줄의 형식이 맞지 않습니다: {p}")
        out.append(_current_ids(e))
    return out, False


def _current_ids(e: dict[str, Any]) -> dict[str, Any]:
    """An entry with today's track ids (``OLD_TRACK_IDS``); the file itself is never rewritten."""
    for key in ("track", "to"):
        if isinstance(e.get(key), str) and e[key] in OLD_TRACK_IDS:
            e[key] = OLD_TRACK_IDS[e[key]]
    note = e.get("note")
    if isinstance(note, dict) and note.get("track") in OLD_TRACK_IDS:
        note["track"] = OLD_TRACK_IDS[note["track"]]
    return e


def read_log(job_dir: Path) -> list[dict[str, Any]]:
    p = log_path(job_dir)
    return _load(p)[0] if p.is_file() else []


def new_record(entry: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``entry`` with the next free id (one past the largest, so a removed line never makes a duplicate) and UTC time."""
    n = max((int(m.group(1)) for e in entries if (m := _ID.match(str(e.get("id", ""))))), default=0)
    return {"v": VERSION, "id": f"e{n + 1:05d}", "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **dict(entry)}


def write_record(job_dir: Path, rec: Mapping[str, Any]) -> None:
    """Append ``rec``, flushed to disk. A torn or unterminated last line is repaired first, so the new line can
    never be glued to it."""
    p = log_path(job_dir)
    if p.is_file():
        data = p.read_bytes()
        good, torn = _load(p)
        if torn or (data and not data.endswith(b"\n")):
            atomic.write_text(p, "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in good))
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(rec), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append(job_dir: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Append one edit to the job's log (tests and tools; the viewer checks an edit before it appends)."""
    rec = new_record(entry, read_log(job_dir))
    write_record(job_dir, rec)
    return rec


def stacks(entries: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """(undo stack, redo stack) of edit ids after the log: a new edit clears the redo stack."""
    undo: list[str] = []
    redo: list[str] = []
    for e in entries:
        op = e["op"]
        if op == "undo":
            if undo and undo[-1] == e.get("target"):
                redo.append(undo.pop())
        elif op == "redo":
            if redo and redo[-1] == e.get("target"):
                undo.append(redo.pop())
        else:
            undo.append(str(e["id"]))
            redo.clear()
    return undo, redo


def active(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The edits in effect, in log order."""
    on = set(stacks(entries)[0])
    return [dict(e) for e in entries if e["op"] not in ("undo", "redo") and str(e["id"]) in on]


# ---------------------------------------------------------------------------------------------- the grid


def bar_starts(grid: Mapping[str, Any]) -> list[int]:
    """Global beat index of each bar's first beat."""
    first: dict[int, int] = {}
    for i, b in enumerate(grid["beats"]):
        first.setdefault(int(b["bar"]), i)
    return [first[int(b["bar"])] for b in grid["bars"] if int(b["bar"]) in first]


def _meter(unit: str, n: int) -> tuple[int, int]:
    return (3 * n, 8) if unit == "dotted_quarter" else ((n, 8) if unit == "eighth" else (n, 4))


def _full_beats(bar: Mapping[str, Any]) -> int:
    """Beats of a full bar of ``bar``'s time signature."""
    num = int(bar["numerator"])
    return num // 3 if bar["beat_unit"] == "dotted_quarter" else num


def _beat_of(times: Sequence[float], m: Mapping[str, Any], key: str) -> int | None:
    """Beat index of a bar-line edit's ``at`` / ``to``: by its time when stored (within a third of a beat), else the
    stored index (logs written before the times were)."""
    t = m.get(f"{key}_s")
    if t is None:
        i = int(m[key])
        return i if 0 <= i < len(times) else None
    j = bisect.bisect_left(times, float(t))
    cand = [k for k in (j - 1, j) if 0 <= k < len(times)]
    if not cand:
        return None
    k = min(cand, key=lambda x: abs(times[x] - float(t)))
    ibi = (times[k + 1] - times[k]) if k + 1 < len(times) else (times[k] - times[k - 1]) if k else 0.5
    return k if abs(times[k] - float(t)) <= max(EXACT_TOL_S, BEAT_TOL * ibi) else None


def edited_grid(grid: Mapping[str, Any], moves: Sequence[Mapping[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    """``grid`` with its bar lines moved: each ``barline`` edit moves the bar line at beat ``at`` to beat ``to``
    and every later bar line by the same number of beats (the bar before it gets longer or shorter). A bar of its
    original length keeps its time signature and pickup flag; the first bar stays a pickup while it is shorter than
    a full bar; any other new length gets a signature of its own."""
    if not moves:
        return dict(grid)
    g = copy.deepcopy(dict(grid))
    beats = g["beats"]
    nb = len(beats)
    times = [float(b["t_s"]) for b in beats]
    first: dict[int, int] = {}
    for i, b in enumerate(beats):
        first.setdefault(int(b["bar"]), i)
    old = [(first[int(b["bar"])], b) for b in g["bars"] if int(b["bar"]) in first]
    old_starts = [s for s, _b in old]
    old_bars = [b for _s, b in old]
    starts = list(old_starts)
    for m in moves:
        at, to = _beat_of(times, m, "at"), _beat_of(times, m, "to")
        if at is None or to is None or at not in starts:
            report["unmatched"].append({"id": m.get("id"), "op": "barline", "why": "no bar line at that beat"})
            continue
        d = to - at
        moved = [s for s in starts if s < at] + [s + d for s in starts if s >= at]
        starts = sorted({s for s in moved if 0 <= s < nb} | {0})
        report["applied"] += 1
    first_no = int(old_bars[0]["bar"])
    new_bars = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else nb
        src = old_bars[max(0, bisect.bisect_right(old_starts, s) - 1)]
        n = e - s
        if n == int(src["n_beats"]):
            num, den, pickup = int(src["numerator"]), int(src["denominator"]), bool(src.get("pickup"))
        elif k == 0 and n < _full_beats(src):
            num, den, pickup = int(src["numerator"]), int(src["denominator"]), True
        else:
            (num, den), pickup = _meter(str(src["beat_unit"]), n), False
        end_s = times[e] if e < nb else float(old_bars[-1]["end_s"])
        no = first_no + k
        new_bars.append({**{x: src[x] for x in src if x not in ("bar", "start_s", "end_s", "n_beats", "numerator",
                                                                  "denominator", "pickup")},
                         "bar": no, "start_s": times[s], "end_s": end_s, "n_beats": n, "numerator": num,
                         "denominator": den, "pickup": pickup})
        for i in range(s, e):
            beats[i]["bar"], beats[i]["beat"], beats[i]["is_downbeat"] = no, i - s + 1, i == s
    g["bars"] = new_bars
    return g


def sections_on(grid: Mapping[str, Any], sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sections re-numbered to ``grid``'s bars by their times (a bar-line edit can merge or add a bar)."""
    bars = grid["bars"]
    if not bars:
        return [dict(s) for s in sections]

    def bar_at(t: float) -> int:
        return int(min(bars, key=lambda b: abs(float(b["start_s"]) - t))["bar"])

    out = []
    for s in sections:
        a = bar_at(float(s["start_s"]))
        e = bar_at(float(s["end_s"])) if float(s["end_s"]) < float(bars[-1]["end_s"]) - 1e-6 else int(bars[-1]["bar"]) + 1
        out.append({**dict(s), "start_bar": a, "end_bar": max(a + 1, e)})
    return out


# --------------------------------------------------------------------------------------------- the notes


def anchor(tid: str, n: Mapping[str, Any]) -> dict[str, Any]:
    a = {"track": tid, "onset_s": round(float(n["onset_s"]), 4), "pitch": int(n["pitch"])}
    if n.get("added"):
        a["added"] = n["added"]
    return a


def find(notes: Mapping[str, list[dict[str, Any]]], a: Mapping[str, Any]) -> tuple[str, int, bool] | None:
    """(track, index, exact) of the note ``a`` anchors: the same added note, else the same raw onset and pitch,
    else the nearest note of that pitch within ``MATCH_TOL_S`` (a re-run's transcription)."""
    tid = str(a.get("track"))
    ns = notes.get(tid) or []
    if a.get("added"):
        hit = [k for k, n in enumerate(ns) if n.get("added") == a["added"]]
        return (tid, hit[0], True) if hit else None
    p, t = int(a["pitch"]), float(a["onset_s"])
    best = None
    for k, n in enumerate(ns):
        if int(n["pitch"]) != p or n.get("added"):
            continue
        d = abs(float(n["onset_s"]) - t)
        if d <= MATCH_TOL_S and (best is None or d < best[0]):
            best = (d, k)
    if best is None:
        return None
    return tid, best[1], best[0] <= EXACT_TOL_S


def _added_note(e: Mapping[str, Any], tm: TempoMap, instrument: str) -> dict[str, Any]:
    g, d = int(e["gtick"]), int(e.get("dur") or ADD_STEP)
    on = float(tm.beat_to_time(g / Q.TPB))
    bar, start = tm._bar_of_beat(g / Q.TPB)
    return {"onset_s": round(on, 4), "offset_s": round(float(tm.beat_to_time((g + d) / Q.TPB)), 4),
            "pitch": int(e["pitch"]), "instrument": instrument, "gtick": g, "dur": d, "bar": int(bar),
            "tick": g - int(round(start * Q.TPB)), "added": str(e["id"])}


def _instrument_of(tid: str, ns: Sequence[Mapping[str, Any]]) -> str:
    if tid == T.BASS:
        return "electric_bass"
    cls = [str(n.get("instrument")) for n in ns if n.get("instrument")]
    return sorted(set(cls), key=lambda c: (-cls.count(c), c))[0] if cls else "distorted_electric_guitar"


def apply_note_edit(notes: dict[str, list[dict[str, Any]]], e: Mapping[str, Any], tm: TempoMap,
                    report: dict[str, Any]) -> None:
    op = e["op"]
    if op == "add":
        tid = str(e["track"])
        notes.setdefault(tid, []).append(_added_note(e, tm, _instrument_of(tid, notes.get(tid) or [])))
        report["applied"] += 1
        return
    hit = find(notes, e["note"])
    if hit is None:
        report["unmatched"].append({"id": e.get("id"), "op": op, "note": dict(e["note"])})
        return
    tid, k, exact = hit
    if not exact:
        report["fuzzy"] += 1
    n = notes[tid][k]
    if op == "delete":
        notes[tid].pop(k)
    elif op == "pitch":
        n["pitch"] = int(e["to"])
        n.pop("pin", None)  # the position cannot sound the new pitch
    elif op == "assign":
        to = str(e["to"])
        notes[tid].pop(k)
        n.pop("pin", None)
        n["instrument"] = _instrument_of(to, notes.get(to) or [])
        notes.setdefault(to, []).append(n)
    elif op == "pin":
        n["pin"] = [int(e["string"]), int(e["fret"])]
    report["applied"] += 1


# --------------------------------------------------------------------------------------------- replay


@dataclass
class State:
    grid: dict[str, Any]
    sections: list[dict[str, Any]]
    quant: dict[str, Any]
    tab: dict[str, Any]
    report: dict[str, Any]
    n_active: int


class _Cache:
    """Last quantisation (per bar-line edit list) and fretting (per track content) of a session."""

    def __init__(self) -> None:
        self.quant_key: str | None = None
        self.quant: dict[str, Any] | None = None
        self.grid: dict[str, Any] | None = None
        self.grid_report: dict[str, Any] = {}
        self.frets: dict[str, tuple[str, dict[str, Any]]] = {}


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def apply(base: Base, edits: Sequence[Mapping[str, Any]], cache: _Cache | None = None) -> State:
    """Replay ``edits`` (active ones, log order) on ``base``."""
    report: dict[str, Any] = {"applied": 0, "unmatched": [], "fuzzy": 0}
    moves = [e for e in edits if e["op"] in GRID_OPS]
    qkey = _digest([[m.get("id"), m["at"], m["to"], m.get("at_s"), m.get("to_s")] for m in moves])
    if cache is not None and cache.quant_key == qkey and cache.quant is not None:
        grid, quant0 = cache.grid, cache.quant
        report["applied"] += int(cache.grid_report.get("applied", 0))
        report["unmatched"] += list(cache.grid_report.get("unmatched", []))
    else:
        grid_report: dict[str, Any] = {"applied": 0, "unmatched": []}
        grid = edited_grid(base.grid, moves, grid_report)
        quant0 = Q.quantize(base.raw, grid, sections_on(grid, base.sections) if moves else base.sections,
                            base.quant_params)
        report["applied"] += grid_report["applied"]
        report["unmatched"] += grid_report["unmatched"]
        if cache is not None:
            cache.quant_key, cache.quant, cache.grid, cache.grid_report = qkey, quant0, grid, grid_report
    sections = sections_on(grid, base.sections) if moves else list(base.sections)
    # note edits change the notes' own fields only: copy the note dicts, share the rest of the quantisation
    quant = {**quant0, "tracks": {tid: {**tr, "notes": [dict(n) for n in tr.get("notes") or []],
                                        "stats": dict(tr.get("stats") or {})}
                                  for tid, tr in (quant0.get("tracks") or {}).items()}}
    tm = TempoMap.from_dict(grid)
    notes = {tid: Q.notes_of(quant, tid) for tid in TRACKS}
    for e in edits:
        if e["op"] in NOTE_OPS:
            apply_note_edit(notes, e, tm, report)
    for tid in TRACKS:
        ns = sorted(notes.get(tid) or [], key=lambda n: (int(n["gtick"]), int(n["pitch"]), float(n["onset_s"])))
        tr = quant["tracks"].setdefault(tid, {"notes": [], "stats": {}})
        tr["notes"] = ns
        tr.setdefault("stats", {})["n"] = len(ns)
    tun = base.tab.get("tuning") or {}
    tracks: dict[str, Any] = {}
    for tid, inst in ((T.BASS, "bass"), ("guitar_all", "guitar")):
        sub = {"tracks": {tid: quant["tracks"][tid]}}
        key = _digest([tid, quant["tracks"][tid]["notes"], tun.get(inst), base.tab_params])
        if cache is not None and tid in cache.frets and cache.frets[tid][0] == key:
            tr = cache.frets[tid][1]
        else:
            tr = T.fret_tracks(sub, {inst: tun.get(inst)}, base.tab_params).get(tid)
            if cache is not None:
                cache.frets[tid] = (key, tr)
        if tr is not None:
            tracks[tid] = tr
    tab = {**{k: v for k, v in base.tab.items() if k != "tracks"}, "tracks": tracks}
    return State(grid=grid, sections=sections, quant=quant, tab=tab, report=report, n_active=len(edits))


def state_hash(state: State) -> str:
    """Hash of what an edited score says: bars, and per track (tick, length, pitch, string, fret) of every note."""
    rows: dict[str, Any] = {"bars": [[b["bar"], b["n_beats"], b["numerator"], b["denominator"]] for b in state.grid["bars"]]}
    for tid in TRACKS:
        tr = state.tab["tracks"].get(tid) or {}
        if tr.get("fretted"):
            rows[tid] = [[n["gtick"], n["dur"], n["pitch"], n["string"], n["fret"]] for n in tr["notes"]]
        else:
            rows[tid] = [[n["gtick"], n["dur"], n["pitch"]] for n in Q.notes_of(state.quant, tid)]
    return _digest(rows)


# ------------------------------------------------------------------------------------------ the score


def note_map(score: ATX.WScore, state: State) -> dict[str, Any]:
    """Where every note is written: ``pos`` = {"<track>:<bar>:<beat>:s<string>|p<pitch>": note id} (alphaTab's
    track/bar/beat indices; strings 1 = highest), ``at`` = {note id: [track, bar, beat, key]} (its first written
    piece), ``notes`` = {note id: what the editor shows}."""
    pos: dict[str, str] = {}
    at: dict[str, list[Any]] = {}
    for ti, tr in enumerate(score.tracks):
        for bi, beats in enumerate(tr.beats):
            for k, be in enumerate(beats):
                for wn in be.notes:
                    key = f"s{wn.string}" if tr.fretted else f"p{wn.pitch}"
                    if wn.src:
                        pos[f"{ti}:{bi}:{k}:{key}"] = wn.src[0]
                        for uid in wn.src:
                            at.setdefault(uid, [ti, bi, k, key])
    notes: dict[str, Any] = {}
    for tid in TRACKS:
        tr = state.tab["tracks"].get(tid) or {}
        frets = {int(n["i"]): n for n in tr.get("notes") or []} if tr.get("fretted") else {}
        for k, n in enumerate(Q.notes_of(state.quant, tid)):
            f = frets.get(k) or {}
            notes[f"{tid}:{k}"] = {"track": tid, "pitch": int(f.get("pitch", n["pitch"])), "bar": int(n["bar"]),
                                   "string": f.get("string"), "fret": f.get("fret"), "pinned": bool(f.get("pinned")),
                                   "added": bool(n.get("added"))}
    tracks = [{"id": tid, "fretted": bool((state.tab["tracks"].get(tid) or {}).get("fretted"))}
              for tid in TRACKS if Q.notes_of(state.quant, tid)]
    return {"tracks": tracks, "pos": pos, "at": at, "notes": notes}


def render(base: Base, state: State) -> tuple[ATX.WScore, str]:
    score, problems = T.build_score(base.title, state.quant, state.tab, state.grid, state.sections, base.score_params)
    if problems:
        raise EditError(f"편집한 악보의 리듬 표기에 문제가 있습니다(버그): {problems[:3]}")
    return score, ATX.write(score)


@dataclass
class Built:
    """A replayed and rendered score, ready to swap into a session."""

    state: State
    tex: str
    map: dict[str, Any]
    bars: list[int]  # musical bar number of each written bar (alphaTab's bar index)
    hash: str


def build(base: Base, edits: Sequence[Mapping[str, Any]], cache: _Cache | None = None) -> Built:
    """Replay and render (raises when the result cannot be written, e.g. every note deleted)."""
    state = apply(base, edits, cache)
    score, tex = render(base, state)
    return Built(state, tex, note_map(score, state), [int(b.bar) for b in score.bars], state_hash(state))


def build_tolerant(base: Base, edits: Sequence[Mapping[str, Any]], cache: _Cache | None = None
                   ) -> tuple[Built, list[str]]:
    """``build`` of every edit, or, when that fails (a log from another run, a hand-edited log), of every edit
    that can be added one by one: (result, ids of the edits left out)."""
    try:
        return build(base, edits, cache), []
    except Exception as e:  # noqa: BLE001 - fall back below; the edits left out are reported
        log.warning("edits: the log does not replay as a whole (%s); dropping the edits that fail", e)
    kept: list[Mapping[str, Any]] = []
    skipped: list[str] = []
    built = build(base, [], cache)  # the run itself (the score stage wrote it)
    for e in edits:
        try:
            built = build(base, [*kept, e], cache)
            kept.append(e)
        except Exception:  # noqa: BLE001
            skipped.append(str(e.get("id")))
    return built, skipped


# ----------------------------------------------------------------------------------------------- export

SCORE_FILES = ("tab/score.gp5", "tab/score.alphatex", "score.json", "tab/tab.json",
               *(f"tab/{t}.txt" for t in TRACKS), *(f"midi/{t}_quantized.mid" for t in TRACKS))


def export(base: Base, state: State, h: str) -> dict[str, Any]:
    """Write the edited score over the job's ``export/`` score files (alphaTex, GP5, text tabs, quantised MIDI,
    score.json, tab.json) and record them in ``export.json``: as made from the edit log, or, with no edit in
    effect, as the run's own files again. Every file is written aside first; the GP5 (the file another program may
    hold open) replaces its old copy first, so a refusal leaves the old export whole."""
    out = base.job_dir / "export"
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".edits-", dir=out))
    try:
        T.write_score(tmp, song_key=base.song_key, meta=base.meta, title=base.title, quant=state.quant,
                      tab=state.tab, grid=state.grid, sections=state.sections, params=base.score_params)
        atomic.write_json(tmp / "tab" / "tab.json", state.tab)
        made = [r for r in SCORE_FILES if (tmp / r).is_file()]
        for rel in made:  # SCORE_FILES lists the GP5 first
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            atomic._replace(str(tmp / rel), out / rel)
        for rel in SCORE_FILES:
            if rel not in made and (out / rel).is_file():  # e.g. no fretted track left: no GP5
                (out / rel).unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    exp_path = out / "export.json"
    try:
        exp = atomic.read_json(exp_path)
    except (OSError, ValueError):
        exp = {"format": "bandscribe.export/1", "files": {}}
    files = exp.setdefault("files", {})
    for rel in SCORE_FILES:
        if not (out / rel).is_file():
            files.pop(rel, None)
            continue
        sha = atomic.sha256_file(out / rel)
        if state.n_active:
            files[rel] = {"stage": "edits", "key12": base.keys.get("score"), "edits": state.n_active,
                          "state": h[:16], "sha256": sha}
        else:
            stage = "tab" if rel == "tab/tab.json" else "score"
            files[rel] = {"stage": stage, "key12": base.keys.get(stage), "sha256": sha}
    if state.n_active:
        exp["edits"] = {"file": EDITS_FILE, "active": state.n_active, "state": h,
                        "unmatched": len(state.report["unmatched"])}
    else:
        exp.pop("edits", None)
    atomic.write_json(exp_path, exp)
    return exp.get("edits") or {"active": 0, "state": h, "unmatched": 0}


def reapply(job_dir: Path) -> dict[str, Any] | None:
    """After ``bandscribe run`` published a fresh export: replay the log on it (None when there is nothing to do)."""
    job_dir = Path(job_dir)
    edits = active(read_log(job_dir))
    try:
        had = bool(atomic.read_json(job_dir / "export" / "export.json").get("edits"))
    except (OSError, ValueError):
        had = False
    if not edits and not had:
        return None
    base = load_base(job_dir)
    built, skipped = build_tolerant(base, edits)
    info = export(base, built.state, built.hash)
    return {**info, "applied": built.state.report["applied"], "fuzzy": built.state.report["fuzzy"],
            "unmatched_edits": built.state.report["unmatched"], "skipped": skipped}


# --------------------------------------------------------------------------------------------- session


class _JobLock:
    """One editor per job: an OS lock on ``edits.lock`` held while a session lives (freed if the process dies)."""

    def __init__(self, job_dir: Path) -> None:
        self.path = Path(job_dir) / LOCK_FILE
        self._f = open(self.path, "a+b")  # noqa: SIM115 - held for the session's life
        try:
            if os.name == "nt":
                import msvcrt

                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self._f.close()
            raise EditError("다른 뷰어가 이 곡을 편집하고 있습니다. 그 창을 닫거나 --read-only 로 여세요.") from e

    def release(self) -> None:
        if self._f.closed:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._f.close()


class _Exporter:
    """Writes the latest edited score to ``export/`` in the background (writing GP5, MIDI and score.json takes
    ~1 s on a 3.5 min song; the page should not wait for it). Bursts of edits are written once."""

    DEBOUNCE_S = 0.3

    def __init__(self, write: Any) -> None:
        self._write = write
        self._cv = threading.Condition()
        self._pending: tuple[State, str] | None = None
        self._busy = False
        self._t = threading.Thread(target=self._loop, name="bandscribe-export", daemon=True)
        self._t.start()

    def submit(self, state: State, h: str) -> None:
        with self._cv:
            self._pending = (state, h)
            self._cv.notify_all()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._pending is None:
                    self._cv.wait()
            time.sleep(self.DEBOUNCE_S)
            with self._cv:
                job, self._pending = self._pending, None
                self._busy = True
            try:
                if job is not None:
                    self._write(*job)
            except Exception:  # noqa: BLE001 - the writer records its own errors; the thread must live on
                log.exception("edits: background export failed")
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait until everything submitted is written (True) or ``timeout`` passed (False)."""
        end = time.monotonic() + timeout
        with self._cv:
            while self._pending is not None or self._busy:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True


class Session:
    """One job open in the viewer: the base, the log, the current state and its rendering (thread-safe)."""

    def __init__(self, job_dir: Path) -> None:
        self.job_dir = Path(job_dir)
        self._lock_file = _JobLock(self.job_dir)
        try:
            self.lock = threading.RLock()
            self.version = 0
            self.export_error = ""
            self.skipped: list[str] = []
            self._load()
            try:
                had = atomic.read_json(self.job_dir / "export" / "export.json").get("edits") or {}
            except (OSError, ValueError):
                had = {}
            # export/ follows the log: after a run rebuilt it, or when the log was emptied by hand
            if (self.state.n_active and had.get("state") != self.hash) or (not self.state.n_active and had):
                self._export(self.state, self.hash)
            self.exporter = _Exporter(self._export)
        except BaseException:
            self._lock_file.release()
            raise

    # state ----------------------------------------------------------------------------------------------

    def _load(self) -> None:
        """(Re)load the run behind ``export/`` and replay the log on it."""
        self.base = load_base(self.job_dir)
        self.cache = _Cache()
        self.entries = read_log(self.job_dir)
        built, self.skipped = build_tolerant(self.base, active(self.entries), self.cache)
        if self.skipped:
            log.warning("edits: %d edits of %s do not apply and are left out: %s", len(self.skipped),
                        self.job_dir.name, self.skipped)
        self._set(built)

    def _set(self, built: Built) -> None:
        self.state, self.tex, self.map, self.score_bars, self.hash = (built.state, built.tex, built.map, built.bars,
                                                                     built.hash)
        self.version += 1

    def _active(self, entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [e for e in active(entries) if str(e["id"]) not in self.skipped]

    def _base_moved(self) -> bool:
        """A newer run published ``export/`` since this session loaded it."""
        key = export_score_key(self.job_dir)
        return key is not None and key != self.base.keys.get("score")

    def _follow_run(self) -> None:
        if self._base_moved():
            self.exporter.flush(30.0)
            self._load()
            raise StaleVersion("새로 실행한 결과로 다시 불러왔습니다. 편집은 그 위에 다시 적용했습니다.")

    def _export(self, state: State, h: str) -> None:
        if self._base_moved():  # never write the old run over a newer one
            self.export_error = "새 실행 결과가 있어 내보내지 않았습니다. 페이지를 새로 고치세요."
            return
        try:
            export(self.base, state, h)
            self.export_error = ""
        except Exception as e:  # noqa: BLE001 - e.g. the GP5 is open in a program that locks it; the log keeps the edit
            self.export_error = f"내보내기 실패(파일이 다른 프로그램에 열려 있을 수 있습니다): {e}"
            log.warning("edits: %s", self.export_error)

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait for the background export."""
        return self.exporter.flush(timeout)

    def close(self, timeout: float = 30.0) -> bool:
        """Finish the export and let another viewer edit the job (the viewer calls this when it stops)."""
        ok = self.flush(timeout)
        self._lock_file.release()
        return ok

    def info(self) -> dict[str, Any]:
        undo, redo = stacks(self.entries)
        st = self.state
        return {"version": self.version, "hash": self.hash, "edits": st.n_active, "undo": len(undo), "redo": len(redo),
                "unmatched": len(st.report["unmatched"]), "skipped": len(self.skipped),
                "export_error": self.export_error}

    def snapshot(self) -> dict[str, Any]:
        """What the page loads: the alphaTex, the note map and the version info, consistent with each other."""
        with self.lock:
            if self._base_moved():
                self.exporter.flush(30.0)
                self._load()
            return {**self.info(), "tex": self.tex, "map": self.map}

    # UI commands -> absolute edits ---------------------------------------------------------------

    def _note(self, uid: str) -> tuple[str, int, dict[str, Any]]:
        try:
            tid, k = str(uid).rsplit(":", 1)
            return tid, int(k), Q.notes_of(self.state.quant, tid)[int(k)]
        except (ValueError, IndexError) as e:
            raise EditError(f"음을 찾지 못했습니다: {uid}") from e

    def _resolve(self, cmd: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
        """(edit to append or None, message). Relative commands (+1 semitone, next string, a tap) become the
        absolute pitch / position / bar line they produce now."""
        c = str(cmd.get("cmd"))
        st = self.state
        if c not in COMMANDS:
            raise EditError(f"알 수 없는 편집: {c}")
        if c in ("undo", "redo"):
            undo, redo = stacks(self.entries)
            stack = undo if c == "undo" else redo
            if not stack:
                return None, "되돌릴 편집이 없습니다" if c == "undo" else "다시 할 편집이 없습니다"
            return {"op": c, "target": stack[-1]}, ""
        if c == "add":
            tid = str(cmd.get("track") or "guitar_all")
            if tid not in TRACKS:
                raise EditError(f"알 수 없는 트랙: {tid}")
            tm = TempoMap.from_dict(st.grid)
            beat = float(tm.time_to_beat(float(cmd["t_s"])))
            g = int(round(beat * Q.TPB / ADD_STEP)) * ADD_STEP
            pitch = int(cmd.get("pitch") or 0)
            if not pitch:
                ps = sorted(int(n["pitch"]) for n in Q.notes_of(st.quant, tid)) or [40 if tid == T.BASS else 52]
                pitch = ps[len(ps) // 2]
            return {"op": "add", "track": tid, "gtick": g, "dur": ADD_STEP, "pitch": pitch}, ""
        if c in ("barline", "tap"):
            starts = bar_starts(st.grid)
            times = [float(b["t_s"]) for b in st.grid["beats"]]
            if c == "barline":
                bi = int(cmd["bar_index"])  # alphaTab's index of the written bar
                bars = [int(b["bar"]) for b in st.grid["bars"]]
                no = self.score_bars[bi] if 0 <= bi < len(self.score_bars) else None
                if no not in bars:
                    return None, "이 마디는 박 격자 밖이라 옮길 수 없습니다"
                k = bars.index(no)
                at, to = starts[k], starts[k] + int(cmd["delta"])
            else:
                tm = TempoMap.from_dict(st.grid)
                beat = int(round(float(tm.time_to_beat(float(cmd["t_s"])))))
                beat = max(0, min(beat, len(times) - 1))
                j = bisect.bisect_right(starts, beat) - 1
                if j >= 0 and starts[j] == beat:
                    return None, "이미 마디 첫 박입니다"
                prev_ = starts[j] if j >= 0 else None
                nxt = starts[j + 1] if j + 1 < len(starts) else None
                at = nxt if prev_ is None or (nxt is not None and nxt - beat <= beat - prev_) else prev_
                to = beat
            if not 0 <= to < len(times) or to == at:
                return None, "마디선을 옮길 수 없습니다"
            return {"op": "barline", "at": int(at), "to": int(to), "at_s": round(times[at], 4),
                    "to_s": round(times[to], 4)}, ""
        tid, k, n = self._note(str(cmd.get("uid")))
        a = anchor(tid, n)
        if c == "delete":
            return {"op": "delete", "note": a}, ""
        if c == "pitch":
            to = int(n["pitch"]) + int(cmd["delta"])
            if not 0 <= to <= 127:
                return None, "더 옮길 수 없습니다"
            return {"op": "pitch", "note": a, "to": to}, ""
        if c == "assign":
            to = str(cmd["track"])
            if to not in TRACKS:
                raise EditError(f"알 수 없는 트랙: {to}")
            if to == tid:
                return None, "이미 그 트랙입니다"
            return {"op": "assign", "note": a, "to": to}, ""
        # string
        tr = st.tab["tracks"].get(tid) or {}
        if not tr.get("fretted"):
            return None, "이 트랙은 오선보라 줄이 없습니다(기타 탭은 --parts 1)"
        row = next((r for r in tr["notes"] if int(r["i"]) == k), None)
        inst = F.Instrument(tuple(tr["tuning"]), max_fret=24, capo=int(tr.get("capo") or 0))
        pitch = int(row["pitch"]) if row else int(n["pitch"])
        opts = sorted(inst.positions(pitch))  # string 1 (highest) first
        if len(opts) < 2:
            return None, "이 음은 다른 줄에서 낼 수 없습니다"
        cur = (row.get("string"), row.get("fret")) if row else None
        i = opts.index(cur) if cur in opts else -1
        s, f = opts[(i + (1 if int(cmd.get("dir", 1)) >= 0 else -1)) % len(opts)]
        return {"op": "pin", "note": a, "string": int(s), "fret": int(f)}, ""

    def command(self, cmd: Mapping[str, Any]) -> dict[str, Any]:
        """Apply one UI command; returns the new version info, the note to select and a message. The edit is
        replayed and rendered before it is written to the log: an edit that cannot be written (the last note
        deleted, say) is refused and leaves the log as it was."""
        with self.lock:
            self._follow_run()
            if cmd.get("version") is not None and int(cmd["version"]) != self.version:
                raise StaleVersion("악보가 그사이 바뀌었습니다. 새로 불러옵니다.")
            entry, msg = self._resolve(cmd)
            select = None
            if entry is not None:
                rec = new_record(entry, self.entries)
                entries = [*self.entries, rec]
                try:
                    built = build(self.base, self._active(entries), self.cache)
                except Exception as e:  # noqa: BLE001 - reported to the page; nothing was written
                    if not isinstance(e, EditError):
                        log.exception("edits: %s failed", rec["op"])
                    raise EditError(f"이 편집은 적용할 수 없습니다: {e}") from e
                write_record(self.job_dir, rec)
                self.entries = entries
                self._set(built)
                self.exporter.submit(self.state, self.hash)
                select = self._selection_after(rec, cmd)
            return {**self.info(), "applied": entry is not None, "message": msg, "select": select}

    def _selection_after(self, rec: Mapping[str, Any], cmd: Mapping[str, Any]) -> str | None:
        """The id of the note the edit left in place (to keep it selected), or None. Undo selects the note as it
        was before the undone edit (a deleted note comes back selected), redo as the edit left it."""
        op = rec["op"]
        if op in ("undo", "redo"):
            tgt = next((e for e in self.entries if e.get("id") == rec.get("target")), None)
            if tgt is None or tgt["op"] not in NOTE_OPS:
                return None
            if op == "undo":
                if tgt["op"] == "add":
                    return None
                hit = find({tid: Q.notes_of(self.state.quant, tid) for tid in TRACKS}, tgt["note"])
                return f"{hit[0]}:{hit[1]}" if hit else None
            rec, op = tgt, tgt["op"]
        if op in ("barline", "delete"):
            return None
        if op == "add":
            a = {"track": rec["track"], "added": rec["id"]}
        else:
            a = dict(rec["note"])
            if op == "pitch":
                a["pitch"] = rec["to"]
            elif op == "assign":
                a["track"] = rec["to"]
        hit = find({tid: Q.notes_of(self.state.quant, tid) for tid in TRACKS}, a)
        return f"{hit[0]}:{hit[1]}" if hit else None
