"""Tab pipeline stages (M3; ``bandscribe.pipeline.graph`` owns names and deps):

- ``a4`` (CPU): ``a4.json`` - the recording's A4 offset in cents from the pitched stem views
  (``bandscribe.tab.a4``); a Korean warning from 25 cents (varispeed is E16, not applied).
- ``quant`` (CPU): ``quant.json`` - the guitar notes (track ``guitar_all``; M7a: from the parts stage, each with its
  part ``line``; without one, ``notes/guitar_all.json``) and the bass stage's ``bass.json`` (track ``bass``; M5: the
  pass-1 bass after the silence gate, cleaned with the F0 tracks; a run from before M5 used ``notes/bass_raw.json``)
  quantised on the grid with one subdivision decision per beat, the meter re-read (shuffle vs 12/8) and the
  triplet feel per section (``bandscribe.tab.quantize``); ``parts`` = the parts stage's count and Line names.
- ``tuning`` (CPU): ``tuning.json`` - the song's guitar and bass tuning chosen jointly (``bandscribe.tab.tuning``,
  fake low guitar notes left out of the evidence) on the merged guitar and the bass, quantised like ``quant`` but
  without the parts stage: another part count (``--parts``) reuses it (2026-10-17; it was inside ``tab``).
- ``tab`` (CPU): ``tab.json`` + ``text/<track>.txt`` - the tuning stage's tunings, the bass fretted
  (``bandscribe.tab.fretting``), each guitar part (``gtr_1``..``gtr_4``, M7a) fretted as one player's part, with a
  tuning of its own when its notes clearly prefer one of the song's best few (``line_tunings``); the merged guitar
  and the Unassigned notes stay staff + MIDI only (DESIGN 4.2).
- ``score`` (CPU): ``tab/score.alphatex`` (every track; staffs where not fretted; one ``\\sync`` per bar on the
  recording's time), ``tab/score.gp5`` (fretted tracks), ``tab/<track>.txt``, ``midi/<track>_quantized.mid`` and
  ``score.json`` (schema v0), all from the same spelled bars (``bandscribe.tab.notation``). Track order and names:
  the Part view (DESIGN 4.9) - the parts, ``All guitars (reference)``, ``Unassigned (review)``, ``Bass``.

Stage functions read only their dep dirs, the job input and ``ctx.params`` (M1_M2_SPEC 3.1).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.schema.grid import TempoMap
from bandscribe.tab import a4 as A4
from bandscribe.tab import alphatex as ATX
from bandscribe.tab import ascii as ascii_tab
from bandscribe.tab import chords as CH
from bandscribe.tab import fretting as F
from bandscribe.tab import gp5 as GP5
from bandscribe.tab import notation as N
from bandscribe.tab import quantize as Q
from bandscribe.tab import tuning as TU

log = logging.getLogger(__name__)

A4_VIEWS = ("guitar_mono", "bass_mono", "piano_other_mono")
BASS_RAW = "bass_raw.json"  # in the notes stage dir: the pass-1 bass minus the notes over a silent stem
BASS_JSON = "bass.json"  # in the bass stage dir (M5): BASS_RAW cleaned with the F0 tracks
BASS = "bass"  # the bass track id (M5; "bass_raw" before, still read from old edit logs)
GUITAR = "guitar_all"  # the merged guitar (all parts); its notes carry the parts stage's ``line`` (M7a)
LINES_JSON = "guitar_lines.json"  # in the parts stage dir (M7a): GUITAR's notes with their line
PARTS_FORMAT = "bandscribe.guitar_lines/2"
PART_LINES = ("1", "2", "3", "4")  # a part's line value is its number in the Part view
UNASSIGNED = "unassigned"  # the staff of the guitar notes no part took (DESIGN 4.2 "Unassigned (review)")
# M7a: tracks derived from GUITAR by its notes' line. Their notes are GUITAR's notes (uid ``guitar_all:<index>``),
# so an edit of a guitar note shows in its part track too.
DERIVED = {**{f"gtr_{x}": x for x in PART_LINES}, UNASSIGNED: "u"}
LINE_TUNING_MIN_NOTES = 150  # a part's own tuning needs this many notes ...
LINE_TUNING_MARGIN = 0.3  # ... and must beat the song's guitar tuning on them by this cost per note
LINE_TUNING_CANDIDATES = 5  # ... among the song's best this many guitar tunings (each costs ~0.1 s per part)
TUNING_JSON = "tuning.json"
TUNING_FORMAT = "bandscribe.tuning/1"


class TabStageError(RuntimeError):
    """User-facing (Korean) problem of a tab stage."""


def _warn(ctx: Any, msg: str) -> None:
    from bandscribe.amt.stage import _notifier

    _notifier(ctx, "warning")(msg)


# ------------------------------------------------------------------------------------------------- a4


def a4_params(cfg: Any) -> dict[str, Any]:
    return dict(A4.DEFAULT_PARAMS)


def run_a4(ctx: Any) -> None:
    views_dir = Path(ctx.dep_dirs["stems"]) / "views"
    views = {v: A4.read_mono(views_dir / f"{v}.wav") for v in A4_VIEWS if (views_dir / f"{v}.wav").is_file()}
    if not views:
        raise TabStageError(f"stems 단계에 기준음을 잴 뷰가 없습니다: {views_dir}")
    doc = A4.estimate(views, dict(ctx.params))
    atomic.write_json(Path(ctx.out_dir) / "a4.json", doc)
    if doc["warning"]:
        _warn(ctx, doc["warning"])


# ---------------------------------------------------------------------------------------------- quant


def quant_params(cfg: Any) -> dict[str, Any]:
    """The quantiser's params plus, when on, the M10 note clean-up (E39 / E40) - left out when off so an old
    quantisation keeps its key."""
    p = dict(Q.DEFAULT_PARAMS)
    clean = {"dedupe_s": float(_cfg_get(cfg, "tab.dedupe_s", 0.0)), "min_dur_s": float(_cfg_get(cfg, "tab.min_dur_s", 0.0))}
    if any(v > 0 for v in clean.values()):
        p["cleanup"] = clean
    return p


def quant_tracks(dep_dirs: Mapping[str, Path], cleanup: Mapping[str, float] | None = None
                 ) -> dict[str, list[dict[str, Any]]]:
    """{track id: notes} the quantiser reads: guitar classes of the guitar pass (notes stage, without the notes over
    a silent stem; from the parts stage when there is one, M7a: the same notes with their ``line``) and the bass
    stage's cleaned bass (M5; without a ``bass`` dep - a run from before M5 - the gated pass-1 bass of the notes
    stage)."""
    lines_dir = dep_dirs.get("parts")
    if lines_dir is not None and (Path(lines_dir) / LINES_JSON).is_file():
        gtr = atomic.read_json(Path(lines_dir) / LINES_JSON).get("notes") or []
    else:
        gtr = atomic.read_json(Path(dep_dirs["notes"]) / "guitar_all.json").get("notes") or []
    bass_dir = dep_dirs.get("bass")
    if bass_dir is not None and (Path(bass_dir) / BASS_JSON).is_file():
        bass = atomic.read_json(Path(bass_dir) / BASS_JSON).get("notes") or []
    else:
        bass_path = Path(dep_dirs["notes"]) / BASS_RAW
        bass = (atomic.read_json(bass_path).get("notes") or []) if bass_path.is_file() else []
    if cleanup:  # M10 (E39 / E40): short notes out of both, then duplicates within a part
        from bandscribe.tab import cleanup as CL

        # keep each note's index in the stage's list: quant's ``i`` must stay the parts stage's index (the part
        # count switch and the repeat links look notes up by it, tab/edits.py), not the position after the clean-up
        gtr = [{**n, Q.SRC_INDEX: k} for k, n in enumerate(gtr)]
        bass = [{**n, Q.SRC_INDEX: k} for k, n in enumerate(bass)]
        gtr = CL.drop_short(gtr, float(cleanup.get("min_dur_s", 0.0)))[0]
        bass = CL.drop_short(bass, float(cleanup.get("min_dur_s", 0.0)))[0]
        gtr = CL.dedupe_within_line(gtr, float(cleanup.get("dedupe_s", 0.0)))[0]
    return {"guitar_all": gtr, BASS: bass}


def parts_meta(dep_dirs: Mapping[str, Path]) -> dict[str, Any] | None:
    """The parts stage's count and Lines (number, name, short, role, ...), or None without one (M7a)."""
    d = dep_dirs.get("parts")
    if d is None or not (Path(d) / LINES_JSON).is_file():
        return None
    doc = atomic.read_json(Path(d) / LINES_JSON)
    if doc.get("format") != PARTS_FORMAT:
        return None
    return {"count": doc.get("count"), "lines": doc.get("lines") or []}


def run_quant(ctx: Any) -> None:
    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    grid = atomic.read_json(dd["grid"] / "grid.json")
    sections = atomic.read_json(dd["sections"] / "sections.json").get("sections") or []
    doc = Q.quantize(quant_tracks(dd, ctx.params.get("cleanup")), grid, sections, dict(ctx.params))
    meta = parts_meta(dd)
    if meta is not None:
        doc["parts"] = meta
    atomic.write_json(Path(ctx.out_dir) / "quant.json", doc)


# ------------------------------------------------------------------------------------------------ tab


def tuning_params(cfg: Any) -> dict[str, Any]:
    """The tuning stage's params: the search, the user's fixed tunings and the quantisation it judges on - not the
    part count, so ``--parts`` does not redo it."""
    tp = tab_params(cfg)
    return {"tuning": tp["tuning"], "guitar_tuning": tp["guitar_tuning"], "bass_tuning": tp["bass_tuning"],
            "quant": quant_params(cfg)}


def song_tuning(q: Mapping[str, Any], p: Mapping[str, Any]) -> tuple[dict[str, Any], set[int]]:
    """(tunings chosen jointly for the song, indices of the fake low guitar notes) of a quantisation ``q``."""
    g_notes = Q.notes_of(q, "guitar_all")
    b_notes = Q.notes_of(q, BASS)
    fake = TU.fake_low_notes(g_notes, b_notes, float(p["tuning"]["bleed_window_s"]))
    evidence = F.group_events([n for i, n in enumerate(g_notes) if i not in fake])
    tun = TU.choose(evidence, F.group_events(b_notes), p["tuning"],
                    fixed={"guitar": p.get("guitar_tuning", "auto"), "bass": p.get("bass_tuning", "auto")})
    return tun, fake


def run_tuning(ctx: Any) -> None:
    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    p = dict(ctx.params)
    grid = atomic.read_json(dd["grid"] / "grid.json")
    sections = atomic.read_json(dd["sections"] / "sections.json").get("sections") or []
    q = Q.quantize(quant_tracks({k: v for k, v in dd.items() if k in ("notes", "bass")}, p["quant"].get("cleanup")),
                   grid, sections, dict(p["quant"]))
    tun, fake = song_tuning(q, p)
    atomic.write_json(Path(ctx.out_dir) / TUNING_JSON, {"format": TUNING_FORMAT, "tuning": tun,
                                                        "fake_low_guitar_notes": len(fake)})


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg.get(key, default) if cfg is not None and hasattr(cfg, "get") else default
    except Exception:  # noqa: BLE001 - a plain dict / partial config in tests
        return default


def tab_params(cfg: Any) -> dict[str, Any]:
    """Stage params; a user tuning (``tab.guitar_tuning`` / ``tab.bass_tuning``, aliases allowed) is stored as
    its canonical label, so "standard" and "E standard" share one cache key. An unknown name is a Korean
    ConfigError before anything runs."""
    from bandscribe.config import ConfigError

    fixed: dict[str, str] = {}
    for name, ko in (("guitar", "기타"), ("bass", "베이스")):
        v = str(_cfg_get(cfg, f"tab.{name}_tuning", "auto")).strip()
        if v.lower() == "auto":
            fixed[name] = "auto"
            continue
        h = TU.resolve(v, name)
        if h is None:
            raise ConfigError(f"{ko} 튜닝 '{v}' 을(를) 모릅니다(tab.{name}_tuning). auto 또는 다음 중 하나: "
                              f"{', '.join(TU.labels(name)[:14])} … (별칭: standard, 하프다운, drop d)")
        fixed[name] = h.label
    return {"guitar_parts": int(_cfg_get(cfg, "tab.guitar_parts", 0)), "tuning": dict(TU.DEFAULT_PARAMS),
            "weights": dict(F.DEFAULT_WEIGHTS), "model": F.DEFAULT_MODEL,
            "guitar_tuning": fixed["guitar"], "bass_tuning": fixed["bass"]}


def _octave_into_range(pitch: int, inst: F.Instrument) -> int:
    """Shift a pitch no string can play by octaves into the instrument's range (0 if it cannot be)."""
    lo = min(inst.open_pitch(s) for s in range(1, inst.n_strings + 1))
    hi = max(inst.open_pitch(s) for s in range(1, inst.n_strings + 1)) + inst.max_fret - inst.capo
    shift = 0
    while pitch + shift < lo:
        shift += 12
    while pitch + shift > hi:
        shift -= 12
    return shift if lo <= pitch + shift <= hi else 0


def _fret_track(notes: list[dict[str, Any]], tun: dict[str, Any], weights: dict[str, float],
                model: str = F.DEFAULT_MODEL) -> dict[str, Any]:
    """Fret one track. A note outside the instrument's range (a transcription octave error below a 4-string
    bass's E1, say) is written an octave (or two) inside it and flagged ``octave_shift`` instead of dropped.
    A note with ``pin`` = [string, fret] (an M4a edit) stays there and its neighbours are fretted around it."""
    inst = F.Instrument(tuple(tun["tuning"]), max_fret=24, capo=int(tun["capo"]))
    shifts = [0 if inst.positions(int(n["pitch"])) else _octave_into_range(int(n["pitch"]), inst) for n in notes]
    shifted = [{**n, "pitch": int(n["pitch"]) + d} for n, d in zip(notes, shifts)]
    events = F.group_events(shifted)
    pins = {i: (int(n["pin"][0]), int(n["pin"][1])) for i, n in enumerate(notes) if n.get("pin")}
    res = F.assign(events, inst, weights, model=model, pins=pins or None, links=slide_links(events, notes) or None)
    by_id = {a.note_id: a for a in res}
    rows = []
    for i, n in enumerate(shifted):
        a = by_id[i]
        rows.append({"i": i, "gtick": n["gtick"], "dur": n["dur"], "bar": n["bar"], "tick": n["tick"],
                     "pitch": n["pitch"], "string": a.string, "fret": a.fret, "conf": a.confidence,
                     **({"octave_shift": shifts[i]} if shifts[i] else {}),
                     **({"pinned": True} if i in pins and (a.string, a.fret) == pins[i] else {}),
                     **({k: n[k] for k in ("tech", "src", "uid") if n.get(k)})})
    check = F.playable(res, events, inst)
    check["octave_shifted"] = sum(1 for d in shifts if d)
    return {"fretted": True, "tuning": list(inst.tuning), "capo": inst.capo, "label": tun["label"], "notes": rows,
            "check": check}


def slide_links(events: list[F.Event], notes: list[dict[str, Any]]) -> set[int]:
    """Events that must share the string of the event before them: the note a legato slide (``tech.slide``) goes
    into, when both ends are single notes (M5)."""
    ev_of = {nid: e for e, ev in enumerate(events) for nid in ev.ids}
    out = set()
    for i, n in enumerate(notes):
        if (n.get("tech") or {}).get("slide") == "legato":
            e = ev_of.get(i)
            if e is not None and e + 1 < len(events) and len(events[e].ids) == 1 and len(events[e + 1].ids) == 1:
                out.add(e + 1)
    return out


def track_notes(q: Mapping[str, Any], tid: str) -> list[dict[str, Any]]:
    """Notes of a track of ``q`` (quant.json or an edited quantisation): a derived track (``DERIVED``) is the merged
    guitar's notes of that part - a ``shared`` note is in every part the song has - or, for the Unassigned staff,
    those in no part (``u``, or a note an edit added or moved to the guitar without one); each a copy carrying
    ``uid`` = ``guitar_all:<index>`` and ``src_k`` = that index. A song without parts has no derived tracks."""
    if tid not in DERIVED:
        return Q.notes_of(q, tid)
    g = Q.notes_of(q, GUITAR)
    present = {n.get("line") for n in g} & set(PART_LINES)
    if not present:
        return []
    line = DERIVED[tid]
    if tid == UNASSIGNED:
        keep = [n.get("line") not in PART_LINES and not n.get("shared") for n in g]
    else:
        keep = [n.get("line") == line or bool(n.get("shared") and line in present) for n in g]
    return [{**n, "uid": f"{GUITAR}:{k}", "src_k": k} for k, (n, ok) in enumerate(zip(g, keep)) if ok]


def split_on(q: Mapping[str, Any]) -> bool:
    """Whether the merged guitar's notes carry their parts (M7a)."""
    return any(n.get("line") in PART_LINES for n in Q.notes_of(q, GUITAR))


def _staff(t: Mapping[str, Any]) -> dict[str, Any]:
    return {"fretted": False, "tuning": t["tuning"], "capo": t["capo"], "label": t["label"], "notes": [],
            "check": None}


def fret_tracks(q: dict[str, Any], tun: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """{track id: fretting} of ``q``'s tracks with the chosen tunings (``tun`` = tab.json "tuning"; a part with a
    tuning of its own in ``tun["lines"]``): the bass, each guitar part fretted as one player's part, the merged guitar
    and the Unassigned notes as staffs (DESIGN 4.2: merged parts cannot be played by one hand)."""
    g_notes = Q.notes_of(q, GUITAR)
    b_notes = Q.notes_of(q, BASS)
    w = p["weights"]
    tracks: dict[str, Any] = {}
    if b_notes and tun.get("bass"):
        tracks[BASS] = _fret_track(b_notes, tun["bass"], w, p["model"])
    if g_notes and tun.get("guitar"):
        tracks[GUITAR] = _staff(tun["guitar"])
        for tid in DERIVED:
            notes = track_notes(q, tid)
            if not notes:
                continue
            own = (tun.get("lines") or {}).get(tid) or tun["guitar"]
            tracks[tid] = _staff(tun["guitar"]) if tid == UNASSIGNED else _fret_track(notes, own, w, p["model"])
    return tracks


def line_tunings(q: Mapping[str, Any], tun: Mapping[str, Any], p: Mapping[str, Any], fake: set[int]
                 ) -> dict[str, Any]:
    """Parts whose own notes clearly prefer another tuning than the song's guitar tuning (DESIGN M7a "Line별 튜닝
    탐색", v0): with ``LINE_TUNING_MIN_NOTES`` notes (fake low notes left out), an automatic choice that is not
    unknown, the bass held at the song's tuning, and the song's tuning at least ``LINE_TUNING_MARGIN`` dearer per
    note on them. Two or more parts only; none when the user named the guitar tuning."""
    g = tun.get("guitar")
    parts = [tid for tid in DERIVED if tid != UNASSIGNED and track_notes(q, tid)]
    if not g or len(parts) < 2 or str(p.get("guitar_tuning", "auto")).lower() != "auto":
        return {}
    # the song's best few (the part's notes are part of the song's evidence), the song's own first
    cands = list(dict.fromkeys([g["label"], *[t["label"] for t in g.get("top") or []]]))[:LINE_TUNING_CANDIDATES]
    bass = tun.get("bass") or {}
    out: dict[str, Any] = {}
    for tid in parts:
        notes = [n for n in track_notes(q, tid) if int(n["src_k"]) not in fake]
        if len(notes) < LINE_TUNING_MIN_NOTES:
            continue
        rows = TU.rank_guitar(F.group_events(notes), cands, p["tuning"], bass.get("downtune"))
        if not rows or rows[0]["label"] == g["label"]:
            continue
        best = rows[0]
        if best["per_note"] > float(p["tuning"]["unknown_cost"]):
            continue
        song_cost = next((r["cost"] for r in rows if r["label"] == g["label"]), None)
        if song_cost is not None and song_cost - best["cost"] < LINE_TUNING_MARGIN:
            continue
        out[tid] = {**{k: v for k, v in best.items() if k not in ("joint", "prior")}, "song_cost": song_cost,
                    "top": [{"label": r["label"], "cost": r["cost"]} for r in rows]}
    return out


def text_tab(tid: str, tr: dict[str, Any], q: dict[str, Any]) -> str:
    """The plain-text tab of a fretted track (``tab/<track>.txt``)."""
    name = track_name(q, tid)[0] if tid in DERIVED else tid  # a part by its Part view name (M7a)
    title = f"{name} — {tr['label']} ({' '.join(ascii_tab.string_names(tr['tuning'])[::-1])})"
    return ascii_tab.render(tr["notes"], q["bars"], tr["tuning"], title=title)


def run_tab(ctx: Any) -> None:
    p = dict(ctx.params)
    q = atomic.read_json(Path(ctx.dep_dirs["quant"]) / "quant.json")
    a4 = atomic.read_json(Path(ctx.dep_dirs["a4"]) / "a4.json")
    tdir = ctx.dep_dirs.get("tuning")
    if tdir is not None and (Path(tdir) / TUNING_JSON).is_file():
        tun = dict(atomic.read_json(Path(tdir) / TUNING_JSON)["tuning"])
        fake = TU.fake_low_notes(Q.notes_of(q, "guitar_all"), Q.notes_of(q, BASS),
                                 float(p["tuning"]["bleed_window_s"]))
    else:  # a caller without the tuning stage (tests, tools): judge here
        tun, fake = song_tuning(q, p)
    tun["lines"] = line_tunings(q, tun, p, fake)
    tracks = fret_tracks(q, tun, p)
    doc = {"format": "bandscribe.tab/1", "guitar_parts": int(p["guitar_parts"]), "tuning": tun,
           "fake_low_guitar_notes": len(fake), "a4": {k: a4.get(k) for k in ("cents", "a4_hz", "confidence")},
           "tracks": tracks}
    out = Path(ctx.out_dir)
    atomic.write_json(out / "tab.json", doc)
    for tid, tr in tracks.items():
        if tr["fretted"]:
            atomic.write_text(out / "text" / f"{tid}.txt", text_tab(tid, tr, q))
    for name, tid in (("guitar", "guitar_all"), ("bass", BASS)):
        t = tun.get(name)
        ko = "기타" if name == "guitar" else "베이스"
        if t and t.get("unknown"):
            _warn(ctx, f"{ko} 튜닝을 확신하지 못합니다(최선 {t['label']}, "
                       f"다음 {t['runner_up']['label'] if t.get('runner_up') else '-'}). 결과의 프렛 번호를 확인하세요.")
        shifted = ((tracks.get(tid) or {}).get("check") or {}).get("octave_shifted") or 0
        if t and t.get("user") and t.get("auto_label") != t["label"] and shifted:
            _warn(ctx, f"{ko} 튜닝을 지정한 대로 {t['label']} 로 썼습니다(자동 판정은 {t['auto_label']}). 이 튜닝으로는 "
                       f"낼 수 없는 음 {shifted}개를 옥타브를 옮겨 적었습니다.")
    for tid in DERIVED:  # the parts (M7a): each is one player's tab
        chk = (tracks.get(tid) or {}).get("check") or {}
        name = track_name(q, tid)[0]
        own = (tun.get("lines") or {}).get(tid)
        if own:
            _warn(ctx, f"{name}: 이 파트의 음은 곡의 기타 튜닝({tun['guitar']['label']})보다 {own['label']} 에 맞아 "
                       f"그 튜닝으로 적었습니다. 확인하세요.")
        if chk.get("dropped"):
            _warn(ctx, f"{name} 에서 한 손으로 칠 수 없는 음 {chk['dropped']}개를 탭에서 뺐습니다(MIDI·오선보에는 남음).")
        t = tun.get("guitar")
        if chk.get("octave_shifted") and t and t.get("user") and t.get("auto_label") != t["label"]:
            _warn(ctx, f"{name}: 지정한 튜닝 {t['label']} 으로 낼 수 없는 음 {chk['octave_shifted']}개를 옥타브를 옮겨 "
                       f"적었습니다.")


# ---------------------------------------------------------------------------------------------- score

GUITAR_INSTRUMENT = {"distorted_electric_guitar": "distortionguitar", "clean_electric_guitar": "electricguitarclean",
                     "acoustic_guitar": "acousticguitarsteel"}
# the Part view (DESIGN 4.9): the parts, the merged reference, the Unassigned notes, the bass
TRACK_NAMES = {GUITAR: ("All guitars (reference)", "All gtr"), UNASSIGNED: ("Unassigned (review)", "Unasg."),
               BASS: ("Bass", "Bass")}
TRACK_ORDER = (*(f"gtr_{x}" for x in PART_LINES), GUITAR, UNASSIGNED, BASS)


def track_name(q: Mapping[str, Any], tid: str) -> tuple[str, str]:
    """(name, short name) of a track: a part's from the parts stage (S9, e.g. "Gtr 1 - Rhythm (dist., L)")."""
    if tid in TRACK_NAMES:
        return TRACK_NAMES[tid]
    num = DERIVED.get(tid, tid)
    for ln in (q.get("parts") or {}).get("lines") or []:
        if str(ln.get("number")) == num:
            return str(ln["name"]), str(ln.get("short") or f"Gtr {num}")
    return f"Gtr {num}", f"Gtr {num}"


def score_params(cfg: Any) -> dict[str, Any]:
    ascii_title = str(_cfg_get(cfg, "hints.title_ascii", "") or "").strip()
    return {"tempo_change_ratio": 0.04, "title": str(_cfg_get(cfg, "hints.title", "") or "").strip(),
            **({"title_ascii": ascii_title} if ascii_title else {}),  # only when set: other songs keep their key
            "views": {"players": int(_cfg_get(cfg, "views.players", 2)),
                      "idle_policy": str(_cfg_get(cfg, "views.idle_policy", "rest")),
                      "unison_policy": str(_cfg_get(cfg, "views.unison_policy", "duplicate"))},
            "chords": {"in_score": bool(_cfg_get(cfg, "tab.chords_in_score", False))},
            "gp7": {"backing": str(_cfg_get(cfg, "tab.gp7_backing", "original"))}}


def _job_meta(ctx: Any) -> dict[str, Any]:
    try:
        return atomic.read_json(Path(ctx.store.job_dir(ctx.song_key)) / "input" / "meta.json")
    except (OSError, ValueError, AttributeError):
        return {}


def title_for(hint: str, meta: dict[str, Any], song_key: str) -> str:
    """``hints.title`` when set, else the input's file name without its extension."""
    if str(hint).strip():
        return str(hint).strip()
    name = (meta.get("source") or {}).get("filename") or meta.get("title") or song_key
    return Path(str(name)).stem


def _title(ctx: Any) -> str:
    return title_for(str((getattr(ctx, "params", None) or {}).get("title") or ""), _job_meta(ctx),
                     str(getattr(ctx, "song_key", "song")))


SLIDE_FX = {"legato": "sl", "out_down": "sod", "out_up": "sou"}


def note_fx(n: Mapping[str, Any]) -> tuple[str, ...]:
    """alphaTex effects of a quantised / fretted note (M5 bass): the ``tech`` suggestions listed in
    ``notation.WRITTEN_TECH`` (the rest only reach score.json), and an F0-only note as a ghost (bracketed) note."""
    t = {k: v for k, v in (n.get("tech") or {}).items() if k in N.WRITTEN_TECH}
    fx = ["x"] if t.get("dead") else []
    if n.get("src") == "f0":
        fx.append("g")
    if t.get("vibrato"):
        fx.append("v")
    if t.get("slide") in SLIDE_FX:
        fx.append(SLIDE_FX[t["slide"]])
    return tuple(fx)


def _events(notes: list[dict[str, Any]], fretted: bool, tid: str = "") -> list[tuple[int, int, list[N.WNote]]]:
    """(gtick, duration, notes) per onset; one note per string in a tab chord, one per pitch on a staff. Each
    written note carries the ids (``<track>:<index in quant.json>``) of the notes it stands for, and their
    technique marks (``note_fx``)."""
    by: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for k, n in enumerate(notes):
        if fretted and n.get("string") is None:
            continue  # dropped from the tab (kept in MIDI and score.json)
        uid = n.get("uid") or f"{tid}:{int(n['i']) if fretted else k}"  # a part track's note is a guitar note
        by.setdefault(int(n["gtick"]), []).append((uid, n))
    out = []
    for g in sorted(by):
        ns = sorted(by[g], key=lambda x: (-(x[1].get("string") or 0), int(x[1]["pitch"])))
        written: dict[int, N.WNote] = {}
        for uid, n in ns:
            key = int(n["string"]) if fretted else int(n["pitch"])
            if key in written:
                written[key].src += (uid,)
                written[key].fx += tuple(f for f in note_fx(n) if f not in written[key].fx)
            else:
                written[key] = N.WNote(int(n["pitch"]), n.get("string"), n.get("fret"), src=(uid,), fx=note_fx(n))
        out.append((g, max(int(n["dur"]) for _u, n in ns), list(written.values())))
    return out


VIEW_DEFAULTS = {"players": 2, "idle_policy": "rest", "unison_policy": "duplicate"}


def view_layout(quant: Mapping[str, Any], tab: Mapping[str, Any], sections: Sequence[Mapping[str, Any]],
                view: str, settings: Mapping[str, Any] | None = None, p: Mapping[str, Any] | None = None
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The tracks of a Role or Player view (M7c, ``bandscribe.parts.views``) and the view's doc (assignment,
    markers, violations): each guitar track is fretted as one player's part - with a part's own tuning when every
    part in it has that one, else the song's - and the Bass follows. Each entry: {tid, name, short, notes (the
    quantised notes, ``uid`` = the merged guitar's note), track (fretting), fretted, markers ([(bar, text)])}."""
    from bandscribe.parts import views as V

    st = {**VIEW_DEFAULTS, **(settings or {})}
    pp = dict(p or tab_params(None))
    g = Q.notes_of(quant, GUITAR)
    tun = tab.get("tuning") or {}
    if view == "role":
        doc: dict[str, Any] = {"tracks": V.role_view(g, sections)}
    elif view == "player":
        pos = {str(ln.get("number")): str(ln.get("position") or "")
               for ln in (quant.get("parts") or {}).get("lines") or []}
        doc = V.player_view(g, sections, int(st["players"]), idle_policy=str(st["idle_policy"]),
                            unison_policy=str(st["unison_policy"]), positions=pos)
    else:
        raise ValueError(f"unknown view {view!r}")
    marks: dict[str, list[tuple[int, str]]] = {}
    for m in doc.get("markers") or []:
        marks.setdefault(str(m["track"]), []).append((int(m["bar"]), str(m["text"])))
    out = []
    for t in doc["tracks"]:
        if not t["notes"] or not tun.get("guitar"):
            continue
        notes = sorted(({**g[k], "uid": f"{GUITAR}:{k}", "src_k": k} for k in t["notes"]),
                       key=lambda n: (int(n["gtick"]), int(n["pitch"])))
        parts_in = sorted({str(g[k].get("line")) for k in t["notes"]})
        owns = [(tun.get("lines") or {}).get(f"gtr_{ln}") for ln in parts_in]
        same = bool(owns) and all(o is not None and o.get("label") == owns[0].get("label") for o in owns)
        own = owns[0] if same else tun["guitar"]
        if t["id"] == "player_extra":  # several parts at once: no one hand plays it - a staff, no note dropped
            out.append({"tid": t["id"], "name": t["name"], "short": t["name"], "notes": notes,
                        "track": _staff(tun["guitar"]), "fretted": False, "markers": marks.get(t["name"], [])})
            continue
        out.append({"tid": t["id"], "name": t["name"], "short": t["name"], "notes": notes,
                    "track": _fret_track(notes, own, pp["weights"], pp["model"]), "fretted": True,
                    "markers": marks.get(t["name"], [])})
    un = track_notes(quant, UNASSIGNED)  # the notes in no part: a staff, as in the Part view
    if un and tun.get("guitar"):
        out.append({"tid": UNASSIGNED, "name": TRACK_NAMES[UNASSIGNED][0], "short": TRACK_NAMES[UNASSIGNED][1],
                    "notes": un, "track": _staff(tun["guitar"]), "fretted": False, "markers": []})
    btr = (tab.get("tracks") or {}).get(BASS)
    bn = Q.notes_of(quant, BASS)
    if bn and btr:
        out.append({"tid": BASS, "name": "Bass", "short": "Bass", "notes": bn, "track": btr,
                    "fretted": bool(btr.get("fretted")), "markers": []})
    return out, doc


def chord_marks(segments: Sequence[Mapping[str, Any]], grid: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    """(bar, tick in the bar, name) where each chord of the chart starts ("N" left out)."""
    beats = grid.get("beats") or []
    out = []
    for g in segments:
        k = int(g["start"])
        if g.get("name") in (None, CH.N) or not 0 <= k < len(beats):
            continue
        out.append((int(beats[k]["bar"]), (int(beats[k].get("beat") or 1) - 1) * Q.TPB, str(g["name"])))
    return out


def build_score(title: str, quant: dict[str, Any], tab: dict[str, Any], grid: dict[str, Any],
                sections: list[dict[str, Any]], params: dict[str, Any],
                layout: Sequence[Mapping[str, Any]] | None = None,
                chords: Sequence[Mapping[str, Any]] | None = None) -> tuple[ATX.WScore, list[str]]:
    """The written score of a job (shared bars, spelled tracks) and spelling problems (empty = consistent).
    ``layout`` (M7c, ``view_layout``) writes those tracks instead of the Part view's; ``chords`` (M9, the chart's
    segments) names the chords on the guitar tracks' beats where they start."""
    tm = TempoMap.from_dict(grid)
    family_of = {int(b): str(f) for b, f in ((quant.get("beat_families") or {}).get("*") or [])}
    tracks_src = []
    names: dict[str, tuple[str, str]] = {}
    markers: dict[str, list[tuple[int, str]]] = {}
    classes_of: dict[str, list[Any]] = {}
    if layout is not None:
        for e in layout:
            tr, fretted = e["track"], bool(e["fretted"])
            tracks_src.append((e["tid"], tr, fretted, tr["notes"] if fretted else list(e["notes"])))
            names[e["tid"]] = (e["name"], e["short"])
            markers[e["tid"]] = list(e.get("markers") or [])
            classes_of[e["tid"]] = [n.get("instrument") for n in e["notes"]]
    for tid in TRACK_ORDER if layout is None else ():
        tr = (tab.get("tracks") or {}).get(tid)
        qn = track_notes(quant, tid)
        if not qn or (tid in DERIVED and not tr):  # a part is written once the tab stage made it (M7a)
            continue
        fretted = bool(tr and tr.get("fretted"))
        tracks_src.append((tid, tr, fretted, tr["notes"] if fretted else qn))
    if not tracks_src:
        raise TabStageError("악보로 쓸 음이 없습니다(기타·베이스 모두 비어 있음)")
    note_bars = [int(n["bar"]) for _t, _r, _f, ns in tracks_src for n in ns]
    grid_first = min(int(b["bar"]) for b in quant["bars"])
    bars = N.bars_from_quant(quant, tm, min(min(note_bars), grid_first), max(note_bars), sections)
    tempi = ATX.written_tempi(bars, section_starts=[int(s["start_bar"]) for s in sections],
                              change_ratio=float(params["tempo_change_ratio"]))
    score = ATX.WScore(title=title, bars=bars, tempo_qpm=tempi)
    problems: list[str] = []
    for tid, tr, fretted, notes in tracks_src:
        spelled = N.spell_track(_events(notes, fretted, tid), bars, family_of)
        for b, beats in zip(bars, spelled):
            problems += N.check_bar(b, beats)
        if markers.get(tid):  # M7c: a view's text markers on the first beat of their bar
            at = {b.bar: i for i, b in enumerate(bars)}
            for bar_no, text in markers[tid]:
                i = at.get(int(bar_no))
                if i is not None and spelled[i]:
                    first = spelled[i][0]
                    first.text = text if not first.text else f"{first.text}; {text}"
        if chords and tid != BASS:  # M9: the chord on the beat that holds its start
            at = {b.bar: i for i, b in enumerate(bars)}
            for bar_no, tick, name in chord_marks(chords, grid):
                i = at.get(bar_no)
                if i is None:
                    continue
                k = next((j for j, x in enumerate(spelled[i]) if x.start <= tick < x.start + x.dur), None)
                free = [x for x in spelled[i][k:] if x.chord is None] if k is not None else []
                if free:  # the beat holding the change, else the next free one of the bar (review 2026-10-18)
                    free[0].chord = name
        if tid == BASS:
            instrument = "electricbassfinger"
        else:
            classes = ([n.get("instrument") for n in track_notes(quant, tid)] if layout is None
                       else classes_of.get(tid, []))
            # most frequent class; ties by name, so the output does not depend on string hashing (review)
            top = sorted(set(classes), key=lambda c: (-classes.count(c), str(c)))[0] if classes else None
            instrument = GUITAR_INSTRUMENT.get(str(top), "electricguitarclean")
        name, short = names.get(tid) or track_name(quant, tid)
        score.tracks.append(ATX.WTrack(name=name, short=short, instrument=instrument, fretted=fretted,
                                       tuning=(tr or {}).get("tuning") if fretted else None,
                                       capo=int((tr or {}).get("capo") or 0) if fretted else 0, beats=spelled,
                                       clef="f4" if tid == BASS and not fretted else None, tid=tid))
    return score, problems


def score_doc(song_key: str, meta: dict[str, Any], title: str, quant: dict[str, Any], tab: dict[str, Any],
              grid: dict[str, Any], sections: Sequence[Mapping[str, Any]] | None = None) -> Any:
    """``score.json`` (schema v0) of a job's score; with ``sections`` each part gets its role per section (M7c)."""
    from bandscribe.parts import views as V
    from bandscribe.schema.score import Confidence, Line, Note, RoleSegment, Score, SongMeta

    tm = TempoMap.from_dict(grid)
    seg_of: dict[str, list[Any]] = {}
    if sections:
        span = {int(s["start_bar"]): (float(s["start_s"]), float(s["end_s"])) for s in sections
                if s.get("start_s") is not None and s.get("end_s") is not None}
        for r in V.section_roles(Q.notes_of(quant, GUITAR), sections):
            if r["start_bar"] in span and span[r["start_bar"]][1] > span[r["start_bar"]][0]:
                a, b = span[r["start_bar"]]
                seg_of.setdefault(f"gtr_{r['line']}", []).append(RoleSegment(start_s=a, end_s=b, role=r["role"]))
    lines = []
    for tid in TRACK_ORDER:
        qn = track_notes(quant, tid)
        if not qn or (tid in DERIVED and not (tab.get("tracks") or {}).get(tid)):
            continue
        tr = (tab.get("tracks") or {}).get(tid) or {}
        fretted = bool(tr.get("fretted"))
        frets = {int(n["i"]): n for n in tr.get("notes") or []} if fretted else {}
        notes = []
        for i, n in enumerate(qn):
            on = max(0.0, tm.beat_time(int(n["gtick"]) / Q.TPB))
            off = max(on, tm.beat_time((int(n["gtick"]) + int(n["dur"])) / Q.TPB))
            f = frets.get(i) or {}
            placed = f.get("string") is not None
            flags = {"octave": True} if f.get("octave_shift") else {}
            tech = n.get("tech") or {}
            flags.update({f"slide_{tech['slide']}" if k == "slide" else k: True for k in tech})  # M5 suggestions
            if n.get("src") == "f0":
                flags["low_conf"] = True  # a note only the F0 trackers found (E13 two-voter gap fill)
            notes.append(Note(onset_s=on, offset_s=off, pitch=int(f.get("pitch", n["pitch"])),
                              string=f.get("string") if placed else None, fret=f.get("fret") if placed else None,
                              confidence=Confidence(fret=f.get("conf") if placed else None,
                                                    part=n.get("p") if tid in DERIVED else None),
                              sources=["f0"] if n.get("src") == "f0" else ["muscriptor"], flags=flags))
        notes.sort(key=lambda x: (x.onset_s, x.pitch))
        kind = ("bass" if tid == BASS else "guitar" if fretted else "unassigned" if tid == UNASSIGNED
                else "reference")
        lines.append(Line(id=tid, name=track_name(quant, tid)[0], kind=kind,
                          tuning=tr.get("tuning") if fretted else None,
                          capo=int(tr.get("capo") or 0) if fretted else None, notes=notes,
                          role_segments=seg_of.get(tid, [])))
    src = meta.get("source") or {}
    return Score(song=SongMeta(song_key=str(song_key), title=title,
                               duration_s=float(meta.get("duration_s") or grid.get("duration_s") or 0.0),
                               source_kind="youtube" if src.get("kind") == "youtube" else "file",
                               source_ref=str(src.get("filename") or src.get("ref") or "")), lines=lines)


def write_score(out: Path, *, song_key: str, meta: dict[str, Any], title: str, quant: dict[str, Any],
                tab: dict[str, Any], grid: dict[str, Any], sections: list[dict[str, Any]],
                params: dict[str, Any], job_dir: Path | None = None, stems_dir: Path | None = None) -> ATX.WScore:
    """Every file of the written score under ``out``: ``tab/score.alphatex``, ``tab/score.gp5``, ``tab/<track>.txt``,
    ``midi/<track>_quantized.mid`` and ``score.json`` (the ``score`` stage, and M4a's re-export of an edited score);
    with parts (M7c) also the Role and Player views, ``tab/views/{role,player}.{alphatex,gp5}`` and
    ``tab/views/views.json`` (each view's tracks, the Player view's assignment, markers and violations)."""
    from bandscribe.amt import midi
    from bandscribe.schema.score import save_score

    out = Path(out)
    chart = write_chords(out, quant=quant, grid=grid, sections=sections)
    in_score = bool((params.get("chords") or {}).get("in_score")) and chart is not None
    segs = chart["segments"] if in_score else None
    score, problems = build_score(title, quant, tab, grid, sections, params, chords=segs)
    if problems:
        raise TabStageError(f"리듬 표기 오류 {len(problems)}건(버그): {problems[:3]}")
    atomic.write_text(out / "tab" / "score.alphatex", ATX.write(score))
    GP5.write(score, out / "tab" / "score.gp5", title_ascii=str(params.get("title_ascii") or ""))
    write_gp7(out, score, grid, backing_audio(job_dir, str((params.get("gp7") or {}).get("backing") or "none"),
                                              stems_dir=stems_dir), job_dir=job_dir, quant=quant)
    for tid, tr in (tab.get("tracks") or {}).items():
        if tr.get("fretted"):
            atomic.write_text(out / "tab" / f"{tid}.txt", text_tab(tid, tr, quant))
    tm = TempoMap.from_dict(grid)
    for tid in TRACK_ORDER:
        qn = track_notes(quant, tid)
        if tid in DERIVED and not (qn and (tab.get("tracks") or {}).get(tid)):
            continue
        notes = [{"onset_s": max(0.0, tm.beat_time(int(n["gtick"]) / Q.TPB)),
                  "offset_s": max(0.0, tm.beat_time((int(n["gtick"]) + int(n["dur"])) / Q.TPB)),
                  "pitch": int(n["pitch"])} for n in qn]
        midi.write_performance_midi(out / "midi" / f"{tid}_quantized.mid",
                                    [(track_name(quant, tid)[0], 33 if tid == BASS else 27, notes)], grid)
    save_score(out / "score.json", score_doc(song_key, meta, title, quant, tab, grid, sections))
    write_views(out, title=title, quant=quant, tab=tab, grid=grid, sections=sections, params=params, chords=segs)
    return score


def backing_audio(job_dir: Path | None, choice: str, *, stems_dir: Path | None = None) -> Path | None:
    """The recording a ``.gp`` embeds (M9, ``tab.gp7_backing``): the job's input mix, or the no-guitar practice mix
    of the score stage's own stems dependency (review: not a newest-by-mtime guess outside the stage key), or None."""
    if choice == "none":
        return None
    if choice == "original":
        p = None if job_dir is None else Path(job_dir) / "input" / "mix_44k_f32.wav"
    else:
        p = None if stems_dir is None else Path(stems_dir) / "practice" / "mix_minus_guitar.wav"
    return p if p is not None and p.is_file() else None


def stems_dir_of(job_dir: Path, score_key: str | None) -> Path | None:
    """The stems stage dir behind a job's score stage (its manifest's deps), for an edit's re-export."""
    if not score_key:
        return None
    try:
        deps = atomic.read_json(Path(job_dir) / "stages" / "score" / str(score_key)[:12] / "manifest.json").get("deps")
        d = Path(job_dir) / "stages" / "stems" / str((deps or {})["stems"])[:12]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return d if d.is_dir() else None


def _backing_mp3(wav: Path, job_dir: Path | None) -> Path:
    """``wav`` as MP3 (libsndfile), cached in the job's ``cache/gp7`` by path, size and mtime."""
    import hashlib

    import soundfile as sf

    st = wav.stat()
    key = hashlib.sha256(f"{wav.resolve()}|{st.st_size}|{st.st_mtime_ns}|mp3/1".encode()).hexdigest()[:16]
    root = (Path(job_dir) / "cache" / "gp7") if job_dir is not None else wav.parent
    mp3 = root / f"{key}.mp3"
    if not mp3.is_file():
        import os
        import tempfile

        root.mkdir(parents=True, exist_ok=True)
        y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{key}.", suffix=".mp3", dir=root)  # a run and the editor may race
        os.close(fd)
        try:
            sf.write(tmp, y[:, :2], sr, format="MP3")
            atomic._replace(tmp, mp3)
        finally:
            Path(tmp).unlink(missing_ok=True)
    return mp3


def gp7_sync_points(score: ATX.WScore, grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One sync point per grid beat of every written bar (its bar position = beat k / the bar's beats), so a
    player follows the recording inside the bar too; a bar without grid beats gets one at its start."""
    by_bar: dict[int, list[float]] = {}
    for b in grid.get("beats") or []:
        by_bar.setdefault(int(b["bar"]), []).append(float(b["t_s"]))
    out = []
    for i, bar in enumerate(score.bars):
        ts = sorted(by_bar.get(int(bar.bar)) or [float(bar.start_s)])
        for k, t in enumerate(ts):
            out.append({"barIndex": i, "barOccurence": 0, "barPosition": round(k / len(ts), 6),
                        "millisecondOffset": round(1000.0 * max(0.0, t), 3)})
    return out


def _stats(d: list[float], unit: str) -> dict[str, float] | None:
    if not d:
        return None
    d = sorted(d)
    return {unit: len(d), "median": round(d[len(d) // 2], 2), "p95": round(d[int(0.95 * (len(d) - 1))], 2),
            "max": round(d[-1], 2)}


def gp7_drift(check: Mapping[str, Any], score: ATX.WScore, grid: Mapping[str, Any],
              quant: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Where a player of the reloaded file (its synth tick -> audio ms) puts things against the recording, |ms|:
    ``notes`` - each written guitar / bass note against its detected onset (the meaningful measure: the beats are
    sync points themselves, review 2026-10-18), ``beats`` - the grid beats (a check that bars line up)."""
    import bisect

    sync = sorted(check.get("sync") or [], key=lambda p: p[1])
    if len(sync) < 2:
        return None
    ticks, ms = [float(p[1]) for p in sync], [float(p[2]) for p in sync]

    def audio_ms(tick: float) -> float:
        j = min(max(bisect.bisect_right(ticks, tick) - 1, 0), len(ticks) - 2)
        return ms[j] + (ms[j + 1] - ms[j]) * (tick - ticks[j]) / max(1e-9, ticks[j + 1] - ticks[j])

    by_bar: dict[int, list[float]] = {}
    for b in grid.get("beats") or []:
        by_bar.setdefault(int(b["bar"]), []).append(float(b["t_s"]))
    mb = {int(bar.bar): (bar, start, dur) for (start, dur), bar in zip(check.get("bars") or [], score.bars)}
    beats = []
    for bar, start, dur in mb.values():
        ts = sorted(by_bar.get(int(bar.bar)) or [])
        beats += [abs(audio_ms(float(start) + k * float(dur) / len(ts)) - 1000.0 * t) for k, t in enumerate(ts)]
    out: dict[str, Any] = {"beats": _stats(beats, "beats")}
    for tid in (GUITAR, BASS):
        d = []
        for n in Q.notes_of(quant, tid) if quant and (quant.get("tracks") or {}).get(tid) else []:
            hit = mb.get(int(n["bar"]))
            if hit is None or n.get("onset_s") is None:
                continue
            bar, start, dur = hit
            frac = (int(n["gtick"]) - int(bar.start_gtick)) / max(1, bar.length)
            d.append(abs(audio_ms(float(start) + frac * float(dur)) - 1000.0 * float(n["onset_s"])))
        out[tid] = _stats(d, "notes")
    return out


def write_gp7(out: Path, score: ATX.WScore, grid: Mapping[str, Any], backing: Path | None, *,
              job_dir: Path | None = None, quant: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """``tab/score.gp`` (Guitar Pro 7/8 by alphaTab's Gp7Exporter from ``tab/score.alphatex``, one sync point per
    beat, ``backing`` embedded as MP3) and ``tab/score_gp.json`` (what a player reads back: sizes, sync points, the
    drift: written notes against their detected onsets). Best effort: no Node / alphaTab or a failure leaves the .gp
    out, and ``score_gp.json`` says why; a backing that cannot be encoded leaves the .gp without audio."""
    from bandscribe.eval import node as NJ

    note = None
    try:
        if not NJ.alphatab_available():
            log.info("Node/alphaTab 이 없어 GP7(.gp)은 건너뜁니다")
            return None
        with NJ._scratch() as tmp:
            sync = Path(tmp) / "sync.json"
            check = Path(tmp) / "check.json"
            atomic.write_json(sync, gp7_sync_points(score, grid))
            args: list[Any] = ["gp7", out / "tab" / "score.alphatex", out / "tab" / "score.gp", "--sync", sync,
                               "--check", check]
            mp3 = None
            if backing is not None:
                try:
                    mp3 = _backing_mp3(backing, job_dir)
                except Exception as e:  # noqa: BLE001 - the .gp without audio rather than no .gp
                    log.warning("백킹 음원을 MP3 로 만들지 못해 소리 없이 씁니다: %s", e)
                    note, backing = f"백킹 인코딩 실패: {e}", None
            if mp3 is not None:
                args += ["--audio", mp3]
            NJ.run_score(args, timeout_s=300)
            chk = atomic.read_json(check)
        doc = {"format": "bandscribe.gp7/1", "backing": None if backing is None else backing.name,
               "bytes": chk.get("bytes"), "backing_bytes": chk.get("backing_bytes"),
               "sync_points": len(chk.get("flat_sync") or []), "master_bars": chk.get("master_bars"),
               "drift_ms": gp7_drift(chk, score, grid, quant),
               "note": note or ("개인용: 백킹으로 원곡 음원이 들어 있다" if backing is not None else None)}
        atomic.write_json(out / "tab" / "score_gp.json", doc)
        return doc
    except Exception as e:  # noqa: BLE001 - logged; the .gp is left out
        log.warning("GP7(.gp)을 쓰지 못했습니다: %s", e)
        (out / "tab" / "score.gp").unlink(missing_ok=True)
        atomic.write_json(out / "tab" / "score_gp.json", {"format": "bandscribe.gp7/1", "error": str(e)[-500:]})
        return None


CHART_NOTE = ("# 자동 추정 코드 차트 (bandscribe M9) - 참고용. 기타 음에서 박마다 추정했다. 사용자 악보 세 곡 대조(근음 + 부류):\n"
              "# seisyun complex 55 %, 다른 두 곡 28 %·45 % (E31, E33). 텐션 코드와 근음 없는 보이싱에 약하다.\n"
              "# 칸 = 마디, 칸 안의 글자 = 박마다 바뀌는 코드(. = 그대로, N.C. = 기타 없음).\n\n")


def write_chords(out: Path, *, quant: dict[str, Any], grid: dict[str, Any],
                 sections: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """``chords/chords.json`` (segments, the bass cross-check) and ``chords/chords.txt`` (M9, DESIGN 6.5); best
    effort - a problem here never fails the score."""
    try:
        edges = CH.beat_edges(grid)
        res = CH.recognize(Q.notes_of(quant, GUITAR), Q.notes_of(quant, BASS) if quant.get("tracks", {}).get(BASS)
                           else [], edges)
        doc = {"format": "bandscribe.chords/1", "params": dict(CH.DEFAULTS), "segments": res["segments"],
               "checks": res["checks"], "measured": "E31/E33 root+class: seisyun complex 0.546, two other songs 0.281 / 0.452"}
        atomic.write_json(out / "chords" / "chords.json", doc)
        atomic.write_text(out / "chords" / "chords.txt", CHART_NOTE + CH.text_chart(res["segments"], grid, sections))
        return doc
    except Exception as e:  # noqa: BLE001 - logged; the chart is left out
        log.warning("코드 차트를 쓰지 못했습니다: %s", e)
        return None


def write_views(out: Path, *, title: str, quant: dict[str, Any], tab: dict[str, Any], grid: dict[str, Any],
                sections: list[dict[str, Any]], params: dict[str, Any],
                chords: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any] | None:
    """The Role and Player views of a song with guitar parts (M7c; module doc of ``bandscribe.parts.views``)."""
    if not split_on(quant) or not sections:
        return None
    settings = {**VIEW_DEFAULTS, **(params.get("views") or {})}
    doc: dict[str, Any] = {"format": "bandscribe.views/1", "settings": settings}
    for view in ("role", "player"):
        try:  # best effort: a view's problem never fails the Part view's score or an edit's export (review)
            lay, vd = view_layout(quant, tab, sections, view, settings)
            if not lay:
                continue
            score, problems = build_score(title, quant, tab, grid, sections, params, layout=lay, chords=chords)
            if problems:
                raise TabStageError(f"리듬 표기 오류 {len(problems)}건(버그): {problems[:3]}")
            atomic.write_text(out / "tab" / "views" / f"{view}.alphatex", ATX.write(score))
            GP5.write(score, out / "tab" / "views" / f"{view}.gp5", title_ascii=str(params.get("title_ascii") or ""))
        except Exception as e:  # noqa: BLE001 - logged; the view is left out
            log.warning("%s 보기를 쓰지 못했습니다: %s", view, e)
            doc.setdefault("errors", {})[view] = str(e)
            continue
        doc[view] = {"tracks": [{"id": e["tid"], "name": e["name"], "notes": len(e["notes"])} for e in lay],
                     **({k: vd[k] for k in ("assign", "markers", "violations", "copies")} if view == "player" else {})}
    atomic.write_json(out / "tab" / "views" / "views.json", doc)
    return doc


def run_score(ctx: Any) -> None:
    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    try:
        job_dir = Path(ctx.store.job_dir(ctx.song_key))
    except (AttributeError, TypeError):
        job_dir = None
    write_score(Path(ctx.out_dir), song_key=str(getattr(ctx, "song_key", "song")), meta=_job_meta(ctx),
                job_dir=job_dir, stems_dir=dd.get("stems"),
                title=_title(ctx), quant=atomic.read_json(dd["quant"] / "quant.json"),
                tab=atomic.read_json(dd["tab"] / "tab.json"), grid=atomic.read_json(dd["grid"] / "grid.json"),
                sections=atomic.read_json(dd["sections"] / "sections.json").get("sections") or [],
                params=dict(ctx.params))


# quant 4 / tab 7 / score 5: M7a Part view (guitar notes and Line names from the parts stage; K part tracks with
# their own tunings, the Unassigned staff). tab 8: the song tuning from the tuning stage, a part's own tuning among
# the song's best few (2026-10-17, the --parts switch). score 6: the Role / Player views, roles per section (M7c); 7: review fixes (view instruments, extra staff); 8: the chord chart and the GP7 file (M9); 9: M9 review (stems dep, note drift); 10: chart header with three songs (E33); 11: GP5 title in cp949 even when partly lossy, hints.title_ascii
STAGES: dict[str, dict] = {
    "tuning": {"run": run_tuning, "code_version": "1", "params": tuning_params, "device": "cpu",
               "models": lambda cfg: {}},
    "a4": {"run": run_a4, "code_version": "1", "params": a4_params, "device": "cpu", "models": lambda cfg: {}},
    "quant": {"run": run_quant, "code_version": "5", "params": quant_params, "device": "cpu",
              "models": lambda cfg: {}},
    "tab": {"run": run_tab, "code_version": "8", "params": tab_params, "device": "cpu", "models": lambda cfg: {}},
    "score": {"run": run_score, "code_version": "11", "params": score_params, "device": "cpu",
              "models": lambda cfg: {}},
}
