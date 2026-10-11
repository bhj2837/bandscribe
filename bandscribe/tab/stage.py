"""Tab pipeline stages (M3; ``bandscribe.pipeline.graph`` owns names and deps):

- ``a4`` (CPU): ``a4.json`` - the recording's A4 offset in cents from the pitched stem views
  (``bandscribe.tab.a4``); a Korean warning from 25 cents (varispeed is E16, not applied).
- ``quant`` (CPU): ``quant.json`` - the guitar notes (track ``guitar_all``; M7a: from the parts stage, each with its
  part ``line``; without one, ``notes/guitar_all.json``) and the bass stage's ``bass.json`` (track ``bass``; M5: the
  pass-1 bass after the silence gate, cleaned with the F0 tracks; a run from before M5 used ``notes/bass_raw.json``)
  quantised on the grid with one subdivision decision per beat, the meter re-read (shuffle vs 12/8) and the
  triplet feel per section (``bandscribe.tab.quantize``); ``parts`` = the parts stage's count and Line names.
- ``tab`` (CPU): ``tab.json`` + ``text/<track>.txt`` - guitar and bass tuning chosen jointly
  (``bandscribe.tab.tuning``, fake low guitar notes left out of the evidence), the bass fretted
  (``bandscribe.tab.fretting``), each guitar part (``gtr_1``..``gtr_4``, M7a) fretted as one player's part, with a
  tuning of its own when its notes clearly prefer one (``line_tunings``); the merged guitar and the Unassigned
  notes stay staff + MIDI only (DESIGN 4.2).
- ``score`` (CPU): ``tab/score.alphatex`` (every track; staffs where not fretted; one ``\\sync`` per bar on the
  recording's time), ``tab/score.gp5`` (fretted tracks), ``tab/<track>.txt``, ``midi/<track>_quantized.mid`` and
  ``score.json`` (schema v0), all from the same spelled bars (``bandscribe.tab.notation``). Track order and names:
  the Part view (DESIGN 4.9) - the parts, ``All guitars (reference)``, ``Unassigned (review)``, ``Bass``.

Stage functions read only their dep dirs, the job input and ``ctx.params`` (M1_M2_SPEC 3.1).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.schema.grid import TempoMap
from bandscribe.tab import a4 as A4
from bandscribe.tab import alphatex as ATX
from bandscribe.tab import ascii as ascii_tab
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
    return dict(Q.DEFAULT_PARAMS)


def quant_tracks(dep_dirs: Mapping[str, Path]) -> dict[str, list[dict[str, Any]]]:
    """{track id: notes} the quantiser reads: guitar classes of the guitar pass (notes stage, without the notes over
    a silent stem; from the parts stage when there is one, M7a: the same notes with their ``line``; M6b's split
    stage for an older run) and the bass stage's cleaned bass (M5; without a ``bass`` dep - a run from before M5 -
    the gated pass-1 bass of the notes stage)."""
    lines_dir = dep_dirs.get("parts") or dep_dirs.get("split")
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
    doc = Q.quantize(quant_tracks(dd), grid, sections, dict(ctx.params))
    meta = parts_meta(dd)
    if meta is not None:
        doc["parts"] = meta
    atomic.write_json(Path(ctx.out_dir) / "quant.json", doc)


# ------------------------------------------------------------------------------------------------ tab


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
    """Notes of a track of ``q`` (quant.json or an edited quantisation): a derived track (``DERIVED``: a part, or the
    Unassigned staff) is the merged guitar's notes on that line (a ``shared`` note is in every part), each a copy
    carrying ``uid`` = ``guitar_all:<index>`` and ``src_k`` = that index."""
    if tid not in DERIVED:
        return Q.notes_of(q, tid)
    line = DERIVED[tid]
    return [{**n, "uid": f"{GUITAR}:{k}", "src_k": k} for k, n in enumerate(Q.notes_of(q, GUITAR))
            if n.get("line") == line or (n.get("shared") and tid != UNASSIGNED)]


def split_on(q: Mapping[str, Any]) -> bool:
    """Whether the merged guitar's notes carry their parts (M7a)."""
    return any(n.get("line") in DERIVED.values() for n in Q.notes_of(q, GUITAR))


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
    b_events = F.group_events(Q.notes_of(q, BASS))
    bass = (tun.get("bass") or {}).get("label") or "auto"
    out: dict[str, Any] = {}
    for tid in parts:
        notes = [n for n in track_notes(q, tid) if int(n["src_k"]) not in fake]
        if len(notes) < LINE_TUNING_MIN_NOTES:
            continue
        got = TU.choose(F.group_events(notes), b_events, p["tuning"], fixed={"guitar": "auto", "bass": bass})
        lg = got.get("guitar")
        if not lg or lg["label"] == g["label"] or lg.get("unknown"):
            continue
        song_cost = next((t["cost"] for t in lg.get("top") or [] if t["label"] == g["label"]), None)
        if song_cost is not None and song_cost - lg["cost"] < LINE_TUNING_MARGIN:
            continue
        out[tid] = {**lg, "song_cost": song_cost}
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
    g_notes = Q.notes_of(q, "guitar_all")
    b_notes = Q.notes_of(q, BASS)
    fake = TU.fake_low_notes(g_notes, b_notes, float(p["tuning"]["bleed_window_s"]))
    evidence = F.group_events([n for i, n in enumerate(g_notes) if i not in fake])
    tun = TU.choose(evidence, F.group_events(b_notes), p["tuning"],
                    fixed={"guitar": p.get("guitar_tuning", "auto"), "bass": p.get("bass_tuning", "auto")})
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
    return {"tempo_change_ratio": 0.04, "title": str(_cfg_get(cfg, "hints.title", "") or "").strip()}


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


def build_score(title: str, quant: dict[str, Any], tab: dict[str, Any], grid: dict[str, Any],
                sections: list[dict[str, Any]], params: dict[str, Any]) -> tuple[ATX.WScore, list[str]]:
    """The written score of a job (shared bars, spelled tracks) and spelling problems (empty = consistent)."""
    tm = TempoMap.from_dict(grid)
    family_of = {int(b): str(f) for b, f in ((quant.get("beat_families") or {}).get("*") or [])}
    tracks_src = []
    for tid in TRACK_ORDER:
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
        if tid == BASS:
            instrument = "electricbassfinger"
        else:
            classes = [n.get("instrument") for n in track_notes(quant, tid)]
            # most frequent class; ties by name, so the output does not depend on string hashing (review)
            top = sorted(set(classes), key=lambda c: (-classes.count(c), str(c)))[0] if classes else None
            instrument = GUITAR_INSTRUMENT.get(str(top), "electricguitarclean")
        name, short = track_name(quant, tid)
        score.tracks.append(ATX.WTrack(name=name, short=short, instrument=instrument, fretted=fretted,
                                       tuning=(tr or {}).get("tuning") if fretted else None,
                                       capo=int((tr or {}).get("capo") or 0) if fretted else 0, beats=spelled,
                                       clef="f4" if tid == BASS and not fretted else None, tid=tid))
    return score, problems


def score_doc(song_key: str, meta: dict[str, Any], title: str, quant: dict[str, Any], tab: dict[str, Any],
              grid: dict[str, Any]) -> Any:
    """``score.json`` (schema v0) of a job's score."""
    from bandscribe.schema.score import Confidence, Line, Note, Score, SongMeta

    tm = TempoMap.from_dict(grid)
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
            on = max(0.0, float(tm.beat_to_time(int(n["gtick"]) / Q.TPB)))
            off = max(on, float(tm.beat_to_time((int(n["gtick"]) + int(n["dur"])) / Q.TPB)))
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
                          capo=int(tr.get("capo") or 0) if fretted else None, notes=notes))
    src = meta.get("source") or {}
    return Score(song=SongMeta(song_key=str(song_key), title=title,
                               duration_s=float(meta.get("duration_s") or grid.get("duration_s") or 0.0),
                               source_kind="youtube" if src.get("kind") == "youtube" else "file",
                               source_ref=str(src.get("filename") or src.get("ref") or "")), lines=lines)


def write_score(out: Path, *, song_key: str, meta: dict[str, Any], title: str, quant: dict[str, Any],
                tab: dict[str, Any], grid: dict[str, Any], sections: list[dict[str, Any]],
                params: dict[str, Any]) -> ATX.WScore:
    """Every file of the written score under ``out``: ``tab/score.alphatex``, ``tab/score.gp5``, ``tab/<track>.txt``,
    ``midi/<track>_quantized.mid`` and ``score.json`` (the ``score`` stage, and M4a's re-export of an edited score)."""
    from bandscribe.amt import midi
    from bandscribe.schema.score import save_score

    score, problems = build_score(title, quant, tab, grid, sections, params)
    if problems:
        raise TabStageError(f"리듬 표기 오류 {len(problems)}건(버그): {problems[:3]}")
    out = Path(out)
    atomic.write_text(out / "tab" / "score.alphatex", ATX.write(score))
    GP5.write(score, out / "tab" / "score.gp5")
    for tid, tr in (tab.get("tracks") or {}).items():
        if tr.get("fretted"):
            atomic.write_text(out / "tab" / f"{tid}.txt", text_tab(tid, tr, quant))
    tm = TempoMap.from_dict(grid)
    for tid in TRACK_ORDER:
        qn = track_notes(quant, tid)
        if tid in DERIVED and not (qn and (tab.get("tracks") or {}).get(tid)):
            continue
        notes = [{"onset_s": max(0.0, float(tm.beat_to_time(int(n["gtick"]) / Q.TPB))),
                  "offset_s": max(0.0, float(tm.beat_to_time((int(n["gtick"]) + int(n["dur"])) / Q.TPB))),
                  "pitch": int(n["pitch"])} for n in qn]
        midi.write_performance_midi(out / "midi" / f"{tid}_quantized.mid",
                                    [(track_name(quant, tid)[0], 33 if tid == BASS else 27, notes)], grid)
    save_score(out / "score.json", score_doc(song_key, meta, title, quant, tab, grid))
    return score


def run_score(ctx: Any) -> None:
    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    write_score(Path(ctx.out_dir), song_key=str(getattr(ctx, "song_key", "song")), meta=_job_meta(ctx),
                title=_title(ctx), quant=atomic.read_json(dd["quant"] / "quant.json"),
                tab=atomic.read_json(dd["tab"] / "tab.json"), grid=atomic.read_json(dd["grid"] / "grid.json"),
                sections=atomic.read_json(dd["sections"] / "sections.json").get("sections") or [],
                params=dict(ctx.params))


# quant 4 / tab 7 / score 5: M7a Part view (guitar notes and Line names from the parts stage; K part tracks with
# their own tunings, the Unassigned staff)
STAGES: dict[str, dict] = {
    "a4": {"run": run_a4, "code_version": "1", "params": a4_params, "device": "cpu", "models": lambda cfg: {}},
    "quant": {"run": run_quant, "code_version": "4", "params": quant_params, "device": "cpu",
              "models": lambda cfg: {}},
    "tab": {"run": run_tab, "code_version": "7", "params": tab_params, "device": "cpu", "models": lambda cfg: {}},
    "score": {"run": run_score, "code_version": "5", "params": score_params, "device": "cpu",
              "models": lambda cfg: {}},
}
