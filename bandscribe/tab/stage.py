"""Tab pipeline stages (M3; ``bandscribe.pipeline.graph`` owns names and deps):

- ``a4`` (CPU): ``a4.json`` - the recording's A4 offset in cents from the pitched stem views
  (``bandscribe.tab.a4``); a Korean warning from 25 cents (varispeed is E16, not applied).
- ``quant`` (CPU): ``quant.json`` - ``notes/guitar_all.json`` (track ``guitar_all``) and ``notes/bass_raw.json``
  (the pass-1 bass, track ``bass_raw``) quantised on the grid with one subdivision decision per beat, the meter
  re-read (shuffle vs 12/8) and the triplet feel per section (``bandscribe.tab.quantize``).
- ``tab`` (CPU): ``tab.json`` + ``text/<track>.txt`` - guitar and bass tuning chosen jointly
  (``bandscribe.tab.tuning``, fake low guitar notes left out of the evidence), the bass fretted
  (``bandscribe.tab.fretting``), the merged guitar fretted only with ``tab.guitar_parts = 1`` (otherwise staff +
  MIDI only, DESIGN 4.2).
- ``score`` (CPU): ``tab/score.alphatex`` (every track; the guitar staff-only unless fretted; one ``\\sync`` per
  bar on the recording's time), ``tab/score.gp5`` (fretted tracks), ``tab/<track>.txt``,
  ``midi/<track>_quantized.mid`` and ``score.json`` (schema v0), all from the same spelled bars
  (``bandscribe.tab.notation``).

Stage functions read only their dep dirs, the job input and ``ctx.params`` (M1_M2_SPEC 3.1).
"""

from __future__ import annotations

import logging
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


def quant_tracks(dep_dirs: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    """{track id: notes} the quantiser reads: guitar classes of the guitar pass, and the pass-1 bass (both from the
    notes stage, without the notes over a silent stem)."""
    gtr = atomic.read_json(Path(dep_dirs["notes"]) / "guitar_all.json").get("notes") or []
    bass_path = Path(dep_dirs["notes"]) / BASS_RAW
    bass = (atomic.read_json(bass_path).get("notes") or []) if bass_path.is_file() else []
    return {"guitar_all": gtr, "bass_raw": bass}


def run_quant(ctx: Any) -> None:
    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    grid = atomic.read_json(dd["grid"] / "grid.json")
    sections = atomic.read_json(dd["sections"] / "sections.json").get("sections") or []
    doc = Q.quantize(quant_tracks(dd), grid, sections, dict(ctx.params))
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
    bass's E1, say) is written an octave (or two) inside it and flagged ``octave_shift`` instead of dropped."""
    inst = F.Instrument(tuple(tun["tuning"]), max_fret=24, capo=int(tun["capo"]))
    shifts = [0 if inst.positions(int(n["pitch"])) else _octave_into_range(int(n["pitch"]), inst) for n in notes]
    shifted = [{**n, "pitch": int(n["pitch"]) + d} for n, d in zip(notes, shifts)]
    events = F.group_events(shifted)
    res = F.assign(events, inst, weights, model=model)
    by_id = {a.note_id: a for a in res}
    rows = []
    for i, n in enumerate(shifted):
        a = by_id[i]
        rows.append({"i": i, "gtick": n["gtick"], "dur": n["dur"], "bar": n["bar"], "tick": n["tick"],
                     "pitch": n["pitch"], "string": a.string, "fret": a.fret, "conf": a.confidence,
                     **({"octave_shift": shifts[i]} if shifts[i] else {})})
    check = F.playable(res, events, inst)
    check["octave_shifted"] = sum(1 for d in shifts if d)
    return {"fretted": True, "tuning": list(inst.tuning), "capo": inst.capo, "label": tun["label"], "notes": rows,
            "check": check}


def run_tab(ctx: Any) -> None:
    p = dict(ctx.params)
    q = atomic.read_json(Path(ctx.dep_dirs["quant"]) / "quant.json")
    a4 = atomic.read_json(Path(ctx.dep_dirs["a4"]) / "a4.json")
    g_notes = Q.notes_of(q, "guitar_all")
    b_notes = Q.notes_of(q, "bass_raw")
    fake = TU.fake_low_notes(g_notes, b_notes, float(p["tuning"]["bleed_window_s"]))
    evidence = F.group_events([n for i, n in enumerate(g_notes) if i not in fake])
    tun = TU.choose(evidence, F.group_events(b_notes), p["tuning"],
                    fixed={"guitar": p.get("guitar_tuning", "auto"), "bass": p.get("bass_tuning", "auto")})
    w = p["weights"]
    tracks: dict[str, Any] = {}
    if b_notes and tun["bass"]:
        tracks["bass_raw"] = _fret_track(b_notes, tun["bass"], w, p["model"])
    if g_notes and tun["guitar"]:
        if int(p["guitar_parts"]) == 1:
            tracks["guitar_all"] = _fret_track(g_notes, tun["guitar"], w, p["model"])
        else:  # staff + MIDI only (DESIGN 4.2): merged parts cannot be played by one hand
            tracks["guitar_all"] = {"fretted": False, "tuning": tun["guitar"]["tuning"], "capo": tun["guitar"]["capo"],
                                    "label": tun["guitar"]["label"], "notes": [], "check": None}
    doc = {"format": "bandscribe.tab/1", "guitar_parts": int(p["guitar_parts"]), "tuning": tun,
           "fake_low_guitar_notes": len(fake), "a4": {k: a4.get(k) for k in ("cents", "a4_hz", "confidence")},
           "tracks": tracks}
    out = Path(ctx.out_dir)
    atomic.write_json(out / "tab.json", doc)
    for tid, tr in tracks.items():
        if tr["fretted"]:
            title = f"{tid} — {tr['label']} ({' '.join(ascii_tab.string_names(tr['tuning'])[::-1])})"
            atomic.write_text(out / "text" / f"{tid}.txt", ascii_tab.render(tr["notes"], q["bars"], tr["tuning"],
                                                                           title=title))
    for name, tid in (("guitar", "guitar_all"), ("bass", "bass_raw")):
        t = tun.get(name)
        ko = "기타" if name == "guitar" else "베이스"
        if t and t.get("unknown"):
            _warn(ctx, f"{ko} 튜닝을 확신하지 못합니다(최선 {t['label']}, "
                       f"다음 {t['runner_up']['label'] if t.get('runner_up') else '-'}). 결과의 프렛 번호를 확인하세요.")
        shifted = ((tracks.get(tid) or {}).get("check") or {}).get("octave_shifted") or 0
        if t and t.get("user") and t.get("auto_label") != t["label"] and shifted:
            _warn(ctx, f"{ko} 튜닝을 지정한 대로 {t['label']} 로 썼습니다(자동 판정은 {t['auto_label']}). 이 튜닝으로는 "
                       f"낼 수 없는 음 {shifted}개를 옥타브를 옮겨 적었습니다.")
    chk = (tracks.get("guitar_all") or {}).get("check")
    if chk and chk.get("dropped"):
        _warn(ctx, f"합친 기타에서 한 손으로 칠 수 없는 음 {chk['dropped']}개를 탭에서 뺐습니다(MIDI·오선보에는 남음).")


# ---------------------------------------------------------------------------------------------- score

GUITAR_INSTRUMENT = {"distorted_electric_guitar": "distortionguitar", "clean_electric_guitar": "electricguitarclean",
                     "acoustic_guitar": "acousticguitarsteel"}
TRACK_NAMES = {"guitar_all": ("Guitar (all parts)", "Gtr"), "bass_raw": ("Bass (raw)", "Bass")}


def score_params(cfg: Any) -> dict[str, Any]:
    return {"tempo_change_ratio": 0.04, "title": str(_cfg_get(cfg, "hints.title", "") or "").strip()}


def _job_meta(ctx: Any) -> dict[str, Any]:
    try:
        return atomic.read_json(Path(ctx.store.job_dir(ctx.song_key)) / "input" / "meta.json")
    except (OSError, ValueError, AttributeError):
        return {}


def _title(ctx: Any) -> str:
    """``hints.title`` when set, else the input's file name without its extension."""
    hint = str((getattr(ctx, "params", None) or {}).get("title") or "").strip()
    if hint:
        return hint
    meta = _job_meta(ctx)
    name = (meta.get("source") or {}).get("filename") or meta.get("title") or getattr(ctx, "song_key", "song")
    return Path(str(name)).stem


def _events(notes: list[dict[str, Any]], fretted: bool) -> list[tuple[int, int, list[N.WNote]]]:
    """(gtick, duration, notes) per onset; one note per string in a tab chord, one per pitch on a staff."""
    by: dict[int, list[dict[str, Any]]] = {}
    for n in notes:
        if fretted and n.get("string") is None:
            continue  # dropped from the tab (kept in MIDI and score.json)
        by.setdefault(int(n["gtick"]), []).append(n)
    out = []
    for g in sorted(by):
        ns = sorted(by[g], key=lambda n: (-(n.get("string") or 0), int(n["pitch"])))
        seen: set[int] = set()
        uniq = []
        for n in ns:
            key = int(n["string"]) if fretted else int(n["pitch"])
            if key not in seen:
                seen.add(key)
                uniq.append(N.WNote(int(n["pitch"]), n.get("string"), n.get("fret")))
        out.append((g, max(int(n["dur"]) for n in ns), uniq))
    return out


def build_score(title: str, quant: dict[str, Any], tab: dict[str, Any], grid: dict[str, Any],
                sections: list[dict[str, Any]], params: dict[str, Any]) -> tuple[ATX.WScore, list[str]]:
    """The written score of a job (shared bars, spelled tracks) and spelling problems (empty = consistent)."""
    tm = TempoMap.from_dict(grid)
    family_of = {int(b): str(f) for b, f in ((quant.get("beat_families") or {}).get("*") or [])}
    tracks_src = []
    for tid in ("guitar_all", "bass_raw"):
        tr = (tab.get("tracks") or {}).get(tid)
        qn = Q.notes_of(quant, tid)
        if not qn:
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
        spelled = N.spell_track(_events(notes, fretted), bars, family_of)
        for b, beats in zip(bars, spelled):
            problems += N.check_bar(b, beats)
        if tid == "bass_raw":
            instrument = "electricbassfinger"
        else:
            classes = [n.get("instrument") for n in Q.notes_of(quant, tid)]
            # most frequent class; ties by name, so the output does not depend on string hashing (review)
            top = sorted(set(classes), key=lambda c: (-classes.count(c), str(c)))[0] if classes else None
            instrument = GUITAR_INSTRUMENT.get(str(top), "electricguitarclean")
        name, short = TRACK_NAMES[tid]
        score.tracks.append(ATX.WTrack(name=name, short=short, instrument=instrument, fretted=fretted,
                                       tuning=(tr or {}).get("tuning") if fretted else None,
                                       capo=int((tr or {}).get("capo") or 0) if fretted else 0, beats=spelled,
                                       clef="f4" if tid == "bass_raw" and not fretted else None))
    return score, problems


def _score_json(ctx: Any, title: str, quant: dict[str, Any], tab: dict[str, Any], grid: dict[str, Any]) -> Any:
    from bandscribe.schema.score import Confidence, Line, Note, Score, SongMeta

    tm = TempoMap.from_dict(grid)
    lines = []
    for tid in ("guitar_all", "bass_raw"):
        qn = Q.notes_of(quant, tid)
        if not qn:
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
            notes.append(Note(onset_s=on, offset_s=off, pitch=int(f.get("pitch", n["pitch"])),
                              string=f.get("string") if placed else None, fret=f.get("fret") if placed else None,
                              confidence=Confidence(fret=f.get("conf") if placed else None), sources=["muscriptor"],
                              flags={"octave": True} if f.get("octave_shift") else {}))
        notes.sort(key=lambda x: (x.onset_s, x.pitch))
        kind = "bass" if tid == "bass_raw" else ("guitar" if fretted else "reference")
        lines.append(Line(id=tid, name=TRACK_NAMES[tid][0], kind=kind, tuning=tr.get("tuning") if fretted else None,
                          capo=int(tr.get("capo") or 0) if fretted else None, notes=notes))
    meta = _job_meta(ctx)
    src = meta.get("source") or {}
    return Score(song=SongMeta(song_key=str(getattr(ctx, "song_key", "song")), title=title,
                               duration_s=float(meta.get("duration_s") or grid.get("duration_s") or 0.0),
                               source_kind="youtube" if src.get("kind") == "youtube" else "file",
                               source_ref=str(src.get("filename") or src.get("ref") or "")), lines=lines)


def run_score(ctx: Any) -> None:
    from bandscribe.amt import midi
    from bandscribe.schema.score import save_score

    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    quant = atomic.read_json(dd["quant"] / "quant.json")
    tab = atomic.read_json(dd["tab"] / "tab.json")
    grid = atomic.read_json(dd["grid"] / "grid.json")
    sections = atomic.read_json(dd["sections"] / "sections.json").get("sections") or []
    title = _title(ctx)
    score, problems = build_score(title, quant, tab, grid, sections, dict(ctx.params))
    if problems:
        raise TabStageError(f"리듬 표기 오류 {len(problems)}건(버그): {problems[:3]}")
    out = Path(ctx.out_dir)
    atomic.write_text(out / "tab" / "score.alphatex", ATX.write(score))
    GP5.write(score, out / "tab" / "score.gp5")
    for tid, tr in (tab.get("tracks") or {}).items():
        src = dd["tab"] / "text" / f"{tid}.txt"
        if tr.get("fretted") and src.is_file():
            atomic.write_text(out / "tab" / f"{tid}.txt", src.read_text(encoding="utf-8"))
    tm = TempoMap.from_dict(grid)
    for tid in ("guitar_all", "bass_raw"):
        qn = Q.notes_of(quant, tid)
        notes = [{"onset_s": max(0.0, float(tm.beat_to_time(int(n["gtick"]) / Q.TPB))),
                  "offset_s": max(0.0, float(tm.beat_to_time((int(n["gtick"]) + int(n["dur"])) / Q.TPB))),
                  "pitch": int(n["pitch"])} for n in qn]
        midi.write_performance_midi(out / "midi" / f"{tid}_quantized.mid",
                                    [(TRACK_NAMES[tid][0], 33 if tid == "bass_raw" else 27, notes)], grid)
    save_score(out / "score.json", _score_json(ctx, title, quant, tab, grid))


STAGES: dict[str, dict] = {
    "a4": {"run": run_a4, "code_version": "1", "params": a4_params, "device": "cpu", "models": lambda cfg: {}},
    "quant": {"run": run_quant, "code_version": "2", "params": quant_params, "device": "cpu",
              "models": lambda cfg: {}},
    "tab": {"run": run_tab, "code_version": "5", "params": tab_params, "device": "cpu", "models": lambda cfg: {}},
    "score": {"run": run_score, "code_version": "3", "params": score_params, "device": "cpu",
              "models": lambda cfg: {}},
}
