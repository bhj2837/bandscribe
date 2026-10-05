"""Tab pipeline stages (M3; ``bandscribe.pipeline.graph`` owns names and deps):

- ``a4`` (CPU): ``a4.json`` - the recording's A4 offset in cents from the pitched stem views
  (``bandscribe.tab.a4``); a Korean warning from 25 cents (varispeed is E16, not applied).
- ``quant`` (CPU): ``quant.json`` - ``notes/guitar_all.json`` (track ``guitar_all``) and the pass-1 bass
  transcription (track ``bass_raw``) quantised on the grid with one subdivision decision per beat, the meter
  re-read (shuffle vs 12/8) and the triplet feel per section (``bandscribe.tab.quantize``).
- ``tab`` (CPU): ``tab.json`` + ``text/<track>.txt`` - guitar and bass tuning chosen jointly
  (``bandscribe.tab.tuning``, fake low guitar notes left out of the evidence), the bass fretted
  (``bandscribe.tab.fretting``), the merged guitar fretted only with ``tab.guitar_parts = 1`` (otherwise staff +
  MIDI only, DESIGN 4.2).

Stage functions read only their dep dirs and ``ctx.params`` (M1_M2_SPEC 3.1).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.tab import a4 as A4
from bandscribe.tab import ascii as ascii_tab
from bandscribe.tab import fretting as F
from bandscribe.tab import quantize as Q
from bandscribe.tab import tuning as TU

log = logging.getLogger(__name__)

A4_VIEWS = ("guitar_mono", "bass_mono", "piano_other_mono")
BASS_RAW = "raw/bass_mono__muscriptor.json"


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
    """{track id: notes} the quantiser reads: guitar classes of the guitar pass, and the pass-1 bass."""
    gtr = atomic.read_json(Path(dep_dirs["notes"]) / "guitar_all.json").get("notes") or []
    bass_path = Path(dep_dirs["amt_ms1"]) / BASS_RAW
    bass = (atomic.read_json(bass_path).get("notes") or []) if bass_path.is_file() else []
    return {"guitar_all": gtr, "bass_raw": bass}


def run_quant(ctx: Any) -> None:
    dd = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    grid = atomic.read_json(dd["grid"] / "grid.json")
    sections = atomic.read_json(dd["sections"] / "sections.json").get("sections") or []
    doc = Q.quantize(quant_tracks(dd), grid, sections, dict(ctx.params))
    atomic.write_json(Path(ctx.out_dir) / "quant.json", doc)


# ------------------------------------------------------------------------------------------------ tab


def tab_params(cfg: Any) -> dict[str, Any]:
    parts = int(cfg.get("tab.guitar_parts")) if cfg is not None and hasattr(cfg, "get") else 0
    return {"guitar_parts": parts, "tuning": dict(TU.DEFAULT_PARAMS), "weights": dict(F.DEFAULT_WEIGHTS)}


def _fret_track(notes: list[dict[str, Any]], tun: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    inst = F.Instrument(tuple(tun["tuning"]), max_fret=24, capo=int(tun["capo"]))
    events = F.group_events(notes)
    res = F.assign(events, inst, weights)
    by_id = {a.note_id: a for a in res}
    rows = []
    for i, n in enumerate(notes):
        a = by_id[i]
        rows.append({"i": i, "gtick": n["gtick"], "dur": n["dur"], "bar": n["bar"], "tick": n["tick"],
                     "pitch": n["pitch"], "string": a.string, "fret": a.fret, "conf": a.confidence})
    return {"fretted": True, "tuning": list(inst.tuning), "capo": inst.capo, "label": tun["label"], "notes": rows,
            "check": F.playable(res, events, inst)}


def run_tab(ctx: Any) -> None:
    p = dict(ctx.params)
    q = atomic.read_json(Path(ctx.dep_dirs["quant"]) / "quant.json")
    a4 = atomic.read_json(Path(ctx.dep_dirs["a4"]) / "a4.json")
    g_notes = Q.notes_of(q, "guitar_all")
    b_notes = Q.notes_of(q, "bass_raw")
    fake = TU.fake_low_notes(g_notes, b_notes, float(p["tuning"]["bleed_window_s"]))
    evidence = F.group_events([n for i, n in enumerate(g_notes) if i not in fake])
    tun = TU.choose(evidence, F.group_events(b_notes), p["tuning"])
    w = p["weights"]
    tracks: dict[str, Any] = {}
    if b_notes and tun["bass"]:
        tracks["bass_raw"] = _fret_track(b_notes, tun["bass"], w)
    if g_notes and tun["guitar"]:
        if int(p["guitar_parts"]) == 1:
            tracks["guitar_all"] = _fret_track(g_notes, tun["guitar"], w)
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
    for name in ("guitar", "bass"):
        t = tun.get(name)
        if t and t.get("unknown"):
            _warn(ctx, f"{'기타' if name == 'guitar' else '베이스'} 튜닝을 확신하지 못합니다(최선 {t['label']}, "
                       f"다음 {t['runner_up']['label'] if t.get('runner_up') else '-'}). 결과의 프렛 번호를 확인하세요.")
    chk = (tracks.get("guitar_all") or {}).get("check")
    if chk and chk.get("dropped"):
        _warn(ctx, f"합친 기타에서 한 손으로 칠 수 없는 음 {chk['dropped']}개를 탭에서 뺐습니다(MIDI·오선보에는 남음).")


STAGES: dict[str, dict] = {
    "a4": {"run": run_a4, "code_version": "1", "params": a4_params, "device": "cpu", "models": lambda cfg: {}},
    "quant": {"run": run_quant, "code_version": "1", "params": quant_params, "device": "cpu",
              "models": lambda cfg: {}},
    "tab": {"run": run_tab, "code_version": "1", "params": tab_params, "device": "cpu", "models": lambda cfg: {}},
}
