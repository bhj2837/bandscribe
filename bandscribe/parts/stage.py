"""Part diarizer stages (M7a; ``bandscribe.pipeline.graph`` owns names and deps):

- ``notesig`` (CPU, seconds): ``notesig.json`` - the stereo signature (``bandscribe.router.notesig``: ILD, coherence,
  mid/side at the note's own harmonics) of every guitar note of the notes stage, in ``guitar_all.json`` order, read
  from the ``sep`` stage's stereo guitar stem. A stage of its own so that another part count (``--parts``) does not
  redo it.
- ``parts`` (CPU, well under a second): ``guitar_lines.json`` - the notes stage's guitar notes, each with its Line
  (``line`` = "1".."4" in Part view order, "u" = Unassigned), the Line's posterior ``p`` and the ``amber`` flag
  (DESIGN 4.2: 0.4-0.6), and the Part view itself: the part count (estimate, two hypotheses, forced, reasons) and
  the Lines with their names (``bandscribe.parts.diarize``). Windows and their labels come from the ``router``
  stage, bars from ``grid``. ``tab.guitar_parts`` = N (``run --parts N``) forces N Lines. ``quant`` reads the guitar
  notes from here; ``tab`` / ``score`` write one fretted track per Line next to the merged reference staff and an
  Unassigned staff.
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path
from typing import Any

from bandscribe import atomic

log = logging.getLogger(__name__)

NOTESIG_FILE = "notesig.json"
NOTESIG_FORMAT = "bandscribe.notesig/1"
LINES_FILE = "guitar_lines.json"
LINES_FORMAT = "bandscribe.guitar_lines/2"
UNASSIGNED = "u"


class PartsStageError(RuntimeError):
    """User-facing (Korean) problem of a parts stage."""


def _cfg(cfg: Any) -> Any:
    from bandscribe.config import Config

    return cfg if isinstance(cfg, Config) else Config.model_validate(cfg or {})


def notesig_params(cfg: Any) -> dict[str, Any]:
    from bandscribe.router import notesig as NS

    return {"n_fft": NS.N_FFT, "hop": NS.HOP, "hold_s": NS.HOLD_S, "low_hz": NS.LOW_HZ, "n_harm": NS.N_HARM}


def run_notesig(ctx: Any) -> None:
    from bandscribe.router import notesig as NS
    from bandscribe.router.stage import _read_stereo

    dep = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    notes = list(atomic.read_json(dep["notes"] / "guitar_all.json").get("notes") or [])
    x, sr = _read_stereo(dep["sep"] / "stems" / "guitar.wav")
    sigs = NS.signatures(x, sr, notes) if notes else []
    atomic.write_json(Path(ctx.out_dir) / NOTESIG_FILE, {"format": NOTESIG_FORMAT, "params": dict(ctx.params),
                                                         "n_notes": len(notes), "sigs": sigs})


def parts_params(cfg: Any) -> dict[str, Any]:
    """``[parts]`` + ``tab.guitar_parts`` (0 = estimate, N = force N Lines)."""
    c = _cfg(cfg)
    p = c.parts
    return {"k": int(c.tab.guitar_parts),
            "count": {"n_min": int(p.n_min), "min_windows": int(p.min_windows), "min_sections": int(p.min_sections),
                      "side_db": float(p.side_db), "use_class": bool(p.use_class),
                      "use_symbolic": bool(p.use_symbolic)},
            "tau": float(p.tau)}


def bar_index(notes: list[dict[str, Any]], bars: list[dict[str, Any]]) -> list[int]:
    """The grid bar each note starts in (the nearest one outside the grid)."""
    if not bars:
        return [int(float(n["onset_s"]) // 2.0) for n in notes]
    starts = [float(b["start_s"]) for b in bars]
    out = []
    for n in notes:
        k = max(0, bisect.bisect_right(starts, float(n["onset_s"])) - 1)
        out.append(int(bars[k]["bar"]))
    return out


def parts_song(notes: list[dict[str, Any]], sigs: list[dict[str, Any]], windows: list[dict[str, Any]],
               bars: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    """The ``guitar_lines.json`` body (module doc)."""
    from bandscribe.parts import diarize as DZ

    if len(sigs) != len(notes):
        raise PartsStageError(f"음표별 스테레오 서명 수({len(sigs)})가 기타 음 수({len(notes)})와 다릅니다(버그)")
    k = int(params.get("k") or 0)
    d = DZ.diarize(notes, sigs, windows, bar_of=bar_index(notes, bars), params=params.get("count"),
                   k_force=k or None, tau=float(params.get("tau", 0.4)))
    out_notes = []
    for n, s, nl in zip(notes, sigs, d["note"]):
        out_notes.append({**n, "line": UNASSIGNED if nl["line"] is None else str(nl["line"]), "p": nl["p"],
                          "amber": bool(nl["amber"]), "shared": False, "ild_db": s.get("ild_db")})
    lines = [{k_: ln[k_] for k_ in ("number", "name", "short", "role", "cls", "position", "notes", "bars", "key",
                                    "kind")} for ln in d["lines"]]
    return {"count": d["count"], "lines": lines, "windows": d["windows"], "notes": out_notes,
            "unassigned": sum(1 for n in out_notes if n["line"] == UNASSIGNED)}


def run_parts(ctx: Any) -> None:
    dep = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    gdoc = atomic.read_json(dep["notes"] / "guitar_all.json")
    notes = list(gdoc.get("notes") or [])
    sigs = list(atomic.read_json(dep["notesig"] / NOTESIG_FILE).get("sigs") or [])
    windows = list(atomic.read_json(dep["router"] / "router.json").get("windows") or [])
    bars = list(atomic.read_json(dep["grid"] / "grid.json").get("bars") or [])
    params = dict(ctx.params)
    body = parts_song(notes, sigs, windows, bars, params)
    doc = {"format": LINES_FORMAT, "params": params, **{k: v for k, v in gdoc.items() if k not in ("notes", "format")},
           **body}
    atomic.write_json(Path(ctx.out_dir) / LINES_FILE, doc)
    c = body["count"]
    hyp = " / ".join(f"{h['k']}파트 {h['p']:.2f}" for h in c.get("hypotheses") or [])
    names = ", ".join(f"{ln['name']} {ln['notes']}음" for ln in body["lines"])
    how = f"지정 {c['forced']}" if c.get("forced") else f"추정 ({hyp})"
    log.info("parts: 기타 %d파트 — %s; %s; 미배정 %d음", c["chosen"], how, names or "-", body["unassigned"])
    for r in c.get("reasons") or []:
        log.info("parts: %s", r)


STAGES: dict[str, dict] = {
    "notesig": {"run": run_notesig, "code_version": "1", "params": notesig_params, "device": "cpu",
                "models": lambda cfg: {}},
    "parts": {"run": run_parts, "code_version": "1", "params": parts_params, "device": "cpu",
              "models": lambda cfg: {}},
}
