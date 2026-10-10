"""Router and part-split stages (M6; ``bandscribe.pipeline.graph`` owns names and deps):

- ``router`` (CPU, a few seconds): ``router.json`` - for every window of the grid (4 bars, cut at sections) the
  stereo features of the separated guitar stem, the scenario label (R0-R3, ``mixed`` or ``silent``) with its rule
  scores, p and margin, and the solo gate v0 (``bandscribe.router.route``). Reads the ``sep`` stage's stereo
  ``stems/guitar.wav``, ``grid.json``, ``sections.json``, the ``vocal`` stage's per-bar activity and the ``resid1``
  stage's per-bar repetition residual.
- ``split`` (CPU, seconds): ``guitar_lines.json`` - the notes stage's guitar notes, each with the ``line`` it goes
  to (``L`` / ``R`` / ``all``) and ``shared``, from its own stereo signature in the router's ``R3`` (P: parts at
  different positions) windows (``bandscribe.router.split``, E22b: the sign of the note's ILD). ``A`` windows stay
  merged (E22b held the A split), so do narrow, ADT and mixed ones. The split is used only when it is worth two
  tracks (``decision``): at least ``split_min_share`` of the notes in P windows and ``split_min_notes`` on each
  side; otherwise every note is ``all``. ``quant`` reads the guitar notes from here; ``tab`` / ``score`` write the
  ``L`` / ``R`` notes as two fretted tracks next to the merged reference staff.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.router import route as RT

log = logging.getLogger(__name__)

ROUTER_FILE = "router.json"
FORMAT = "bandscribe.router/1"
CODE_VERSION = "1"
LINES_FILE = "guitar_lines.json"
LINES_FORMAT = "bandscribe.guitar_lines/1"
SPLIT_CODE_VERSION = "1"


class SplitStageError(RuntimeError):
    """User-facing (Korean) problem of the split stage."""


def _router_cfg(cfg: Any) -> Any:
    from bandscribe.config import Config

    return (cfg if isinstance(cfg, Config) else Config.model_validate(cfg or {})).router


def router_params(cfg: Any) -> dict[str, Any]:
    """``[router]`` -> route params (thresholds merged over the synthetic starting values)."""
    r = _router_cfg(cfg)
    return {"window_bars": int(r.window_bars), "silence_db": float(r.silence_db),
            "thresholds": {**RT.THRESHOLDS, **dict(r.thresholds)},
            "solo_coh_rise": float(r.solo_coh_rise), "solo_resid_db": float(r.solo_resid_db),
            "solo_vocal_max": float(r.solo_vocal_max)}


def _per_bar(doc: dict[str, Any] | None, key: str) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for row in (doc or {}).get("per_bar") or []:
        v = row.get(key)
        out[int(row["bar"])] = None if v is None else float(v)
    return out


def _read_stereo(path: Path) -> tuple[Any, int]:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if x.shape[1] == 1:  # a mono stem has no stereo image: every window is narrow by construction
        x = x.repeat(2, axis=1)
    return x[:, :2], int(sr)


def run_router(ctx: Any) -> None:
    dep = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    grid = atomic.read_json(dep["grid"] / "grid.json")
    sections = atomic.read_json(dep["sections"] / "sections.json")
    vocal_path = dep["vocal"] / "vocal_activity.json"
    resid_path = dep["resid1"] / "resid1.json"
    vocal = atomic.read_json(vocal_path) if vocal_path.is_file() else None
    resid = atomic.read_json(resid_path) if resid_path.is_file() else None
    x, sr = _read_stereo(dep["sep"] / "stems" / "guitar.wav")
    params = dict(ctx.params)
    body = RT.route_song(x, sr, grid.get("bars") or [], sections.get("sections") or [], params,
                         vocal_per_bar=_per_bar(vocal, "active_ratio"),
                         resid_per_bar=_per_bar(resid, "resid_rel_db"))
    doc = {"format": FORMAT, "params": params, **body}
    atomic.write_json(Path(ctx.out_dir) / ROUTER_FILE, doc)
    s = body["summary"]
    log.info("router: %s (solo windows %d)", ", ".join(f"{k} {v}" for k, v in sorted(s["windows"].items())),
             s["solo_windows"])


# ------------------------------------------------------------------------------------------------ split


def split_params(cfg: Any) -> dict[str, Any]:
    """``[router] split*`` -> split params; ``tab.guitar_parts = 1`` (the user says one guitar) turns it off."""
    from bandscribe.config import Config

    c = cfg if isinstance(cfg, Config) else Config.model_validate(cfg or {})
    r = c.router
    return {"enabled": bool(r.split) and int(c.tab.guitar_parts) != 1, "p_ild_db": float(r.split_p_ild_db),
            "a_split": False, "min_share": float(r.split_min_share), "min_notes": int(r.split_min_notes)}


def split_song(notes: list[dict[str, Any]], windows: list[dict[str, Any]], stereo: Any, sr: int,
               params: dict[str, Any]) -> dict[str, Any]:
    """The ``guitar_lines.json`` body (module doc): notes with ``line`` / ``shared``, the summary and the decision."""
    from bandscribe.router import notesig as NS
    from bandscribe.router import split as SP

    wins = [w if (w.get("label") != "R2" or params.get("a_split")) else {**w, "label": "R2-merged"}
            for w in windows]
    sigs = NS.signatures(stereo, sr, notes) if notes else []
    assigned = SP.assign(notes, sigs, wins, {"p_ild_db": float(params["p_ild_db"])})
    summ = SP.summary(assigned, wins)
    n_l = sum(1 for a in assigned if "L" in a["lines"])
    n_r = sum(1 for a in assigned if "R" in a["lines"])
    p_share = summ["notes"].get("R3", 0) / max(1, len(assigned))
    reasons = []
    if not params.get("enabled"):
        reasons.append("꺼짐(설정 또는 기타 파트 수 1)")
    if p_share < float(params["min_share"]):
        reasons.append(f"좌우 분리 구간(P)의 음이 {100 * p_share:.0f} %로 {100 * float(params['min_share']):.0f} % 미만")
    if min(n_l, n_r) < int(params["min_notes"]):
        reasons.append(f"한쪽 음이 {min(n_l, n_r)}개로 {int(params['min_notes'])}개 미만")
    use = not reasons
    out_notes = []
    for n, a, s in zip(notes, assigned, sigs):
        line = ("L" if a["lines"] == ["L"] else "R" if a["lines"] == ["R"] else "all") if use else "all"
        out_notes.append({**n, "line": line, "shared": bool(use and a["shared"]),
                          "ild_db": s.get("ild_db")})
    return {"decision": {"split": use, "reasons": reasons, "p_share": round(p_share, 4),
                         "notes_l": n_l, "notes_r": n_r},
            "summary": summ, "notes": out_notes}


def run_split(ctx: Any) -> None:
    dep = {k: Path(v) for k, v in ctx.dep_dirs.items()}
    gdoc = atomic.read_json(dep["notes"] / "guitar_all.json")
    notes = list(gdoc.get("notes") or [])
    router = atomic.read_json(dep["router"] / ROUTER_FILE)
    x, sr = _read_stereo(dep["sep"] / "stems" / "guitar.wav")
    params = dict(ctx.params)
    body = split_song(notes, list(router.get("windows") or []), x, sr, params)
    doc = {"format": LINES_FORMAT, "params": params, **{k: v for k, v in gdoc.items() if k != "notes"},
           "split": body["decision"], "summary": body["summary"], "notes": body["notes"]}
    doc["format"] = LINES_FORMAT
    atomic.write_json(Path(ctx.out_dir) / LINES_FILE, doc)
    d = body["decision"]
    if d["split"]:
        log.info("split: 기타를 좌우 두 파트로 나눕니다 (왼쪽 %d음, 오른쪽 %d음, P 구간 음 %.0f %%)",
                 d["notes_l"], d["notes_r"], 100 * d["p_share"])
    else:
        log.info("split: 나누지 않습니다 (%s)", "; ".join(d["reasons"]))


STAGES: dict[str, dict] = {
    "router": {"run": run_router, "code_version": CODE_VERSION, "params": router_params, "device": "cpu",
               "models": lambda cfg: {}},
    "split": {"run": run_split, "code_version": SPLIT_CODE_VERSION, "params": split_params, "device": "cpu",
              "models": lambda cfg: {}},
}
