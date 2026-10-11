"""E6b and E7 (M6 gaps, docs/decisions.md): channel and centre views of E6's Tier B remixes.

Both read an E6 run (its P / A remixes and their cached SW guitar stems) and use the first ``EXCERPT_S`` seconds
(E22's excerpt), MuScriptor with the guitar-only mask at the top rung, the eval transcription cache.

- **E6b** (R1 self-agreement): the left and right channels of the separated stem transcribed apart; per 8 s window
  the chroma-onset F1 between the two (``chroma_onset_f1``: same pitch class, onsets within 50 ms, greedy) when
  both sides have at least ``MIN_NOTES`` notes. A remixes' R2 windows = one part played twice (a double), P
  remixes' R3 windows = two parts. Reported: the doubles' quantiles (5 % = the R1 veto threshold, 5 % / 25 % = the
  variation boundaries, DESIGN 4.2) and how many two-part windows fall under the veto threshold.
- **E7** (centre extractor for the A path): for each A remix, a centre view per arm (``mid``, ``ipd_ild``,
  ``cw_bs_v2``, ``cw_bs_v3beta``; ``bandscribe.router.centre``) and the side view S = (L-R)/2. DESIGN 4.5's A rule:
  lead = centre notes without an S note within 1 semitone and 50 ms, rhythm = S notes. Scored like E22 against the
  takes' own transcriptions (the double's takes = rhythm, the centred takes = lead): ``lead_f1``, ``rhythm_f1``,
  ``leak_lead``, ``leak_rhythm``, next to ``merged`` (the mid view's notes in both lines).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic
from bandscribe.router import centre as C
from bandscribe.router import experiments as X
from bandscribe.router import split_eval as SE
from bandscribe.router import tierb as T

log = logging.getLogger(__name__)

EXCERPT_S = SE.EXCERPT_S
ONSET_TOL_S = 0.05
MIN_NOTES = 4
LEAD_SEMITONES = 1
E7_ARMS = ("mid", "ipd_ild", "cw_bs_v2", "cw_bs_v3beta")
WIN_S = T.WINDOW_S


def chroma_onset_f1(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], tol: float = ONSET_TOL_S
                    ) -> float | None:
    """F1 between two note lists by pitch class and onset (greedy, nearest first, each note once)."""
    if not a and not b:
        return None
    pairs = []
    for i, n in enumerate(a):
        for j, m in enumerate(b):
            if int(n["pitch"]) % 12 == int(m["pitch"]) % 12:
                d = abs(float(n["onset_s"]) - float(m["onset_s"]))
                if d <= tol:
                    pairs.append((d, i, j))
    ui: set[int] = set()
    uj: set[int] = set()
    tp = 0
    for _d, i, j in sorted(pairs):
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        tp += 1
    return 2 * tp / (len(a) + len(b)) if (len(a) + len(b)) else None


def lead_minus_side(centre: Sequence[Mapping[str, Any]], side_notes: Sequence[Mapping[str, Any]],
                    semis: int = LEAD_SEMITONES, tol: float = ONSET_TOL_S) -> list[bool]:
    """Per centre note: True when no side note lies within ``semis`` semitones and ``tol`` seconds (DESIGN 4.5 A)."""
    s_on = np.array([float(m["onset_s"]) for m in side_notes]) if side_notes else np.zeros(0)
    s_p = np.array([int(m["pitch"]) for m in side_notes]) if side_notes else np.zeros(0, dtype=int)
    out = []
    for n in centre:
        if s_on.size == 0:
            out.append(True)
            continue
        near = (np.abs(s_on - float(n["onset_s"])) <= tol) & (np.abs(s_p - int(n["pitch"])) <= semis)
        out.append(not bool(near.any()))
    return out


def _remixes(e6: Path, scenarios: Sequence[str], limit: int | None = None) -> list[dict[str, Any]]:
    """E6 remixes of these scenarios with plan, relations, the first EXCERPT_S s of the cached SW stem."""
    import soundfile as sf

    from bandscribe import datasets

    wins = json.loads((e6 / "e6_windows.json").read_text(encoding="utf-8"))
    root = Path(datasets.dataset_root("cambridge_mt"))
    tracks = {t.track_id: t for t in datasets.iter_tracks("cambridge_mt")}
    order = sorted(tracks)
    rel_dir = X.cache_root() / "relations"
    keys = sorted({(w["song"], w["scenario"]) for w in wins if w["scenario"] in scenarios})
    if limit:
        keys = keys[:limit]
    from bandscribe.amt import experiments as AE
    from bandscribe.config import load_config

    sep_rung = AE.sep_top_rung(load_config())
    out = []
    for song, sc in keys:
        rel = T.relations(tracks[song], root, rel_dir)
        pl = T.plan(rel, sc, song_index=order.index(song))
        mix_wav = e6 / "work" / f"{song}_{sc}" / "mix.wav"
        mix, _ = sf.read(str(mix_wav), dtype="float32", always_2d=True)
        stems, _ = X.separate_guitar([(X._pcm_sha(mix), mix_wav)], load_config(), force_rung=sep_rung,
                                     cache_dir=X.cache_root())
        stem, sr = sf.read(str(next(iter(stems.values()))), dtype="float32", always_2d=True)
        out.append({"song": song, "scenario": sc, "pl": pl, "rel": rel, "track": tracks[song], "root": root,
                    "stem": stem[: int(EXCERPT_S * sr)], "sr": sr,
                    "windows": [w for w in wins if w["song"] == song and w["scenario"] == sc
                                and w["end_s"] <= EXCERPT_S + 1e-6]})
    return out


def _mono_view(vid: str, x: np.ndarray, sr: int, d: Path, views: list[dict[str, Any]]) -> None:
    from bandscribe.amt import instruments as I
    from bandscribe.amt.reftx import write_mono_f32

    d.mkdir(parents=True, exist_ok=True)
    wav = d / f"{vid.rsplit('.', 1)[-1]}_mono.wav"
    sha = write_mono_f32(wav, np.asarray(x, dtype=np.float32), sr)
    views.append({"id": vid, "wav": wav, "pcm_sha256": sha, "instruments": I.mask_guitar_only()})


def _transcribe(views: list[dict[str, Any]], work: Path) -> tuple[dict[str, dict], str]:
    from bandscribe.amt import experiments as AE
    from bandscribe.amt import stage as S
    from bandscribe.config import load_config

    cfg = load_config()
    params = S.ms_params(cfg)
    top = AE.ms_rung(params)
    raws, info = S.transcribe_views_ms(views, params, cfg, work_dir=work / "ms" / "_worker", cache=AE._eval_cache(),
                                       job=None, force_rung=top)
    return raws, f"{AE.sep_top_rung(cfg)}|{info.get('rung_label') or top}"


def _guitar(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    from bandscribe.amt import instruments as I

    g = set(I.GUITAR_CLASSES)
    return [n for n in raw.get("notes") or [] if n.get("instrument") in g]


def _in(notes: Sequence[Mapping[str, Any]], a: float, b: float) -> list[Mapping[str, Any]]:
    return [n for n in notes if a <= float(n["onset_s"]) < b]


def _quantiles(v: Sequence[float]) -> dict[str, float] | None:
    if not v:
        return None
    q = np.quantile(np.asarray(v, dtype=float), [0.05, 0.25, 0.5, 0.75, 0.95])
    return {k: round(float(x), 4) for k, x in zip(("q05", "q25", "q50", "q75", "q95"), q)}


# ------------------------------------------------------------------------------------------------ E6b


def run_e6b(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    out_dir = Path(out_dir)
    work = out_dir / "work"
    e6 = Path(args["e6_run"]) if args.get("e6_run") else SE.latest_run("e6")
    if e6 is None:
        from bandscribe.amt import experiments as AE

        raise AE.ExperimentDataMissing("E6b: E6 실행 결과가 없습니다. 먼저 bandscribe eval exp E6")
    rms = _remixes(e6, ("A", "P"), int(args["limit"]) if args.get("limit") else None)
    views: list[dict[str, Any]] = []
    for r in rms:
        d = work / f"{r['song']}_{r['scenario']}"
        _mono_view(f"{r['song']}.{r['scenario']}.L", r["stem"][:, 0], r["sr"], d, views)
        _mono_view(f"{r['song']}.{r['scenario']}.R", r["stem"][:, 1], r["sr"], d, views)
    if progress:
        progress(f"전사: 뷰 {len(views)}개 ({len(views) * EXCERPT_S / 60:.0f}분 오디오)")
    raws, rung = _transcribe(views, work)
    rows: list[dict] = []
    vals: dict[str, list[float]] = {"double": [], "different": []}
    detail = []
    for r in rms:
        left, right = (_guitar(raws[f"{r['song']}.{r['scenario']}.{c}"]) for c in ("L", "R"))
        want = "R2" if r["scenario"] == "A" else "R3"
        kind = "double" if r["scenario"] == "A" else "different"
        for w in r["windows"]:
            if w["truth"] != want:
                continue
            a, b = _in(left, w["start_s"], w["end_s"]), _in(right, w["start_s"], w["end_s"])
            if len(a) < MIN_NOTES or len(b) < MIN_NOTES:
                continue
            f = chroma_onset_f1(a, b)
            if f is None:
                continue
            vals[kind].append(f)
            item = f"{r['song']}:{r['scenario']}:{int(w['start_s'] // WIN_S)}"
            rows.append({"dataset": "cambridge_mt", "item": item, "group": r["song"], "scenario": r["scenario"],
                         "section": "", "line": "guitar", "rung": rung, "system": kind, "metric": "lr_f1",
                         "value": round(f, 4)})
            detail.append({"item": item, "kind": kind, "f1": round(f, 4), "n_left": len(a), "n_right": len(b)})
    dq = _quantiles(vals["double"])
    veto = dq["q05"] if dq else None
    summary = {"double": dq, "different": _quantiles(vals["different"]), "n_double": len(vals["double"]),
               "n_different": len(vals["different"]),
               "r1_veto_threshold": veto,
               "variation_boundaries": [dq["q05"], dq["q25"]] if dq else None,
               "different_under_veto": (round(sum(1 for v in vals["different"] if v < veto) / len(vals["different"]), 4)
                                        if veto is not None and vals["different"] else None),
               "doubles_under_veto": (round(sum(1 for v in vals["double"] if v < veto) / len(vals["double"]), 4)
                                      if veto is not None and vals["double"] else None)}
    atomic.write_json(out_dir / "e6b_summary.json", {**summary, "e6_run": str(e6), "excerpt_s": EXCERPT_S,
                                                     "min_notes": MIN_NOTES, "windows": detail})
    log.info("E6b: veto %s (doubles %d, two-part windows %d)", veto, len(vals["double"]), len(vals["different"]))
    return rows


# ------------------------------------------------------------------------------------------------ E7

E7_GAIN, E7_LEAK, E7_DROP = 10.0, 0.25, 2.0


def decide_e7(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pre-registered E7 rule (decisions.md): among the arms whose mean ``leak_lead`` <= 0.25 and rhythm drop <=
    2 points, the best mean lead-F1 gain over ``merged``; the A path is adopted when that gain >= 10 points with a
    CI low > 0 (song-block bootstrap over the A remixes); else held."""
    per: dict[str, Any] = {}
    for arm in E7_ARMS:
        gain = SE._paired(rows, "lead_f1", "A", arm, "merged")
        drop = SE._paired(rows, "rhythm_f1", "A", arm, "merged", sign=-1.0)
        leak = SE._mean(rows, "leak_lead", "A", arm)
        ok = bool(gain and drop and leak is not None and leak <= E7_LEAK and drop["delta"] <= E7_DROP)
        per[arm] = {"lead_gain": gain, "rhythm_drop": drop, "leak_lead": leak,
                    "leak_rhythm": SE._mean(rows, "leak_rhythm", "A", arm), "within_limits": ok}
    cands = [a for a in E7_ARMS if per[a]["within_limits"]]
    best = max(cands, key=lambda a: (per[a]["lead_gain"]["delta"], -E7_ARMS.index(a))) if cands else None
    adopt = bool(best and per[best]["lead_gain"]["delta"] >= E7_GAIN and per[best]["lead_gain"]["ci"][0] > 0)
    return {"arms": per, "chosen": best, "decision": "adopt" if adopt else "hold",
            "rule": f"best lead gain among arms with leak_lead <= {E7_LEAK} and drop <= {E7_DROP}; adopt if gain >= "
                    f"{E7_GAIN} and CI low > 0"}


def run_e7(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    from bandscribe.config import load_config

    out_dir = Path(out_dir)
    work = out_dir / "work"
    e6 = Path(args["e6_run"]) if args.get("e6_run") else SE.latest_run("e6")
    if e6 is None:
        from bandscribe.amt import experiments as AE

        raise AE.ExperimentDataMissing("E7: E6 실행 결과가 없습니다. 먼저 bandscribe eval exp E6")
    rms = _remixes(e6, ("A",), int(args["limit"]) if args.get("limit") else None)
    # stereo excerpts for the CenterWide models (cached by pcm sha)
    from bandscribe import audio as AU

    stereo_items = []
    for r in rms:
        d = work / f"{r['song']}_A"
        d.mkdir(parents=True, exist_ok=True)
        AU.write_f32(d / "stem.wav", r["stem"], r["sr"])
        r["stem_sha"] = X._pcm_sha(r["stem"])
        stereo_items.append((r["stem_sha"], d / "stem.wav"))
    centre_info = {}
    centres: dict[str, dict[str, Path]] = {}
    for mid_ in ("cw_bs_v2", "cw_bs_v3beta"):
        centres[mid_], centre_info[mid_] = C.extract(stereo_items, load_config(), mid_, X.cache_root() / "centre",
                                                     progress=progress)
    import soundfile as sf

    views: list[dict[str, Any]] = []
    for r in rms:
        d = work / f"{r['song']}_A"
        base = f"{r['song']}.A"
        _mono_view(f"{base}.mid", C.mid(r["stem"]), r["sr"], d, views)
        _mono_view(f"{base}.side", C.side(r["stem"]), r["sr"], d, views)
        _mono_view(f"{base}.ipd_ild", C.ipd_ild(r["stem"], r["sr"]), r["sr"], d, views)
        for mid_ in ("cw_bs_v2", "cw_bs_v3beta"):
            c, _ = sf.read(str(centres[mid_][r["stem_sha"]]), dtype="float32", always_2d=True)
            _mono_view(f"{base}.{mid_}", C.mid(c), r["sr"], d, views)
        # the takes' own excerpts (E22's references; the same pcm, so cache hits)
        for k in sorted(r["pl"].pans):
            rep = r["rel"]["takes"][k]["parts"][0]
            part = next(p for p in r["track"].parts if p.id == rep)
            y = T._read(r["root"] / part.audio, r["pl"].start_s, EXCERPT_S).mean(axis=1)
            _mono_view(f"{base}.t{k}", y, T.SR, d, views)
    if progress:
        progress(f"전사: 뷰 {len(views)}개 ({len(views) * EXCERPT_S / 60:.0f}분 오디오, 참조는 캐시)")
    raws, rung = _transcribe(views, work)
    rows: list[dict] = []
    detail: dict[str, Any] = {}
    for r in rms:
        base = f"{r['song']}.A"
        ref: dict[str, list] = {}
        for k in r["pl"].pans:
            line = "double" if k in r["pl"].roles["double"] else "centre"
            ref.setdefault(line, []).extend(_guitar(raws[f"{base}.t{k}"]))
        side_notes = _guitar(raws[f"{base}.side"])
        mid_notes = _guitar(raws[f"{base}.mid"])
        common = {"dataset": "cambridge_mt", "item": f"{r['song']}:A", "group": r["song"], "scenario": "A",
                  "section": "", "line": "guitar", "rung": rung}
        det: dict[str, Any] = {"ref": {k: len(v) for k, v in ref.items()}, "side_notes": len(side_notes), "arms": {}}

        def add(arm: str, lead: list, rhythm: list) -> None:
            notes = list(lead) + list(rhythm)
            pred = [["centre"]] * len(lead) + [["double"]] * len(rhythm)
            truth = SE.match(notes, ref)
            vals = {"lead_f1": SE.line_f1(notes, pred, "centre", ref.get("centre", [])),
                    "rhythm_f1": SE.line_f1(notes, pred, "double", ref.get("double", [])),
                    "leak_lead": SE.leakage(truth, pred, "centre", "double"),
                    "leak_rhythm": SE.leakage(truth, pred, "double", "centre")}
            det["arms"][arm] = {**{k: (None if v is None else round(v, 4)) for k, v in vals.items()},
                                "lead_notes": len(lead), "rhythm_notes": len(rhythm)}
            for k, v in vals.items():
                if v is not None:
                    rows.append({**common, "system": arm, "metric": k, "value": round(v, 4)})

        # merged: the mid view's notes in both lines (no split)
        notes = mid_notes
        pred = [["double", "centre"]] * len(notes)
        truth = SE.match(notes, ref)
        for k, v in (("lead_f1", SE.line_f1(notes, pred, "centre", ref.get("centre", []))),
                     ("rhythm_f1", SE.line_f1(notes, pred, "double", ref.get("double", [])))):
            if v is not None:
                rows.append({**common, "system": "merged", "metric": k, "value": round(v, 4)})
        det["arms"]["merged"] = {"notes": len(notes)}
        for arm in E7_ARMS:
            cn = _guitar(raws[f"{base}.{arm}"])
            keep = lead_minus_side(cn, side_notes)
            add(arm, [n for n, k in zip(cn, keep) if k], side_notes)
        detail[f"{r['song']}:A"] = det
    decision = decide_e7(rows)
    atomic.write_json(out_dir / "e7_items.json", detail)
    atomic.write_json(out_dir / "e7_decision.json", {**decision, "centre": centre_info, "e6_run": str(e6),
                                                     "excerpt_s": EXCERPT_S})
    log.info("E7: %s (chosen %s)", decision["decision"], decision["chosen"])
    return rows
