"""E22 (M6b, docs/decisions.md): the per-note two-Line split on E6's Tier B remixes.

For every P and A remix of an E6 run (the same 60 s mixes and their cached SW guitar stems), the first
``EXCERPT_S`` seconds:

- reference: MuScriptor on each take's own excerpt (the take's representative part, mono, guitar-only mask) - the
  E1 "reftx" idea at take level: what the transcriber hears without separation or routing;
- system: MuScriptor on the SW guitar stem's mono down-mix (guitar-only mask), every note with its stereo signature
  (``notesig``) and a line from each arm:
  - ``majority``: every note to the remix's largest reference line (the macro-accuracy floor, DESIGN 8.5);
  - ``window_pan`` (P only): every note of an 8 s window to the side holding more of the window's energy;
  - ``merged`` (A only): no split (the "합친 기준선" of the M6 A criterion);
  - ``note_sig|oracle``: ``split.assign`` with the windows labelled by construction (E6 truth);
  - ``note_sig|router``: ``split.assign`` with the router's labels from the SW features (design thresholds).
- scoring: a system note matches a reference note of the same pitch within ``ONSET_TOL_S`` (greedy, nearest first,
  each reference note once per take). Its reference line set = the lines of the takes it matched (P: the take
  panned left -> ``L``; A: the double's takes -> ``double``, the centred takes -> ``centre``); a set of two lines
  is a unison. Unmatched system notes are not scored here (they are transcription errors, not routing errors).
  - ``assign_macro``: over the lines present, the share of the line's exclusive notes predicted exactly that line
    (a note written to both lines when it belongs to one is wrong), averaged; ``unison_hit``: unison notes
    predicted in both lines.
  - A: ``lead_f1`` / ``rhythm_f1`` = note F1 of the predicted ``centre`` / ``double`` line against the centred /
    double takes' reference; ``leak_lead`` = share of predicted centre-line notes whose reference is the double
    only, ``leak_rhythm`` = the reverse.
Rows: one per remix and arm (``item`` = remix, ``group`` = song, ``scenario`` = P / A).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic, paths
from bandscribe.router import experiments as X
from bandscribe.router import notesig as N
from bandscribe.router import split as SP
from bandscribe.router import tierb as T
from bandscribe.router.thresholds import THRESHOLDS

log = logging.getLogger(__name__)

EXCERPT_S = 32.0
ONSET_TOL_S = 0.05
P_ARMS = ("majority", "window_pan", "note_sig|oracle", "note_sig|router")
A_ARMS = ("majority", "merged", "note_sig|oracle", "note_sig|router")


def latest_run(exp: str = "e6") -> Path | None:
    runs = sorted(Path(paths.RUNS).glob(f"*_exp-{exp}"))
    return runs[-1] if runs else None


def match(sys_notes: Sequence[Mapping[str, Any]], ref: Mapping[str, Sequence[Mapping[str, Any]]],
          tol: float = ONSET_TOL_S) -> list[set[str]]:
    """Reference line set per system note (module doc): greedy nearest-onset matching per line, each reference note
    used once per line."""
    out: list[set[str]] = [set() for _ in sys_notes]
    for line, notes in ref.items():
        by_pitch: dict[int, list[tuple[float, int]]] = {}
        for j, r in enumerate(notes):
            by_pitch.setdefault(int(r["pitch"]), []).append((float(r["onset_s"]), j))
        pairs = []
        for i, n in enumerate(sys_notes):
            for t, j in by_pitch.get(int(n["pitch"]), []):
                d = abs(t - float(n["onset_s"]))
                if d <= tol:
                    pairs.append((d, i, j))
        used_i: set[int] = set()
        used_j: set[int] = set()
        for d, i, j in sorted(pairs):
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            out[i].add(line)
    return out


def assign_macro(truth: Sequence[set[str]], pred: Sequence[Sequence[str]]) -> tuple[float | None, dict[str, Any]]:
    """Macro accuracy over the lines' exclusive notes, plus the unison hit rate (module doc)."""
    lines = sorted({ln for t in truth if len(t) == 1 for ln in t})
    per: dict[str, float] = {}
    for ln in lines:
        idx = [i for i, t in enumerate(truth) if t == {ln}]
        if idx:
            per[ln] = sum(1 for i in idx if set(pred[i]) == {ln}) / len(idx)
    uni = [i for i, t in enumerate(truth) if len(t) > 1]
    hit = (sum(1 for i in uni if set(pred[i]) >= truth[i]) / len(uni)) if uni else None
    macro = float(np.mean(list(per.values()))) if per else None
    return macro, {"per_line": {k: round(v, 4) for k, v in per.items()}, "unison_n": len(uni),
                   "unison_hit": None if hit is None else round(hit, 4),
                   "n": sum(1 for t in truth if t)}


def line_f1(sys_notes: Sequence[Mapping[str, Any]], pred: Sequence[Sequence[str]], line: str,
            ref_notes: Sequence[Mapping[str, Any]]) -> float | None:
    """Note F1 (pitch, onset +-50 ms) of the system notes predicted ``line`` against ``ref_notes``."""
    est = [n for n, p in zip(sys_notes, pred) if line in p]
    if not ref_notes and not est:
        return None
    m = match(est, {"x": ref_notes})
    tp = sum(1 for s in m if s)
    prec = tp / len(est) if est else 0.0
    rec = tp / len(ref_notes) if ref_notes else 0.0
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def leakage(truth: Sequence[set[str]], pred: Sequence[Sequence[str]], line: str, other: str) -> float | None:
    """Share of notes predicted ``line`` whose reference is ``other`` only."""
    idx = [i for i, p in enumerate(pred) if line in p and truth[i]]
    if not idx:
        return None
    return sum(1 for i in idx if truth[i] == {other}) / len(idx)


def window_pan_lines(stem: np.ndarray, sr: int, sys_notes: Sequence[Mapping[str, Any]],
                     windows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Every note of a window to the side with more of the window's energy."""
    side = []
    for w in windows:
        a, b = int(round(w["start_s"] * sr)), int(round(w["end_s"] * sr))
        el = float(np.sum(stem[a:b, 0].astype(np.float64) ** 2))
        er = float(np.sum(stem[a:b, 1].astype(np.float64) ** 2))
        side.append((float(w["start_s"]), float(w["end_s"]), "L" if el >= er else "R"))
    out = []
    for n in sys_notes:
        t = float(n["onset_s"])
        lab = next((s for a, b, s in side if a <= t < b), "all")
        out.append([lab])
    return out


P_GAIN, A_LEAD_GAIN, A_LEAK, A_RHYTHM_DROP = 20.0, 10.0, 0.25, 2.0  # points / share (decisions.md E22)


def _paired(rows: Sequence[Mapping[str, Any]], metric: str, scenario: str, a: str, b: str, *, sign: float = 1.0,
            n: int = X.BOOT_N, seed: int = X.BOOT_SEED) -> dict[str, Any] | None:
    """Mean of sign * (a - b) in points over the scenario's remixes, with a song-block bootstrap CI."""
    va = {r["item"]: (r["group"], float(r["value"])) for r in rows
          if r["metric"] == metric and r["scenario"] == scenario and r["system"] == a}
    vb = {r["item"]: float(r["value"]) for r in rows
          if r["metric"] == metric and r["scenario"] == scenario and r["system"] == b}
    items = sorted(set(va) & set(vb))
    if not items:
        return None
    diffs = {i: sign * 100.0 * (va[i][1] - vb[i]) for i in items}
    songs = sorted({va[i][0] for i in items})
    by = {g: [diffs[i] for i in items if va[i][0] == g] for g in songs}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n):
        pick = rng.integers(0, len(songs), len(songs))
        vals = [v for k in pick for v in by[songs[k]]]
        boot.append(float(np.mean(vals)))
    return {"delta": round(float(np.mean(list(diffs.values()))), 2),
            "ci": [round(float(np.quantile(boot, 0.025)), 2), round(float(np.quantile(boot, 0.975)), 2)],
            "remixes": len(items), "songs": len(songs)}


def _mean(rows: Sequence[Mapping[str, Any]], metric: str, scenario: str, arm: str) -> float | None:
    v = [float(r["value"]) for r in rows if r["metric"] == metric and r["scenario"] == scenario and r["system"] == arm]
    return round(float(np.mean(v)), 4) if v else None


def decide(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pre-registered E22 rules (decisions.md) on the rows."""
    out: dict[str, Any] = {}
    p_maj = _paired(rows, "assign_macro", "P", "note_sig|oracle", "majority")
    p_win = _paired(rows, "assign_macro", "P", "note_sig|oracle", "window_pan")
    p_ok = bool(p_maj and p_maj["delta"] >= P_GAIN and p_maj["ci"][0] > 0)
    out["P"] = {"vs_majority": p_maj, "vs_window_pan": p_win,
                "per_note_beats_window": bool(p_win and p_win["ci"][0] > 0),
                "router_vs_majority": _paired(rows, "assign_macro", "P", "note_sig|router", "majority"),
                "decision": "adopt" if p_ok else "hold",
                "rule": f"note_sig|oracle - majority >= {P_GAIN} points and CI low > 0"}
    lead = _paired(rows, "lead_f1", "A", "note_sig|oracle", "merged")
    drop = _paired(rows, "rhythm_f1", "A", "note_sig|oracle", "merged", sign=-1.0)
    leak = _mean(rows, "leak_lead", "A", "note_sig|oracle")
    a_ok = bool(lead and lead["delta"] >= A_LEAD_GAIN and leak is not None and leak <= A_LEAK and drop
                and drop["delta"] <= A_RHYTHM_DROP)
    out["A"] = {"lead_gain": lead, "rhythm_drop": drop, "leak_lead": leak,
                "leak_rhythm": _mean(rows, "leak_rhythm", "A", "note_sig|oracle"),
                "router_lead_gain": _paired(rows, "lead_f1", "A", "note_sig|router", "merged"),
                "decision": "adopt" if a_ok else "hold",
                "rule": f"lead F1 gain >= {A_LEAD_GAIN} points, leak_lead <= {A_LEAK}, rhythm drop <= {A_RHYTHM_DROP}"}
    return out


def _prepare(out_dir: Path, cfg: Any, args: Mapping[str, Any], progress: Callable[[str], None] | None
             ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per P / A remix of the E6 run: system notes with signatures, the reference line set of each, the oracle and
    router windows (module doc). Transcriptions come from (and go to) the eval cache."""
    import soundfile as sf

    from bandscribe import audio as AU
    from bandscribe import datasets
    from bandscribe.amt import experiments as AE
    from bandscribe.amt import instruments as I
    from bandscribe.amt import stage as S
    from bandscribe.amt.reftx import write_mono_f32

    work = Path(out_dir) / "work"
    e6 = Path(args["e6_run"]) if args.get("e6_run") else latest_run("e6")
    if e6 is None or not (e6 / "e6_windows.json").is_file():
        raise AE.ExperimentDataMissing("E22: E6 실행 결과(e6_windows.json)가 없습니다. 먼저 bandscribe eval exp E6")
    wins = json.loads((e6 / "e6_windows.json").read_text(encoding="utf-8"))
    split_of = {s: v["split"] for s, v in json.loads((e6 / "relations.json").read_text(encoding="utf-8")).items()}
    root = Path(datasets.dataset_root("cambridge_mt"))
    tracks = {t.track_id: t for t in datasets.iter_tracks("cambridge_mt")}
    order = sorted(tracks)  # E6's song_index
    rel_dir = X.cache_root() / "relations"
    remixes = sorted({(w["song"], w["scenario"]) for w in wins if w["scenario"] in ("A", "P")})
    if args.get("limit"):  # smoke tests only
        remixes = remixes[: int(args["limit"])]
    sep_rung = AE.sep_top_rung(cfg)
    gmask = I.mask_guitar_only()
    plans, views = {}, []
    for song, sc in remixes:
        rel = T.relations(tracks[song], root, rel_dir)
        pl = T.plan(rel, sc, song_index=order.index(song))
        mix_wav = e6 / "work" / f"{song}_{sc}" / "mix.wav"
        mix, _ = sf.read(str(mix_wav), dtype="float32", always_2d=True)
        stems, _ = X.separate_guitar([(X._pcm_sha(mix), mix_wav)], cfg, force_rung=sep_rung,
                                     cache_dir=X.cache_root())
        stem, sr = sf.read(str(next(iter(stems.values()))), dtype="float32", always_2d=True)
        stem = stem[: int(EXCERPT_S * sr)]
        d = work / f"{song}_{sc}"
        d.mkdir(parents=True, exist_ok=True)
        sha = write_mono_f32(d / "sys_mono.wav", stem.mean(axis=1), sr)
        views.append({"id": f"{song}.{sc}.sys", "wav": d / "sys_mono.wav", "pcm_sha256": sha, "instruments": gmask})
        for k in sorted(pl.pans):
            rep = rel["takes"][k]["parts"][0]
            part = next(p for p in tracks[song].parts if p.id == rep)
            y = T._read(root / part.audio, pl.start_s, EXCERPT_S).mean(axis=1)
            sha = write_mono_f32(d / f"take{k}_mono.wav", y, T.SR)
            views.append({"id": f"{song}.{sc}.t{k}", "wav": d / f"take{k}_mono.wav", "pcm_sha256": sha,
                          "instruments": gmask})
        plans[(song, sc)] = (pl, d)
        AU.write_f32(d / "stem.wav", stem, sr)
    if progress:
        progress(f"전사: 뷰 {len(views)}개 ({len(views) * EXCERPT_S / 60:.0f}분 오디오)")
    params = S.ms_params(cfg)
    ms_top = AE.ms_rung(params)
    raws, info_ms = S.transcribe_views_ms(views, params, cfg, work_dir=work / "ms" / "_worker",
                                          cache=AE._eval_cache(), job=None, force_rung=ms_top)
    rung = f"{sep_rung}|{info_ms.get('rung_label') or ms_top}"
    gclasses = set(I.GUITAR_CLASSES)
    items = []
    for (song, sc), (pl, d) in plans.items():
        stem, sr = sf.read(str(d / "stem.wav"), dtype="float32", always_2d=True)
        sys_notes = [n for n in raws[f"{song}.{sc}.sys"].get("notes") or [] if n.get("instrument") in gclasses]
        if sc == "P":
            line_of = {k: ("L" if pan < 0 else "R") for k, pan in pl.pans.items()}
        else:
            line_of = {k: ("double" if k in pl.roles["double"] else "centre") for k in pl.pans}
        ref: dict[str, list] = {}
        for k in pl.pans:
            notes = [n for n in raws[f"{song}.{sc}.t{k}"].get("notes") or [] if n.get("instrument") in gclasses]
            ref.setdefault(line_of[k], []).extend(notes)
        ws = [w for w in wins if w["song"] == song and w["scenario"] == sc and w["end_s"] <= EXCERPT_S + 1e-6]
        items.append({
            "song": song, "scenario": sc, "split": split_of.get(song, ""), "sys": sys_notes, "ref": ref,
            "truth": match(sys_notes, ref), "sigs": N.signatures(stem, sr, sys_notes),
            "oracle": [{"start_s": w["start_s"], "end_s": w["end_s"], "label": w["truth"]} for w in ws],
            "routed": [{"start_s": w["start_s"], "end_s": w["end_s"], "label": X.predict(w["sw"], THRESHOLDS)}
                       for w in ws],
            "window_pan": window_pan_lines(stem, sr, sys_notes, ws) if sc == "P" else None})
    return items, {"e6_run": str(e6), "excerpt_s": EXCERPT_S, "views": len(views), "rung": rung}


def _score(it: Mapping[str, Any], split_params: Mapping[str, float], rung: str, section: str = ""
           ) -> tuple[list[dict], dict[str, Any]]:
    """Rows and detail of one prepared remix under ``split_params`` (module doc)."""
    sys_notes, truth, sigs, ref, sc = it["sys"], it["truth"], it["sigs"], it["ref"], it["scenario"]
    counts: dict[str, int] = {}
    for t in truth:
        for ln in t:
            counts[ln] = counts.get(ln, 0) + 1
    major = max(sorted(counts), key=lambda k: counts[k]) if counts else "all"
    preds: dict[str, list[list[str]]] = {
        "majority": [[major] for _ in sys_notes],
        "note_sig|oracle": [a["lines"] for a in SP.assign(sys_notes, sigs, it["oracle"], split_params)],
        "note_sig|router": [a["lines"] for a in SP.assign(sys_notes, sigs, it["routed"], split_params)],
    }
    if sc == "P":
        preds["window_pan"] = it["window_pan"]
    else:
        preds["merged"] = [["double", "centre"] for _ in sys_notes]  # every note in both lines = no split
    item = f"{it['song']}:{sc}"
    common = {"dataset": "cambridge_mt", "item": item, "group": it["song"], "scenario": sc, "rung": rung,
              "line": "guitar", "section": section}
    rows: list[dict] = []
    det: dict[str, Any] = {"notes": len(sys_notes), "matched": sum(1 for t in truth if t),
                           "ref": {k: len(v) for k, v in ref.items()}, "split": it["split"], "arms": {}}
    for arm in (P_ARMS if sc == "P" else A_ARMS):
        pred = preds[arm]
        macro, extra = assign_macro(truth, pred)
        arm_d: dict[str, Any] = {"assign_macro": macro, **extra}
        if macro is not None:
            rows.append({**common, "system": arm, "metric": "assign_macro", "value": round(macro, 4)})
        if sc == "A":
            for name, val in (("lead_f1", line_f1(sys_notes, pred, "centre", ref.get("centre", []))),
                              ("rhythm_f1", line_f1(sys_notes, pred, "double", ref.get("double", []))),
                              ("leak_lead", leakage(truth, pred, "centre", "double")),
                              ("leak_rhythm", leakage(truth, pred, "double", "centre"))):
                arm_d[name] = None if val is None else round(val, 4)
                if val is not None:
                    rows.append({**common, "system": arm, "metric": name, "value": round(val, 4)})
        det["arms"][arm] = arm_d
    return rows, det


def run_e22(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    out_dir = Path(out_dir)
    items, info = _prepare(out_dir, cfg, args, progress)
    rows: list[dict] = []
    detail: dict[str, Any] = {}
    for it in items:
        r, det = _score(it, SP.DEFAULT_PARAMS, info["rung"])
        rows += r
        detail[f"{it['song']}:{it['scenario']}"] = det
    atomic.write_json(out_dir / "e22_items.json", detail)
    atomic.write_json(out_dir / "e22_decision.json", decide(rows))
    atomic.write_json(out_dir / "e22_info.json", {**info, "split_params": dict(SP.DEFAULT_PARAMS)})
    return rows


# ----------------------------------------------------------------------------------------------- E22b

E22B_P_ILD = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
E22B_A_COH = (0.6, 0.7, 0.8, 0.9)
E22B_A_MS = (3.0, 6.0, 10.0, 15.0)


def _mean_metric(rows: Sequence[Mapping[str, Any]], metric: str, arm: str) -> float | None:
    v = [float(r["value"]) for r in rows if r["metric"] == metric and r["system"] == arm]
    return float(np.mean(v)) if v else None


def choose_params(dev: Sequence[Mapping[str, Any]], rung: str) -> tuple[dict[str, float], dict[str, Any]]:
    """E22b (decisions.md): the split thresholds chosen on the dev remixes only. P: the ``p_ild_db`` with the best
    mean ``note_sig|oracle`` macro accuracy (ties: the larger). A: among (``a_coh``, ``a_ms_db``) whose mean
    ``leak_lead`` <= 0.25 and mean rhythm drop <= 2 points, the best mean lead-F1 gain over ``merged`` (ties: the
    first in grid order); none passing -> the a-priori values."""
    trace: dict[str, Any] = {"P": {}, "A": {}}
    p_items = [it for it in dev if it["scenario"] == "P"]
    a_items = [it for it in dev if it["scenario"] == "A"]
    best_p, best_v = SP.DEFAULT_PARAMS["p_ild_db"], None
    for t in E22B_P_ILD:
        rows = [r for it in p_items for r in _score(it, {**SP.DEFAULT_PARAMS, "p_ild_db": t}, rung)[0]]
        v = _mean_metric(rows, "assign_macro", "note_sig|oracle")
        trace["P"][str(t)] = None if v is None else round(v, 4)
        if v is not None and (best_v is None or v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and t > best_p)):
            best_p, best_v = t, v
    best_a = (SP.DEFAULT_PARAMS["a_coh"], SP.DEFAULT_PARAMS["a_ms_db"])
    best_gain = None
    for c in E22B_A_COH:
        for m in E22B_A_MS:
            prm = {**SP.DEFAULT_PARAMS, "a_coh": c, "a_ms_db": m}
            rows = [r for it in a_items for r in _score(it, prm, rung)[0]]
            lead_s, lead_m = _mean_metric(rows, "lead_f1", "note_sig|oracle"), _mean_metric(rows, "lead_f1", "merged")
            rh_s, rh_m = _mean_metric(rows, "rhythm_f1", "note_sig|oracle"), _mean_metric(rows, "rhythm_f1", "merged")
            leak = _mean_metric(rows, "leak_lead", "note_sig|oracle")
            gain = None if lead_s is None or lead_m is None else 100 * (lead_s - lead_m)
            drop = None if rh_s is None or rh_m is None else 100 * (rh_m - rh_s)
            ok = bool(gain is not None and leak is not None and leak <= A_LEAK and drop is not None
                      and drop <= A_RHYTHM_DROP)
            trace["A"][f"{c}/{m}"] = {"gain": None if gain is None else round(gain, 2),
                                      "leak": None if leak is None else round(leak, 4),
                                      "drop": None if drop is None else round(drop, 2), "ok": ok}
            if ok and gain is not None and (best_gain is None or gain > best_gain + 1e-9):
                best_a, best_gain = (c, m), gain
    return {**SP.DEFAULT_PARAMS, "p_ild_db": best_p, "a_coh": best_a[0], "a_ms_db": best_a[1]}, trace


def run_e22b(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    out_dir = Path(out_dir)
    items, info = _prepare(out_dir, cfg, args, progress)
    dev = [it for it in items if it["split"] == "dev"]
    chosen, trace = choose_params(dev, info["rung"])
    rows: list[dict] = []
    detail: dict[str, Any] = {}
    for it in items:
        r, det = _score(it, chosen, info["rung"], section=it["split"])
        rows += r
        detail[f"{it['song']}:{it['scenario']}"] = det
    atomic.write_json(out_dir / "e22_items.json", detail)
    atomic.write_json(out_dir / "e22_decision.json", {
        "chosen": chosen, "trace": trace, "test": decide([r for r in rows if r["section"] == "test"]),
        "dev": decide([r for r in rows if r["section"] == "dev"])})
    atomic.write_json(out_dir / "e22_info.json", {**info, "split_params": chosen})
    return rows
