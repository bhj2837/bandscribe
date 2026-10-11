"""E25 (M7a, docs/decisions.md): Part view v0 - the part count and the note-to-Line assignment on Tier B remixes.

Remixes (from E6's run; the first ``EXCERPT_S`` seconds, its 8 s windows, one section):

- ``P``, ``ADT``: E6's own remixes (mix and SW stem from the eval cache);
- ``A1``, ``B2``, ``N1`` (``tierb.VARIANTS``): rendered here from the same relations, takes and starts and separated
  with E6's SW rung (eval cache) - DESIGN's A (wide double + one centred part), B (two parts at the centre) and a mono
  single guitar.

Reference: MuScriptor on each included take's own excerpt (E22's take views - cached where E22 made them). Lines: the
included takes joined by the relations' double links (R1: a double is one Line). A Line with ``MIN_REF_NOTES``
reference notes is active (DESIGN 4.4: smaller Lines are dissolved); the true part count is the number of active
Lines and a remix with none is not scored. System: MuScriptor on the SW stem's mono down-mix (guitar classes), each
note with its stereo signature; its truth = the active Lines whose reference it matches (E22's ``match``: pitch,
onset +-50 ms). Router labels: the SW window features with the pipeline's thresholds (``[router.thresholds]``).

Arms: ``majority`` (every note to one Line, count 1), ``symbolic`` (``layers.tabforge_split``, count 2),
``s8v0`` (``diarize`` with the dev-chosen params) and ``s8v0|oracle_k`` (the true count forced; diagnostic).
Scores per remix and arm: ``k_correct``, ``k_abs_err``, ``assign_macro`` (predicted Lines matched to the active
Lines by the Hungarian method on the exclusive notes; a Line's accuracy = its exclusive notes in its matched
prediction, Unassigned and unmatched predictions are wrong; averaged over the Lines with exclusive notes),
``unassigned`` and ``unison_hit``. The dev selection and the decision rules are decisions.md E25's.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic
from bandscribe.parts import diarize as DZ
from bandscribe.parts import layers as LY
from bandscribe.parts import reid as R

log = logging.getLogger(__name__)

EXCERPT_S = 32.0
MIN_REF_NOTES = 16
SCENARIOS = ("P", "ADT", "A1", "B2", "N1")
ONE_GUITAR = ("ADT", "N1")
ARMS = ("majority", "symbolic", "s8v0", "s8v0|oracle_k")
TAUS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# dev grid, conservative first (ties keep the earlier): notes per stream, windows, side margin, evidence kinds
GRID = [{"n_min": n, "min_windows": mw, "side_db": sd, "use_class": uc, "use_symbolic": us}
        for n in (12, 8, 6) for mw in (3, 2) for sd in (3.0, 0.0) for uc in (False, True) for us in (False, True)]
K_ACC, K_ACC_AP, K_MAE, GAIN_AP, GAIN_B, GAIN_SYM, UNASSIGNED_MAX, REGRESS = 0.70, 0.90, 0.4, 20.0, 10.0, 10.0, 0.15, -2.0


# ---------------------------------------------------------------------------------------------- scoring


def macro_hungarian(truth: Sequence[set[str]], pred: Sequence[Any]) -> tuple[float | None, dict[str, Any]]:
    """Macro accuracy (module doc) of predicted labels (None = Unassigned) against truth Line sets."""
    from scipy.optimize import linear_sum_assignment

    lines = sorted({ln for t in truth if len(t) == 1 for ln in t})
    if not lines:
        return None, {"per_line": {}, "match": {}}
    labs = sorted({str(p) for p in pred if p is not None})
    cmat = np.zeros((max(1, len(labs)), len(lines)))
    for t, p in zip(truth, pred):
        if len(t) == 1 and p is not None:
            cmat[labs.index(str(p)), lines.index(next(iter(t)))] += 1
    rows, cols = linear_sum_assignment(-cmat)
    n_ex = {ln: sum(1 for t in truth if t == {ln}) for ln in lines}
    acc = {lines[c]: float(cmat[r, c] / n_ex[lines[c]]) for r, c in zip(rows, cols) if labs}
    per = {ln: round(acc.get(ln, 0.0), 4) for ln in lines}
    match = {labs[r]: lines[c] for r, c in zip(rows, cols) if labs and cmat[r, c] > 0}
    return float(np.mean(list(per.values()))), {"per_line": per, "match": match}


def unison_hit(truth: Sequence[set[str]], pred: Sequence[Any], match: Mapping[str, str]) -> float | None:
    uni = [i for i, t in enumerate(truth) if len(t) > 1]
    if not uni:
        return None
    return sum(1 for i in uni if pred[i] is not None and match.get(str(pred[i])) in truth[i]) / len(uni)


def coverage_accuracy(truth: Sequence[set[str]], pred: Sequence[Any], p: Sequence[float], match: Mapping[str, str],
                      taus: Sequence[float] = TAUS) -> list[dict[str, Any]]:
    """Per tau: the share of scored (exclusive) notes with posterior >= tau and the accuracy among them, with the
    Line matching fixed at tau = 0."""
    idx = [i for i, t in enumerate(truth) if len(t) == 1]
    out = []
    for tau in taus:
        kept = [i for i in idx if pred[i] is not None and p[i] >= tau]
        ok = sum(1 for i in kept if match.get(str(pred[i])) in truth[i])
        out.append({"tau": tau, "coverage": round(len(kept) / len(idx), 4) if idx else None,
                    "accuracy": round(ok / len(kept), 4) if kept else None})
    return out


def arm_preds(it: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """{arm: {pred (label or None per note), k, p}} for one prepared remix."""
    notes, sigs, wins = it["sys"], it["sigs"], it["windows"]
    feats = LY.note_features(notes)
    out: dict[str, dict[str, Any]] = {"majority": {"pred": ["1"] * len(notes), "k": 1 if notes else 0}}
    sym = LY.tabforge_split(notes, feats)
    out["symbolic"] = {"pred": sym, "k": len(set(sym))}
    for arm, kf in (("s8v0", None), ("s8v0|oracle_k", it["k_true"] or None)):
        d = DZ.diarize(notes, sigs, wins, params=params, k_force=kf)
        out[arm] = {"pred": [None if n["line"] is None else str(n["line"]) for n in d["note"]],
                    "p": [n["p"] for n in d["note"]], "k": d["count"]["chosen"], "count": d["count"],
                    "lines": [{k: ln[k] for k in ("number", "name", "notes", "key", "kind")} for ln in d["lines"]]}
    return out


def score_item(it: Mapping[str, Any], params: Mapping[str, Any], rung: str = "") -> tuple[list[dict], dict]:
    """Rows and detail of one prepared remix (module doc)."""
    preds = arm_preds(it, params)
    item = f"{it['song']}:{it['scenario']}"
    common = {"dataset": "cambridge_mt", "item": item, "group": it["song"], "scenario": it["scenario"],
              "rung": rung, "line": "guitar", "section": it["split"]}
    rows: list[dict] = []
    det: dict[str, Any] = {"k_true": it["k_true"], "notes": len(it["sys"]), "lines_ref": it["ref_counts"],
                           "arms": {}}
    for arm in ARMS:
        pr = preds[arm]
        macro, extra = macro_hungarian(it["truth"], pr["pred"])
        un = sum(1 for x in pr["pred"] if x is None)
        vals = {"k_pred": pr["k"], "k_correct": float(pr["k"] == it["k_true"]),
                "k_abs_err": float(abs(pr["k"] - it["k_true"])), "assign_macro": macro,
                "unassigned": un / len(pr["pred"]) if pr["pred"] else 0.0,
                "unison_hit": unison_hit(it["truth"], pr["pred"], extra["match"])}
        d = {**vals, **extra, "n_unassigned": un}
        if "p" in pr:
            d["coverage_accuracy"] = coverage_accuracy(it["truth"], pr["pred"], pr["p"], extra["match"])
            d["count"] = pr["count"]
            d["lines"] = pr["lines"]
        det["arms"][arm] = d
        for m in ("k_correct", "k_abs_err", "assign_macro", "unassigned", "unison_hit"):
            if vals[m] is not None:
                rows.append({**common, "system": arm, "metric": m, "value": round(float(vals[m]), 4)})
    return rows, det


# ------------------------------------------------------------------------------------------- dev / decide


def objective(items: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> tuple[int, float, float]:
    """(1-guitar remixes kept at one part, part-count accuracy, mean macro over A1 / P / B2) of s8v0 (decisions.md
    E25 dev selection)."""
    one, kok, macros = 0, [], []
    for it in items:
        d = DZ.diarize(it["sys"], it["sigs"], it["windows"], params=params)
        k = d["count"]["chosen"]
        kok.append(k == it["k_true"])
        if it["scenario"] in ONE_GUITAR and it["k_true"] == 1 and k == 1:
            one += 1
        if it["scenario"] in ("A1", "P", "B2"):
            m, _ = macro_hungarian(it["truth"], [None if n["line"] is None else str(n["line"]) for n in d["note"]])
            if m is not None:
                macros.append(m)
    return one, (float(np.mean(kok)) if kok else 0.0), (float(np.mean(macros)) if macros else 0.0)


def choose_params(dev: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best, best_obj, trace = None, None, []
    for g in GRID:
        prm = {**R.DEFAULT_PARAMS, **g}
        obj = objective(dev, prm)
        trace.append({**g, "one_guitar_k1": obj[0], "k_acc": round(obj[1], 4), "macro": round(obj[2], 4)})
        if best_obj is None or obj > best_obj:
            best, best_obj = prm, obj
    return dict(best or R.DEFAULT_PARAMS), trace


def decide(rows: Sequence[Mapping[str, Any]], detail: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """decisions.md E25's rules on the given (test) rows; ``detail`` for the pooled Unassigned share."""
    from bandscribe.router import split_eval as SE

    def mean(metric: str, arm: str, scs: Sequence[str]) -> float | None:
        v = [float(r["value"]) for r in rows if r["metric"] == metric and r["system"] == arm and r["scenario"] in scs]
        return round(float(np.mean(v)), 4) if v else None

    out: dict[str, Any] = {}
    k_all = mean("k_correct", "s8v0", SCENARIOS)
    k_ap = mean("k_correct", "s8v0", ("A1", "P"))
    mae = mean("k_abs_err", "s8v0", SCENARIOS)
    out["count"] = {"accuracy": k_all, "accuracy_A1_P": k_ap, "mae": mae,
                    "by_scenario": {sc: mean("k_correct", "s8v0", (sc,)) for sc in SCENARIOS},
                    "pass": bool(k_all is not None and k_all >= K_ACC and k_ap is not None and k_ap >= K_ACC_AP
                                 and mae is not None and mae <= K_MAE),
                    "rule": f"accuracy >= {K_ACC}, A1+P >= {K_ACC_AP}, MAE <= {K_MAE}"}
    gains: dict[str, Any] = {}
    for sc, need in (("A1", GAIN_AP), ("P", GAIN_AP), ("B2", GAIN_B)):
        maj = SE._paired(rows, "assign_macro", sc, "s8v0", "majority")
        sym = SE._paired(rows, "assign_macro", sc, "s8v0", "symbolic") if sc in ("A1", "P") else None
        ok = bool(maj and maj["delta"] >= need and maj["ci"][0] > 0)
        rule = f"s8v0 - majority >= {need} points, CI low > 0"
        if sc in ("A1", "P"):
            ok = ok and bool(sym and sym["delta"] >= GAIN_SYM and sym["ci"][0] > 0)
            rule += f"; s8v0 - symbolic >= {GAIN_SYM} points, CI low > 0"
        gains[sc] = {"vs_majority": maj, "vs_symbolic": sym, "pass": ok, "rule": rule}
    out["assign"] = gains
    one = [it for key, it in detail.items() if key.split(":")[1] in ONE_GUITAR and it["k_true"] == 1]
    kept = sum(1 for it in one if it["arms"]["s8v0"]["k_pred"] == 1)
    out["one_guitar"] = {"remixes": len(one), "kept_one": kept, "pass": bool(one) and kept == len(one)}
    n = sum(it["notes"] for it in detail.values())
    un = sum(it["arms"]["s8v0"]["n_unassigned"] for it in detail.values())
    out["unassigned"] = {"share": round(un / n, 4) if n else None, "pass": bool(n) and un / n <= UNASSIGNED_MAX}
    reg = {}
    for sc in SCENARIOS:
        d = SE._paired(rows, "assign_macro", sc, "s8v0", "majority")
        reg[sc] = None if d is None else d["ci"][1]
    out["no_regression"] = {"ci_high_vs_majority": reg,
                            "pass": all(v is None or v >= REGRESS for v in reg.values())}
    out["oracle_k"] = {sc: SE._paired(rows, "assign_macro", sc, "s8v0|oracle_k", "majority") for sc in SCENARIOS}
    return out


# ------------------------------------------------------------------------------------------------ data


def line_groups(rel: Mapping[str, Any], takes: Sequence[int]) -> dict[int, str]:
    """Take -> Line id: the included takes joined by double links (R1); the id is ``g<lowest take>``."""
    par = {k: k for k in takes}

    def find(x: int) -> int:
        while par[x] != x:
            x = par[x]
        return x

    for ln in rel["links"]:
        a, b = ln["takes"]
        if ln["relation"] == "double" and a in par and b in par:
            ra, rb = find(a), find(b)
            par[max(ra, rb)] = min(ra, rb)
    return {k: f"g{find(k)}" for k in takes}


def _prepare(out_dir: Path, cfg: Any, args: Mapping[str, Any], progress: Callable[[str], None] | None
             ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import soundfile as sf

    from bandscribe import audio as AU
    from bandscribe import datasets
    from bandscribe.amt import experiments as AE
    from bandscribe.amt import instruments as I
    from bandscribe.amt import stage as S
    from bandscribe.amt.reftx import write_mono_f32
    from bandscribe.router import experiments as X
    from bandscribe.router import notesig as N
    from bandscribe.router import split_eval as SE
    from bandscribe.router import stage as RS
    from bandscribe.router import tierb as T

    work = Path(out_dir) / "work"
    e6 = Path(args["e6_run"]) if args.get("e6_run") else SE.latest_run("e6")
    if e6 is None or not (e6 / "e6_windows.json").is_file():
        raise AE.ExperimentDataMissing("E25: E6 실행 결과(e6_windows.json)가 없습니다. 먼저 bandscribe eval exp E6")
    e6w = json.loads((e6 / "e6_windows.json").read_text(encoding="utf-8"))
    split_of = {s: v["split"] for s, v in json.loads((e6 / "relations.json").read_text(encoding="utf-8")).items()}
    root = Path(datasets.dataset_root("cambridge_mt"))
    tracks = {t.track_id: t for t in datasets.iter_tracks("cambridge_mt")}
    order = sorted(tracks)  # E6's song_index
    songs = sorted(s for s in split_of if s in tracks)
    if args.get("songs"):  # smoke tests only
        want = {x.strip() for x in str(args["songs"]).split(",")}
        songs = [s for s in songs if s in want]
    thr = RS.router_params(cfg)["thresholds"]
    rel_dir = X.cache_root() / "relations"
    sep_rung = AE.sep_top_rung(cfg)
    remixes = []
    for song in songs:
        rel = T.relations(tracks[song], root, rel_dir, code_version=SE.E6_REL_VERSION)
        k = order.index(song)
        for sc in SCENARIOS:
            if sc in T.VARIANTS:
                pl = T.variant_plan(rel, sc, song_index=k, eval_s=EXCERPT_S)
                if pl is None:
                    continue
                if progress:
                    progress(f"리믹스: {song} {sc}")
                rm = T.build(tracks[song], root, rel, pl)
                d = work / f"{song}_{sc}"
                d.mkdir(parents=True, exist_ok=True)
                AU.write_f32(d / "mix.wav", rm.mix, T.SR)
                wins = [{"start_s": w["start_s"], "end_s": w["end_s"], "truth": w["truth"]} for w in rm.windows]
                remixes.append({"song": song, "scenario": sc, "rel": rel, "plan": pl, "mix": d / "mix.wav",
                                "sha": X._pcm_sha(rm.mix), "windows": wins, "sw": None})
                del rm
            else:
                pl = T.plan(rel, sc, song_index=k)
                mix_wav = e6 / "work" / f"{song}_{sc}" / "mix.wav"
                if pl is None or not mix_wav.is_file():
                    continue
                mix, _ = sf.read(str(mix_wav), dtype="float32", always_2d=True)
                ws = [w for w in e6w if w["song"] == song and w["scenario"] == sc]
                remixes.append({"song": song, "scenario": sc, "rel": rel, "plan": pl, "mix": mix_wav,
                                "sha": X._pcm_sha(mix), "sw": [w["sw"] for w in ws],
                                "windows": [{"start_s": w["start_s"], "end_s": w["end_s"], "truth": w["truth"]}
                                            for w in ws]})
                del mix
    stems, sep_info = X.separate_guitar([(r["sha"], Path(r["mix"])) for r in remixes], cfg, force_rung=sep_rung,
                                        cache_dir=X.cache_root(), progress=progress)
    gmask = I.mask_guitar_only()
    views: list[dict[str, Any]] = []
    for r in remixes:
        g, sr = sf.read(str(stems[r["sha"]]), dtype="float32", always_2d=True)
        if r["sw"] is None:
            r["sw"] = [X._window_feats(g, int(round(w["start_s"] * sr)), int(round(w["end_s"] * sr)))
                       for w in r["windows"]]
        stem = g[: int(EXCERPT_S * sr)]
        d = work / f"{r['song']}_{r['scenario']}"
        d.mkdir(parents=True, exist_ok=True)
        AU.write_f32(d / "stem.wav", stem, sr)
        sha = write_mono_f32(d / "sys_mono.wav", stem.mean(axis=1), sr)
        views.append({"id": f"{r['song']}.{r['scenario']}.sys", "wav": d / "sys_mono.wav", "pcm_sha256": sha,
                      "instruments": gmask})
        pl = r["plan"]
        for t in sorted(pl.pans):
            rep = r["rel"]["takes"][t]["parts"][0]
            part = next(p for p in tracks[r["song"]].parts if p.id == rep)
            y = T._read(root / part.audio, pl.start_s, EXCERPT_S).mean(axis=1)
            sha_t = write_mono_f32(d / f"take{t}_mono.wav", y, T.SR)
            views.append({"id": f"{r['song']}.{r['scenario']}.t{t}", "wav": d / f"take{t}_mono.wav",
                          "pcm_sha256": sha_t, "instruments": gmask})
        del g, stem
    if progress:
        progress(f"전사: 뷰 {len(views)}개 ({len(views) * EXCERPT_S / 60:.0f}분 오디오, 캐시에 있는 것은 건너뜀)")
    params = S.ms_params(cfg)
    ms_top = AE.ms_rung(params)
    raws, info_ms = S.transcribe_views_ms(views, params, cfg, work_dir=work / "ms" / "_worker",
                                          cache=AE._eval_cache(), job=None, force_rung=ms_top)
    rung = f"{sep_rung}|{info_ms.get('rung_label') or ms_top}"
    gclasses = set(I.GUITAR_CLASSES)
    items = []
    for r in remixes:
        song, sc, pl = r["song"], r["scenario"], r["plan"]
        d = work / f"{song}_{sc}"
        stem, sr = sf.read(str(d / "stem.wav"), dtype="float32", always_2d=True)
        sys_notes = [n for n in raws[f"{song}.{sc}.sys"].get("notes") or [] if n.get("instrument") in gclasses]
        line_of = line_groups(r["rel"], sorted(pl.pans))
        ref: dict[str, list] = {}
        for t in sorted(pl.pans):
            notes = [n for n in raws[f"{song}.{sc}.t{t}"].get("notes") or [] if n.get("instrument") in gclasses]
            ref.setdefault(line_of[t], []).extend(notes)
        active = {ln for ln, ns in ref.items() if len(ns) >= MIN_REF_NOTES}
        truth = [t & active for t in SE.match(sys_notes, ref)]
        wins = []
        for w, f in zip(r["windows"], r["sw"]):
            if float(w["end_s"]) <= EXCERPT_S + 1e-6:
                wins.append({"start_s": float(w["start_s"]), "end_s": float(w["end_s"]), "section": "S1",
                             "label": X.predict(f or {}, thr), "truth_label": w["truth"]})
        items.append({"song": song, "scenario": sc, "split": split_of.get(song, ""), "sys": sys_notes,
                      "sigs": N.signatures(stem, sr, sys_notes) if sys_notes else [], "truth": truth,
                      "k_true": len(active), "ref_counts": {ln: len(ns) for ln, ns in ref.items()},
                      "takes": {str(t): line_of[t] for t in sorted(pl.pans)}, "windows": wins})
    return items, {"e6_run": str(e6), "excerpt_s": EXCERPT_S, "views": len(views), "rung": rung, "sep": sep_info,
                   "remixes": len(remixes), "thresholds": thr}


def run_e25(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    out_dir = Path(out_dir)
    items, info = _prepare(out_dir, cfg, args, progress)
    scored = [it for it in items if it["k_true"] >= 1]
    dev = [it for it in scored if it["split"] == "dev"]
    chosen, trace = choose_params(dev)
    rows: list[dict] = []
    detail: dict[str, Any] = {}
    for it in scored:
        r, det = score_item(it, chosen, info["rung"])
        rows += r
        detail[f"{it['song']}:{it['scenario']}"] = {**det, "split": it["split"], "windows": it["windows"],
                                                    "takes": it["takes"]}
    test_rows = [r for r in rows if r["section"] == "test"]
    dev_rows = [r for r in rows if r["section"] == "dev"]
    atomic.write_json(out_dir / "e25_items.json", detail)
    atomic.write_json(out_dir / "e25_decision.json", {
        "chosen": chosen, "trace": trace,
        "test": decide(test_rows, {k: v for k, v in detail.items() if v["split"] == "test"}),
        "dev": decide(dev_rows, {k: v for k, v in detail.items() if v["split"] == "dev"}),
        "excluded_no_guitar": sorted(f"{it['song']}:{it['scenario']}" for it in items if it["k_true"] < 1)})
    atomic.write_json(out_dir / "e25_info.json", {**info, "params": chosen})
    return rows
