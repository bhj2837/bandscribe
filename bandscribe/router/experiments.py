"""E6 (M6, docs/decisions.md): the router on Tier B remixes. Do the stereo features survive separation (SW), and how
often does the rule label match the way the remix was mixed, with the synthetic thresholds and with thresholds
re-fit on separated stems? The go/no-go of the spatial path (DESIGN 10 M6).

- Remixes: ``bandscribe.router.tierb`` (13 Cambridge-MT songs; B / A / P / ADT where the song can make them; 60 s
  excerpts; 8 s windows labelled from the construction).
- Arms: ``true|design`` (features of the true guitar bus, DESIGN S4 thresholds), ``sw|design`` (features of SW's
  guitar stem, same thresholds), ``sw|refit`` (SW features, thresholds re-fit on the dev songs by a coordinate
  search, scored on both splits; the decision reads test).
- Rows (metric ``router_acc``): one per window and arm, value 1/0, ``scenario`` = the true label, ``section`` =
  dev/test, ``group`` = song. Preservation rows ``d_<feature>`` = |true - SW| per window (system ``sw``).
- Side reports: ``e6_windows.json`` (every window: features both ways, truth, labels), ``refit.json``,
  ``confusion.json``, ``go_nogo.json`` (the pre-registered rule computed), ``relations.json`` (takes / doubles).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic, paths
from bandscribe.router import route as RT
from bandscribe.router import tierb as T
from bandscribe.router.thresholds import THRESHOLDS

log = logging.getLogger(__name__)

CODE_VERSION = "1"
ARMS = ("true|design", "sw|design", "sw|refit")
CLASSES = ("R0", "R1", "R2", "R3")
GO_MACRO, GO_R2, PARTIAL_MACRO = 0.80, 0.70, 0.60
REFIT_SWEEPS = 3
REFIT_QUANTILES = tuple(round(q, 2) for q in np.linspace(0.05, 0.95, 11))
# threshold -> (feature, absolute value?) it is compared with
FEATURE_OF: dict[str, tuple[str, bool]] = {
    "r0_sm_db": ("SM", False), "r0_coh": ("coh", False), "adt_xcorr": ("xcoff", True),
    "adt_min_lag_ms": ("lag", True), "r2_coh": ("coh", False), "r2_hard": ("hard", False),
    "r2_cen": ("cen", False), "r3_ild_db": ("ildcoh", False), "r3_hard": ("hard", False), "r3_coh": ("coh", False)}
PRESERVED = ("SM", "coh", "cen", "hard", "fcoh", "ildcoh", "xcoff")
BOOT_N, BOOT_SEED = 10000, 20260930
SEP_BATCH_S = 600.0


def cache_root() -> Path:
    return Path(paths.EVAL) / "router"


# ------------------------------------------------------------------------------------------------ split


def split_songs(track_ids: Sequence[str], can_a: Mapping[str, bool]) -> dict[str, str]:
    """dev/test by song: songs ordered by sha256(id); the ones that can make an A remix alternate dev/test (dev
    first), then the others continue the alternation so both splits get about half of each kind."""
    order = sorted(track_ids, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    out: dict[str, str] = {}
    a = [t for t in order if can_a.get(t)]
    rest = [t for t in order if not can_a.get(t)]
    for i, t in enumerate(a):
        out[t] = "dev" if i % 2 == 0 else "test"
    start = len(a) % 2  # continue the alternation
    for i, t in enumerate(rest):
        out[t] = "dev" if (i + start) % 2 == 0 else "test"
    return out


# ------------------------------------------------------------------------------------------------ labels


def predict(feat: Mapping[str, Any], thr: Mapping[str, float]) -> str:
    return RT.label(feat, thr)[0] if feat else "mixed"


def macro_accuracy(truth: Sequence[str], pred: Sequence[str]) -> tuple[float, dict[str, float]]:
    """Mean over the true classes present of the share predicted right (balanced accuracy) -> (macro, per class)."""
    per: dict[str, float] = {}
    for c in CLASSES:
        idx = [i for i, t in enumerate(truth) if t == c]
        if idx:
            per[c] = sum(1 for i in idx if pred[i] == c) / len(idx)
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


GROUPS: dict[str, tuple[str, ...]] = {  # thresholds searched together: one label's AND / OR conditions
    "R0": ("r0_sm_db", "r0_coh"), "R1": ("adt_xcorr", "adt_min_lag_ms", "adt_over_r0"),
    "R2": ("r2_coh", "r2_hard", "r2_cen"), "R3": ("r3_ild_db", "r3_hard", "r3_coh")}
SCALE_OF = {"r0_sm_db": "sm_db", "r0_coh": "coh", "adt_xcorr": "xcorr", "adt_min_lag_ms": "lag_ms",
            "adt_over_r0": "xcorr", "r2_coh": "coh", "r2_hard": "hard", "r2_cen": "cen", "r3_ild_db": "ild_db",
            "r3_hard": "hard", "r3_coh": "coh"}


def _feature_values(key: str, fm: Mapping[str, np.ndarray]) -> np.ndarray:
    if key == "adt_over_r0":
        v = np.abs(fm["xcoff"]) - fm["r0"]
    else:
        feat, absolute = FEATURE_OF[key]
        v = np.abs(fm[feat]) if absolute else fm[feat]
    return v[np.isfinite(v)]


def _candidates(key: str, fm: Mapping[str, np.ndarray], design: float) -> list[float]:
    vals = _feature_values(key, fm)
    cands = {round(float(design), 4)}
    if vals.size:
        cands |= {round(float(np.quantile(vals, q)), 4) for q in REFIT_QUANTILES}
    return sorted(cands)


def _macro_vec(truth: np.ndarray, pred: np.ndarray) -> float:
    per = [float(np.mean(pred[truth == c] == c)) for c in CLASSES if np.any(truth == c)]
    return float(np.mean(per)) if per else float("nan")


def refit(dev: Sequence[Mapping[str, Any]], start: Mapping[str, float] = THRESHOLDS) -> tuple[dict[str, float], dict]:
    """Grid search of each label's thresholds together (``GROUPS``; candidates = the design value and
    ``REFIT_QUANTILES`` of the dev windows' SW feature), label by label, up to ``REFIT_SWEEPS`` sweeps.
    Objective: dev macro accuracy; ties -> the smallest summed distance to the design values in ``SCALES`` units,
    then the smaller values. Deterministic. -> (thresholds, trace)."""
    import itertools

    thr = {k: float(v) for k, v in start.items()}
    fm = RT.feature_matrix([w["sw"] or {} for w in dev])
    truth = np.array([w["truth"] for w in dev], dtype=object)
    trace: dict[str, Any] = {"start": round(_macro_vec(truth, RT.labels_from_scores(RT.score_matrix(fm, thr))), 4),
                             "sweeps": []}
    for _sweep in range(REFIT_SWEEPS):
        changed = False
        for lab, keys in GROUPS.items():
            base = RT.score_matrix(fm, thr)
            cands = [_candidates(k, fm, THRESHOLDS[k]) for k in keys]
            best: tuple | None = None
            for combo in itertools.product(*cands):
                t2 = {**thr, **dict(zip(keys, combo))}
                sm = {**base, lab: RT.score_matrix(fm, t2)[lab]}
                m = _macro_vec(truth, RT.labels_from_scores(sm))
                dist = sum(abs(v - THRESHOLDS[k]) / RT.SCALES[SCALE_OF[k]] for k, v in zip(keys, combo))
                cand = (-round(m, 9), round(dist, 9), combo)
                if best is None or cand < best:
                    best = cand
            assert best is not None
            new = dict(zip(keys, best[2]))
            if any(abs(new[k] - thr[k]) > 1e-12 for k in keys):
                thr.update(new)
                changed = True
        trace["sweeps"].append(round(_macro_vec(truth, RT.labels_from_scores(RT.score_matrix(fm, thr))), 4))
        if not changed:
            break
    return thr, trace


def bootstrap_macro(wins: Sequence[Mapping[str, Any]], pred_key: str, n: int = BOOT_N, seed: int = BOOT_SEED
                    ) -> dict[str, Any]:
    """Macro accuracy and R2 recall with a song-block bootstrap (resample songs) -> point, 95 % CI."""
    songs = sorted({w["song"] for w in wins})
    by = {s: [w for w in wins if w["song"] == s] for s in songs}
    truth = [w["truth"] for w in wins]
    pred = [w[pred_key] for w in wins]
    point, per = macro_accuracy(truth, pred)
    rng = np.random.default_rng(seed)
    mac, r2 = [], []
    for _ in range(n):
        pick = rng.integers(0, len(songs), len(songs))
        ws = [w for i in pick for w in by[songs[i]]]
        m, pc = macro_accuracy([w["truth"] for w in ws], [w[pred_key] for w in ws])
        if math.isfinite(m):
            mac.append(m)
        if "R2" in pc:
            r2.append(pc["R2"])

    def ci(xs: list[float]) -> list[float] | None:
        return [round(float(np.quantile(xs, 0.025)), 4), round(float(np.quantile(xs, 0.975)), 4)] if xs else None

    return {"macro": round(point, 4), "macro_ci": ci(mac), "per_class": {k: round(v, 4) for k, v in per.items()},
            "r2": round(per["R2"], 4) if "R2" in per else None, "r2_ci": ci(r2), "songs": len(songs),
            "windows": len(wins)}


def go_nogo(test: Mapping[str, Any]) -> dict[str, Any]:
    """The pre-registered rule on the ``sw|refit`` test values (point estimates; CIs reported)."""
    m, r2 = test.get("macro"), test.get("r2")
    if m is not None and m >= GO_MACRO and r2 is not None and r2 >= GO_R2:
        d, ko = "go", "GO: 공간 경로를 설계대로 쓴다"
    elif m is not None and m >= PARTIAL_MACRO:
        d, ko = "partial", "부분: 여유(margin)가 큰 창만 라우터 판정을 쓰고 나머지는 모든 경로"
    else:
        d, ko = "no-go", "NO-GO: 공간 경로 대신 N 경로와 기호 분할"
    return {"decision": d, "decision_ko": ko, "macro": m, "r2": r2,
            "rule": f"go: macro >= {GO_MACRO} and R2 >= {GO_R2}; partial: macro >= {PARTIAL_MACRO}; else no-go"}


# ------------------------------------------------------------------------------------------------ separation


def separate_guitar(items: Sequence[tuple[str, Path]], cfg: Any, *, force_rung: str, cache_dir: Path,
                    progress: Callable[[str], None] | None = None) -> tuple[dict[str, Path], dict[str, Any]]:
    """{pcm sha: stereo SW guitar stem} for (sha, mix wav) items, batched like the distractor suite (every item
    sees the chunk grid it would see alone); cached under ``cache_dir`` by (sha, sep params, rung)."""
    import time

    import soundfile as sf

    from bandscribe import audio as AU
    from bandscribe.datasets import store
    from bandscribe.eval.distractors import _separate, batch_layout, batch_total
    from bandscribe.sep.run import sep_params

    params = sep_params(cfg)
    out: dict[str, Path] = {}
    todo: list[tuple[str, Path, Path]] = []
    for sha, wav in items:
        key = hashlib.sha256(json.dumps({"pcm": sha, "sep": dict(params), "rung": force_rung, "v": CODE_VERSION},
                                        sort_keys=True).encode()).hexdigest()[:20]
        d = Path(cache_dir) / "sep" / key
        if (d / "done.json").is_file():
            out[sha] = d / "guitar.wav"
        elif sha not in {t[0] for t in todo}:
            todo.append((sha, Path(wav), d))
    info: dict[str, Any] = {"cached": len(out), "separated": len(todo), "gpu_s": 0.0, "audio_s": 0.0, "batches": 0}
    if not todo:
        return out, info
    chunk = int(force_rung.split("=", 1)[1]) if force_rung.startswith("chunk=") else int(params["chunk_ladder"][0])
    step = chunk // int(params["num_overlap"])
    lengths = [sf.info(str(w)).frames for _s, w, _d in todo]
    layout = batch_layout(lengths, chunk=chunk, step=step, max_samples=int(SEP_BATCH_S * T.SR))
    for b, lay in enumerate(layout):
        total = batch_total(lengths, lay, chunk=chunk, step=step)
        mix = np.zeros((total, 2), dtype=np.float32)
        for i, off in lay:
            y, _ = sf.read(str(todo[i][1]), dtype="float32", always_2d=True)
            mix[off:off + len(y)] = y[:, :2]
        bdir = Path(store.make_scratch_dir(Path(cache_dir) / "tmp", f"sepbatch{b}"))
        try:
            AU.write_f32(bdir / "mix.wav", mix, T.SR)
            del mix
            if progress:
                progress(f"분리 {b + 1}/{len(layout)}: 리믹스 {len(lay)}개, {total / T.SR / 60:.1f}분")
            t0 = time.perf_counter()
            res = _separate(bdir / "mix.wav", bdir / "sep", cfg, force_rung=force_rung)
            info["gpu_s"] += time.perf_counter() - t0
            info["audio_s"] += total / T.SR
            g, _ = sf.read(str(res.stems["guitar"]), dtype="float32", always_2d=True)
            for i, off in lay:
                sha, _w, d = todo[i]
                d.mkdir(parents=True, exist_ok=True)
                AU.write_f32(d / "guitar.wav", g[off:off + lengths[i]], T.SR)
                atomic.write_json(d / "done.json", {"sha": sha, "rung": str((getattr(res, "rung", {}) or {}).get(
                    "label") or force_rung)})
                out[sha] = d / "guitar.wav"
            del g
            info["batches"] += 1
        finally:
            store.remove_tree(bdir)
    return out, info


# ------------------------------------------------------------------------------------------------ run


def _pcm_sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


def _window_feats(x: np.ndarray, a: int, b: int) -> dict[str, Any]:
    seg = x[a:b]
    if seg.shape[0] < T.SR // 2 or float(np.mean(seg.astype(np.float64) ** 2)) < 1e-14:
        return {}
    return RT.features(seg, T.SR)


def _level_db(x: np.ndarray) -> float:
    return round(10 * math.log10(float(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-20), 2)


def run_e6(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    from bandscribe import audio as AU
    from bandscribe import datasets
    from bandscribe.amt import experiments as AE

    out_dir = Path(out_dir)
    work = out_dir / "work"
    root = Path(datasets.dataset_root("cambridge_mt"))
    tracks = list(datasets.iter_tracks("cambridge_mt"))
    if args.get("songs"):  # smoke tests of the runner only, never a registered run
        want = {x.strip() for x in str(args["songs"]).split(",")}
        tracks = [t for t in tracks if t.track_id in want]
    if args.get("limit"):  # smoke tests only
        tracks = tracks[: int(args["limit"])]
    if not tracks:
        raise AE.ExperimentDataMissing("E6: Cambridge-MT 멀티트랙이 없습니다 (bandscribe data list)")
    rel_dir = cache_root() / "relations"
    rels = {}
    for t in tracks:
        if progress:
            progress(f"트랙 관계: {t.track_id}")
        rels[t.track_id] = T.relations(t, root, rel_dir)
    can_a = {tid: any(ln["relation"] == "double" for ln in r["links"]) for tid, r in rels.items()}
    split = split_songs([t.track_id for t in tracks], can_a)
    atomic.write_json(out_dir / "relations.json", {
        tid: {"split": split[tid], "takes": [{"parts": tk["parts"], "line": tk["line"],
                                              "active_s": len(tk["active_s"])} for tk in r["takes"]],
              "links": r["links"]} for tid, r in rels.items()})
    # remixes: true-bus features now, the mix to disk (no audio kept in memory)
    remixes: list[dict[str, Any]] = []
    for k, t in enumerate(sorted(tracks, key=lambda t: t.track_id)):
        for sc in T.SCENARIOS:
            pl = T.plan(rels[t.track_id], sc, song_index=k)
            if pl is None:
                continue
            if progress:
                progress(f"리믹스: {t.track_id} {sc}")
            rm = T.build(t, root, rels[t.track_id], pl)
            d = work / f"{t.track_id}_{sc}"
            d.mkdir(parents=True, exist_ok=True)
            AU.write_f32(d / "mix.wav", rm.mix, T.SR)
            wins = []
            for w in rm.windows:
                a, b = int(round(w["start_s"] * T.SR)), int(round(w["end_s"] * T.SR))
                wins.append({**w, "true": _window_feats(rm.guitar, a, b), "true_db": _level_db(rm.guitar[a:b])})
            remixes.append({"song": t.track_id, "scenario": sc, "start_s": rm.start_s, "sources": rm.sources,
                            "mix": str(d / "mix.wav"), "sha": _pcm_sha(rm.mix), "windows": wins})
            del rm
    sep_rung = AE.sep_top_rung(cfg)
    stems, sep_info = separate_guitar([(r["sha"], Path(r["mix"])) for r in remixes], cfg, force_rung=sep_rung,
                                      cache_dir=cache_root(), progress=progress)
    import soundfile as sf

    flat: list[dict[str, Any]] = []
    for r in remixes:
        g, _ = sf.read(str(stems[r["sha"]]), dtype="float32", always_2d=True)
        for i, w in enumerate(r["windows"]):
            a, b = int(round(w["start_s"] * T.SR)), int(round(w["end_s"] * T.SR))
            w["sw"] = _window_feats(g, a, b)
            w["sw_db"] = _level_db(g[a:b])
            flat.append({"song": r["song"], "scenario": r["scenario"], "w": i, "split": split[r["song"]], **w})
    scored = [w for w in flat if w["truth"] in CLASSES]
    dev = [w for w in scored if w["split"] == "dev"]
    thr_refit, trace = refit(dev)
    for w in scored:
        w["pred_true_design"] = predict(w["true"], THRESHOLDS)
        w["pred_sw_design"] = predict(w["sw"], THRESHOLDS)
        w["pred_sw_refit"] = predict(w["sw"], thr_refit)
    rung = f"{sep_rung}|cpu"
    rows: list[dict] = []
    for w in scored:
        item = f"{w['song']}:{w['scenario']}:{w['w']}"
        common = {"dataset": "cambridge_mt", "item": item, "group": w["song"], "scenario": w["truth"],
                  "section": w["split"], "rung": rung, "line": "guitar"}
        for arm, key in zip(ARMS, ("pred_true_design", "pred_sw_design", "pred_sw_refit")):
            rows.append({**common, "system": arm, "metric": "router_acc", "value": float(w[key] == w["truth"])})
        for f in PRESERVED:
            a_, b_ = (w["true"] or {}).get(f), (w["sw"] or {}).get(f)
            if a_ is not None and b_ is not None and math.isfinite(a_) and math.isfinite(b_):
                rows.append({**common, "system": "sw", "metric": f"d_{f}", "value": round(abs(a_ - b_), 4)})
    report: dict[str, Any] = {}
    for arm, key in zip(ARMS, ("pred_true_design", "pred_sw_design", "pred_sw_refit")):
        for part in ("dev", "test"):
            ws = [w for w in scored if w["split"] == part]
            if ws:
                report[f"{arm}|{part}"] = bootstrap_macro(ws, key)
    confusion: dict[str, dict[str, dict[str, int]]] = {}
    for arm, key in zip(ARMS, ("pred_true_design", "pred_sw_design", "pred_sw_refit")):
        m: dict[str, dict[str, int]] = {}
        for w in scored:
            m.setdefault(w["truth"], {}).setdefault(w[key], 0)
            m[w["truth"]][w[key]] += 1
        confusion[arm] = m
    test_vals = report.get("sw|refit|test", {})
    decision = go_nogo(test_vals)
    atomic.write_json(out_dir / "e6_windows.json", flat)
    atomic.write_json(out_dir / "refit.json", {"thresholds": thr_refit, "design": dict(THRESHOLDS), "trace": trace,
                                               "dev_songs": sorted({w["song"] for w in dev}),
                                               "test_songs": sorted({w["song"] for w in scored
                                                                     if w["split"] == "test"})})
    atomic.write_json(out_dir / "confusion.json", confusion)
    excluded = {k: sum(1 for w in flat if w["truth"] == k) for k in ("mixed", "silent")}
    atomic.write_json(out_dir / "go_nogo.json", {**decision, "report": report, "excluded_windows": excluded,
                                                 "sep": sep_info, "remixes": len(remixes)})
    log.info("E6: %s (sw|refit test macro %s, R2 %s)", decision["decision"], decision["macro"], decision["r2"])
    return rows


def _run_e22(out_dir: Path, cfg: Any, args: dict, **kw: Any) -> list[dict]:
    from bandscribe.router.split_eval import run_e22  # imports this module

    return run_e22(out_dir, cfg, args, **kw)


def _run_e22b(out_dir: Path, cfg: Any, args: dict, **kw: Any) -> list[dict]:
    from bandscribe.router.split_eval import run_e22b

    return run_e22b(out_dir, cfg, args, **kw)


EXPERIMENTS = {"E6": run_e6, "E22": _run_e22, "E22b": _run_e22b}
