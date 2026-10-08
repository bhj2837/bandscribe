"""M5 bass experiments (ids = the ``## <id>`` headings of docs/decisions.md): ``E13``.

E13 compares the clean-up arms of ``bandscribe.bass.clean`` (``raw`` = MuScriptor's bass pass as M2 wrote it,
``anchor``, ``anchor_merge``, ``anchor_merge_kick``, ``two_voter``, ``and``) on three data sets:

- ``idmt_bass_st``: all 17 IDMT-SMT-Bass-Single-Track lines (DI electric bass, GT notes, beats, string/fret,
  playing styles); input = the recording itself.
- ``filobass``: ``FILO_SONGS`` FiloBass songs (jazz upright, the set's own separated bass stem) chosen by a seeded
  hash of the song name, window ``FILO_WINDOW_S``; GT = ``midi_fully_aligned`` notes starting in the window.
- ``cambridge_mt`` (Tier B, the six E1 songs): the remix (``tierb_mix``, master -8 LUFS) separated by SW; input = the
  bass stem, drums stem for the kick arm; reference = MuScriptor's transcription of the true bass track (reftx:
  compares arms, never an absolute score).

MuScriptor goes through the eval transcription cache (rung pinned to the top), the F0 trackers through
``data/eval/f0_cache``; separation runs into the run dir. Rows (metrics.csv): ``system`` = arm, ``item`` =
``<dataset>:<track>``, ``group`` = track / song, ``scenario`` = dataset, ``line`` = bass, ``rung`` =
``<MuScriptor rung>|<F0 rung>`` (the separation rung of Tier B items is in ``note``). Metrics: ``onset_f1_50``
(primary; reference and estimate merged as in E1-E3b), ``chroma_f1_50``, ``octave_error_rate`` (chroma F1 - pitch
F1, points), ``tick_f1`` (GT grid; IDMT and FiloBass).

Written next to the rows (M5 acceptance, descriptive): ``tracker_accuracy.json`` (per tracker and GT band:
raw pitch / chroma accuracy and voicing on IDMT and FiloBass), ``techniques.json`` (slide, vibrato and dead-note
recall / precision on IDMT with the current ``[bass]`` defaults), ``fretting.json`` (IDMT string and string/fret
agreement on GT notes), ``policy_check.json`` (two_voter vs and: recall and precision, paired block bootstrap),
``steps.json`` (each step against the arm before it, with the registered rule: ``step_decisions``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from bandscribe import atomic
from bandscribe.bass import clean as CL
from bandscribe.bass import f0 as F0
from bandscribe.bass import stage as BS

log = logging.getLogger(__name__)

FILO_SONGS = 12
FILO_WINDOW_S = (30.0, 90.0)
TIER_B_SONGS = ("butterflyeffect", "jetb", "mistrusted", "younggriffo", "secretariat", "woodfire")
SPLIT_SEED = "20260930"
METRIC = "onset_f1_50"


@dataclass
class Item:
    dataset: str
    item: str
    group: str
    view: dict[str, Any]
    gt: list[dict[str, Any]] | None = None  # true notes (IDMT, FiloBass)
    ref: tuple[np.ndarray, np.ndarray] | None = None  # reftx arrays (Tier B)
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    drums: Path | None = None
    styles: list[dict[str, Any]] | None = None
    string_fret: list[tuple[int, int]] | None = None
    tuning: list[int] | None = None
    note: str = ""


# ------------------------------------------------------------------------------------------------ data


def _hash_order(name: str, ids: Sequence[str]) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"{SPLIT_SEED}:{name}:{i}".encode()).hexdigest())


def _gt_rows(track: Any) -> list[dict[str, Any]]:
    (line, ns), = track.notes.items()
    return [{"onset_s": float(n.onset_s), "offset_s": float(n.offset_s), "pitch": int(n.pitch)} for n in ns.notes]


def _audio(dataset: str, track: Any) -> Path:
    from bandscribe import datasets

    return Path(datasets.dataset_root(dataset)) / track.parts[0].audio


def idmt_items(work: Path) -> list[Item]:
    from bandscribe import datasets
    from bandscribe.amt import experiments as AE

    out = []
    for tid in sorted(datasets.track_ids("idmt_bass_st")):
        tr = datasets.load_track("idmt_bass_st", tid)
        x, sr = AE._read(_audio("idmt_bass_st", tr))
        view = AE.view_entry("bass_mono", AE._mono(x), sr, work / "idmt" / f"{tid}.wav", None)
        sf = (tr.string_fret or {}).get("bass")
        out.append(Item("idmt_bass_st", f"idmt_bass_st:{tid}", tid, view, gt=_gt_rows(tr), beats=list(tr.beats_s or []),
                        downbeats=list(tr.downbeats_s or []), styles=list(tr.meta.get("styles") or []),
                        string_fret=[(int(s), int(f)) for s, f in sf] if sf else None, tuning=list(tr.tuning or [])))
    return out


def filobass_items(work: Path, n_songs: int = FILO_SONGS) -> list[Item]:
    from bandscribe import datasets
    from bandscribe.amt import experiments as AE

    a, b = FILO_WINDOW_S
    out = []
    for song in _hash_order("filobass", datasets.track_ids("filobass"))[:n_songs]:
        tr = datasets.load_track("filobass", song)
        x, sr = AE._read(_audio("filobass", tr))
        seg = AE._mono(x)[int(a * sr):int(b * sr)]
        view = AE.view_entry("bass_mono", seg, sr, work / "filobass" / f"{AE._slug(song)}.wav", None)
        gt = [{"onset_s": n["onset_s"] - a, "offset_s": min(n["offset_s"], b) - a, "pitch": n["pitch"]}
              for n in _gt_rows(tr) if a <= n["onset_s"] < b]
        beats = [t - a for t in tr.beats_s or [] if a - 2.0 <= t <= b + 2.0]
        downs = [t - a for t in tr.downbeats_s or [] if a - 2.0 <= t <= b + 2.0]
        out.append(Item("filobass", f"filobass:{song}", song, view, gt=gt, beats=beats, downbeats=downs))
    return out


def tierb_items(work: Path, cfg: Any, args: Mapping[str, Any]) -> list[Item]:
    from bandscribe.amt import experiments as AE

    sep_rung = AE.sep_top_rung(cfg)
    ms_top = AE.ms_rung(AE.S.ms_params(cfg))
    out = []
    for dataset, track in AE._tier_b({"data": "cambridge_mt"}, need="E13"):
        tid = str(track.track_id)
        if not any(tid.lower().startswith(s) for s in TIER_B_SONGS):
            continue
        item = AE._item(dataset, track)
        w = work / "tierb" / AE._slug(item)
        mix_wav, refs = AE._prep_tierb(dataset, track, w, cfg, ms_top)
        if "bass" not in refs:
            log.warning("E13: %s has no bass line in its reference; skipped", item)
            continue
        sep = AE._separate(mix_wav, w / "sep", cfg, force_rung=sep_rung, input_lufs="off")
        x, sr = AE._read(Path(sep.stems["bass"]))
        view = AE.view_entry("bass_mono", AE._mono(x), sr, w / "bass_mono.wav", None)
        out.append(Item(dataset, item, AE._group(track), view, ref=refs["bass"], drums=Path(sep.stems["drums"]),
                        note=f"sep {sep.rung.get('label')}"))
    return out


# ------------------------------------------------------------------------------------------- scoring


def tempo_map(beats: Sequence[float], downbeats: Sequence[float], tpb: int = 48) -> Any:
    """A TempoMap from beat and downbeat times (bars counted from the first downbeat; beats before it form a
    pickup bar)."""
    from bandscribe.schema.grid import TempoMap

    bt = sorted(float(t) for t in beats)
    dn = sorted(float(t) for t in downbeats)
    if len(bt) < 2 or not dn:
        return None
    idx = [int(np.argmin(np.abs(np.asarray(bt) - d))) for d in dn]
    counts = np.diff(idx) if len(idx) > 1 else np.array([4])
    bpb = int(np.median(counts)) if len(counts) else 4
    bars, pos = [], []
    first = idx[0]
    for i in range(len(bt)):
        if i < first:
            bars.append(0)
            pos.append(max(1, bpb - (first - i) + 1))
        else:
            k = int(np.searchsorted(idx, i, side="right")) - 1
            bars.append(k + 1)
            pos.append(i - idx[k] + 1)
    bpb_of = {b: max(p for bb, p in zip(bars, pos) if bb == b) for b in set(bars)}
    return TempoMap(bt, bars, pos, bpb_of, tpb)


def gt_ticks(gt: Sequence[Mapping[str, Any]], tm: Any) -> Any:
    bars, ticks = tm.time_to_bar_tick(np.array([float(n["onset_s"]) for n in gt], dtype=float))
    notes = [SimpleNamespace(bar=int(b), tick=float(t), pitch=int(n["pitch"]), line="bass")
             for n, b, t in zip(gt, np.atleast_1d(bars), np.atleast_1d(ticks))]
    return SimpleNamespace(tpb=tm.tpb, notes=notes)


def score_rows(it: Item, notes: Sequence[Mapping[str, Any]], *, system: str, rung: str) -> list[dict[str, Any]]:
    from bandscribe.amt import experiments as AE
    from bandscribe.eval import metrics

    ref = it.ref if it.ref is not None else AE.note_arrays(it.gt or [])
    est = AE.note_arrays(notes)
    rm, em = AE.merged(ref), AE.merged(est)
    base = {"system": system, "dataset": it.dataset, "item": it.item, "group": it.group, "scenario": it.dataset,
            "line": "bass", "rung": rung, "note": it.note}
    pitch = metrics.note_prf(rm, em, onset_tol=0.05)
    chroma = metrics.note_prf(rm, em, onset_tol=0.05, chroma=True)
    rows = [AE.f1_row(pitch, **base),
            AE.row(metric="chroma_f1_50", value=float(chroma.f1), tp=chroma.tp, fp=chroma.fp, fn=chroma.fn,
                   n=chroma.tp + chroma.fn, **base),
            AE.row(metric="octave_error_rate", value=100.0 * (float(chroma.f1) - float(pitch.f1)), **base)]
    if it.gt is not None and it.beats and it.downbeats:
        tm = tempo_map(it.beats, it.downbeats)
        if tm is not None and it.gt:
            r = metrics.tick_prf(gt_ticks(it.gt, tm), em, tm)
            rows.append(AE.row(metric="tick_f1", value=float(r.f1), tp=r.tp, fp=r.fp, fn=r.fn, n=r.tp + r.fn, **base))
    return rows


# ----------------------------------------------------------------------------------- side reports


def _match(gt: Sequence[Mapping[str, Any]], est: Sequence[Mapping[str, Any]], tol: float = 0.05,
           same_pitch: bool = True) -> dict[int, int]:
    """Greedy one-to-one onset matching (nearest first): {gt index: est index}."""
    pairs = []
    for i, g in enumerate(gt):
        for j, e in enumerate(est):
            d = abs(float(g["onset_s"]) - float(e["onset_s"]))
            if d <= tol and (not same_pitch or int(g["pitch"]) == int(e["pitch"])):
                pairs.append((d, i, j))
    used_g: set[int] = set()
    used_e: set[int] = set()
    out = {}
    for _d, i, j in sorted(pairs):
        if i not in used_g and j not in used_e:
            out[i] = j
            used_g.add(i)
            used_e.add(j)
    return out


def technique_counts(it: Item, notes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """IDMT styles vs our technique marks: per technique the GT notes with the style (``gt``), those matched by a
    marked estimate (``hit``, onset within 50 ms, any pitch for dead notes), and the marked estimates (``marked``)
    with how many of them sit on a GT note of that style (``marked_hit``). A slide counts on the GT ``SL`` note or
    the note before it (IDMT labels one end)."""
    styles = it.styles or []
    out: dict[str, dict[str, int]] = {}
    for tech, style in (("dead", "DN"), ("vibrato", "VI"), ("slide", "SL")):
        gt_idx = [i for i, s in enumerate(styles) if s.get("expression") == style]
        m = _match(it.gt or [], notes, same_pitch=tech != "dead")
        inv = {j: i for i, j in m.items()}
        marked = [j for j, n in enumerate(notes) if (n.get("tech") or {}).get(tech)]
        if tech == "slide":
            near = set(gt_idx) | {i - 1 for i in gt_idx if i > 0}
            hit = sum(1 for i in gt_idx if any(m.get(k) in marked for k in (i, i - 1)))
            marked_hit = sum(1 for j in marked if inv.get(j) in near)
        else:
            hit = sum(1 for i in gt_idx if m.get(i) in marked)
            marked_hit = sum(1 for j in marked if inv.get(j) in set(gt_idx))
        out[tech] = {"gt": len(gt_idx), "hit": hit, "marked": len(marked), "marked_hit": marked_hit}
    return out


def fretting_counts(it: Item) -> dict[str, int]:
    """String and string/fret agreement of our fretter on the GT notes (oracle pitches), harmonics left out."""
    from bandscribe.tab import fretting as F

    if not it.string_fret or not it.gt or not it.tuning:
        return {}
    inst = F.Instrument(tuple(reversed(it.tuning)), max_fret=24)
    ev = F.group_events(it.gt, key="onset_s", tol=0.05)
    res = F.assign(ev, inst)
    pred = {a.note_id: (a.string, a.fret) for a in res if a.string is not None}
    keep = [i for i, s in enumerate(it.styles or [{}] * len(it.gt)) if s.get("expression") != "HA"]
    return {"notes": len(keep), "string": sum(1 for i in keep if i in pred and pred[i][0] == it.string_fret[i][0]),
            "string_fret": sum(1 for i in keep if pred.get(i) == it.string_fret[i])}


def tracker_frames(it: Item, track: F0.F0Track) -> dict[str, Any]:
    skip = {i for i, s in enumerate(it.styles or []) if s.get("expression") in ("DN", "HA")}
    ref = F0.frame_reference(it.gt or [], track.n, track.hop_s, skip=skip)
    return F0.accuracy(track, ref)


def _pool_tracker(per_item: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for acc in per_item:
        for name, bands in acc.items():
            for band, v in bands.items():
                d = out.setdefault(name, {}).setdefault(band, {"frames": 0})
                k = int(v.get("frames") or 0)
                d["frames"] += k
                for key in ("rpa", "rca", "vr", "rpa_any", "octave"):
                    if key in v:
                        d[key] = d.get(key, 0.0) + float(v[key]) * k
    for name, bands in out.items():
        for band, d in bands.items():
            k = d["frames"]
            for key in ("rpa", "rca", "vr", "rpa_any", "octave"):
                if key in d:
                    d[key] = round(100.0 * d[key] / k, 2) if k else None
    return out


def policy_check(rows: Sequence[Mapping[str, Any]], n: int, seed: int) -> dict[str, Any]:
    """two_voter vs and on the primary metric's counts: recall and precision deltas with paired CIs."""
    from bandscribe.eval import experiments as EX
    from bandscribe.eval import stats

    def items(arm: str) -> list[Any]:
        return EX.items_from_rows([r for r in rows if r["metric"] == METRIC and r["system"] == arm])

    a, b = items("and"), items("two_voter")
    if not a or not b:
        return {"available": False}
    out: dict[str, Any] = {"available": True}
    for name, stat in (("recall", stats.recall_points), ("precision", stats.precision_points)):
        d, lo, hi = stats.paired_bootstrap(a, b, stat, n=n, seed=seed)
        out[name] = {"delta_two_voter_minus_and": round(d, 3), "ci": [round(lo, 3), round(hi, 3)]}
    return out


# E13's step rules (docs/decisions.md): (step, candidate arm, baseline arm, rule, items' dataset filter or None)
STEP_RULES: tuple[tuple[str, str, str, str, str | None], ...] = (
    ("anchor", "anchor", "raw", "superiority", None),
    ("merge", "anchor_merge", "anchor", "noninferiority", None),
    ("kick", "anchor_merge_kick", "anchor_merge", "superiority", "cambridge_mt"),
    ("gap_fill", "two_voter", "anchor_merge", "superiority", None),
)
MDE = 1.0
NI_MARGIN = -1.0
SUBSET_FLOOR = -2.0


def step_decisions(rows: Sequence[Mapping[str, Any]], n: int, seed: int) -> dict[str, Any]:
    """Each step of E13 on its own (paired block bootstrap of the primary metric, per-dataset subsets): DESIGN 8.1
    superiority (MDE 1 point) for anchor, kick and the gap fill; non-inferiority (CI lower bound > -1 point, no
    dataset subset with its CI upper bound < -2) for the merges, which are there for the written slide."""
    from bandscribe.eval import experiments as EX
    from bandscribe.eval import stats

    out: dict[str, Any] = {}
    for step, cand, base, rule, only in STEP_RULES:
        sel = [r for r in rows if r["metric"] == METRIC and (only is None or r["dataset"] == only)]
        a = [r for r in sel if r["system"] == base]
        b = [r for r in sel if r["system"] == cand]
        if not a or not b:
            out[step] = {"decision": "hold", "why": "no rows"}
            continue
        res = EX._paired(a, b, n, seed, "scenario")
        lo, hi = res["ci"]
        if rule == "superiority":
            d = stats.decide(res["delta"], lo, hi, MDE, {k: tuple(v) for k, v in res["subsets"].items()})
        else:
            ok = stats.noninferior(lo, NI_MARGIN) and all(
                not np.isfinite(v[2]) or v[2] >= SUBSET_FLOOR for v in res["subsets"].values())
            d = "adopt" if ok else "reject"
        out[step] = {"decision": d, "rule": rule, "candidate": cand, "baseline": base, "items": only or "all",
                     "delta": round(res["delta"], 3), "ci": [round(lo, 3), round(hi, 3)],
                     "subsets": {k: [round(x, 3) for x in v] for k, v in res["subsets"].items()}}
    return out


# ------------------------------------------------------------------------------------------------- E13


def _rung(info_ms: Mapping[str, Any], info_f0: Mapping[str, Any]) -> str:
    return f"{info_ms.get('rung_label')}|{info_f0.get('rung_label') or 'crepe-full'}"


def run_e13(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    from bandscribe.amt import experiments as AE
    from bandscribe.amt import instruments as I

    out_dir = Path(out_dir)
    work = out_dir / "work"
    want = set(AE._data_names(args) or ["idmt_bass_st", "filobass", "cambridge_mt"])
    items: list[Item] = []
    if "idmt_bass_st" in want:
        items += idmt_items(work)
    if "filobass" in want:
        items += filobass_items(work, int(args.get("filo_songs") or FILO_SONGS))
    if "cambridge_mt" in want:
        items += tierb_items(work, cfg, args)
    if args.get("limit"):  # smoke tests of the runner only, never a registered run
        items = items[: int(args["limit"])]
    if not items:
        raise AE.ExperimentDataMissing("E13: 평가할 베이스 데이터가 없습니다 (bandscribe data list)")
    views = [{**it.view, "id": f"v{k:03d}", "instruments": I.mask_bass()} for k, it in enumerate(items)]
    ms_params = AE.S.ms_params(cfg)
    ms_top = AE.ms_rung(ms_params)
    raws, info_ms = AE._ms(views, ms_params, cfg, work_dir=work / "ms" / "_worker", cache=AE._eval_cache(),
                           force_rung=ms_top)
    tracks, info_f0 = BS.run_trackers(views, BS.f0_params(cfg), cfg, work_dir=work / "f0",
                                      cache_root=BS.eval_cache_root(), force_rung="crepe-full")
    rung = _rung(info_ms, info_f0)
    defaults = BS.bass_params(cfg)
    rows: list[dict] = []
    tech: dict[str, dict[str, int]] = {}
    fret: dict[str, int] = {}
    trk: dict[str, list[dict[str, Any]]] = {}
    per_item: dict[str, Any] = {}
    for it, v in zip(items, views):
        if progress:
            progress(it.item)
        raw = AE.S.apply_latency(raws[v["id"]], ms_params["latency_s"])["notes"]
        mono, sr = BS._read_mono(Path(v["wav"]))
        attack = CL.Attack(mono, sr)
        kicks = None
        if it.drums is not None:
            dm, dsr = BS._read_mono(it.drums)
            kicks = CL.kick_onsets(dm, dsr)
        track = tracks[v["id"]]
        reports = {}
        for arm in CL.ARMS:
            notes, rep = CL.clean(raw, track, CL.arm_params(arm), f0_params=defaults["f0"], attack=attack, kicks=kicks)
            rows += score_rows(it, notes, system=arm, rung=rung)
            reports[arm] = {k: rep.get(k) for k in ("out", "vote", "moved", "dupes_dropped", "kick_dropped",
                                                    "fragments_merged", "slides", "gap_filled", "and_dropped")}
        per_item[it.item] = reports
        if it.gt is not None:
            trk.setdefault(it.dataset, []).append(tracker_frames(it, track))
        if it.styles:
            notes, _rep = CL.clean(raw, track, defaults["clean"], f0_params=defaults["f0"], attack=attack, kicks=kicks)
            for k, d in technique_counts(it, notes).items():
                acc = tech.setdefault(k, {})
                for kk, vv in d.items():
                    acc[kk] = acc.get(kk, 0) + vv
            for k, vv in fretting_counts(it).items():
                fret[k] = fret.get(k, 0) + vv
    n_boot = int(BS._cfg_get(cfg, "eval.bootstrap_iterations", 10000))
    seed = int(BS._cfg_get(cfg, "eval.seed", 20260930))
    atomic.write_json(out_dir / "tracker_accuracy.json", {d: _pool_tracker(v) for d, v in trk.items()})
    atomic.write_json(out_dir / "techniques.json", {"config": defaults["clean"], "idmt": tech})
    atomic.write_json(out_dir / "fretting.json", {
        "idmt": fret, "string_pct": round(100.0 * fret["string"] / fret["notes"], 2) if fret.get("notes") else None,
        "string_fret_pct": round(100.0 * fret["string_fret"] / fret["notes"], 2) if fret.get("notes") else None})
    atomic.write_json(out_dir / "policy_check.json", policy_check(rows, n_boot, seed))
    atomic.write_json(out_dir / "steps.json", step_decisions(rows, n_boot, seed))
    atomic.write_json(out_dir / "e13_items.json", {"items": [it.item for it in items], "rung": rung,
                                                   "filo_window_s": list(FILO_WINDOW_S), "reports": per_item,
                                                   "f0": {"cached": info_f0.get("cached"),
                                                          "computed": info_f0.get("computed")},
                                                   "ms": {"cached": info_ms.get("cached_views"),
                                                          "fresh": info_ms.get("fresh_views")}})
    return rows


EXPERIMENTS: dict[str, Callable[[Path, Any, dict], list[dict]]] = {"E13": run_e13}
