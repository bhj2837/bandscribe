"""Transcription-scored experiments (M1_M2_SPEC 9.2 item 10, §10; ids = the ``## <id>`` headings of docs/decisions.md).

``EXPERIMENTS[id](out_dir, cfg, args) -> list[dict]`` returns metrics.csv rows (``bandscribe.eval.runs.METRIC_COLUMNS``);
``bandscribe eval exp <id>`` (EVAL) runs them, computes the pre-registered decision and writes the run dir. Row
conventions (docs/decisions.md): ``system`` = arm, ``item`` / ``group`` = item and bootstrap block, ``scenario`` =
subset tag, ``section`` = dev/test split, ``line`` = guitar/bass, ``rung`` = the GPU rung(s) that produced the
prediction (``<sep rung>|<MuScriptor rung>``), ``onnx-cpu`` for Basic Pitch.

Every experiment pins the rung of each GPU backend it measures (``force_rung`` = top rung) and never reuses
``bandscribe run`` stage caches: separation runs into the run dir; MuScriptor goes through the eval transcription cache
(``data/eval/reftx_cache``, rung in the key) except E24-amt, which uses a fresh cache so resources are measured.

Data (pre-registered): Tier B = ``cambridge_mt`` (+ ``medleydb`` when present) remixed with
``bandscribe.eval.remix.tierb_mix(master_lufs=-8)``; references = ``bandscribe.amt.reftx`` of the true solo tracks,
pooled per kind and merged like ``eval.suites`` (same pitch within 50 ms = one note, references and estimates
alike; E1-E3 ran before this 2026-10-05 fix with unmerged references). EGDB for
E23 (dev = official train clips, test = official test clips; amp renders, DI excluded) and gate-m2 (test clips,
``scenario`` = ``distortion`` for Marshall/Mesa/Plexi renders, ``clean`` otherwise). GuitarSet for ``latency``
(estimate on players 00-02, report on 03-05) and as the Tier C GT set of E24-amt.

E3 note: experiments run outside jobs, so there is no Beat This! grid; the instrumentation estimator gets
pseudo-bars of 2 s (``PSEUDO_BAR_S``) and stem energy computed here at 10 Hz. E3 (decided 2026-10-04) estimated
presence from the full-mix pass (the M2 estimator, ``mix_as_evidence``); E3b measures what ``bandscribe run`` does since
the 2026-10-04 speed/presence change: the sampled piano+other presence pass (``bandscribe.amt.presence`` windows on the
pseudo-bars, no sections) and the ``guitar+present+strings`` policy, with a strings family row. E2 transcribes every candidate view
with the guitar-only mask so that only the input changes.

Missing data raises ``ExperimentDataMissing`` with the Korean steps to get it (nothing is downloaded here).
The backend calls are module-level seams (``_tracks``, ``_ms``, ``_bp_sweep``, ``_separate``, ``_reftx``) so the
experiment logic is unit-tested with fakes.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic, vram
from bandscribe.amt import cache as amt_cache
from bandscribe.amt import instruments as I
from bandscribe.amt import latency as L
from bandscribe.amt import stage as S

log = logging.getLogger(__name__)

METRIC_F1 = "onset_f1_50"
TIER_B = ("cambridge_mt", "medleydb")
E1_ARMS: tuple[tuple[str, Any], ...] = (("off", "off"), ("-14", -14.0), ("-16", -16.0))
E2_VIEWS = ("guitar_mono", "mix_mono", "nonvox_mono", "guitar_other_mono")
E3_POLICIES = ("guitar+present", "guitar_only", "all")
E3B_POLICIES = ("guitar+present", "guitar_only", "all", "guitar+present+strings")
E3_FAMILY_ROWS = ("keys", "synth", "strings")
E3_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
E3_DEFAULT_THRESHOLD = 0.5
E23_FRAMES = (3, 4, 5, 6, 7, 12)
E23_ONSET = (0.3, 0.4, 0.5, 0.6)
E23_FRAME = (0.2, 0.3, 0.4)
E23_BATCH = 16
BP_RUNG = S.BP_RUNG
PSEUDO_BAR_S = 2.0
SPLIT_SEED = "20260930"


class ExperimentDataMissing(RuntimeError):
    """The pre-registered data is not on this PC (Korean message with the steps to get it)."""


# ------------------------------------------------------------------------------------------- seams


def _available(name: str) -> bool:
    from bandscribe import datasets

    try:
        return bool(datasets.is_available(name))
    except Exception:  # unknown name / registry problem = not available
        return False


def _tracks(name: str, split: str | None = None) -> list[Any]:
    from bandscribe import datasets

    return list(datasets.iter_tracks(name, split=split))


def _root(name: str) -> Path:
    from bandscribe import datasets

    return Path(datasets.dataset_root(name))


def _ms(views: list[dict[str, Any]], params: Mapping[str, Any], cfg: Any, *, work_dir: Path,
        cache: amt_cache.AmtCache, force_rung: str) -> tuple[dict[str, dict], dict[str, Any]]:
    return S.transcribe_views_ms(views, params, cfg, work_dir=work_dir, cache=cache, job=None, force_rung=force_rung)


def _separate(mix_wav: Path, out_dir: Path, cfg: Any, *, force_rung: str, input_lufs: Any = None,
              dtype: str | None = None) -> Any:
    from bandscribe.sep.run import separate

    return separate(mix_wav, out_dir, cfg, force_rung=force_rung, input_lufs=input_lufs, dtype=dtype)


def _reftx(track: Any, out_dir: Path, cfg: Any, *, force_rung: str) -> dict[str, dict[str, Any]]:
    from bandscribe.amt.reftx import reference_transcribe

    return reference_transcribe(track, out_dir, cfg, force_rung=force_rung)


def _bp_sweep(views: list[dict[str, Any]], settings: list[dict[str, Any]], cfg: Any, *, work_dir: Path,
              model_output_cache: Path) -> dict[str, list[dict]]:
    """{view id: [NoteSet per setting]} from one ``amt_basicpitch`` sweep worker (one model load)."""
    from bandscribe.gpu import run_worker

    raw = Path(work_dir) / "raw"
    req = {"format": "bandscribe.worker/1", "job": None, "out_dir": str(Path(work_dir).parent), "device": "cpu",
           "mode": "sweep", "threads": int(S.amt_value(cfg, "amt.bp_threads")),
           "model_output_cache": str(model_output_cache),
           "views": [{"id": v["id"], "wav": str(v["wav"]), "out": str(raw / f"{v['id']}__basicpitch.json"),
                      "view_sha256": v["pcm_sha256"]} for v in views],
           "settings": settings, "allow_default_min_len": True, "determinism": True}
    res = run_worker("amt_basicpitch", req, Path(work_dir), timeout_s=float(vram.cfg_get(cfg, "gpu.worker_timeout_s",
                     7200.0)), python=S.bp310_python(), use_lock=False)
    if not res.ok:
        raise RuntimeError(f"Basic Pitch sweep 워커 실패: {res.error}. 자세한 내용: {res.stderr_path}")
    out: dict[str, list[dict]] = {}
    for row in res.output.get("views") or []:
        docs = []
        for p in row["outs"]:
            docs.append(atomic.read_json(Path(p)))
            Path(p).unlink(missing_ok=True)  # 72 files per track: score, then drop
        out[row["id"]] = docs
    return out


def _bp_transcribe(views: list[dict[str, Any]], params: dict[str, Any], cfg: Any, *, work_dir: Path,
                   model_output_cache: Path) -> dict[str, dict]:
    return {k: v[0] for k, v in _bp_sweep(views, [params], cfg, work_dir=work_dir,
                                          model_output_cache=model_output_cache).items()}


# ----------------------------------------------------------------------------------------- helpers


def _slug(s: str) -> str:
    from bandscribe.eval.runs import slug

    return slug(s)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def note_arrays(notes: Iterable[Any], *, classes: Sequence[str] | None = None,
                latency_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(intervals (n, 2) s, pitches (n,) MIDI) of dict/model notes, optionally class-filtered / latency-corrected."""
    keep = None if classes is None else set(classes)
    iv, pp = [], []
    for n in notes:
        if keep is not None and _get(n, "instrument") not in keep:
            continue
        on = max(0.0, float(_get(n, "onset_s")) - latency_s)
        off = _get(n, "offset_s")
        off = on if off is None else max(on, float(off) - latency_s)
        iv.append((on, off))
        pp.append(float(_get(n, "pitch")))
    return (np.asarray(iv, dtype=float).reshape(-1, 2), np.asarray(pp, dtype=float))


def _concat(*arrs: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not arrs:
        return np.zeros((0, 2)), np.zeros(0)
    return (np.concatenate([a[0] for a in arrs]).reshape(-1, 2), np.concatenate([a[1] for a in arrs]))


def _score(ref: tuple[np.ndarray, np.ndarray], est: tuple[np.ndarray, np.ndarray]) -> Any:
    from bandscribe.eval.metrics import note_prf

    return note_prf(ref, est, onset_tol=0.05, pitch_tol_cents=50.0)


# Copies of one note (same pitch, onsets within this tolerance of the first) count once, on both sides.
# = the onset tolerance of the metric: two such reference notes can never both be matched by one transcription
# of the mixed guitar stem, so unmerged they cap recall. Measured 2026-10-05 on the 22,382 reftx guitar notes of
# the 6 Tier B songs: ~11 % are redundant cross-line copies within 50 ms (Mistrusted 23 %; double-tracked parts
# are offset by human timing, so the 1-tick tolerance of eval.suites, 10.5 ms, merges only ~5 %), and another
# 11 % are exact within-line duplicates (one onset labelled with two guitar classes).
MERGE_TOL_S = 0.05


def merged(arrs: tuple[np.ndarray, np.ndarray], tol: float = MERGE_TOL_S) -> tuple[np.ndarray, np.ndarray]:
    """``eval.suites._merged_arrays`` (same pitch, onset within ``tol`` of the cluster's first onset -> one note
    with the latest offset) on (intervals, pitches) arrays. Idempotent."""
    from bandscribe.eval.suites import _merged_arrays

    iv, pp = arrs
    return _merged_arrays(((float(a), float(b), int(round(float(p)))) for (a, b), p in zip(iv, pp)), tol=tol)


def _score_merged(ref: tuple[np.ndarray, np.ndarray], est: tuple[np.ndarray, np.ndarray]) -> Any:
    """Note P/R/F1 of the Tier B / scene experiments (E1-E3b, E24-sep): reference (pooled over its guitar or
    bass lines) and estimate both merged with :func:`merged` first (decisions.md amendment 2026-10-05)."""
    return _score(merged(ref), merged(est))


def row(*, system: str, dataset: str, item: str, metric: str, value: Any, group: str = "", line: str = "",
        scenario: str = "", section: str = "", rung: str = "", tp: Any = None, fp: Any = None, fn: Any = None,
        n: Any = None, note: str = "") -> dict[str, Any]:
    return {"system": system, "dataset": dataset, "item": item, "group": group or item, "scenario": scenario,
            "section": section, "line": line, "rung": rung, "metric": metric, "value": value, "tp": tp, "fp": fp,
            "fn": fn, "n": n, "note": note}


def f1_row(prf: Any, **kw: Any) -> dict[str, Any]:
    return row(metric=METRIC_F1, value=float(prf.f1), tp=int(prf.tp), fp=int(prf.fp), fn=int(prf.fn),
               n=int(prf.tp + prf.fn), **kw)


def hash_split(item: str, test_ratio: float = 0.3) -> str:
    """Seeded, order-independent dev/test split by item id (used where no official split exists)."""
    h = int(hashlib.sha256(f"{SPLIT_SEED}:{item}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_ratio else "dev"


def ms_rung(params: Mapping[str, Any]) -> str:
    return S.ms_ladder(params)[0][0]


def sep_top_rung(cfg: Any) -> str:
    ladder = vram.cfg_get(cfg, "sep.chunk_ladder", [588800, 352256, 262144])
    return f"chunk={int(list(ladder)[0])}"


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1).astype(np.float32) if x.ndim == 2 and x.shape[1] > 1 else x.reshape(-1)


def _read(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, int(sr)


def _write_stereo(path: Path, x: np.ndarray, sr: int) -> None:
    from bandscribe.audio import write_f32

    write_f32(path, np.ascontiguousarray(np.asarray(x, dtype="<f4")), int(sr))


def view_entry(vid: str, mono: np.ndarray, sr: int, path: Path, instruments: list[str] | None) -> dict[str, Any]:
    from bandscribe.amt.reftx import write_mono_f32

    sha = write_mono_f32(path, mono, sr)
    return {"id": vid, "wav": path, "pcm_sha256": sha, "instruments": instruments}


def _data_names(args: Mapping[str, Any]) -> list[str]:
    """Dataset names from ``bandscribe eval exp --data a,b`` (``args["data"]``) or ``args["datasets"]`` (API)."""
    raw = args.get("datasets") or args.get("data") or []
    if isinstance(raw, str):
        raw = raw.split(",")
    return [str(n).strip() for n in raw if str(n).strip()]


def _tier_b(args: Mapping[str, Any], *, need: str) -> list[tuple[str, Any]]:
    names = _data_names(args) or [d for d in TIER_B if _available(d)]
    missing = [n for n in names if not _available(n)]
    if missing:
        raise ExperimentDataMissing(f"{need}: 데이터셋이 없습니다: {', '.join(missing)} (bandscribe data list 로 확인하세요).")
    if not names:
        raise ExperimentDataMissing(
            f"{need}: Tier B 데이터가 없습니다. Cambridge-MT 록 멀티트랙 2곡 이상을 브라우저에서 받아(약관 동의) "
            "`bandscribe data import-local cambridge_mt <폴더> --song <이름>` 으로 등록하고 lines.yaml 을 채우세요.")
    out = []
    for name in names:
        for t in _tracks(name):
            out.append((name, t))
    limit = args.get("max_tracks")
    return out[: int(limit)] if limit else out


def _refs_by_kind(track: Any, refs: Mapping[str, dict], latency_s: float = 0.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reference arrays per kind, the lines of a kind pooled and **merged** (:func:`merged`: a note that two
    guitar lines both play counts once; decisions.md amendment 2026-10-05). ``latency_s`` must be the same
    constant the estimates get: reftx is itself a raw MuScriptor transcription with the same lag, so correcting
    only the estimates would shift every estimate against its reference by the constant and cost F1 at the
    50 ms tolerance in every arm."""
    kinds: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"guitar": [], "bass": []}
    for line in _get(track, "lines") or []:
        lid, kind = str(_get(line, "id")), str(_get(line, "kind"))
        if kind in kinds and lid in refs:
            kinds[kind].append(note_arrays(refs[lid].get("notes") or [], latency_s=latency_s))
    return {k: merged(_concat(*v)) for k, v in kinds.items() if v}


def _item(dataset: str, track: Any) -> str:
    return f"{dataset}:{_get(track, 'track_id')}"


def _group(track: Any) -> str:
    return str(_get(track, "group") or _get(track, "track_id"))


def _stem_views(stems: Mapping[str, Path], wanted: Sequence[str], work: Path,
                masks: Mapping[str, list[str] | None]) -> list[dict[str, Any]]:
    """Mono views computed from SW stems (and the mix for mix_mono) for the experiment arms."""
    cache: dict[str, tuple[np.ndarray, int]] = {}

    def get(name: str) -> np.ndarray:
        if name not in cache:
            cache[name] = _read(stems[name])
        return cache[name][0]

    views = []
    for vid in wanted:
        if vid == "guitar_mono":
            x = get("guitar")
        elif vid == "bass_mono":
            x = get("bass")
        elif vid == "mix_mono":
            x = get("mix")
        elif vid == "nonvox_mono":
            x = get("mix") - get("vocals") - get("drums") - get("bass")
        elif vid == "guitar_other_mono":
            x = get("guitar") + get("other")
        elif vid == "piano_other_mono":
            x = get("piano") + get("other")
        else:
            raise ValueError(f"unknown view {vid}")
        sr = next(iter(cache.values()))[1]
        views.append(view_entry(vid, _mono(x), sr, work / f"{vid}.wav", masks.get(vid)))
    return views


def _sep_resources(sep: Any) -> dict[str, float | None]:
    doc = getattr(sep, "sep_json", {}) or {}
    attempts = ((doc.get("vram") or {}).get("attempts") or [])
    ok = [a for a in attempts if a.get("status") == "ok"]
    return {"reserved_peak_mb": ok[-1].get("max_reserved_mb") if ok else None, "rtf": doc.get("rtf")}


def _ms_resources(info: Mapping[str, Any]) -> dict[str, float | None]:
    out = info.get("worker_output") or {}
    attempts = ((out.get("vram") or {}).get("attempts") or [])
    ok = [a for a in attempts if a.get("status") == "ok"]
    return {"reserved_peak_mb": ok[-1].get("max_reserved_mb") if ok else None,
            "rtf": (out.get("timing") or {}).get("rtf")}


def _eval_cache() -> amt_cache.AmtCache:
    return amt_cache.AmtCache(amt_cache.eval_cache_root())


# ----------------------------------------------------------------------------------------------- E1


def _prep_tierb(dataset: str, track: Any, work: Path, cfg: Any, ms_top: str) -> tuple[Path, dict]:
    from bandscribe.eval.remix import SR, tierb_mix

    mix_wav = work / "mix.wav"
    if not mix_wav.is_file():
        _write_stereo(mix_wav, tierb_mix(track, "original", master_lufs=-8.0), SR)
    refs = _refs_by_kind(track, _reftx(track, work / "reftx", cfg, force_rung=ms_top),
                         latency_s=float(S.ms_params(cfg)["latency_s"]))
    return mix_wav, refs


def run_e1(out_dir: Path, cfg: Any, args: dict) -> list[dict]:
    """SW input loudness (``sep.input_lufs``): off vs -14 vs -16 LUFS, scored on stem transcriptions vs reftx."""
    params = S.ms_params(cfg)
    top, sep_rung = ms_rung(params), sep_top_rung(cfg)
    rows: list[dict] = []
    masks = {"guitar_mono": I.mask_guitar_only(), "bass_mono": I.mask_bass()}
    for dataset, track in _tier_b(args, need="E1"):
        item = _item(dataset, track)
        work = Path(out_dir) / "work" / _slug(item)
        mix_wav, refs = _prep_tierb(dataset, track, work, cfg, top)
        for label, lufs in E1_ARMS:
            sep = _separate(mix_wav, work / f"sep_{_slug(label)}", cfg, force_rung=sep_rung, input_lufs=lufs)
            views = _stem_views(sep.stems, ["guitar_mono", "bass_mono"], work / f"views_{_slug(label)}", masks)
            raws, info = _ms(views, params, cfg, work_dir=work / f"ms_{_slug(label)}" / "_worker", cache=_eval_cache(),
                             force_rung=top)
            rung = f"{sep.rung.get('label')}|{info['rung_label']}"
            for kind, vid, classes in (("guitar", "guitar_mono", I.GUITAR_CLASSES), ("bass", "bass_mono", I.BASS_CLASSES)):
                if kind not in refs:
                    continue
                est = note_arrays(raws[vid].get("notes") or [], classes=classes, latency_s=params["latency_s"])
                ref = refs[kind]
                rows.append(f1_row(_score_merged(ref, est), system=label, dataset=dataset, item=item, group=_group(track),
                                   line=kind, rung=rung))
    return rows


# ----------------------------------------------------------------------------------------------- E2


def run_e2(out_dir: Path, cfg: Any, args: dict) -> list[dict]:
    """MuScriptor guitar input view: guitar stem / mix / nonvox / guitar+other (guitar-only mask for all)."""
    params = S.ms_params(cfg)
    top, sep_rung = ms_rung(params), sep_top_rung(cfg)
    rows: list[dict] = []
    masks = {v: I.mask_guitar_only() for v in E2_VIEWS}
    for dataset, track in _tier_b(args, need="E2"):
        item = _item(dataset, track)
        work = Path(out_dir) / "work" / _slug(item)
        mix_wav, refs = _prep_tierb(dataset, track, work, cfg, top)
        if "guitar" not in refs:
            continue
        sep = _separate(mix_wav, work / "sep", cfg, force_rung=sep_rung, input_lufs="off")
        stems = dict(sep.stems)
        stems["mix"] = mix_wav
        views = _stem_views(stems, list(E2_VIEWS), work / "views", masks)
        raws, info = _ms(views, params, cfg, work_dir=work / "ms" / "_worker", cache=_eval_cache(), force_rung=top)
        rung = f"{sep.rung.get('label')}|{info['rung_label']}"
        for vid in E2_VIEWS:
            est = note_arrays(raws[vid].get("notes") or [], classes=I.GUITAR_CLASSES, latency_s=params["latency_s"])
            rows.append(f1_row(_score_merged(refs["guitar"], est), system=vid, dataset=dataset, item=item,
                               group=_group(track), line="guitar", rung=rung))
    return rows


# ----------------------------------------------------------------------------------------------- E3


def energy_db(x: np.ndarray, sr: int, fps: float = 10.0) -> np.ndarray:
    """RMS in dB at ``fps`` of a (n, ch) or (n,) signal (same definition as the ``stems`` stage's energy.npz)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    hop = int(round(sr / fps))
    n = -(-len(x) // hop)
    pad = np.zeros((n * hop - len(x), x.shape[1]))
    y = np.concatenate([x, pad]).reshape(n, hop, x.shape[1])
    counts = np.full(n, hop * x.shape[1], dtype=np.float64)
    counts[-1] = (len(x) - (n - 1) * hop) * x.shape[1]
    return (10.0 * np.log10((y ** 2).sum(axis=(1, 2)) / counts + 1e-20)).astype(np.float32)


def pseudo_bars(duration_s: float, bar_s: float = PSEUDO_BAR_S) -> list[tuple[int, float, float]]:
    n = max(1, int(np.ceil(duration_s / bar_s)))
    return [(i + 1, i * bar_s, min(duration_s, (i + 1) * bar_s)) for i in range(n)]


def _e3_items(args: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Tier B tracks with keys/synth, plus synthetic full-band scenes (GT notes) unless disabled."""
    items: list[dict[str, Any]] = []
    try:
        tb = _tier_b(args, need="E3")
    except ExperimentDataMissing:
        tb = []
    for dataset, track in tb:
        fams = set(_get(track, "families_present") or [])
        if fams & {"keys", "synth"}:
            items.append({"kind": "tierb", "dataset": dataset, "track": track, "families": sorted(fams)})
    if args.get("synthetic", True):
        from bandscribe.eval import remix

        for scenario in args.get("scenes") or ("A", "P70", "B", "twin"):
            for seed in args.get("seeds") or (0, 1):
                items.append({"kind": "synthetic", "dataset": "synthetic", "scenario": scenario, "seed": int(seed),
                              "factory": remix.make_scene})
    if not items:
        raise ExperimentDataMissing("E3: 건반·신스가 있는 Tier B 곡이 없고 합성 장면도 꺼져 있습니다.")
    return items


def _e3_presence(stems: Mapping[str, Path], energy: Mapping[str, np.ndarray], bars: list, dur: float,
                 params: Mapping[str, Any], cfg: Any, work: Path, top: str) -> dict[str, Any]:
    """E3b: the production presence source (``bandscribe run`` quality settings) for one experiment item."""
    from bandscribe.amt import presence as P

    settings = S.presence_settings(cfg, "quality")
    sel = P.select_windows(bars=bars, sections=None, energy=energy, fps=10.0, song_s=dur,
                           **{k: settings[k] for k in P.DEFAULTS})
    segs = P.spans(sel)
    out: dict[str, Any] = {"mode": "sampled", "skipped": None if segs else (sel.get("reason") or "no_window"),
                           "notes": [], "spans": segs or None,
                           "window_sections": [w["section"] for w in sel["windows"]] or None,
                           "summary": {"windows": segs, "transcribed_s": sel.get("transcribed_s")}}
    if segs:
        mode = str(S.amt_value(cfg, "amt.piano_other_instruments"))
        po = _stem_views(stems, ["piano_other_mono"], work / "views", {"piano_other_mono": I.mask_piano_other(mode)})
        raws, _info = _ms([dict(po[0], id=P.PRESENCE_VIEW, segments=segs)], params, cfg,
                          work_dir=work / "ms_presence" / "_worker", cache=_eval_cache(), force_rung=top)
        out["notes"] = raws[P.PRESENCE_VIEW].get("notes") or []
    return out


def run_e3(out_dir: Path, cfg: Any, args: dict) -> list[dict]:
    """Guitar-view hard mask policy (+ instrumentation threshold) on keys-containing items; b13 family table.

    ``args["presence"]``: ``mix`` (E3: presence from the full-mix pass, the M2 estimator) or ``sampled`` (E3b: the
    production presence pass); ``args["policies"]`` overrides the arm list (E3b adds ``guitar+present+strings``).
    """
    from bandscribe.amt import instrumentation as INS

    presence_src = str(args.get("presence") or "mix")
    if presence_src not in ("mix", "sampled"):
        raise ValueError(f"E3 presence source must be mix | sampled, not {presence_src!r}")
    policies = tuple(args.get("policies") or E3_POLICIES)
    params = S.ms_params(cfg)
    top, sep_rung = ms_rung(params), sep_top_rung(cfg)
    thresholds = tuple(float(t) for t in (args.get("thresholds") or E3_THRESHOLDS))
    base_prm = {k: S.amt_value(cfg, f"instr.{k}") for k in INS.DEFAULT_PARAMS}
    rows: list[dict] = []
    for it in _e3_items(args):
        if it["kind"] == "tierb":
            dataset, track = it["dataset"], it["track"]
            item, group = _item(dataset, track), _group(track)
            work = Path(out_dir) / "work" / _slug(item)
            mix_wav, refs = _prep_tierb(dataset, track, work, cfg, top)
            if "guitar" not in refs:
                continue
            ref = refs["guitar"]
            gt_families = it["families"]
        else:
            scene = it["factory"](it["scenario"], it["seed"], full_band=True)
            item = f"synthetic:{scene.scenario}:s{scene.seed}"
            group, dataset = item, "synthetic"
            work = Path(out_dir) / "work" / _slug(item)
            mix_wav = work / "mix.wav"
            _write_stereo(mix_wav, np.asarray(scene.mix, dtype=np.float32).T, scene.sr)
            from bandscribe.eval.metrics import as_arrays

            glines = [ln.id for ln in scene.lines if ln.kind == "guitar"]
            ref = as_arrays(scene.gt, lines=glines)
            gt_families = sorted({t.kind for t in scene.takes if t.kind in I.FAMILIES} | {"guitar"})
        section = hash_split(item)
        sep = _separate(mix_wav, work / "sep", cfg, force_rung=sep_rung, input_lufs="off")
        stems = dict(sep.stems)
        stems["mix"] = mix_wav
        energy: dict[str, np.ndarray] = {}
        dur = 0.0
        for name in ("mix", "bass", "drums", "other", "vocals", "guitar", "piano"):
            x, sr = _read(stems[name])
            energy[name] = energy_db(x, sr)
            dur = len(x) / sr
        bars = pseudo_bars(dur)
        if presence_src == "mix":
            pass1 = _stem_views(stems, ["mix_mono", "guitar_mono"], work / "views", {"mix_mono": None})
            mix_raw, _info = _ms([pass1[0]], params, cfg, work_dir=work / "ms_mix" / "_worker", cache=_eval_cache(),
                                 force_rung=top)
            est_src: dict[str, Any] = {"mix_notes": mix_raw["mix_mono"].get("notes") or [], "mix_as_evidence": True}
            gview = pass1[1]
        else:
            gview = _stem_views(stems, ["guitar_mono"], work / "views", {})[0]
            est_src = {"mix_notes": None, "presence": _e3_presence(stems, energy, bars, dur, params, cfg, work, top)}
        rung_sep = sep.rung.get("label")
        for policy in policies:
            thr_list = thresholds if policy == "guitar+present" else (E3_DEFAULT_THRESHOLD,)
            for thr in thr_list:
                prm = dict(base_prm, threshold=thr)
                inst = INS.estimate(profile="quality", bars=bars, energy=energy, sections=[], hint="auto", params=prm,
                                    guitar_mask=policy, **est_src)
                mask = inst["masks"]["guitar_pass"]
                raws, info = _ms([dict(gview, instruments=mask)], params, cfg, work_dir=work / "ms_gtr" / "_worker",
                                 cache=_eval_cache(), force_rung=top)
                est = note_arrays(raws["guitar_mono"].get("notes") or [], classes=I.GUITAR_CLASSES,
                                  latency_s=params["latency_s"])
                arm = policy if thr == E3_DEFAULT_THRESHOLD else f"{policy}|t={thr:g}"
                rung = f"{rung_sep}|{info['rung_label']}"
                scen = "keys" if dataset != "synthetic" else f"synthetic_{it.get('scenario')}"
                rows.append(f1_row(_score_merged(ref, est), system=arm, dataset=dataset, item=item, group=group,
                                   line="guitar", rung=rung, scenario=scen, section=section,
                                   note=f"mask={'+'.join(mask) if mask else 'all'}"))
                rows.append(row(system=arm, dataset=dataset, item=item, group=group, line="guitar", rung=rung,
                                scenario=scen, section=section, metric="guitar_class_omissions",
                                value=INS.guitar_class_omissions(inst)))
                if policy == "guitar+present":
                    est_f = INS.estimated_families(inst)
                    for fam in E3_FAMILY_ROWS:
                        rows.append(row(system=arm, dataset=dataset, item=item, group=group, scenario=scen,
                                        section=section, rung=rung, metric=f"{fam}_presence_correct",
                                        value=int((fam in est_f) == (fam in gt_families)),
                                        note=f"est={','.join(est_f)};gt={','.join(gt_families)}"))
    return rows


# ---------------------------------------------------------------------------------------------- E23


def e23_settings() -> list[dict[str, Any]]:
    """The 72 pre-registered settings, in a fixed order."""
    out = []
    for f, o, fr in itertools.product(E23_FRAMES, E23_ONSET, E23_FRAME):
        out.append({"onset_threshold": o, "frame_threshold": fr, "min_note_frames": f})
    return out


def e23_arm(s: Mapping[str, Any]) -> str:
    return f"f{int(s['min_note_frames'])}_o{float(s['onset_threshold']):g}_fr{float(s['frame_threshold']):g}"


def _egdb_tracks(args: Mapping[str, Any], split: str | None, need: str) -> list[Any]:
    if not _available("egdb"):
        raise ExperimentDataMissing(
            f"{need}: EGDB 가 없거나 일부만 받아졌습니다. `bandscribe data fetch egdb --yes` 로 받으세요(공개 Google "
            "Drive 폴더, 이어받기 지원, 약 9 GB; 라이선스 미기재: 개인 연구용).")
    tracks = [t for t in _tracks("egdb", split=split) if (_get(t, "meta") or {}).get("timbre") != "DI"]
    limit = args.get("max_tracks_per_split")
    return tracks[: int(limit)] if limit else tracks


def _track_view(dataset: str, track: Any, work: Path) -> dict[str, Any]:
    """Mono view of a Tier C track's first guitar part (read + re-written as float32 for a stable PCM hash)."""
    parts = [p for p in _get(track, "parts") or [] if _get(p, "kind") in ("guitar", "di")]
    src = _get(parts[0], "audio") if parts else _get(track, "mix")
    x, sr = _read(_root(dataset) / src)
    item = _item(dataset, track)
    return view_entry(_slug(item), _mono(x), sr, work / f"{_slug(item)}.wav", I.mask_guitar_only())


def _gt(track: Any) -> tuple[np.ndarray, np.ndarray]:
    return _concat(*[note_arrays(ns.notes if hasattr(ns, "notes") else ns.get("notes") or [])
                     for ns in (_get(track, "notes") or {}).values()])


def run_e23(out_dir: Path, cfg: Any, args: dict) -> list[dict]:
    """Basic Pitch min note length x onset/frame thresholds on EGDB: tune on dev (train clips), confirm on test."""
    settings = e23_settings()
    rows: list[dict] = []
    for official, section in (("train", "dev"), ("test", "test")):
        tracks = _egdb_tracks(args, official, "E23")
        work = Path(out_dir) / "work" / f"e23_{section}"
        for k in range(0, len(tracks), E23_BATCH):
            batch = tracks[k:k + E23_BATCH]
            views = [_track_view("egdb", t, work / "views") for t in batch]
            res = _bp_sweep(views, settings, cfg, work_dir=work / f"batch{k // E23_BATCH:04d}" / "_worker",
                            model_output_cache=amt_cache.eval_cache_root().parent / "bp_model_output")
            for t, v in zip(batch, views):
                ref = _gt(t)
                for s, doc in zip(settings, res[v["id"]]):
                    rows.append(f1_row(_score(ref, note_arrays(doc.get("notes") or [])), system=e23_arm(s),
                                       dataset="egdb", item=_item("egdb", t), group=_group(t), line="guitar",
                                       rung=BP_RUNG, section=section,
                                       scenario=str((_get(t, "meta") or {}).get("drive_guess") or "")))
    return rows


# ------------------------------------------------------------------------------------------ gate-m2


def _bp_tuned(cfg: Any, args: Mapping[str, Any]) -> dict[str, Any]:
    """E23-tuned Basic Pitch params: ``args["bp_params"]`` or the config (defaults.toml after E23)."""
    from bandscribe.workers.amt_basicpitch import normalize_params

    if args.get("bp_params"):
        return normalize_params(dict(args["bp_params"]), allow_default_min_len=True)
    return dict(S.bp_params(cfg)["params"])


def run_gate_m2(out_dir: Path, cfg: Any, args: dict) -> list[dict]:
    """MuScriptor (medium, top rung, latency-corrected) vs tuned Basic Pitch on EGDB test amp renders."""
    params = S.ms_params(cfg)
    top = ms_rung(params)
    tracks = _egdb_tracks(args, "test", "gate-m2")
    work = Path(out_dir) / "work" / "gate"
    views = [_track_view("egdb", t, work / "views") for t in tracks]
    raws, info = _ms(views, params, cfg, work_dir=work / "ms" / "_worker", cache=_eval_cache(), force_rung=top)
    bp = _bp_transcribe(views, _bp_tuned(cfg, args), cfg, work_dir=work / "bp" / "_worker",
                        model_output_cache=amt_cache.eval_cache_root().parent / "bp_model_output")
    rows: list[dict] = []
    for t, v in zip(tracks, views):
        ref = _gt(t)
        scen = "distortion" if (_get(t, "meta") or {}).get("drive_guess") == "distorted" else "clean"
        base = {"dataset": "egdb", "item": _item("egdb", t), "group": _group(t), "line": "guitar", "scenario": scen,
                "section": "test"}
        ms_est = note_arrays(raws[v["id"]].get("notes") or [], classes=I.GUITAR_CLASSES, latency_s=params["latency_s"])
        bp_est = note_arrays(bp[v["id"]].get("notes") or [])
        rows.append(f1_row(_score(ref, ms_est), system="muscriptor", rung=info["rung_label"], **base))
        rows.append(f1_row(_score(ref, bp_est), system="basicpitch", rung=BP_RUNG, **base))
        # diagnostic (registered 2026-10-03 before the first run, not part of the gate): Basic Pitch is never
        # latency-corrected, so a lag of its own would show up here rather than hide inside the F1 gap
        for system, est, rung in (("muscriptor", ms_est, info["rung_label"]), ("basicpitch", bp_est, BP_RUNG)):
            lat = L.estimate_latency(est, ref)
            rows.append(row(system=system, rung=rung, metric="median_residual_ms", value=lat["median_ms"],
                            n=lat["n"], **base))
    return rows


# ------------------------------------------------------------------------------------------ latency


def _guitarset(args: Mapping[str, Any], players: Sequence[str]) -> list[Any]:
    if not _available("guitarset"):
        raise ExperimentDataMissing("latency: GuitarSet 이 없습니다. `bandscribe data fetch guitarset` 로 받으세요.")
    out = [t for t in _tracks("guitarset") if str((_get(t, "meta") or {}).get("player", _get(t, "group"))) in players]
    limit = args.get("max_tracks_per_split")
    return out[: int(limit)] if limit else out


def run_latency(out_dir: Path, cfg: Any, args: dict) -> list[dict]:
    """Estimate the MuScriptor lag on dev (players 00-02; EGDB dev only with ``egdb_dev=true``); report
    |residual| on test (03-05)."""
    params = S.ms_params(cfg)
    top = ms_rung(params)
    splits = {"dev": _guitarset(args, L.DEV_PLAYERS), "test": _guitarset(args, L.TEST_PLAYERS)}
    dev_egdb = []
    # Off by default: the 2026-10-03 registration amendment excludes EGDB dev (cost); only an explicit
    # ``--arg egdb_dev=true`` adds it (review 2026-10-05: the default used to contradict the registration).
    if args.get("egdb_dev", False) and _available("egdb"):
        dev_egdb = _egdb_tracks(args, "train", "latency")
    work = Path(out_dir) / "work" / "latency"
    per_track: dict[str, list[tuple[str, Any, dict, tuple, Path]]] = {"dev": [], "test": []}
    rung = top
    for section, entries in (("dev", [("guitarset", t) for t in splits["dev"]] + [("egdb", t) for t in dev_egdb]),
                             ("test", [("guitarset", t) for t in splits["test"]])):
        views = [_track_view(ds, t, work / "views") for ds, t in entries]
        if not views:
            continue
        raws, info = _ms(views, params, cfg, work_dir=work / f"ms_{section}" / "_worker", cache=_eval_cache(),
                         force_rung=top)
        for (ds, t), v in zip(entries, views):
            per_track[section].append((ds, t, raws[v["id"]], _gt(t), Path(v["wav"])))
        rung = info["rung_label"]
    if not per_track["dev"]:
        raise ExperimentDataMissing("latency: dev 트랙(GuitarSet 연주자 00-02)이 없습니다.")

    dev_res = []
    for _ds, _t, raw, ref, _w in per_track["dev"]:
        est = note_arrays(raw.get("notes") or [], classes=I.GUITAR_CLASSES)
        dev_res.append(L.match_residuals(est, ref))
    pooled = np.concatenate(dev_res) if dev_res else np.zeros(0)
    if len(pooled) == 0:
        raise RuntimeError("latency: dev 세트에서 짝지은 onset 이 없습니다.")
    const_s = float(np.median(pooled))
    rows = [row(system="muscriptor", dataset="guitarset", item="dev-pooled", group="dev-pooled", section="dev",
                rung=rung, metric="latency_ms", value=round(const_s * 1000.0, 3), n=int(len(pooled)),
                note="median residual (transcribed - GT) on the dev split = amt.latency_s")]
    for section in ("dev", "test"):
        for ds, t, raw, ref, wav in per_track[section]:
            est = note_arrays(raw.get("notes") or [], classes=I.GUITAR_CLASSES)
            # decisions.md registers a *track* block bootstrap for this entry (the loader's group is the player,
            # which would leave 3 blocks per split)
            item = _item(ds, t)
            group = item
            raw_stat = L.estimate_latency(est, ref)
            corr = L.median_abs_residual_ms(est, ref, const_s)
            flux = L.estimate_latency((est[0][:, 0], None), L.spectral_flux_onsets(wav))
            common = {"system": "muscriptor", "dataset": ds, "item": item, "group": group, "section": section,
                      "rung": rung, "line": "guitar"}
            rows.append(row(metric="median_residual_ms", value=raw_stat["median_ms"], n=raw_stat["n"], **common))
            rows.append(row(metric="median_abs_residual_ms", value=corr["median_abs_ms"], n=corr["n"], **common))
            rows.append(row(metric="flux_median_residual_ms", value=flux["median_ms"], n=flux["n"], **common,
                            note="cross-check: transcribed onsets vs spectral-flux onsets"))
    atomic.write_json(Path(out_dir) / "latency_constant.json", {
        "format": "bandscribe.latency_constant/1", "latency_s": round(const_s, 6), "n_dev_matches": int(len(pooled)),
        "dev_players": list(L.DEV_PLAYERS), "test_players": list(L.TEST_PLAYERS), "rung": rung,
        "integration_request": f"defaults.toml [amt] latency_s = {round(const_s, 4)}"})
    return rows


# ------------------------------------------------------------------------------------------ E24-sep


def run_e24_sep(out_dir: Path, cfg: Any, args: dict) -> list[dict]:
    """SW dtype upstream vs fp16 weights: stem transcription vs reftx (non-inferiority) + reserved / rtf rows."""
    params = S.ms_params(cfg)
    top, sep_rung = ms_rung(params), sep_top_rung(cfg)
    masks = {"guitar_mono": I.mask_guitar_only(), "bass_mono": I.mask_bass()}
    rows: list[dict] = []
    for dataset, track in _tier_b(args, need="E24-sep"):
        item, group = _item(dataset, track), _group(track)
        work = Path(out_dir) / "work" / _slug(item)
        mix_wav, refs = _prep_tierb(dataset, track, work, cfg, top)
        for dtype in ("upstream", "fp16"):
            sep = _separate(mix_wav, work / f"sep_{dtype}", cfg, force_rung=sep_rung, input_lufs="off", dtype=dtype)
            views = _stem_views(sep.stems, ["guitar_mono", "bass_mono"], work / f"views_{dtype}", masks)
            raws, info = _ms(views, params, cfg, work_dir=work / f"ms_{dtype}" / "_worker", cache=_eval_cache(),
                             force_rung=top)
            rung = f"{sep.rung.get('label')}|{info['rung_label']}"
            for kind, vid, classes in (("guitar", "guitar_mono", I.GUITAR_CLASSES), ("bass", "bass_mono", I.BASS_CLASSES)):
                if kind in refs:
                    est = note_arrays(raws[vid].get("notes") or [], classes=classes, latency_s=params["latency_s"])
                    rows.append(f1_row(_score_merged(refs[kind], est), system=dtype, dataset=dataset, item=item, group=group,
                                       line=kind, rung=rung))
            for metric, value in _sep_resources(sep).items():
                if value is not None:
                    rows.append(row(system=dtype, dataset=dataset, item=item, group=group, rung=sep.rung.get("label"),
                                    metric=metric, value=float(value)))
    return rows


# ------------------------------------------------------------------------------------------ E24-amt


def run_e24_amt(out_dir: Path, cfg: Any, args: dict) -> list[dict]:
    """MuScriptor dtype upstream vs fp16 weights (conditioners fp32) vs GT (GuitarSet) [+ Tier B reftx]."""
    params0 = S.ms_params(cfg)
    top = ms_rung(params0)
    limit = int(args.get("max_tracks", 30))
    players = L.DEV_PLAYERS + L.TEST_PLAYERS
    every = _guitarset({"max_tracks_per_split": None}, players)
    # seeded, order-independent subset with the same number of tracks per player (the loader order is by
    # player, so a plain prefix would be player 00 only = one bootstrap block)
    per_player = max(1, -(-limit // len(players)))
    by_player: dict[str, list[Any]] = {}
    for t in every:
        by_player.setdefault(str((_get(t, "meta") or {}).get("player", _get(t, "group"))), []).append(t)
    picked = []
    for p in sorted(by_player):
        ranked = sorted(by_player[p], key=lambda t: hashlib.sha256(
            f"{SPLIT_SEED}:{_item('guitarset', t)}".encode()).hexdigest())
        picked.extend(ranked[:per_player])
    picked = sorted(picked, key=lambda t: str(_get(t, "track_id")))[:limit]
    tracks = [("guitarset", t) for t in picked]
    work = Path(out_dir) / "work" / "e24amt"
    views = [_track_view(ds, t, work / "views") for ds, t in tracks]
    rows: list[dict] = []
    for dtype in ("upstream", "fp16"):
        params = dict(params0, dtype=dtype)
        fresh = amt_cache.AmtCache(work / f"cache_{dtype}")  # fresh: resources must be measured, not cached
        raws, info = _ms(views, params, cfg, work_dir=work / f"ms_{dtype}" / "_worker", cache=fresh, force_rung=top)
        for (ds, t), v in zip(tracks, views):
            est = note_arrays(raws[v["id"]].get("notes") or [], classes=I.GUITAR_CLASSES, latency_s=params["latency_s"])
            rows.append(f1_row(_score(_gt(t), est), system=dtype, dataset=ds, item=_item(ds, t), group=_group(t),
                               line="guitar", rung=info["rung_label"]))
        for metric, value in _ms_resources(info).items():
            if value is not None:
                rows.append(row(system=dtype, dataset="guitarset", item="all", group="all", rung=info["rung_label"],
                                metric=metric, value=float(value)))
    return rows


# E3c (docs/decisions.md): windows of the distractor songs that the distractor run did NOT use (held-out), chosen
# before any E3c result by fixed rules: no overlap with the run's window, guitar active >= 50 %, a keys/organ/synth/
# strings category active >= 10 %, the strongest such 75 s window per song (5 of the 10 songs have one).
E3C_WINDOWS = ("jameselder_englishactor@35-110", "jetb_tothewolves@5-80", "moosmusic_bigdummyshake@0-75",
               "secretariat_overthetop@145-220", "zeno_signs@0-75")
E3C_METRICS = ("note_f1_onset50", "mask_non_guitar", "non_guitar_dropped")


def _num(v: Any) -> Any:
    if v in ("", None):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() and "." not in str(v) else f


def run_e3c(out_dir: Path, cfg: Any, args: dict) -> list[dict]:
    """E3c: production mask vs guitar-only mask on the full mix of the held-out windows (distractor machinery,
    conditions base + full, no null control). Returns the per-window rows of the full condition."""
    import csv

    from bandscribe.eval import distractors as D

    sub = Path(out_dir) / "distractors"
    sub.mkdir(parents=True, exist_ok=True)
    D.run(sub, cfg, {"windows": list(args.get("windows") or E3C_WINDOWS), "conditions": "full", "null": False,
                     "gpu_budget_min": float(args.get("gpu_budget_min", 30.0)),
                     **({"allow_over_budget": True} if args.get("allow_over_budget") else {})},
          progress=lambda s: log.info("E3c: %s", s))
    rows: list[dict] = []
    with open(sub / "metrics.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("item") == "*" or r.get("scenario") != "full" or r.get("metric") not in E3C_METRICS:
                continue
            rows.append({k: _num(v) for k, v in r.items() if k != "suite"})
    return rows


EXPERIMENTS: dict[str, Callable[[Path, Any, dict], list[dict]]] = {
    "E1": run_e1,
    "E2": run_e2,
    "E3": run_e3,
    "E3b": lambda out_dir, cfg, args: run_e3(out_dir, cfg, {**args, "presence": "sampled",
                                                             "policies": args.get("policies") or E3B_POLICIES}),
    "E3c": run_e3c,
    "E23": run_e23,
    "E24-sep": run_e24_sep,
    "E24-amt": run_e24_amt,
    "latency": run_latency,
    "gate-m2": run_gate_m2,
}
