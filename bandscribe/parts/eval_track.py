"""E27 data (M7b, docs/decisions.md): whole-song Tier B remixes for identity tracking.

For every Cambridge-MT song of E6's run with two guitar Lines, the scenarios ``P`` (E6's plan), ``A1`` and ``B2``
(``tierb.VARIANTS``) are rendered over the **whole song** (start 0), then, per remix:

- the SW guitar stem (E6's rung, eval cache; GPU), the beats of the mix (Beat This!, the pipeline's ``beats`` stage
  on a scratch store; GPU, seconds), the grid, the sections and the repeat map (the pipeline's CPU stages);
- router windows: 4 bars inside a section (the pipeline's window size), labelled from the SW stem's window features
  with the pipeline's thresholds (as E25);
- MuScriptor on the stem's mono down-mix and on each included take's own mono (E25's views, now whole-song; GPU,
  eval cache; a take's view is the same pcm for every scenario, so it is transcribed once);
- truth as E25: Lines = takes joined by double links, active with ``MIN_REF_NOTES`` reference notes, a system note's
  truth = the active Lines whose reference it matches.

``prepare_song`` writes one JSON per song (``cache_dir() / <song>.json``) so the scoring is CPU only and a song can be
filled at a time (``bandscribe eval exp E27`` with ``songs=...``; the cache fill runs per song between games).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bandscribe import atomic, paths

log = logging.getLogger(__name__)

SCENARIOS = ("P", "A1", "B2")
WINDOW_BARS = 4
FORMAT = "bandscribe.e27_song/1"


def cache_dir() -> Path:
    return Path(paths.EVAL) / "parts_track"


def _ctx(stage: str, out_dir: Path, mix_dir: Path, params: dict, cfg: Any, deps: dict | None = None) -> Any:
    out_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(song_key="e27", stage=stage, key=stage * 8, out_dir=out_dir, dep_dirs=deps or {},
                           params=params, config=cfg, store=SimpleNamespace(input_dir=lambda key: mix_dir))


def grid_sections(mix_dir: Path, work: Path, cfg: Any) -> tuple[dict, list, dict]:
    """(grid, sections, repeat map) of ``mix_dir/mix_44k_f32.wav`` through the pipeline's stage functions (beats on
    the GPU worker, cached in ``work``)."""
    from bandscribe.analysis import stage as A

    bd, gd, sd = work / "beats", work / "grid", work / "sections"
    if not (bd / "beats.json").is_file():
        A.run_beats(_ctx("beats", bd, mix_dir, A.beats_params(cfg), cfg))
    if not (gd / "grid.json").is_file():
        A.run_grid(_ctx("grid", gd, mix_dir, A.grid_params(cfg), cfg, {"beats": bd}))
    if not (sd / "repeat_map.json").is_file():
        A.run_sections(_ctx("sections", sd, mix_dir, A.sections_params(cfg), cfg, {"grid": gd}))
    return (atomic.read_json(gd / "grid.json"), atomic.read_json(sd / "sections.json").get("sections") or [],
            atomic.read_json(sd / "repeat_map.json"))


def bar_windows(grid: Mapping[str, Any], sections: list[Mapping[str, Any]], dur: float) -> list[dict[str, Any]]:
    """4-bar windows that never cross a section boundary (the router's windows): [{start_s, end_s, section,
    start_bar}]."""
    bars = [b for b in grid.get("bars") or [] if float(b["start_s"]) < dur]
    out = []
    for s in sections or [{"id": "S1", "start_bar": bars[0]["bar"] if bars else 0, "end_bar": 10**9}]:
        inside = [b for b in bars if int(s["start_bar"]) <= int(b["bar"]) < int(s["end_bar"])]
        groups = [inside[k:k + WINDOW_BARS] for k in range(0, len(inside), WINDOW_BARS)]
        if len(groups) > 1 and len(groups[-1]) == 1:  # as the router: a one-bar leftover joins the window before
            last = groups.pop()
            groups[-1] = groups[-1] + last
        for grp in groups:
            out.append({"start_s": float(grp[0]["start_s"]), "end_s": min(dur, float(grp[-1]["end_s"])),
                        "section": str(s["id"]), "start_bar": int(grp[0]["bar"]),
                        "end_bar": int(grp[-1]["bar"]) + 1})
    return out


def prepare_song(song: str, cfg: Any, *, e6: Path, progress: Callable[[str], None] | None = None,
                 scenarios: tuple[str, ...] = SCENARIOS, out_name: str | None = None) -> dict | None:
    """Fill the caches of one song and write its items (module doc) to ``out_name`` (default ``<song>.json``);
    returns the doc (None: no remix). ``scenarios``: E27's P / A1 / B2, or E28's one-guitar control ``D1``."""
    import numpy as np
    import soundfile as sf

    from bandscribe import audio as AU
    from bandscribe import datasets
    from bandscribe.amt import experiments as AE
    from bandscribe.amt import instruments as I
    from bandscribe.amt import stage as S
    from bandscribe.amt.reftx import write_mono_f32
    from bandscribe.parts import eval as PE
    from bandscribe.router import experiments as X
    from bandscribe.router import notesig as N
    from bandscribe.router import split_eval as SE
    from bandscribe.router import stage as RS
    from bandscribe.router import tierb as T

    split_of = {s: v["split"] for s, v in json.loads((e6 / "relations.json").read_text(encoding="utf-8")).items()}
    root = Path(datasets.dataset_root("cambridge_mt"))
    tracks = {t.track_id: t for t in datasets.iter_tracks("cambridge_mt")}
    order = sorted(tracks)
    rel = T.relations(tracks[song], root, X.cache_root() / "relations", code_version=SE.E6_REL_VERSION)
    full = float(math.floor(float(rel["duration_s"])))
    k = order.index(song)
    thr = RS.router_params(cfg)["thresholds"]
    work = cache_dir() / "work" / song
    plans = []
    for sc in scenarios:
        pl = (T.variant_plan(rel, sc, song_index=k, excerpt_s=full, eval_s=full) if sc in T.VARIANTS
              else T.plan(rel, sc, song_index=k, excerpt_s=full))
        if pl is not None:
            plans.append((sc, pl))
    if not plans:
        return None
    remixes = []
    for sc, pl in plans:
        d = work / sc
        mix_dir = d / "input"
        mix_wav = mix_dir / "mix_44k_f32.wav"
        if not mix_wav.is_file():
            if progress:
                progress(f"E27 리믹스: {song} {sc} ({full:.0f} s)")
            rm = T.build(tracks[song], root, rel, pl, excerpt_s=full)
            mix_dir.mkdir(parents=True, exist_ok=True)
            AU.write_f32(mix_wav, rm.mix, T.SR)
            del rm
        mix, _ = sf.read(str(mix_wav), dtype="float32", always_2d=True)
        remixes.append({"scenario": sc, "plan": pl, "dir": d, "mix": mix_wav, "sha": X._pcm_sha(mix)})
        del mix
    stems, sep_info = X.separate_guitar([(r["sha"], r["mix"]) for r in remixes], cfg, force_rung=AE.sep_top_rung(cfg),
                                        cache_dir=X.cache_root(), progress=progress)
    gmask = I.mask_guitar_only()
    views: list[dict[str, Any]] = []
    takes_done: dict[int, dict[str, Any]] = {}
    for r in remixes:
        if progress:
            progress(f"E27 박·구간: {song} {r['scenario']}")
        r["grid"], r["sections"], r["repeat"] = grid_sections(r["dir"] / "input", r["dir"], cfg)
        g, sr = sf.read(str(stems[r["sha"]]), dtype="float32", always_2d=True)
        r["windows"] = bar_windows(r["grid"], r["sections"], full)
        for w in r["windows"]:
            w["label"] = X.predict(X._window_feats(g, int(round(w["start_s"] * sr)), int(round(w["end_s"] * sr)))
                                   or {}, thr)
        sha = write_mono_f32(r["dir"] / "sys_mono.wav", g.mean(axis=1), sr)
        AU.write_f32(r["dir"] / "stem.wav", g, sr)
        views.append({"id": f"{song}.{r['scenario']}.sys", "wav": r["dir"] / "sys_mono.wav", "pcm_sha256": sha,
                      "instruments": gmask})
        for t in sorted(r["plan"].pans):
            if t in takes_done:
                continue
            rep = rel["takes"][t]["parts"][0]
            part = next(p for p in tracks[song].parts if p.id == rep)
            y = T._read(root / part.audio, r["plan"].start_s, full).mean(axis=1)
            tw = work / f"take{t}_mono.wav"
            sha_t = write_mono_f32(tw, y, T.SR)
            takes_done[t] = {"id": f"{song}.t{t}", "start_s": r["plan"].start_s}
            views.append({"id": f"{song}.t{t}", "wav": tw, "pcm_sha256": sha_t, "instruments": gmask})
        del g
    if progress:
        progress(f"E27 전사: {song} 뷰 {len(views)}개 ({len(views) * full / 60:.0f}분 오디오, 캐시에 있는 것은 건너뜀)")
    params = S.ms_params(cfg)
    raws, info_ms = S.transcribe_views_ms(views, params, cfg, work_dir=work / "ms" / "_worker",
                                          cache=AE._eval_cache(), job=None, force_rung=AE.ms_rung(params))
    gclasses = set(I.GUITAR_CLASSES)
    items = []
    for r in remixes:
        pl = r["plan"]
        if any(takes_done[t]["start_s"] != pl.start_s for t in pl.pans):
            raise RuntimeError(f"E27: {song} {r['scenario']} 의 시작이 테이크 뷰와 다릅니다")
        stem, sr = sf.read(str(r["dir"] / "stem.wav"), dtype="float32", always_2d=True)
        sys_notes = [n for n in raws[f"{song}.{r['scenario']}.sys"].get("notes") or []
                     if n.get("instrument") in gclasses]
        line_of = PE.line_groups(rel, sorted(pl.pans))
        ref: dict[str, list] = {}
        for t in sorted(pl.pans):
            ref.setdefault(line_of[t], []).extend(n for n in raws[f"{song}.t{t}"].get("notes") or []
                                                  if n.get("instrument") in gclasses)
        active = {ln for ln, ns in ref.items() if len(ns) >= PE.MIN_REF_NOTES}
        truth = [sorted(t & active) for t in SE.match(sys_notes, ref)]
        items.append({"song": song, "scenario": r["scenario"], "split": split_of.get(song, ""), "sys": sys_notes,
                      "sigs": N.signatures(stem, sr, sys_notes) if sys_notes else [], "truth": truth,
                      "k_true": len(active), "ref_counts": {ln: len(ns) for ln, ns in ref.items()},
                      "takes": {str(t): line_of[t] for t in sorted(pl.pans)}, "windows": r["windows"],
                      "grid": {"bars": r["grid"].get("bars") or [], "beats": r["grid"].get("beats") or []},
                      "sections": r["sections"], "repeat": r["repeat"], "duration_s": full})
        del stem
    doc = {"format": FORMAT, "song": song, "items": items, "sep": sep_info,
           "rung": f"{AE.sep_top_rung(cfg)}|{info_ms.get('rung_label') or AE.ms_rung(params)}",
           "views": len(views), "fresh_views": info_ms.get("fresh_views") or [], "audio_min": len(views) * full / 60}
    atomic.write_json(cache_dir() / (out_name or f"{song}.json"), doc)
    return doc


def two_line_songs(e6: Path) -> list[str]:
    """E6's songs (sorted), each tried; a song without a two-Line remix writes no items."""
    return sorted(json.loads((Path(e6) / "relations.json").read_text(encoding="utf-8")))


def load_items(songs: list[str] | None = None, pattern: str = "*.json") -> list[dict[str, Any]]:
    """The cached items (``pattern``: E27's ``<song>.json`` files match ``*.json`` only without a ``.d1``
    suffix; E28's controls are ``<song>.d1.json``)."""
    out = []
    for f in sorted(cache_dir().glob(pattern)):
        if pattern == "*.json" and f.name.endswith(".d1.json"):
            continue
        doc = atomic.read_json(f)
        if doc.get("format") != FORMAT or (songs and doc["song"] not in songs):
            continue
        for it in doc["items"]:
            it["truth"] = [set(t) for t in it["truth"]]
            out.append(it)
    return out
