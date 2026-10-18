"""E30 (docs/decisions.md): bleed flags (``bandscribe.amt.bleed``) on the merged guitar notes of E29's whole-song
Tier B remixes. A flag is right when no guitar take holds the note (E25's truth is empty). The vocal / bass f0 come
from the song's own vocal / bass parts summed (CPU pyin, cached in ``data/eval/bleed``)."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic, paths
from bandscribe.parts.eval_e28 import SEED, block_bootstrap

RULES = ("vocal", "bass")


def cache_dir() -> Path:
    return Path(paths.EVAL) / "bleed"


def _parts_sum(song: str, kinds: Sequence[str]) -> tuple[np.ndarray, int] | None:
    import soundfile as sf

    from bandscribe import datasets
    from bandscribe.eval.remix import mix_parts
    from bandscribe.router import tierb as T

    tr = {t.track_id: t for t in datasets.iter_tracks("cambridge_mt")}[song]
    root = Path(datasets.dataset_root("cambridge_mt"))
    parts = []
    for k in kinds:
        parts = [p for p in mix_parts(tr) if p.kind == k]
        if parts:
            break
    if not parts:
        return None
    dur = float(np.floor(max(sf.info(str(root / p.audio)).duration for p in parts)))
    y = None
    for p in parts:
        x = T._read(root / p.audio, 0.0, dur).mean(axis=1)
        y = x if y is None else y[:len(x)] + x[:len(y)]
    return y, T.SR


def f0_tracks(song: str, progress: Callable[[str], None] | None = None) -> dict[str, dict[str, Any] | None]:
    from bandscribe.amt import bleed as B
    from bandscribe.bass import f0 as F

    out: dict[str, dict[str, Any] | None] = {}
    for name, kinds in (("vocal", ("vocals",)), ("bass", ("bass", "di"))):
        f = cache_dir() / f"{song}.{name}.npz"
        if not f.is_file():
            got = _parts_sum(song, kinds)
            if got is None:
                out[name] = None
                continue
            if progress:
                progress(f"E30: {song} {name} f0")
            if name == "vocal":
                tr = B.vocal_f0(*got)
                doc = {"hz": tr["hz"], "conf": tr["vprob"], "hop_s": tr["hop_s"]}
            else:
                t = F.pyin_track(got[0].astype(np.float32), got[1])
                doc = {"hz": t.hz["pyin"].astype(np.float32), "conf": t.conf["pyin"].astype(np.float32),
                       "hop_s": t.hop_s}
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, **doc)
        z = np.load(f)
        out[name] = {"hz": z["hz"], "conf": z["conf"], "hop_s": float(z["hop_s"])}
    return out


def flags(rule: str, notes: Sequence[Mapping[str, Any]], tr: Mapping[str, Any] | None) -> list[int]:
    from bandscribe.amt import bleed as B

    if tr is None:
        return []
    fn = B.vocal_flags if rule == "vocal" else B.bass_flags
    return fn(notes, tr["hz"], tr["conf"], tr["hop_s"])


def pooled_precision(per_song: Mapping[str, tuple[int, int]], n: int = 10_000, seed: int = SEED
                     ) -> tuple[float, float, float]:
    """(pooled precision, CI) over songs: per_song = {song: (flagged, false)}; song-block bootstrap."""
    songs = sorted(per_song)
    tot = sum(per_song[s][0] for s in songs)
    if not tot:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [per_song[songs[rng.randrange(len(songs))]] for _ in songs]
        f = sum(x[0] for x in pick)
        vals.append(sum(x[1] for x in pick) / f if f else float("nan"))
    vals = sorted(v for v in vals if v == v)
    return (sum(per_song[s][1] for s in songs) / tot, vals[int(0.025 * len(vals))],
            vals[min(len(vals) - 1, int(0.975 * len(vals)))])


def decide(rows: Sequence[Mapping[str, Any]], rule: str) -> dict[str, Any]:
    per: dict[str, list[int]] = {}
    diffs: dict[str, list[float]] = {}
    for r in rows:
        x = r[rule]
        acc = per.setdefault(r["song"], [0, 0])
        acc[0] += x["flagged"]
        acc[1] += x["false"]
        diffs.setdefault(r["song"], []).append(x["f1_diff"])
    prec, lo, hi = pooled_precision({s: (v[0], v[1]) for s, v in per.items()})
    dm, dlo, dhi = block_bootstrap(diffs)
    flagged = sum(v[0] for v in per.values())
    rule1 = prec == prec and prec >= 0.80
    rule2 = dm == dm and dm >= -0.5
    verdict = "HOLD" if flagged < 20 else ("PASS" if rule1 and rule2 else "FAIL")
    return {"flagged": flagged, "false": sum(v[1] for v in per.values()),
            "precision": {"value": round(prec, 4) if prec == prec else None, "ci_low": round(lo, 4) if lo == lo else None,
                          "ci_high": round(hi, 4) if hi == hi else None},
            "f1_diff": {"mean": round(dm, 3), "ci_low": round(dlo, 3), "ci_high": round(dhi, 3)},
            "per_song": {s: {"flagged": v[0], "false": v[1]} for s, v in sorted(per.items())},
            "rule1_precision": rule1, "rule2_f1": rule2, "verdict": verdict}


def run_e30(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    from bandscribe.parts import eval_e29 as E29
    from bandscribe.parts import eval_track as ET

    items = [*ET.load_items(), *ET.load_items(pattern="*.d1.json")]
    refs: dict[str, dict[str, list]] = {}
    tracks: dict[str, dict] = {}
    per, rows_out = [], []
    for it in items:
        song = it["song"]
        if song not in tracks:
            tracks[song] = f0_tracks(song, progress)
            refs[song] = {}
        need = [t for t in sorted(it["takes"]) if t not in refs[song]]
        if need:
            refs[song].update(E29.take_notes(song, need, cfg))
        ref = E29.union_ref([refs[song][t] for t in sorted(it["takes"])])
        base = 100.0 * E29.note_f1(ref, it["sys"]).f1
        r: dict[str, Any] = {"song": song, "scenario": it["scenario"], "split": it["split"], "sys_notes": len(it["sys"])}
        for rule in RULES:
            idx = flags(rule, it["sys"], tracks[song][rule])
            drop = set(idx)
            kept = [n for i, n in enumerate(it["sys"]) if i not in drop]
            r[rule] = {"flagged": len(idx), "false": sum(1 for i in idx if not it["truth"][i]),
                       "f1_diff": round(100.0 * E29.note_f1(ref, kept).f1 - base, 4),
                       "examples": [{"onset_s": round(float(it["sys"][i]["onset_s"]), 3), "pitch": int(it["sys"][i]["pitch"]),
                                     "false": not it["truth"][i]} for i in idx[:10]]}
            rows_out.append({"system": rule, "dataset": "cambridge_mt", "item": f"{song}:{it['scenario']}",
                             "group": song, "scenario": it["scenario"], "section": it["split"], "line": "full",
                             "metric": "precision", "value": round(r[rule]["false"] / r[rule]["flagged"], 4)
                             if r[rule]["flagged"] else None})
        per.append(r)
    atomic.write_json(Path(out_dir) / "e30_items.json", per)
    atomic.write_json(Path(out_dir) / "e30_decision.json", {
        split: {rule: decide([r for r in per if r["split"] == split], rule) for rule in RULES}
        for split in ("test", "dev")})
    return [r for r in rows_out if r["value"] is not None]


# ------------------------------------------------------------------------------------------------ E30b

GHOST_BELOW = 40
GHOST_ONSET_S = 0.05
GHOST_DB = 20.0
GHOST_SPAN_S = 0.15
NFFT, HOP = 8192, 1024


def band_power(x: np.ndarray, sr: int, onset_s: float, midi: float) -> float:
    """Power of the fundamental's band (+-50 cents, the nearest bin always in) over ``GHOST_SPAN_S`` from the onset:
    Hann ``NFFT`` frames every ``HOP`` samples."""
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / sr)
    band = (freqs >= f0 * 2 ** (-0.5 / 12)) & (freqs <= f0 * 2 ** (0.5 / 12))
    band[int(np.argmin(np.abs(freqs - f0)))] = True
    win = np.hanning(NFFT)
    a = int(onset_s * sr)
    tot = 0.0
    for s in range(a, a + int(GHOST_SPAN_S * sr) + 1, HOP):
        seg = x[s:s + NFFT]
        if len(seg) < NFFT:
            seg = np.pad(seg, (0, NFFT - len(seg)))
        tot += float(np.sum(np.abs(np.fft.rfft(seg * win))[band] ** 2))
    return tot


def ghost_notes(notes: Sequence[Mapping[str, Any]], x: np.ndarray, sr: int) -> set[int]:
    """Indices of a take's notes under E2 that are the transcriber's octave-below ghosts (E30b's rule)."""
    out = set()
    for i, n in enumerate(notes):
        p = int(n["pitch"])
        if p >= GHOST_BELOW:
            continue
        on = float(n["onset_s"])
        ups = [m for m in notes if int(m["pitch"]) == p + 12 and abs(float(m["onset_s"]) - on) <= GHOST_ONSET_S]
        if not ups:
            continue
        low = band_power(x, sr, on, p)
        high = band_power(x, sr, float(ups[0]["onset_s"]), p + 12)
        if high > 0 and 10 * np.log10(max(low, 1e-30) / high) < -GHOST_DB:
            out.add(i)
    return out


def run_e30b(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    import soundfile as sf

    from bandscribe.parts import eval as PE
    from bandscribe.parts import eval_e29 as E29
    from bandscribe.parts import eval_track as ET
    from bandscribe.router import split_eval as SE

    items = [*ET.load_items(), *ET.load_items(pattern="*.d1.json")]
    refs: dict[str, dict[str, list]] = {}
    removed: dict[str, dict[str, int]] = {}
    tracks: dict[str, dict] = {}
    per, rows_out = [], []
    for it in items:
        song = it["song"]
        if song not in tracks:
            tracks[song] = f0_tracks(song, progress)
            refs[song], removed[song] = {}, {}
        need = [t for t in sorted(it["takes"]) if t not in refs[song]]
        if need:
            raw = E29.take_notes(song, need, cfg)
            for t in need:
                x, sr = sf.read(str(ET.cache_dir() / "work" / song / f"take{t}_mono.wav"), dtype="float32")
                g = ghost_notes(raw[t], x, sr)
                refs[song][t] = [n for i, n in enumerate(raw[t]) if i not in g]
                removed[song][t] = len(g)
                if progress:
                    progress(f"E30b: {song} 테이크 {t}: 저음 유령 {len(g)}개 뺌")
        by_line: dict[str, list] = {}
        for t, line in it["takes"].items():
            by_line.setdefault(line, []).extend(refs[song][t])
        active = {ln for ln, ns in by_line.items() if len(ns) >= PE.MIN_REF_NOTES}
        truth = [t & active for t in SE.match(it["sys"], by_line)]
        ref = E29.union_ref([refs[song][t] for t in sorted(it["takes"])])
        base = 100.0 * E29.note_f1(ref, it["sys"]).f1
        idx = flags("bass", it["sys"], tracks[song]["bass"])
        drop = set(idx)
        kept = [n for i, n in enumerate(it["sys"]) if i not in drop]
        r = {"song": song, "scenario": it["scenario"], "split": it["split"], "sys_notes": len(it["sys"]),
             "bass": {"flagged": len(idx), "false": sum(1 for i in idx if not truth[i]),
                      "false_e30": sum(1 for i in idx if not it["truth"][i]),
                      "f1_diff": round(100.0 * E29.note_f1(ref, kept).f1 - base, 4)}}
        per.append(r)
        if idx:
            rows_out.append({"system": "bass", "dataset": "cambridge_mt", "item": f"{song}:{it['scenario']}",
                             "group": song, "scenario": it["scenario"], "section": it["split"], "line": "full",
                             "metric": "precision", "value": round(r["bass"]["false"] / len(idx), 4)})
    atomic.write_json(Path(out_dir) / "e30b_items.json", per)
    atomic.write_json(Path(out_dir) / "e30b_decision.json", {
        "removed_ref_notes": removed,
        **{split: decide([r for r in per if r["split"] == split], "bass") for split in ("test", "dev")}})
    return rows_out
