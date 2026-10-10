"""Tier B remixes for the router (E6, M6): Cambridge-MT multitracks re-panned into the DESIGN 4.5 scenarios, with
the router label each window should get from the way it was mixed.

**Takes and doubles (machine labels).** The lines.yaml drafts were guessed from file names and are wrong both ways
(Woodfire's "1a/1b" are one performance on two mics 15 ms apart, labelled as a double; Jokers' "Gtr6 Mic1/Mic2"
behave like two takes; Mu's Gtr1/Gtr2 are a double under two line ids). So every pair of the song's guitar parts
(``remix.mix_parts``: a DI is left out when its line has an amped part) is compared where both play (100 ms RMS
within ``ACT_DB`` of each part's loudest frame), at 22.05 kHz:

- ``r``: correlation of the spectral-flux onset envelopes (5 ms hop) at the best lag within ±``MAX_LAG_MS``;
- ``chroma``: median cosine of the 100 ms chroma vectors over the jointly active frames.

One performance (two mics, amp + DI, a copy): ``r >= SAME_TAKE_R``. A double (the same part played again):
``DOUBLE_R_MIN <= r < SAME_TAKE_R`` and ``chroma >= DOUBLE_CHROMA``. Other pairs with at least ``MIN_OVERLAP_S`` of
joint playing are different parts. Parts of one performance form a take (union of the "same" pairs; its
representative is the first part in lines.yaml order); takes relate through their representatives. The values
come from looking at all 13 songs (2026-10-10: same performance 0.78-1.00, doubles 0.30-0.73 with chroma
0.88-1.00, different parts below 0.3 or chroma below 0.88); they label data, they are not a tested hypothesis,
and the labels still need the user's ear (DESIGN 8.2 "귀로 역할 라벨").

**Scenarios** (60 s excerpt each, non-guitar parts as in the ``original`` mix, guitars at unity gain with a
constant-power pan, mastered to -8 LUFS like E1; the true guitar bus gets the master's own gain curve):

- ``B``: every take at the centre -> ``R0`` wherever a guitar plays.
- ``A``: the double with the most joint playing hard left / hard right, every other take at the centre.
  Per window: both double takes playing and >= half the guitar energy -> ``R2``; one of them alone with >= half
  -> ``R3`` (one hard-panned source looks like a panned part); the centre takes >= 80 % -> ``R0``; else ``mixed``.
- ``P``: two different takes with the most joint playing at -/+``P_PAN`` (hard L/R on odd song indices), others
  muted -> ``R3`` wherever one plays.
- ``ADT``: the take that plays most, left, and itself delayed by ``ADT_DELAY_MS`` right, others muted -> ``R1``.

A window (``WINDOW_S``, non-overlapping) whose guitar bus is below ``SILENT_DBFS`` is ``silent``. The excerpt
starts on the whole second that maximises the scenario's playing time (A: both double takes; P: both takes
+ half of either alone; B: any take; ADT: its take); ties -> earliest.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic

CODE_VERSION = "2"  # 2: onset frames mapped to activity frames by time (review 2026-10-10)
REL_FORMAT = "bandscribe.router_tierb_relations/1"
SR = 44100
SR_REL = 22050
ONSET_HOP = 110  # 5 ms at 22.05 kHz
ACT_HOP = 2205  # 100 ms
ACT_DB = -40.0
MAX_LAG_MS = 30
MIN_OVERLAP_S = 8.0
SAME_TAKE_R = 0.75
DOUBLE_R_MIN = 0.30
DOUBLE_CHROMA = 0.88
EXCERPT_S = 60.0
WINDOW_S = 8.0
SILENT_DBFS = -50.0
ACTIVE_REL_DB = -25.0  # a source plays in a window when within 25 dB of the window's loudest source
P_PAN = 0.7
ADT_DELAY_MS = 15.0
MASTER_LUFS = -8.0
SCENARIOS = ("B", "A", "P", "ADT")


# ------------------------------------------------------------------------------------------- relations


def guitar_parts(track: Any) -> list[Any]:
    """The guitar parts that go into a remix: ``remix.mix_parts`` restricted to guitar lines, lines.yaml order."""
    from bandscribe.eval.remix import mix_parts

    guitar_lines = {str(ln.id) for ln in track.lines if ln.kind == "guitar"}
    return [p for p in mix_parts(track) if p.kind in ("guitar", "di") and p.line in guitar_lines]


def part_features(y: np.ndarray) -> dict[str, np.ndarray]:
    """Activity (100 ms), onset envelope (5 ms) and chroma (100 ms) of a mono 22.05 kHz signal."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    rms = librosa.feature.rms(y=y, frame_length=2 * ACT_HOP, hop_length=ACT_HOP, center=True)[0]
    act = rms > (rms.max() if rms.size else 0.0) * 10 ** (ACT_DB / 20) if rms.size and rms.max() > 0 else \
        np.zeros(rms.size, dtype=bool)
    spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=ONSET_HOP, center=True))
    flux = np.concatenate([[0.0], np.maximum(np.diff(np.log1p(100 * spec), axis=1), 0).sum(axis=0)])
    big = np.abs(librosa.stft(y, n_fft=4096, hop_length=ACT_HOP, center=True)) ** 2
    chroma = librosa.feature.chroma_stft(S=big, sr=SR_REL)
    return {"act": act.astype(bool), "flux": flux.astype(np.float32), "chroma": chroma.astype(np.float32)}


def pair_relation(fa: Mapping[str, np.ndarray], fb: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Relation of two parts (module doc) -> {overlap_s, r, lag_ms, chroma, relation}."""
    n = min(fa["act"].size, fb["act"].size)
    both = fa["act"][:n] & fb["act"][:n]
    overlap = round(float(both.sum()) * ACT_HOP / SR_REL, 2)
    out: dict[str, Any] = {"overlap_s": overlap, "r": None, "lag_ms": None, "chroma": None, "relation": "none"}
    if overlap < MIN_OVERLAP_S:
        return out
    n_on = min(fa["flux"].size, fb["flux"].size)
    idx = np.minimum((np.arange(n_on) * ONSET_HOP) // ACT_HOP, both.size - 1)  # the activity frame of each onset frame
    m = both[idx]
    x, y = fa["flux"][: m.size][m].astype(np.float64), fb["flux"][: m.size][m].astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    max_lag = int(round(MAX_LAG_MS / 1000 * SR_REL / ONSET_HOP))
    best, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = float(np.mean(x[lag:] * y[: y.size - lag])) if y.size > lag else -2.0
        else:
            r = float(np.mean(x[: x.size + lag] * y[-lag:])) if x.size > -lag else -2.0
        if r > best + 1e-12:
            best, best_lag = r, lag
    kc = min(fa["chroma"].shape[1], fb["chroma"].shape[1], both.size)
    sel = both[:kc]
    a, b = fa["chroma"][:, :kc][:, sel], fb["chroma"][:, :kc][:, sel]
    cos = np.sum(a * b, axis=0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9)
    chroma = float(np.median(cos)) if cos.size else 0.0
    if best >= SAME_TAKE_R:
        rel = "same"
    elif best >= DOUBLE_R_MIN and chroma >= DOUBLE_CHROMA:
        rel = "double"
    else:
        rel = "different"
    out.update(r=round(best, 4), lag_ms=round(best_lag * ONSET_HOP / SR_REL * 1000, 1), chroma=round(chroma, 4),
               relation=rel)
    return out


def takes_from_pairs(ids: Sequence[str], pairs: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Union of the "same" pairs -> takes (lists of part ids, lines.yaml order; takes ordered by first part)."""
    parent = {i: i for i in ids}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for p in pairs:
        if p["relation"] == "same":
            ra, rb = find(p["a"]), find(p["b"])
            if ra != rb:
                parent[max(ra, rb, key=ids.index)] = min(ra, rb, key=ids.index)
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: ids.index(g[0]))


def _activity_seconds(act: np.ndarray, n_s: int) -> np.ndarray:
    """100 ms activity -> per whole second: playing for >= half of it."""
    per = int(round(SR_REL / ACT_HOP))  # 10 frames per second
    out = np.zeros(n_s, dtype=bool)
    for s in range(n_s):
        seg = act[s * per:(s + 1) * per]
        out[s] = seg.size > 0 and seg.mean() >= 0.5
    return out


def _rel_key(track: Any, parts: Sequence[Any], root: Path) -> str:
    sig = [(p.id, p.audio, (Path(root) / p.audio).stat().st_size) for p in parts]
    blob = json.dumps({"code": CODE_VERSION, "track": track.track_id, "parts": sig,
                       "consts": [SAME_TAKE_R, DOUBLE_R_MIN, DOUBLE_CHROMA, MIN_OVERLAP_S, ACT_DB, MAX_LAG_MS]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def relations(track: Any, root: Path, cache_dir: Path | None = None) -> dict[str, Any]:
    """Takes, doubles and different-part pairs of a Tier B track (module doc), cached as JSON under ``cache_dir``."""
    import librosa
    import soundfile as sf

    parts = guitar_parts(track)
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{track.track_id}_{_rel_key(track, parts, root)}.json"
        if cache.is_file():
            return atomic.read_json(cache)
    feats: dict[str, dict[str, np.ndarray]] = {}
    dur_s = 0.0
    for p in parts:
        y, sr = sf.read(str(Path(root) / p.audio), dtype="float32", always_2d=True)
        dur_s = max(dur_s, len(y) / sr)
        y = librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=SR_REL)
        feats[p.id] = part_features(y)
    ids = [p.id for p in parts]
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pairs.append({"a": a, "b": b, **pair_relation(feats[a], feats[b])})
    takes = takes_from_pairs(ids, pairs)
    rep = [t[0] for t in takes]
    by_pair = {(p["a"], p["b"]): p for p in pairs}

    def rel(x: str, y: str) -> Mapping[str, Any]:
        return by_pair.get((x, y)) or by_pair.get((y, x)) or {"relation": "none", "overlap_s": 0.0}

    n_s = int(math.floor(dur_s))
    take_rows = []
    for t in takes:
        sec = np.zeros(n_s, dtype=bool)
        for pid in t:
            sec |= _activity_seconds(feats[pid]["act"], n_s)
        take_rows.append({"parts": t, "line": next(p.line for p in parts if p.id == t[0]),
                          "active_s": [int(s) for s in np.flatnonzero(sec)]})
    links = []
    for i in range(len(takes)):
        for j in range(i + 1, len(takes)):
            r = rel(rep[i], rep[j])
            if r["relation"] in ("double", "different"):
                links.append({"takes": [i, j], "relation": r["relation"], "overlap_s": r["overlap_s"],
                              "r": r.get("r"), "chroma": r.get("chroma")})
    doc = {"format": REL_FORMAT, "code_version": CODE_VERSION, "track": track.track_id, "duration_s": round(dur_s, 3),
           "parts": [{"id": p.id, "file": Path(p.audio).name, "kind": p.kind, "line": p.line, "take": p.take}
                     for p in parts],
           "pairs": pairs, "takes": take_rows, "links": links}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        atomic.write_json(cache, doc)
    return doc


# ------------------------------------------------------------------------------------------- scenarios


@dataclass
class Plan:
    """What a scenario remix puts where: take index -> pan (None = muted), the excerpt start, the takes whose
    playing decides the label, and the ADT delay."""

    scenario: str
    start_s: int
    pans: dict[int, float]
    roles: dict[str, list[int]] = field(default_factory=dict)  # "double" / "pair" / "adt" / "centre"
    delay_ms: float = 0.0


def _best_start(score: np.ndarray, dur_s: int) -> int:
    n = score.size
    if n <= dur_s:
        return 0
    c = np.concatenate([[0.0], np.cumsum(score)])
    tot = c[dur_s:] - c[:-dur_s]
    return int(np.argmax(tot))  # first maximum = earliest


def plan(rel: Mapping[str, Any], scenario: str, song_index: int = 0, excerpt_s: float = EXCERPT_S) -> Plan | None:
    """The scenario's remix plan for a song (module doc), or None when the song cannot make it (no double for A,
    no two different takes for P, no guitar)."""
    takes = rel["takes"]
    if not takes:
        return None
    n_s = int(math.floor(float(rel["duration_s"])))
    dur = int(excerpt_s)
    sec = [np.zeros(n_s, dtype=float) for _ in takes]
    for i, t in enumerate(takes):
        for s in t["active_s"]:
            if s < n_s:
                sec[i][s] = 1.0
    if scenario == "B":
        any_ = np.clip(sum(sec), 0, 1)
        return Plan("B", _best_start(any_, dur), {i: 0.0 for i in range(len(takes))},
                    {"centre": list(range(len(takes)))})
    if scenario == "ADT":
        i = int(np.argmax([s.sum() for s in sec]))
        return Plan("ADT", _best_start(sec[i], dur), {i: -1.0}, {"adt": [i]}, delay_ms=ADT_DELAY_MS)
    kind = {"A": "double", "P": "different"}.get(scenario)
    if kind is None:
        raise ValueError(f"unknown scenario {scenario!r}")
    links = [ln for ln in rel["links"] if ln["relation"] == kind]
    if not links:
        return None
    best = max(links, key=lambda ln: (float(np.sum(sec[ln["takes"][0]] * sec[ln["takes"][1]])), -ln["takes"][0],
                                       -ln["takes"][1]))
    i, j = best["takes"]
    both = sec[i] * sec[j]
    if scenario == "A":
        pans = {k: 0.0 for k in range(len(takes))}
        pans[i], pans[j] = -1.0, 1.0
        return Plan("A", _best_start(both, dur), pans,
                    {"double": [i, j], "centre": [k for k in range(len(takes)) if k not in (i, j)]})
    p = 1.0 if song_index % 2 else P_PAN
    either = np.clip(sec[i] + sec[j], 0, 1)
    return Plan("P", _best_start(both + 0.5 * either, dur), {i: -p, j: p}, {"pair": [i, j]})


def _pan(x: np.ndarray, p: float) -> np.ndarray:
    """Constant-power pan of a mono (n,) signal -> (n, 2) (``remix.pan`` is channel-first)."""
    th = (p + 1) * np.pi / 4
    return np.stack([np.cos(th) * x, np.sin(th) * x], axis=1)


def _read(path: Path, start_s: float, dur_s: float) -> np.ndarray:
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate != SR:
        raise ValueError(f"{path}: {info.samplerate} Hz (Tier B remixes expect {SR})")
    a = int(round(start_s * SR))
    n = int(round(dur_s * SR))
    y, _ = sf.read(str(path), start=min(a, info.frames), stop=min(a + n, info.frames), dtype="float32",
                   always_2d=True)
    if len(y) < n:
        y = np.pad(y, ((0, n - len(y)), (0, 0)))
    return y


@dataclass
class Remix:
    song: str
    scenario: str
    start_s: int
    mix: np.ndarray  # (n, 2) float32, mastered
    guitar: np.ndarray  # (n, 2) float32, the true guitar bus with the master's gain
    sources: list[dict[str, Any]]  # per take: {take, parts, pan, delay_ms, role}
    windows: list[dict[str, Any]]  # per WINDOW_S window: {start_s, end_s, truth, shares, bus_dbfs}


def window_truth(scenario: str, roles: Mapping[str, Sequence[int]], src_energy: Mapping[int, float],
                 bus_dbfs: float) -> tuple[str, dict[str, float]]:
    """Label a window from its sources' energies (module doc) -> (truth, energy shares by role)."""
    if bus_dbfs < SILENT_DBFS or not src_energy:
        return "silent", {}
    top = max(src_energy.values())
    if top <= 0:
        return "silent", {}
    active = {k for k, e in src_energy.items() if e > 0 and 10 * math.log10(e / top) >= ACTIVE_REL_DB}
    total = sum(src_energy.values())
    shares = {role: round(sum(src_energy.get(k, 0.0) for k in ks) / total, 4) for role, ks in roles.items()}
    if scenario == "B":
        return "R0", shares
    if scenario == "ADT":
        return ("R1" if set(roles["adt"]) & active else "silent"), shares
    if scenario == "P":
        return ("R3" if set(roles["pair"]) & active else "silent"), shares
    i, j = roles["double"]
    dbl = shares.get("double", 0.0)
    if i in active and j in active and dbl >= 0.5:
        return "R2", shares
    if (i in active) != (j in active) and dbl >= 0.5:
        return "R3", shares
    if shares.get("centre", 0.0) >= 0.8:
        return "R0", shares
    return "mixed", shares


def build(track: Any, root: Path, rel: Mapping[str, Any], pl: Plan, excerpt_s: float = EXCERPT_S) -> Remix:
    """Render a planned remix (module doc)."""
    from bandscribe.eval.remix import master_gain, mix_parts

    guitar_ids = {p["id"] for p in rel["parts"]}
    guitar_lines = {str(ln.id) for ln in track.lines if ln.kind == "guitar"}
    backing = np.zeros((int(round(excerpt_s * SR)), 2), dtype=np.float64)
    for p in mix_parts(track):  # everything but the guitars (a bass recorded only as a DI stays in)
        if p.id in guitar_ids or p.kind in ("guitar", "mix") or (p.kind == "di" and p.line in guitar_lines):
            continue
        y = _read(Path(root) / p.audio, pl.start_s, excerpt_s)
        backing += np.repeat(y, 2, axis=1) if y.shape[1] == 1 else y[:, :2]
    n = backing.shape[0]
    bus = np.zeros((n, 2), dtype=np.float64)
    sources = []
    per_take: dict[int, np.ndarray] = {}
    for k, pan in sorted(pl.pans.items()):
        rep = rel["takes"][k]["parts"][0]
        part = next(p for p in track.parts if p.id == rep)
        y = _read(Path(root) / part.audio, pl.start_s, excerpt_s).mean(axis=1).astype(np.float64)
        st = _pan(y, pan)
        if pl.scenario == "ADT":
            d = int(round(pl.delay_ms / 1000 * SR))
            st = np.stack([y, np.concatenate([np.zeros(d), y[: n - d]])], axis=1)
        per_take[k] = st
        bus += st
        role = next((r for r, ks in pl.roles.items() if k in ks), None)
        sources.append({"take": k, "parts": rel["takes"][k]["parts"], "line": rel["takes"][k]["line"],
                        "pan": pan if pl.scenario != "ADT" else None, "delay_ms": pl.delay_ms or None,
                        "role": role})
    full = backing + bus
    g = master_gain(full, SR, lufs=MASTER_LUFS)
    mix = np.clip(full * g[:, None], -1.0, 1.0).astype(np.float32)
    guitar = (bus * g[:, None]).astype(np.float32)
    wins = []
    w = int(round(WINDOW_S * SR))
    for a in range(0, n - w + 1, w):
        seg = guitar[a:a + w].astype(np.float64)
        bus_db = 10 * math.log10(float(np.mean(seg ** 2)) + 1e-20)
        energy = {k: float(np.mean((st[a:a + w] * g[a:a + w, None]) ** 2)) for k, st in per_take.items()}
        truth, shares = window_truth(pl.scenario, pl.roles, energy, bus_db)
        wins.append({"start_s": round(a / SR, 3), "end_s": round((a + w) / SR, 3), "truth": truth,
                     "shares": shares, "bus_dbfs": round(bus_db, 2)})
    return Remix(track.track_id, pl.scenario, pl.start_s, mix, guitar, sources, wins)
