"""Distractor suite: which kinds of sound make the guitar transcription wrong, and by how much (DESIGN §8.8).

``bandscribe eval run --suite distractors``. Pre-registered as ``distractors`` in docs/decisions.md.

Band recordings (J-pop / J-rock especially) put synth pads and leads, keys, strings and effects (risers, sweeps)
on top of the guitars. A single "share of guitar notes also found on the backing" cannot say *which* sound makes
false notes or hides real ones, so this suite rebuilds band mixes from multitracks with and without each kind of
sound and runs the production guitar path on every version.

Per Tier B multitrack (Cambridge-MT) with at least one distractor stem:

1. **Stems** are grouped by sound category from their file names (``bandscribe.eval.stem_taxonomy``; a part on a
   guitar line of ``lines.yaml`` is always a guitar; redundant DI and mix files are left out like ``tierb_mix``).
2. **Excerpts:** up to ``max_windows`` (2) deterministic windows of ``window_s`` (75 s), start every 5 s. A window
   qualifies when the guitar is active (>= -20 dB rel. the stem sum, 10 Hz frames) in >= 50 % of its frames and
   at least one distractor category is active (>= -30 dB, above -60 dBFS) in >= 10 % ("present"). The first window
   has the most present categories (then the most activity); the second must not overlap it and must add a
   present category the first lacks.
3. **Conditions** (stems summed at unity = the multitrack's own balance, like ``tierb_mix``): ``base`` = guitars +
   drums + bass + vocals; ``+<category>`` = base + one present distractor category; ``full`` = every stem;
   optionally ``guitar_alone``. Every condition is multiplied by the **same** mastering gain curve, computed on
   ``full`` with ``remix.master_gain`` (-8 LUFS, -1 dBTP limiter): the guitar signal is identical in all conditions
   and only the other sounds change.
4. **Reference:** MuScriptor (guitar-only mask) on the true summed guitar stems of the excerpt (same gain), so the
   numbers measure what the mixture and the separation add, not MuScriptor's own errors.
5. **System under test** = the production guitar path, in batched workers under the GPU lock with content caches:
   SW separation (``bandscribe.sep.run.separate``; conditions are laid end to end with >= 1 chunk of silence between
   them, aligned to the chunk step, so each is separated exactly as if alone with silence around it) -> energy and
   ``guitar_mono`` / ``piano_other_mono`` views as the ``stems`` stage makes them -> the sampled presence pass and
   ``instrumentation.estimate`` with the configured profile and guitar-mask policy (2 s pseudo-bars, no sections:
   the experiments' convention, E3b) -> MuScriptor on ``guitar_mono`` with that mask -> guitar classes only, latency
   corrected like the reference. A second arm, ``guitar_only``, transcribes the same view with the guitar-only mask
   (a cache hit whenever the production mask is guitar-only), so audio effects and mask effects can be told apart.
6. **Scoring and causes:** mir_eval onset 50 ms / pitch 50 cents per (excerpt, condition, arm); FP and FN per
   minute; every false positive and missed note gets a cause (``bandscribe.eval.attribution``); separation leakage of
   each category into the SW guitar stem. Deltas against ``base`` per category with song-block bootstrap CIs
   (``bandscribe.eval.stats``, 10,000 draws).

Outputs (run dir): ``metrics.csv`` (one row per item x metric, plus ``*`` aggregate rows with CIs),
``summary.json``, ``report.md`` + ``report.html`` (Korean), ``windows.json``, ``run.json``.
Caches: ``data/eval/distractors/`` (song energy, separated views, reference views) and the MuScriptor eval cache.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic, paths
from bandscribe.eval import attribution as A
from bandscribe.eval import stats
from bandscribe.eval import stem_taxonomy as T

log = logging.getLogger(__name__)

SUITE = "distractors"
CODE_VERSION = "1"
SR = 44100
FPS = 10.0
WINDOW_S = 75.0
STEP_S = 5.0
GUITAR_ACTIVE_DB = -20.0
ACTIVE_DB = -30.0
SILENCE_DBFS = -60.0
PRESENT_SHARE = 0.1
GUITAR_SHARE = 0.5
MASTER_LUFS = -8.0
CEILING_DBTP = -1.0
SEP_BATCH_S = 600.0
ONSET_TOL = 0.05
PITCH_TOL_CENTS = 50.0
ARMS = ("production", "guitar_only")
# GPU time model for the budget check (RTX 2060 measurements, docs/PROGRESS.md): SW ~4.4x realtime + ~40 s per
# worker; MuScriptor guitar view ~0.6 s per audio s (0.5-1.2) + ~40 s per worker; presence windows ~1.5 s per s.
EST_SEP_RTF = 4.4
EST_SEP_OVERHEAD_S = 40.0
EST_MS_S_PER_S = 0.6
EST_MS_OVERHEAD_S = 40.0
EST_PRESENCE_S_PER_S = 1.5


def root_dir() -> Path:
    return Path(paths.EVAL) / "distractors"


class DistractorError(RuntimeError):
    """User-facing (Korean) problem of the distractor suite."""


# ------------------------------------------------------------------------------------------------ songs


@dataclass
class StemFile:
    path: Path
    name: str
    category: str


@dataclass
class SongInfo:
    dataset: str
    track_id: str
    group: str
    title: str
    stems: list[StemFile]
    unknown: list[str] = field(default_factory=list)

    @property
    def item(self) -> str:
        return f"{self.dataset}:{self.track_id}"

    def categories(self) -> list[str]:
        have = {s.category for s in self.stems}
        return [c for c in T.CATEGORIES if c in have]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def song_from_track(track: Any, root: Path) -> SongInfo:
    """Stems of a Tier B ``DatasetTrack`` with their sound categories (DI and mix parts as in ``tierb_mix``)."""
    from bandscribe.eval.remix import mix_parts

    guitar_lines = {str(_get(ln, "id")) for ln in _get(track, "lines") or [] if _get(ln, "kind") == "guitar"}
    stems: list[StemFile] = []
    names: list[str] = []
    for p in mix_parts(track):
        audio = str(_get(p, "audio"))
        name = Path(audio).name
        names.append(name)
        if T.is_excluded(name):
            continue
        cat = T.categorize(name)
        if _get(p, "line") in guitar_lines and cat not in T.GUITAR:
            cat = "guitar_acoustic" if set(T.tokens(name)) & {"ac", "acoustic"} else "guitar_electric"
        stems.append(StemFile(Path(root) / audio, name, cat))
    meta = _get(track, "meta") or {}
    return SongInfo(dataset=str(_get(track, "dataset")), track_id=str(_get(track, "track_id")),
                    group=str(_get(track, "group") or _get(track, "track_id")), title=str(meta.get("title") or ""),
                    stems=stems, unknown=T.unknown_names(names))


def load_songs(dataset: str = "cambridge_mt", only: Sequence[str] | None = None) -> list[SongInfo]:
    from bandscribe import datasets

    try:
        root = Path(datasets.dataset_root(dataset))
        tracks = list(datasets.iter_tracks(dataset))
    except datasets.DatasetUnavailable as e:
        raise DistractorError(str(e)) from e
    out = []
    for t in tracks:
        if only and str(_get(t, "track_id")) not in only:
            continue
        out.append(song_from_track(t, root))
    return sorted(out, key=lambda s: s.track_id)


# ------------------------------------------------------------------------------------- energy, windows


def _read_info(path: Path) -> tuple[int, int, int]:
    import soundfile as sf

    info = sf.info(str(path))
    return int(info.samplerate), int(info.frames), int(info.channels)


def stem_power(path: Path, fps: float = FPS, block_s: float = 20.0) -> np.ndarray:
    """Mean-square power per 1/fps frame (channels averaged), read block by block (any sample rate)."""
    import soundfile as sf

    sr, frames, _ch = _read_info(path)
    hop = int(round(sr / fps))
    n = -(-frames // hop)
    out = np.zeros(n, dtype=np.float64)
    block = max(hop, int(block_s * sr) // hop * hop)
    pos = 0
    with sf.SoundFile(str(path)) as f:
        while pos < frames:
            x = f.read(min(block, frames - pos), dtype="float32", always_2d=True)
            if not len(x):
                break
            x2 = (x.astype(np.float64) ** 2).mean(axis=1)
            k0 = pos // hop
            m = -(-len(x2) // hop)
            pad = np.zeros(m * hop - len(x2))
            seg = np.concatenate([x2, pad]).reshape(m, hop)
            cnt = np.full(m, hop, dtype=np.float64)
            cnt[-1] = len(x2) - (m - 1) * hop
            out[k0:k0 + m] += seg.sum(axis=1) / cnt
            pos += len(x)
    return out


def _energy_cache_key(song: SongInfo) -> str:
    h = hashlib.sha256(f"{CODE_VERSION}|{FPS}".encode())
    for s in sorted(song.stems, key=lambda s: s.name):
        st = s.path.stat()
        h.update(f"|{s.name}|{s.category}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def category_power(song: SongInfo, *, cache_root: Path | None = None) -> dict[str, np.ndarray]:
    """{category: power per 10 Hz frame} = sum of its stems' mean-square power (cached per stem set)."""
    from bandscribe.eval.runs import slug

    cdir = Path(cache_root or root_dir()) / "songs" / slug(song.track_id)
    f = cdir / f"power_{_energy_cache_key(song)}.npz"
    if f.is_file():
        with np.load(f) as z:
            return {k: np.array(z[k]) for k in z.files}
    acc: dict[str, np.ndarray] = {}
    for s in song.stems:
        p = stem_power(s.path)
        if s.category in acc:
            n = max(len(acc[s.category]), len(p))
            acc[s.category] = np.pad(acc[s.category], (0, n - len(acc[s.category]))) + np.pad(p, (0, n - len(p)))
        else:
            acc[s.category] = p
    n = max((len(v) for v in acc.values()), default=0)
    acc = {k: np.pad(v, (0, n - len(v))) for k, v in acc.items()}
    cdir.mkdir(parents=True, exist_ok=True)
    from bandscribe.sep.derive import write_npz_deterministic

    write_npz_deterministic(f, {k: v.astype(np.float64) for k, v in sorted(acc.items())})
    return acc


@dataclass
class WindowChoice:
    wid: str
    start_s: float
    end_s: float
    guitar_share: float
    shares: dict[str, float]          # every non-base category's active share in the window
    present: list[str]                # distractor categories with share >= PRESENT_SHARE (CATEGORIES order)


def activity(power: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per category: bool per frame (guitar: >= -20 dB rel. the stem sum; others >= -30 dB; all above -60 dBFS)."""
    if not power:
        return {}
    n = max(len(v) for v in power.values())
    tot = np.zeros(n)
    for v in power.values():
        tot[: len(v)] += v
    floor = 10 ** (SILENCE_DBFS / 10)
    out = {}
    for c, v in power.items():
        v = np.pad(v, (0, n - len(v)))
        thr = GUITAR_ACTIVE_DB if c in T.GUITAR else ACTIVE_DB
        out[c] = (v >= floor) & (v >= tot * 10 ** (thr / 10))
    if any(c in out for c in T.GUITAR):
        g = sum(power.get(c, np.zeros(n)) for c in T.GUITAR)
        g = np.pad(g, (0, n - len(g)))
        out["_guitar"] = (g >= floor) & (g >= tot * 10 ** (GUITAR_ACTIVE_DB / 10))
    return out


def select_windows(power: Mapping[str, np.ndarray], *, window_s: float = WINDOW_S, step_s: float = STEP_S,
                   max_windows: int = 2, fps: float = FPS) -> list[WindowChoice]:
    """Deterministic excerpt windows (see the module docstring, point 2). Pure numpy."""
    act = activity(power)
    if "_guitar" not in act:
        return []
    n = len(act["_guitar"])
    dur = n / fps
    win = min(window_s, dur)
    wf = max(1, int(round(win * fps)))
    starts = np.arange(0.0, max(0.0, dur - win) + 1e-9, step_s) if dur > win else np.array([0.0])
    distract = [c for c in T.DISTRACTORS if c in act]
    cands = []
    for s in starts:
        a = int(round(s * fps))
        z = min(n, a + wf)
        g = float(act["_guitar"][a:z].mean())
        shares = {c: round(float(act[c][a:z].mean()), 4) for c in distract}
        present = [c for c in distract if shares[c] >= PRESENT_SHARE]
        if g >= GUITAR_SHARE and present:
            cands.append(WindowChoice("", round(float(s), 3), round(float(s) + win, 3), round(g, 4), shares, present))
    if not cands:
        return []

    def strength(c: WindowChoice) -> float:
        return round(sum(c.shares[k] for k in c.present), 4)

    first = max(cands, key=lambda c: (len(c.present), strength(c), c.guitar_share, -c.start_s))
    first.wid = "w1"
    out = [first]
    if max_windows >= 2:
        rest = [c for c in cands if c.start_s >= first.end_s - 1e-9 or c.end_s <= first.start_s + 1e-9]
        rest = [c for c in rest if set(c.present) - set(first.present)]
        if rest:
            second = max(rest, key=lambda c: (len(set(c.present) - set(first.present)), len(c.present), strength(c),
                                              c.guitar_share, -c.start_s))
            second.wid = "w2"
            out.append(second)
    return out


# ------------------------------------------------------------------------------------------ conditions


@dataclass
class Condition:
    name: str
    categories: tuple[str, ...]


def conditions_for(song_categories: Sequence[str], present: Sequence[str], *,
                   guitar_alone: bool = False) -> list[Condition]:
    """base, +<present category>..., full (+ guitar_alone); ``full`` is dropped when equal to the base set."""
    have = [c for c in T.CATEGORIES if c in song_categories]
    base = tuple(c for c in have if c in T.BASE)
    out = [Condition("base", base)]
    for c in present:
        if c in have and c not in T.BASE:
            out.append(Condition(f"+{c}", tuple(x for x in have if x in base or x == c)))
    full = tuple(have)
    if full != base:
        out.append(Condition("full", full))
    if guitar_alone:
        out.append(Condition("guitar_alone", tuple(c for c in have if c in T.GUITAR)))
    return out


def read_segment(path: Path, start_s: float, n_out: int) -> np.ndarray:
    """(n_out, 2) float64 at 44.1 kHz from ``start_s`` (resampled with soxr if needed; mono duplicated)."""
    import soundfile as sf

    sr, frames, _ch = _read_info(path)
    a = int(round(start_s * sr))
    need = int(math.ceil(n_out * sr / SR)) + 64
    with sf.SoundFile(str(path)) as f:
        f.seek(min(a, frames))
        x = f.read(max(0, min(need, frames - a)), dtype="float32", always_2d=True)
    if sr != SR and len(x):
        import soxr

        x = soxr.resample(x, sr, SR).astype(np.float32)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    x = x[:n_out].astype(np.float64)
    if len(x) < n_out:
        x = np.pad(x, ((0, n_out - len(x)), (0, 0)))
    return x


def category_audio(song: SongInfo, start_s: float, end_s: float) -> dict[str, np.ndarray]:
    """{category: (n, 2) float64} stem sums of one window (unity gain)."""
    n = int(round((end_s - start_s) * SR))
    out: dict[str, np.ndarray] = {}
    for s in song.stems:
        x = read_segment(s.path, start_s, n)
        out[s.category] = out[s.category] + x if s.category in out else x
    return out


def condition_audio(cat_audio: Mapping[str, np.ndarray], gain: np.ndarray, cats: Iterable[str]) -> np.ndarray:
    """(n, 2) float32 = clip(sum of the categories x gain, ceiling)."""
    cats = [c for c in cats if c in cat_audio]
    n = len(gain)
    x = np.zeros((n, 2))
    for c in cats:
        x += cat_audio[c]
    ceiling = 10 ** (CEILING_DBTP / 20)
    return np.clip(x * gain[:, None], -ceiling, ceiling).astype(np.float32)


def pcm_sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, dtype="<f4")).tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------ separation


def batch_layout(lengths: Sequence[int], *, chunk: int, step: int, max_samples: int) -> list[list[tuple[int, int]]]:
    """Group segments into batches of (index, offset): every offset is a multiple of ``step`` with >= ``chunk``
    samples of silence before each segment and after the last, so the chunked overlap-add sees each segment
    exactly as it would alone (same chunk grid relative to the segment, only zeros next to it)."""
    if chunk <= 0 or step <= 0 or chunk % step:
        raise ValueError("chunk must be a positive multiple of step")
    batches: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    pos = chunk
    for i, n in enumerate(lengths):
        slot = -(-(int(n) + chunk) // step) * step
        if cur and pos + slot > max_samples:
            batches.append(cur)
            cur, pos = [], chunk
        cur.append((i, pos))
        pos += slot
    if cur:
        batches.append(cur)
    return batches


def batch_total(lengths: Sequence[int], layout: Sequence[tuple[int, int]], *, chunk: int, step: int) -> int:
    last_i, last_off = layout[-1]
    return last_off + -(-(int(lengths[last_i]) + chunk) // step) * step


def _separate(mix_wav: Path, out_dir: Path, cfg: Any, *, force_rung: str) -> Any:
    from bandscribe.sep.run import separate

    return separate(mix_wav, out_dir, cfg, force_rung=force_rung)


def _ms(views: list[dict[str, Any]], params: Mapping[str, Any], cfg: Any, *, work_dir: Path, force_rung: str,
        cache: Any = None) -> tuple[dict[str, dict], dict[str, Any]]:
    from bandscribe.amt import cache as amt_cache
    from bandscribe.amt import stage as S

    cache = cache or amt_cache.AmtCache(amt_cache.eval_cache_root())
    return S.transcribe_views_ms(views, params, cfg, work_dir=work_dir, cache=cache, job=None, force_rung=force_rung)


def _energy_db(x: np.ndarray) -> np.ndarray:
    from bandscribe.amt.experiments import energy_db

    return energy_db(x, SR)


def _sep_key(sha: str, sep_params: Mapping[str, Any], rung: str) -> str:
    blob = json.dumps({"pcm": sha, "sep": dict(sep_params), "rung": rung, "v": CODE_VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


@dataclass
class SepViews:
    key: str
    dir: Path
    guitar_mono: Path
    guitar_sha: str
    piano_other_mono: Path
    piano_other_sha: str
    energy: dict[str, np.ndarray]
    rung: str

    @staticmethod
    def load(d: Path) -> SepViews | None:
        done = d / "done.json"
        if not done.is_file():
            return None
        doc = atomic.read_json(done)
        with np.load(d / "energy.npz") as z:
            energy = {k: np.array(z[k]) for k in z.files}
        return SepViews(doc["key"], d, d / "guitar_mono.wav", doc["guitar_sha"], d / "piano_other_mono.wav",
                        doc["piano_other_sha"], energy, doc["rung"])


def _write_views(d: Path, key: str, mix: np.ndarray, stems: Mapping[str, np.ndarray], rung: str) -> SepViews:
    """The ``stems`` stage's derived data for one separated condition: energy (10 Hz dB) and the two mono views."""
    from bandscribe.amt.reftx import write_mono_f32
    from bandscribe.audio import to_mono
    from bandscribe.sep.derive import write_npz_deterministic

    d.mkdir(parents=True, exist_ok=True)
    energy = {"mix": _energy_db(mix)}
    for name, x in stems.items():
        energy[name] = _energy_db(x)
    write_npz_deterministic(d / "energy.npz", {k: v.astype(np.float32) for k, v in sorted(energy.items())})
    g_sha = write_mono_f32(d / "guitar_mono.wav", to_mono(stems["guitar"]), SR)
    po_sha = write_mono_f32(d / "piano_other_mono.wav", to_mono(stems["piano"] + stems["other"]), SR)
    atomic.write_json(d / "done.json", {"format": "bandscribe.distractor_sep/1", "key": key, "guitar_sha": g_sha,
                                        "piano_other_sha": po_sha, "rung": rung})
    return SepViews.load(d)  # type: ignore[return-value]


def separate_conditions(items: Sequence[tuple[str, np.ndarray]], cfg: Any, *, force_rung: str,
                        cache_root: Path | None = None, batch_s: float = SEP_BATCH_S,
                        progress: Callable[[str], None] | None = None) -> tuple[dict[str, SepViews], dict[str, Any]]:
    """{pcm sha: SepViews} for (pcm sha, (n, 2) float32 audio) items; cached ones are not separated again."""
    from bandscribe import audio as AU
    from bandscribe.datasets import store
    from bandscribe.sep.run import STEM_ORDER, sep_params

    params = sep_params(cfg)
    base = Path(cache_root or root_dir()) / "sep"
    out: dict[str, SepViews] = {}
    todo: list[tuple[str, np.ndarray, str]] = []
    seen: set[str] = set()
    for sha, x in items:
        if sha in seen:
            continue
        seen.add(sha)
        key = _sep_key(sha, params, force_rung)
        hit = SepViews.load(base / key)
        if hit is not None:
            out[sha] = hit
        else:
            todo.append((sha, x, key))
    info: dict[str, Any] = {"cached": len(out), "separated": len(todo), "batches": 0, "gpu_s": 0.0, "waited_s": 0.0,
                            "audio_s": 0.0, "process_s": 0.0, "rung": force_rung}
    if not todo:
        return out, info
    chunk = int(force_rung.split("=", 1)[1]) if force_rung.startswith("chunk=") else int(params["chunk_ladder"][0])
    step = chunk // int(params["num_overlap"])
    layout = batch_layout([len(x) for _s, x, _k in todo], chunk=chunk, step=step, max_samples=int(batch_s * SR))
    for b, lay in enumerate(layout):
        total = batch_total([len(x) for _s, x, _k in todo], lay, chunk=chunk, step=step)
        mix = np.zeros((total, 2), dtype=np.float32)
        for i, off in lay:
            x = todo[i][1]
            mix[off:off + len(x)] = x
        bdir = Path(store.make_scratch_dir(base.parent / "tmp", f"sepbatch{b}"))
        try:
            AU.write_f32(bdir / "mix.wav", mix, SR)
            del mix
            if progress:
                progress(f"분리 {b + 1}/{len(layout)}: 조건 {len(lay)}개, {total / SR / 60:.1f}분")
            t0 = time.perf_counter()
            res = _separate(bdir / "mix.wav", bdir / "sep", cfg, force_rung=force_rung)
            info["gpu_s"] += time.perf_counter() - t0
            doc = getattr(res, "sep_json", {}) or {}
            info["waited_s"] += float((getattr(res, "gpu_run", {}) or {}).get("waited_s") or 0.0)
            info["audio_s"] += float(doc.get("audio_s") or 0.0)
            info["process_s"] += float(doc.get("process_s") or 0.0)
            rung = str((getattr(res, "rung", {}) or {}).get("label") or force_rung)
            stems_full = {}
            import soundfile as sf

            for name in STEM_ORDER:
                y, _sr = sf.read(str(res.stems[name]), dtype="float32", always_2d=True)
                stems_full[name] = y
            for i, off in lay:
                sha, x, key = todo[i]
                seg = {name: y[off:off + len(x)] for name, y in stems_full.items()}
                out[sha] = _write_views(base / key, key, x, seg, rung)
            del stems_full
            info["batches"] += 1
        finally:
            store.remove_tree(bdir)
    return out, info


# ----------------------------------------------------------------------------------------- the run plan


@dataclass
class CondRun:
    window_item: str
    cond: Condition
    sha: str
    n: int
    sep: SepViews | None = None
    presence: dict[str, Any] | None = None
    masks: dict[str, list[str]] = field(default_factory=dict)        # arm -> mask
    notes: dict[str, list[dict]] = field(default_factory=dict)       # arm -> raw notes (all classes)
    instrumentation: dict[str, Any] | None = None


@dataclass
class WindowRun:
    song: SongInfo
    choice: WindowChoice
    gain: np.ndarray | None = None
    cat_audio: dict[str, np.ndarray] | None = None
    ref_sha: str = ""
    ref_wav: Path | None = None
    ref_notes: list[dict] = field(default_factory=list)
    conds: list[CondRun] = field(default_factory=list)

    @property
    def item(self) -> str:
        return f"{self.song.item}:{self.choice.wid}"

    @property
    def minutes(self) -> float:
        return (self.choice.end_s - self.choice.start_s) / 60.0


def estimate_gpu_s(windows: Sequence[WindowRun], *, sep_cached: set[str], ref_fresh: set[str],
                   null: bool = True) -> float:
    """Rough GPU seconds of the uncached work (conditions with equal audio counted once; see the EST_* constants).
    A condition whose separation is cached counts as fully cached (its transcriptions were made in the same run);
    the decoder-null views are always counted (an upper bound)."""
    shas = {c.sha: c.n for w in windows for c in w.conds if c.sha not in sep_cached}
    sep_audio = sum(shas.values()) / SR
    sep = (sep_audio / EST_SEP_RTF + EST_SEP_OVERHEAD_S * max(1, math.ceil(sep_audio / SEP_BATCH_S))) if shas else 0.0
    ref_audio = sum(w.gain.shape[0] for w in windows if w.ref_sha in ref_fresh) / SR if ref_fresh else 0.0
    ms_audio = sep_audio + ref_audio + (sum(w.gain.shape[0] for w in windows) / SR if null else 0.0)
    presence = 10.0 * len(shas)
    ms = ms_audio * EST_MS_S_PER_S + presence * EST_PRESENCE_S_PER_S + (2 * EST_MS_OVERHEAD_S if ms_audio else 0.0)
    return sep + ms


def _view(vid: str, wav: Path, sha: str, mask: list[str] | None, segments: Any = None) -> dict[str, Any]:
    v = {"id": vid, "wav": wav, "pcm_sha256": sha, "instruments": mask}
    if segments:
        v["segments"] = segments
    return v


def _ms_cached(views: Sequence[Mapping[str, Any]], params: Mapping[str, Any], rung: str) -> set[str]:
    """Ids of views already in the MuScriptor eval cache (no worker needed)."""
    from bandscribe.amt import cache as amt_cache
    from bandscribe.amt import instruments as I
    from bandscribe.amt import stage as S

    try:
        sha = S._ms_model_sha(params)
    except Exception:  # noqa: BLE001 - weights missing: nothing is cached, the real call reports it
        return set()
    cache = amt_cache.AmtCache(amt_cache.eval_cache_root())
    cp = S._ms_cache_params(params)
    out = set()
    for v in views:
        mask = None if v.get("instruments") is None else I.sort_mt3(v["instruments"])
        key = amt_cache.amt_key("muscriptor", params.get("muscriptor_version"), sha, v["pcm_sha256"], mask,
                                S._view_cparams(cp, v), S.CACHE_CODE_VERSION, rung)
        if cache.get(key) is not None:
            out.add(str(v["id"]))
    return out


def _presence_selection(sep: SepViews, n: int, cfg: Any) -> dict[str, Any]:
    from bandscribe.amt import presence as P
    from bandscribe.amt import stage as S
    from bandscribe.amt.experiments import pseudo_bars

    dur = n / SR
    profile = str(S.amt_value(cfg, "run.profile"))
    settings = S.presence_settings(cfg, profile if profile in ("fast", "quality", "eval") else "quality")
    return P.select_windows(bars=pseudo_bars(dur), sections=None, energy=sep.energy, fps=FPS, song_s=dur,
                            **{k: settings[k] for k in P.DEFAULTS})


def _instrumentation(cr: CondRun, sel: Mapping[str, Any], presence_notes: list[dict] | None, cfg: Any) -> dict:
    from bandscribe.amt import instrumentation as INS
    from bandscribe.amt import presence as P
    from bandscribe.amt import stage as S
    from bandscribe.amt.experiments import pseudo_bars

    segs = P.spans(sel)
    pres = {"mode": "sampled", "skipped": None if segs else (sel.get("reason") or "no_window"),
            "notes": presence_notes or [], "spans": segs or None,
            "window_sections": [w["section"] for w in sel.get("windows") or []] or None,
            "summary": {"windows": segs, "transcribed_s": sel.get("transcribed_s")}}
    prm = S.instr_params(cfg)
    return INS.estimate(profile=str(prm["profile"]), bars=pseudo_bars(cr.n / SR), mix_notes=None,
                        energy=cr.sep.energy if cr.sep else {}, sections=[], hint=prm["hints_instruments"],
                        params=prm, guitar_view="guitar_mono", guitar_mask=str(prm["guitar_mask"]),
                        piano_other_mode=str(prm["piano_other_instruments"]), presence=pres)


# ------------------------------------------------------------------------------------------- scoring


def _arrays(notes: Sequence[Mapping[str, Any]], latency_s: float) -> np.ndarray:
    from bandscribe.amt import instruments as I
    from bandscribe.amt.experiments import note_arrays

    iv, pp = note_arrays(notes, classes=I.GUITAR_CLASSES, latency_s=latency_s)
    return np.column_stack([iv, pp]) if len(pp) else np.zeros((0, 3))


def match(ref: np.ndarray, est: np.ndarray) -> list[tuple[int, int]]:
    from bandscribe.eval.matching import match_note_pairs

    if not len(ref) or not len(est):
        return []
    return match_note_pairs(ref[:, :2], ref[:, 2], est[:, :2], est[:, 2], onset_tol=ONSET_TOL,
                            pitch_tol_cents=PITCH_TOL_CENTS)


@dataclass
class Score:
    tp: int
    fp: int
    fn: int
    minutes: float
    fp_causes: Counter
    fn_causes: Counter
    fp_dominant: Counter
    fn_dominant: Counter
    fp_pairs: Counter  # (cause, dominant) -> n

    def counts(self) -> dict[str, float]:
        c: dict[str, float] = {"tp": self.tp, "fp": self.fp, "fn": self.fn, "min": self.minutes}
        for k, v in self.fp_causes.items():
            c[f"fp:{k}"] = v
        for k, v in self.fn_causes.items():
            c[f"fn:{k}"] = v
        return c


def score(ref: np.ndarray, est: np.ndarray, bands: Mapping[str, A.BandEnergy], *, floor: float,
          minutes: float, mix: A.BandEnergy | None = None) -> Score:
    pairs = match(ref, est)
    mr = {i for i, _ in pairs}
    me = {j for _, j in pairs}
    fpc: Counter = Counter()
    fnc: Counter = Counter()
    fpd: Counter = Counter()
    fnd: Counter = Counter()
    fpp: Counter = Counter()
    for j in range(len(est)):
        if j in me:
            continue
        c = A.attribute_fp(tuple(est[j]), ref, bands, guitar=T.GUITAR, floor=floor, mix=mix)
        fpc[c.label] += 1
        fpd[c.dominant or A.UNKNOWN] += 1
        fpp[(c.label, c.dominant or A.UNKNOWN)] += 1
    for i in range(len(ref)):
        if i in mr:
            continue
        c = A.attribute_fn(tuple(ref[i]), est, bands, guitar=T.GUITAR, floor=floor, mix=mix)
        fnc[c.label] += 1
        fnd[c.dominant or "none"] += 1
    return Score(len(pairs), len(est) - len(pairs), len(ref) - len(pairs), minutes, fpc, fnc, fpd, fnd, fpp)


# --------------------------------------------------------------------------------------------- stats


def fp_per_min(c: Mapping[str, np.ndarray]) -> np.ndarray:
    m = np.asarray(c["min"], dtype=float)
    return np.where(m > 0, np.asarray(c["fp"], dtype=float) / np.where(m > 0, m, 1), np.nan)


def fn_per_min(c: Mapping[str, np.ndarray]) -> np.ndarray:
    m = np.asarray(c["min"], dtype=float)
    return np.where(m > 0, np.asarray(c["fn"], dtype=float) / np.where(m > 0, m, 1), np.nan)


def per_min(key: str) -> Callable[[Mapping[str, np.ndarray]], np.ndarray]:
    def f(c: Mapping[str, np.ndarray]) -> np.ndarray:
        m = np.asarray(c["min"], dtype=float)
        v = np.asarray(c[key], dtype=float) if key in c else np.zeros_like(m)
        return np.where(m > 0, v / np.where(m > 0, m, 1), np.nan)
    return f


STATS_MAIN: dict[str, Callable] = {"f1": stats.f1_points, "precision": stats.precision_points,
                                   "recall": stats.recall_points, "fp_per_min": fp_per_min, "fn_per_min": fn_per_min}


def _items(scores: Mapping[tuple[str, str, str], Score], windows: Sequence[WindowRun], cond: str,
           arm: str) -> list[stats.ItemCounts]:
    out = []
    for w in windows:
        s = scores.get((w.item, cond, arm))
        if s is not None:
            out.append(stats.ItemCounts(w.item, w.song.group, s.counts()))
    return out


def _padkeys(a: list[stats.ItemCounts], keys: Iterable[str]) -> list[stats.ItemCounts]:
    keys = list(keys)
    return [stats.ItemCounts(i.item, i.group, {**{k: 0.0 for k in keys}, **dict(i.counts)}) for i in a]


def category_deltas(scores: Mapping[tuple[str, str, str], Score], windows: Sequence[WindowRun], *, arm: str,
                    n: int, seed: int) -> dict[str, dict[str, Any]]:
    """Per added category (and ``full``): paired song-block bootstrap of candidate - base for the main stats and
    the per-cause FP / FN rates."""
    conds = sorted({c.cond.name for w in windows for c in w.conds if c.cond.name != "base"}, key=cond_order)
    out: dict[str, dict[str, Any]] = {}
    for cond in conds:
        ws = [w for w in windows if any(c.cond.name == cond for c in w.conds)
              and (w.item, cond, arm) in scores and (w.item, "base", arm) in scores]
        if not ws:
            continue
        a = _items(scores, ws, "base", arm)
        b = _items(scores, ws, cond, arm)
        keys = sorted({k for i in a + b for k in i.counts})
        a, b = _padkeys(a, keys), _padkeys(b, keys)
        res: dict[str, Any] = {"n_items": len(ws), "n_groups": len({w.song.group for w in ws}),
                               "songs": sorted({w.song.track_id for w in ws})}
        for name, fn in STATS_MAIN.items():
            d, lo, hi = stats.paired_bootstrap(a, b, fn, n=n, seed=seed)
            pb = stats.block_bootstrap(a, fn, n=0)[0]
            pc = stats.block_bootstrap(b, fn, n=0)[0]
            res[name] = {"base": pb, "cand": pc, "delta": d, "ci": [lo, hi]}
        causes = {}
        for k in keys:
            if k.startswith(("fp:", "fn:")):
                d, lo, hi = stats.paired_bootstrap(a, b, per_min(k), n=n, seed=seed)
                causes[k] = {"base": float(stats.block_bootstrap(a, per_min(k), n=0)[0]),
                             "cand": float(stats.block_bootstrap(b, per_min(k), n=0)[0]), "delta": d, "ci": [lo, hi]}
        res["causes"] = causes
        cat = cond[1:] if cond.startswith("+") else ""
        if cat:  # the same deltas per minute of the category's own activity (sparse sounds are not diluted)
            act_items = []
            for w, ia, ib in zip(ws, a, b):
                act = float(w.choice.shares.get(cat, 0.0)) * w.minutes
                act_items.append(stats.ItemCounts(w.item, w.song.group, {
                    "d_fp": ib.counts["fp"] - ia.counts["fp"], "d_fn": ib.counts["fn"] - ia.counts["fn"], "act": act}))
            res["active_min"] = sum(i.counts["act"] for i in act_items)
            for key, name in (("d_fp", "fp_per_active_min"), ("d_fn", "fn_per_active_min")):
                def f(c: Mapping[str, np.ndarray], key: str = key) -> np.ndarray:
                    m = np.asarray(c["act"], dtype=float)
                    return np.where(m > 0, np.asarray(c[key], dtype=float) / np.where(m > 0, m, 1), np.nan)
                p, lo, hi = stats.block_bootstrap(act_items, f, n=n, seed=seed)
                res[name] = {"delta": p, "ci": [lo, hi]}
        out[cond] = res
    return out


def pooled(scores: Mapping[tuple[str, str, str], Score], windows: Sequence[WindowRun], cond: str, arm: str, *,
           n: int, seed: int) -> dict[str, Any]:
    its = _items(scores, windows, cond, arm)
    if not its:
        return {}
    keys = sorted({k for i in its for k in i.counts})
    its = _padkeys(its, keys)
    out = {"n_items": len(its), "n_groups": len({i.group for i in its})}
    for name, fn in STATS_MAIN.items():
        p, lo, hi = stats.block_bootstrap(its, fn, n=n, seed=seed)
        out[name] = {"value": p, "ci": [lo, hi]}
    out["tp"] = sum(i.counts["tp"] for i in its)
    out["fp"] = sum(i.counts["fp"] for i in its)
    out["fn"] = sum(i.counts["fn"] for i in its)
    out["minutes"] = sum(i.counts["min"] for i in its)
    return out


# ------------------------------------------------------------------------------------------------- run


def cond_order(name: str) -> tuple:
    """Report order of conditions: base, +category in CATEGORIES order, full, guitar_alone."""
    if name == "base":
        return (0, 0, name)
    if name.startswith("+"):
        return (1, T.CATEGORIES.index(name[1:]) if name[1:] in T.CATEGORIES else 99, name)
    return ({"full": 2, NULL_COND: 3}.get(name, 4), 0, name)


NULL_COND = "base_null"
NULL_RMS_DBFS = -90.0


def null_view(x: np.ndarray, item: str) -> np.ndarray:
    """The decoder-null input: a guitar view plus seeded white noise at -90 dBFS RMS (about 16-bit quantisation
    noise, inaudible under a -8 LUFS master). Any change in the transcription is MuScriptor's own sensitivity to an
    input change nobody can hear: the yardstick for every other delta."""
    import zlib

    rng = np.random.Generator(np.random.PCG64(zlib.crc32(f"null:{item}".encode("utf-8"))))
    noise = rng.normal(0.0, 10 ** (NULL_RMS_DBFS / 20), len(x))
    return (np.asarray(x, dtype=np.float64) + noise).astype(np.float32)


def _null_vid(sha: str, mask: Sequence[str]) -> str:
    return f"n{sha[:12]}{hashlib.sha256('+'.join(mask).encode()).hexdigest()[:6]}"


def _row(**kw: Any) -> dict[str, Any]:
    base = {"suite": SUITE, "dataset": "cambridge_mt", "line": "guitar"}
    base.update(kw)
    return base


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg.get(key, default) if cfg is not None else default
    except Exception:  # noqa: BLE001
        return default


def run(run_dir: Path, cfg: Any, args: Mapping[str, Any] | None = None,
        progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The whole measurement into ``run_dir``; returns the summary (also written as summary.json)."""
    from bandscribe.amt import instruments as I
    from bandscribe.amt import stage as S
    from bandscribe.amt.experiments import ms_rung, sep_top_rung
    from bandscribe.amt.reftx import write_mono_f32
    from bandscribe.audio import to_mono
    from bandscribe.eval import runs
    from bandscribe.eval.remix import master_gain

    args = dict(args or {})
    say = progress or (lambda s: log.info("%s", s))
    run_dir = Path(run_dir)
    croot = Path(args.get("cache_root") or root_dir())
    n_boot = int(_cfg_get(cfg, "eval.bootstrap_iterations", stats.DEFAULT_ITERATIONS))
    seed = int(_cfg_get(cfg, "eval.seed", stats.DEFAULT_SEED))
    max_windows = int(args.get("max_windows", 2))
    window_s = float(args.get("window_s", WINDOW_S))
    budget_min = float(args.get("gpu_budget_min", 60.0))
    only = args.get("songs")
    if isinstance(only, str):
        only = [s.strip() for s in only.split(",") if s.strip()]
    songs = load_songs(str(args.get("dataset", "cambridge_mt")), only=only)
    if not songs:
        raise DistractorError("평가할 멀티트랙이 없습니다 (bandscribe data fetch cambridge_mt --part all).")

    # ---- 1. windows and conditions (CPU) ----
    windows: list[WindowRun] = []
    skipped: dict[str, str] = {}
    unknown: dict[str, list[str]] = {}
    for song in songs:
        if song.unknown:
            unknown[song.track_id] = song.unknown
        if not any(c in T.DISTRACTORS for c in song.categories()):
            skipped[song.track_id] = "방해 소리 스템 없음"
            continue
        if not any(c in T.GUITAR for c in song.categories()):
            skipped[song.track_id] = "기타 스템 없음"
            continue
        say(f"구간 고르기: {song.track_id}")
        power = category_power(song, cache_root=croot)
        choices = select_windows(power, window_s=window_s, max_windows=max_windows)
        if not choices:
            skipped[song.track_id] = "기타와 방해 소리가 함께 나오는 구간 없음"
            continue
        for ch in choices:
            windows.append(WindowRun(song, ch))

    params = S.ms_params(cfg)
    ms_top, sep_top = ms_rung(params), sep_top_rung(cfg)
    lat = float(params["latency_s"])
    guitar_alone = bool(args.get("guitar_alone", False))
    for w in windows:
        cats = w.song.categories()
        w.cat_audio = category_audio(w.song, w.choice.start_s, w.choice.end_s)
        w.gain = master_gain(sum(w.cat_audio.values()), SR, lufs=MASTER_LUFS, ceiling_dbtp=CEILING_DBTP)
        for cond in conditions_for(cats, w.choice.present, guitar_alone=guitar_alone):
            x = condition_audio(w.cat_audio, w.gain, cond.categories)
            w.conds.append(CondRun(w.item, cond, pcm_sha(x), len(x)))
        g = sum((w.cat_audio[c] for c in T.GUITAR if c in w.cat_audio), np.zeros_like(w.gain[:, None].repeat(2, 1)))
        gm = to_mono((g * w.gain[:, None]).astype(np.float32))
        w.ref_sha = pcm_sha(gm.reshape(-1, 1))
        w.ref_wav = croot / "views" / f"ref_{w.ref_sha[:20]}.wav"
        if not w.ref_wav.is_file():
            write_mono_f32(w.ref_wav, gm, SR)

    # ---- budget (only uncached work counts) ----
    from bandscribe.sep.run import sep_params

    sp = sep_params(cfg)
    sep_cached = {c.sha for w in windows for c in w.conds
                  if SepViews.load(croot / "sep" / _sep_key(c.sha, sp, sep_top)) is not None}
    ref_views = [_view(f"r{w.ref_sha[:16]}", w.ref_wav, w.ref_sha, I.mask_guitar_only()) for w in windows]
    cached_ids = _ms_cached(ref_views, params, ms_top)
    ref_fresh = {str(v["pcm_sha256"]) for v in ref_views if v["id"] not in cached_ids}
    est = estimate_gpu_s(windows, sep_cached=sep_cached, ref_fresh=ref_fresh, null=bool(args.get("null", True)))
    while est / 60.0 > budget_min and any(w.choice.wid == "w2" for w in windows):
        drop = [w for w in windows if w.choice.wid == "w2"][-1]
        windows.remove(drop)
        ref_views = [v for v in ref_views if v["pcm_sha256"] != drop.ref_sha]
        say(f"GPU 예산({budget_min:.0f}분)을 넘어 두 번째 구간을 뺍니다: {drop.item}")
        est = estimate_gpu_s(windows, sep_cached=sep_cached, ref_fresh=ref_fresh, null=bool(args.get("null", True)))
    plan = {"windows": [{"item": w.item, "start_s": w.choice.start_s, "end_s": w.choice.end_s,
                         "guitar_share": w.choice.guitar_share, "shares": w.choice.shares, "present": w.choice.present,
                         "conditions": {c.cond.name: list(c.cond.categories) for c in w.conds}} for w in windows],
            "skipped": skipped, "unknown_stem_names": unknown, "gpu_estimate_min": round(est / 60.0, 1),
            "stems": {s.track_id: {st.name: st.category for st in s.stems} for s in songs}}
    atomic.write_json(run_dir / "windows.json", runs.round_floats(plan))
    say(f"구간 {len(windows)}개, 조건 {sum(len(w.conds) for w in windows)}개, 예상 GPU 약 {est / 60:.0f}분")
    if args.get("dry_run"):
        return {"suite": SUITE, "dry_run": True, **plan}
    if est / 60.0 > budget_min and not args.get("allow_over_budget"):
        raise DistractorError(f"예상 GPU 시간 {est / 60:.0f}분이 예산 {budget_min:.0f}분을 넘습니다. "
                              "--arg gpu_budget_min=<분> 으로 늘리거나 --arg songs=<곡,...> 으로 줄이세요.")

    gpu = {"sep_s": 0.0, "ms_s": 0.0, "waited_s": 0.0}
    # ---- 2. separation (GPU, batched, cached) ----
    t0 = time.perf_counter()
    items = []
    for w in windows:
        for c in w.conds:
            if c.sha not in sep_cached:
                items.append((c.sha, condition_audio(w.cat_audio, w.gain, c.cond.categories)))
            else:
                items.append((c.sha, np.zeros((0, 2), dtype=np.float32)))
    seps, sep_info = separate_conditions(items, cfg, force_rung=sep_top, cache_root=croot, progress=say)
    del items
    gpu["sep_s"] = sep_info["gpu_s"]
    gpu["waited_s"] += sep_info["waited_s"]
    for w in windows:
        for c in w.conds:
            c.sep = seps[c.sha]

    # ---- 3. MuScriptor call 1: references, guitar-only views, presence windows ----
    from bandscribe.amt import presence as P

    po_mask = I.mask_piano_other(str(S.amt_value(cfg, "amt.piano_other_instruments")))
    sels: dict[str, dict] = {}
    views1: dict[str, dict] = {v["id"]: v for v in ref_views}
    for w in windows:
        for c in w.conds:
            gid = f"g{c.sep.guitar_sha[:16]}"
            views1[gid] = _view(gid, c.sep.guitar_mono, c.sep.guitar_sha, I.mask_guitar_only())
            sel = _presence_selection(c.sep, c.n, cfg)
            sels[c.sha] = sel
            segs = P.spans(sel)
            if segs:
                pid = f"p{c.sep.piano_other_sha[:16]}"
                views1[pid] = _view(pid, c.sep.piano_other_mono, c.sep.piano_other_sha, po_mask, segs)
    say(f"MuScriptor 1차: 보기 {len(views1)}개(참조 {len(ref_views)}개)")
    t0 = time.perf_counter()
    raws1, info1 = _ms(list(views1.values()), params, cfg, work_dir=run_dir / "work" / "ms1" / "_worker",
                       force_rung=ms_top)
    if info1.get("worker_output") is not None:
        gpu["ms_s"] += time.perf_counter() - t0
        gpu["waited_s"] += float(info1.get("waited_s") or 0.0)
    ms_rung_label = str(info1.get("rung_label") or ms_top)
    for w in windows:
        w.ref_notes = raws1[f"r{w.ref_sha[:16]}"].get("notes") or []

    # ---- 4. instrumentation per condition (CPU), MuScriptor call 2 for production masks that differ ----
    views2: dict[str, dict] = {}
    guitar_only = I.mask_guitar_only()
    for w in windows:
        for c in w.conds:
            sel = sels[c.sha]
            pid = f"p{c.sep.piano_other_sha[:16]}"
            pnotes = (raws1.get(pid) or {}).get("notes") if P.spans(sel) else []
            c.instrumentation = _instrumentation(c, sel, pnotes, cfg)
            mask = list(c.instrumentation["masks"]["guitar_pass"] or [])
            c.masks = {"production": mask, "guitar_only": guitar_only}
            c.notes["guitar_only"] = raws1[f"g{c.sep.guitar_sha[:16]}"].get("notes") or []
            if I.sort_mt3(mask) != I.sort_mt3(guitar_only):
                vid = f"m{c.sep.guitar_sha[:12]}{hashlib.sha256('+'.join(mask).encode()).hexdigest()[:6]}"
                views2[vid] = _view(vid, c.sep.guitar_mono, c.sep.guitar_sha, mask)
    raws2: dict[str, dict] = {}
    if views2:
        say(f"MuScriptor 2차(제품 마스크가 기타만이 아닌 조건): 보기 {len(views2)}개")
        t0 = time.perf_counter()
        raws2, info2 = _ms(list(views2.values()), params, cfg, work_dir=run_dir / "work" / "ms2" / "_worker",
                           force_rung=ms_top)
        if info2.get("worker_output") is not None:
            gpu["ms_s"] += time.perf_counter() - t0
            gpu["waited_s"] += float(info2.get("waited_s") or 0.0)
    for w in windows:
        for c in w.conds:
            mask = c.masks["production"]
            if I.sort_mt3(mask) == I.sort_mt3(guitar_only):
                c.notes["production"] = c.notes["guitar_only"]
            else:
                vid = f"m{c.sep.guitar_sha[:12]}{hashlib.sha256('+'.join(mask).encode()).hexdigest()[:6]}"
                c.notes["production"] = raws2[vid].get("notes") or []

    # ---- 4b. decoder null (diagnostic, added after the first full run): the base guitar view + inaudible noise ----
    if args.get("null", True):
        import soundfile as sf

        views3: dict[str, dict] = {}
        nulls: list[tuple[WindowRun, CondRun, str]] = []
        for w in windows:
            base = next((c for c in w.conds if c.cond.name == "base"), None)
            if base is None or base.sep is None:
                continue
            x, _sr = sf.read(str(base.sep.guitar_mono), dtype="float32")
            y = null_view(x, w.item)
            sha = pcm_sha(y.reshape(-1, 1))
            path = croot / "views" / f"null_{sha[:20]}.wav"
            if not path.is_file():
                write_mono_f32(path, y, SR)
            nc = CondRun(w.item, Condition(NULL_COND, base.cond.categories), base.sha, base.n, sep=base.sep,
                         masks=dict(base.masks), instrumentation=base.instrumentation)
            for mask in {tuple(m) for m in nc.masks.values()}:
                vid = _null_vid(sha, list(mask))
                views3[vid] = _view(vid, path, sha, list(mask))
            nulls.append((w, nc, sha))
        if views3:
            say(f"MuScriptor 잡음 기준(null): 보기 {len(views3)}개")
            t0 = time.perf_counter()
            raws3, info3 = _ms(list(views3.values()), params, cfg, work_dir=run_dir / "work" / "ms3" / "_worker",
                               force_rung=ms_top)
            if info3.get("worker_output") is not None:
                gpu["ms_s"] += time.perf_counter() - t0
                gpu["waited_s"] += float(info3.get("waited_s") or 0.0)
            for w, nc, sha in nulls:
                for arm, mask in nc.masks.items():
                    nc.notes[arm] = raws3[_null_vid(sha, mask)].get("notes") or []
                w.conds.append(nc)

    # ---- 5. scoring, causes, leakage (CPU) ----
    say("채점·원인 분석")
    scores: dict[tuple[str, str, str], Score] = {}
    leaks: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    rows: list[dict[str, Any]] = []
    rung = f"{sep_info.get('rung') or sep_top}|{ms_rung_label}"
    group_of = {c: ("guitar" if c in T.GUITAR else c) for c in T.CATEGORIES}
    for w in windows:
        mono = {c: to_mono((x * w.gain[:, None]).astype(np.float32)) for c, x in w.cat_audio.items()}
        bands = {c: A.band_energy(x) for c, x in mono.items()}
        mix_be = A.band_energy(sum(mono.values()))
        floor = A.floor_power(bands, mix_be.frame_power_p95)
        ref = _arrays(w.ref_notes, lat)
        for c in w.conds:
            cb = {k: v for k, v in bands.items() if k in c.cond.categories}
            cmix = A.band_energy(sum(mono[k] for k in c.cond.categories if k in mono))
            common = {"item": w.item, "group": w.song.group, "scenario": c.cond.name, "section": w.choice.wid,
                      "rung": rung}
            for arm in ARMS:
                est = _arrays(c.notes[arm], lat)
                s = score(ref, est, cb, floor=floor, minutes=w.minutes, mix=cmix)
                scores[(w.item, c.cond.name, arm)] = s
                prf_f1 = 2 * s.tp / (2 * s.tp + s.fp + s.fn) if (2 * s.tp + s.fp + s.fn) else 1.0
                rows.append(_row(system=arm, metric="note_f1_onset50", value=prf_f1, tp=s.tp, fp=s.fp, fn=s.fn,
                                 n=s.tp + s.fn, note="+".join(c.masks[arm]), **common))
                rows.append(_row(system=arm, metric="fp_per_min", value=s.fp / w.minutes, n=s.fp, **common))
                rows.append(_row(system=arm, metric="fn_per_min", value=s.fn / w.minutes, n=s.fn, **common))
                rows.append(_row(system=arm, metric="minutes", value=w.minutes, **common))
                rows.append(_row(system=arm, metric="mask_non_guitar",
                                 value=sum(1 for m in c.masks[arm] if m not in I.GUITAR_CLASSES), **common))
                dropped = sum(1 for n_ in c.notes[arm] if n_.get("instrument") not in I.GUITAR_CLASSES)
                rows.append(_row(system=arm, metric="non_guitar_dropped", value=dropped, **common))
                for k, v in sorted(s.fp_causes.items()):
                    rows.append(_row(system=arm, metric=f"fp_cause:{k}", value=v, **common))
                for k, v in sorted(s.fn_causes.items()):
                    rows.append(_row(system=arm, metric=f"fn_cause:{k}", value=v, **common))
                for k, v in sorted(s.fp_dominant.items()):
                    rows.append(_row(system=arm, metric=f"fp_dominant:{k}", value=v, **common))
            import soundfile as sf

            gsep, _sr = sf.read(str(c.sep.guitar_mono), dtype="float32")
            lk = A.leakage({k: mono[k] for k in c.cond.categories if k in mono}, gsep, group=group_of)
            leaks[(w.item, c.cond.name)] = lk
            for k, v in sorted(lk.items()):
                rows.append(_row(system="separation", metric=f"leak_db:{k}", value=A.ratio_db(v["leak"]),
                                 n=v["bins"], **common))
                rows.append(_row(system="separation", metric=f"stem_share:{k}", value=v["share"], **common))
        w.cat_audio = None  # free the window's audio

    # ---- 6. statistics and outputs ----
    summary: dict[str, Any] = {"suite": SUITE, "code_version": CODE_VERSION, "rung": rung, "arms": list(ARMS),
                               "n_windows": len(windows), "n_songs": len({w.song.track_id for w in windows}),
                               "windows": plan["windows"], "skipped": skipped, "unknown_stem_names": unknown,
                               "gpu": {k: round(v, 1) for k, v in gpu.items()}, "sep": sep_info,
                               "gpu_estimate_min": plan["gpu_estimate_min"]}
    summary["conditions"] = {}
    summary["deltas"] = {}
    for arm in ARMS:
        names = sorted({c.cond.name for w in windows for c in w.conds}, key=cond_order)
        summary["conditions"][arm] = {cn: pooled(scores, windows, cn, arm, n=n_boot, seed=seed) for cn in names}
        summary["deltas"][arm] = category_deltas(scores, windows, arm=arm, n=n_boot, seed=seed)
    summary["leakage"] = _leak_summary(leaks, windows)
    summary["fp_cause_matrix"] = _cause_matrix(scores, windows, "production", "fp")
    summary["fn_cause_matrix"] = _cause_matrix(scores, windows, "production", "fn")
    summary["fp_cause_vs_dominant"] = _cause_vs_dominant(scores, "production")
    summary["per_window"] = _per_window(scores, windows, leaks)
    summary["masks"] = {f"{w.item}|{c.cond.name}": {"production": c.masks["production"],
                                                     "present": [k for k, v in (c.instrumentation or {}).get(
                                                         "classes", {}).items() if v.get("present")]}
                        for w in windows for c in w.conds}
    summary["null"] = _null_spread(scores, windows)
    summary["ranking"] = fix_first(summary)
    for arm in ARMS:
        for cond, d in summary["deltas"][arm].items():
            for name in STATS_MAIN:
                rows.append(_row(system=arm, item="*", group="*", scenario=cond, metric=f"delta_{name}",
                                 value=d[name]["delta"], ci_low=d[name]["ci"][0], ci_high=d[name]["ci"][1],
                                 n=d["n_items"], rung=rung))
    runs.write_metrics_csv(run_dir / "metrics.csv", rows)
    runs.write_summary(run_dir / "summary.json", summary)
    from bandscribe.eval import distractor_report

    distractor_report.write(run_dir, runs.round_floats(summary))
    return summary


def _leak_summary(leaks: Mapping[tuple[str, str], Mapping[str, Mapping[str, float]]],
                  windows: Sequence[WindowRun]) -> dict[str, Any]:
    """Per category, in the conditions that contain it: mean leak dB and mean guitar-stem share over windows, plus
    the value in its own ``+category`` condition."""
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (item, cond), lk in leaks.items():
        for c, v in lk.items():
            if cond == "full":
                by[c]["full_leak_db"].append(A.ratio_db(v["leak"]))
                by[c]["full_share"].append(v["share"])
            if cond == f"+{c}":
                by[c]["own_leak_db"].append(A.ratio_db(v["leak"]))
                by[c]["own_share"].append(v["share"])
            if cond == "base":
                by[c]["base_leak_db"].append(A.ratio_db(v["leak"]))
                by[c]["base_share"].append(v["share"])
    out: dict[str, Any] = {}
    for c, d in by.items():
        vals: dict[str, Any] = {}
        for k, v in d.items():
            a = np.asarray(v, dtype=float)
            vals[k] = float(np.nanmean(a)) if a.size and np.isfinite(a).any() else None
        vals["n"] = {k: len(v) for k, v in d.items()}
        out[c] = vals
    return out


def _null_spread(scores: Mapping[tuple[str, str, str], Score], windows: Sequence[WindowRun],
                 arm: str = "production") -> dict[str, Any]:
    """Per window: base_null - base (FP/min, FN/min, F1 points) and the median absolute change over windows."""

    def f1(s: Score) -> float:
        den = 2 * s.tp + s.fp + s.fn
        return 100.0 * 2 * s.tp / den if den else 100.0

    rows = []
    for w in windows:
        a, b = scores.get((w.item, "base", arm)), scores.get((w.item, NULL_COND, arm))
        if a is None or b is None:
            continue
        rows.append({"item": w.item, "d_fp_per_min": (b.fp - a.fp) / w.minutes,
                     "d_fn_per_min": (b.fn - a.fn) / w.minutes, "d_f1": f1(b) - f1(a)})
    med = {k: (float(np.median([abs(r[k]) for r in rows])) if rows else None)
           for k in ("d_fp_per_min", "d_fn_per_min", "d_f1")}
    return {"windows": rows, "median_abs": med, "noise_rms_dbfs": NULL_RMS_DBFS}


def _cause_matrix(scores: Mapping[tuple[str, str, str], Score], windows: Sequence[WindowRun], arm: str,
                  kind: str) -> dict[str, dict[str, float]]:
    """{condition: {cause: errors per minute}} pooled over the windows that have the condition."""
    out: dict[str, dict[str, float]] = {}
    conds = sorted({c.cond.name for w in windows for c in w.conds}, key=cond_order)
    for cond in conds:
        tot: Counter = Counter()
        mins = 0.0
        for w in windows:
            s = scores.get((w.item, cond, arm))
            if s is None:
                continue
            tot.update(s.fp_causes if kind == "fp" else s.fn_causes)
            mins += s.minutes
        if mins > 0:
            out[cond] = {k: v / mins for k, v in sorted(tot.items())}
    return out


def _cause_vs_dominant(scores: Mapping[tuple[str, str, str], Score], arm: str) -> dict[str, dict[str, int]]:
    tot: Counter = Counter()
    for (_item, _cond, a), s in scores.items():
        if a == arm:
            tot.update(s.fp_pairs)
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for (cause, dom), v in sorted(tot.items()):
        out[cause][dom] = v
    return dict(out)


def _per_window(scores: Mapping[tuple[str, str, str], Score], windows: Sequence[WindowRun],
                leaks: Mapping[tuple[str, str], Mapping[str, Mapping[str, float]]]) -> list[dict[str, Any]]:
    out = []
    for w in windows:
        d: dict[str, Any] = {"item": w.item, "song": w.song.track_id, "title": w.song.title, "group": w.song.group,
                             "start_s": w.choice.start_s, "end_s": w.choice.end_s, "present": w.choice.present,
                             "n_ref": len(w.ref_notes), "conditions": {}}
        for c in w.conds:
            cd = {}
            for arm in ARMS:
                s = scores[(w.item, c.cond.name, arm)]
                f1 = 2 * s.tp / (2 * s.tp + s.fp + s.fn) if (2 * s.tp + s.fp + s.fn) else 1.0
                from bandscribe.amt import instruments as I

                cd[arm] = {"f1": f1, "recall": s.tp / (s.tp + s.fn) if s.tp + s.fn else None,
                           "precision": s.tp / (s.tp + s.fp) if s.tp + s.fp else None,
                           "fp_per_min": s.fp / s.minutes, "fn_per_min": s.fn / s.minutes,
                           "top_fp": s.fp_causes.most_common(3), "mask": c.masks[arm],
                           "non_guitar_dropped": sum(1 for x in c.notes.get(arm) or []
                                                     if x.get("instrument") not in I.GUITAR_CLASSES),
                           "tp": s.tp, "fp": s.fp, "fn": s.fn}
            lk = leaks.get((w.item, c.cond.name)) or {}
            cd["leak_db"] = {k: A.ratio_db(v["leak"]) for k, v in lk.items()}
            cd["share"] = {k: v["share"] for k, v in lk.items()}
            d["conditions"][c.cond.name] = cd
        out.append(d)
    return out


FIX_HINT_KO: dict[str, str] = {
    "fp": "S7 블리드 감점: 그 소리가 가는 스템(piano/other)의 전사와 겹치는 기타 음을 감점",
    "fn": "S3/S6: 그 소리에 가린 기타 음을 살리는 분리·뷰 개선(가림 원인)",
    "guitar_octave": "S7 옥타브·배음 정리(같은 시각 ±12/19/24 반음 음 정리)",
    "guitar_timing": "S7 음 분할·중복 정리(같은 음높이가 50 ms 밖에서 다시 잡힘)",
    "guitar_near": "음높이 미세 오차(벤딩·비브라토): 주법 처리(M9)",
    "guitar_unref": "참조 전사 자체의 누락일 수 있음: 판정 보류(GP 정답 필요)",
    "est_timing": "onset 위치 오차: 지연 보정·onset 정밀화",
    "est_octave": "옥타브 오류: S7 옥타브 정리",
    "guitar_clear": "가림 없이 놓침: 분리(S3) 또는 전사기 변동",
    "mask": "편성 마스크 탓: 같은 오디오를 기타만 마스크로 전사하면 늘지 않는다. 기타 패스 마스크에 비기타 클래스를 넣는 규칙을 E3b 로 재검토",
    "decoder_null": "MuScriptor 디코딩 불안정: 겹치는 창 재디코딩·다수결, beam(E21) 검토. 모든 범주 효과를 흐린다",
}


def fix_first(summary: Mapping[str, Any], *, arm: str = "production", top: int = 9) -> list[dict[str, Any]]:
    """'What to fix first': error sources ranked by errors per minute.

    Two kinds of rows: (a) a distractor category: errors/min it **adds** (ΔFP/min + ΔFN/min of ``+category`` vs
    ``base``, with the CIs), weighted by the share of measured windows where it is present, so a rare sound that
    hurts a lot ranks below a common one that hurts as much; (b) an error cause that remains in ``full``
    (octave, timing, ...): its FP or FN per minute there; (c) the decoder null: the mean |ΔFP| + |ΔFN| per minute
    that an inaudible change of the guitar view moves (``base_null``)."""
    out: list[dict[str, Any]] = []
    deltas = (summary.get("deltas") or {}).get(arm) or {}
    n_win = max(1, int(summary.get("n_windows") or 1))
    for cond, d in deltas.items():
        if not cond.startswith("+"):
            continue
        cat = cond[1:]
        dfp, dfn = d["fp_per_min"]["delta"], d["fn_per_min"]["delta"]
        prevalence = d["n_items"] / n_win
        hint = FIX_HINT_KO["fp"] if dfp >= dfn else FIX_HINT_KO["fn"]
        go = ((summary.get("deltas") or {}).get("guitar_only") or {}).get(cond) if arm != "guitar_only" else None
        go_err = (go["fp_per_min"]["delta"] + go["fn_per_min"]["delta"]) if go else None
        if go_err is not None and dfp + dfn > 0 and (dfp + dfn) - go_err > 0.5 * (dfp + dfn):
            hint = FIX_HINT_KO["mask"]  # the same audio with the guitar-only mask does not show the damage
        out.append({"source": cat, "kind": "category", "errors_per_min": dfp + dfn,
                    "weighted": (dfp + dfn) * prevalence, "fp": d["fp_per_min"], "fn": d["fn_per_min"],
                    "guitar_only_errors_per_min": go_err, "n_items": d["n_items"], "n_groups": d["n_groups"],
                    "hint": hint})
    full = ((summary.get("conditions") or {}).get(arm) or {}).get("full")
    fpm = (summary.get("fp_cause_matrix") or {}).get("full") or {}
    fnm = (summary.get("fn_cause_matrix") or {}).get("full") or {}
    if full:
        for cause in ("guitar_octave", "guitar_timing", "guitar_near", "guitar_unref"):
            if fpm.get(cause):
                out.append({"source": cause, "kind": "fp_cause", "errors_per_min": fpm[cause],
                            "weighted": fpm[cause], "hint": FIX_HINT_KO.get(cause, "")})
        for cause in ("est_timing", "est_octave", "guitar_clear"):
            if fnm.get(cause):
                out.append({"source": cause, "kind": "fn_cause", "errors_per_min": fnm[cause],
                            "weighted": fnm[cause], "hint": FIX_HINT_KO.get(cause, "")})
    nul = (summary.get("null") or {}).get("windows") or []
    if nul:  # how many errors per minute an inaudible change moves, on average over the windows
        moved = float(np.mean([abs(w["d_fp_per_min"]) + abs(w["d_fn_per_min"]) for w in nul]))
        worst = max(nul, key=lambda w: abs(w["d_fp_per_min"]) + abs(w["d_fn_per_min"]))
        out.append({"source": "decoder_null", "kind": "null", "errors_per_min": moved, "weighted": moved,
                    "worst": worst, "hint": FIX_HINT_KO["decoder_null"]})
    out.sort(key=lambda r: (-float(r["weighted"] if r["weighted"] == r["weighted"] else -1e9), r["source"]))
    return out[:top]


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")
