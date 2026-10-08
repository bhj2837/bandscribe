"""Bass F0 tracks (DESIGN 5.3): three trackers on one 10 ms grid, per-note evidence, the octave/pitch anchor vote
and frame-level tracker accuracy. numpy only, except ``pyin_track`` (librosa, imported when called).

Frame ``i`` of every tracker is centred at ``i * HOP_S`` seconds.

- ``crepe``: torchcrepe 0.0.24 'full' in the gpu env (worker ``f0_bass``). Activations in batches, then one Viterbi
  over the whole file on the salience normalised to sum 1, as the original CREPE does (``torchcrepe.predict``
  decodes each batch on its own, and its decoder softmaxes the sigmoid outputs, which is so flat that on bass stems
  the path sat on the lowest bin through whole phrases), and the pitch as the activation-weighted mean of the bins
  within ±4 of the path bin (torchcrepe adds random dither instead, so its output is not reproducible).
  ``conf`` = the activation at the path bin.
  fmin 32 Hz: CREPE's lowest bin is 31.70 Hz and torchcrepe turns a lower fmin into a negative bin index, which
  masks every bin but the top few (fmin 30 gives garbage, DESIGN 5.3); the worker refuses fmin < 31.71.
- ``pesto``: PESTO 2.0.1 ``mir-1k_g7`` (same worker; the checkpoint is loaded with ``weights_only=True``). Trained
  on singing (MIR-1K) and never measured on 30-100 Hz before M5: one vote, never alone.
- ``pyin``: librosa.pyin in the core env (CPU): fmin 30 Hz, fmax 400 Hz at 8 kHz with 128 ms frames (two periods
  of 30 Hz). ``hz`` = 0 where pyin's HMM calls the frame unvoiced, ``conf`` = its voiced probability.

A tracker frame is *valid* when ``hz > 0`` and ``conf >= conf_min[tracker]``. A note's estimate from one tracker
is the median MIDI pitch of its valid frames over ``onset + 30 ms .. onset + 120 ms`` (DESIGN 5.3), at least three
of them. The anchor vote (``vote``) moves a note only when two trackers agree within half a semitone on a pitch
an octave (or two) or one semitone away from MuScriptor's.
"""

from __future__ import annotations

import io
import math
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TRACKERS: tuple[str, ...] = ("crepe", "pesto", "pyin")
HOP_S = 0.01
CREPE_LOWEST_BIN_HZ = 31.70  # 10 Hz * 2 ** (1997.38 cents / 1200): CREPE's bin 0
CREPE_MIN_FMIN_HZ = 31.71  # the worker refuses a lower fmin (negative bin index in torchcrepe)
FORMAT = "bandscribe.bass_f0/1"
# Bump when the worker's decoding, PESTO chunking or pyin_track changes what a track holds: part of the bass_f0
# stage key and of the eval F0 cache key (review 2026-10-08: the cache key had no code version).
CODE_VERSION = "1"

DEFAULT_PARAMS: dict[str, Any] = {
    "hop_s": HOP_S,
    "crepe": {"model": "full", "fmin_hz": 32.0, "fmax_hz": 400.0, "batch_frames": 256, "path_halfwidth_bins": 4},
    "pesto": {"model": "mir-1k_g7"},
    "pyin": {"fmin_hz": 30.0, "fmax_hz": 400.0, "sr": 8000, "frame_length": 1024},
    "conf_min": {"crepe": 0.5, "pesto": 0.5, "pyin": 0.5},
    "evidence_s": [0.03, 0.12],
    "min_frames": 3,
    "agree_st": 0.5,
    "min_votes": 2,
}


def hz_to_midi(hz: Any) -> np.ndarray:
    """MIDI (float) of ``hz``; NaN where hz <= 0 or not finite."""
    f = np.asarray(hz, dtype=np.float64)
    out = np.full(f.shape, np.nan)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = 69.0 + 12.0 * np.log2(f[ok] / 440.0)
    return out


def midi_to_hz(m: Any) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def fit_length(a: Any, n: int, fill: float = 0.0) -> np.ndarray:
    """``a`` cut or padded with ``fill`` to ``n`` frames (float32)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if len(a) >= n:
        return a[:n].copy()
    return np.concatenate([a, np.full(n - len(a), fill, dtype=np.float32)])


def n_frames(duration_s: float, hop_s: float = HOP_S) -> int:
    return int(math.floor(float(duration_s) / hop_s)) + 1


@dataclass
class F0Track:
    """Per-tracker ``hz`` (0 = unvoiced) and ``conf`` on frames ``i * hop_s``."""

    hz: dict[str, np.ndarray]
    conf: dict[str, np.ndarray]
    hop_s: float = HOP_S
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return min((len(v) for v in self.hz.values()), default=0)

    @property
    def names(self) -> list[str]:
        return [t for t in TRACKERS if t in self.hz] + sorted(set(self.hz) - set(TRACKERS))

    def times(self) -> np.ndarray:
        return np.arange(self.n) * self.hop_s

    def frame(self, t: float) -> int:
        return int(round(float(t) / self.hop_s))

    def span(self, a: float, b: float) -> tuple[int, int]:
        """Frames ``i`` with ``a <= i * hop < b`` (clipped to the track)."""
        i0 = max(0, int(math.ceil(float(a) / self.hop_s - 1e-9)))
        i1 = min(self.n, int(math.ceil(float(b) / self.hop_s - 1e-9)))
        return i0, max(i0, i1)

    def midi(self, name: str) -> np.ndarray:
        return hz_to_midi(self.hz[name][: self.n])

    def valid(self, name: str, conf_min: Mapping[str, float] | None = None) -> np.ndarray:
        thr = float((conf_min or DEFAULT_PARAMS["conf_min"]).get(name, 0.5))
        hz, c = self.hz[name][: self.n], self.conf[name][: self.n]
        return np.isfinite(hz) & (hz > 0) & np.isfinite(c) & (c >= thr)

    def valid_midi(self, name: str, conf_min: Mapping[str, float] | None = None) -> np.ndarray:
        """MIDI where the frame is valid, NaN elsewhere."""
        m = self.midi(name)
        m[~self.valid(name, conf_min)] = np.nan
        return m


# --------------------------------------------------------------------------------------------- storage


def save_npz(path: Path, track: F0Track) -> int:
    """npz with fixed member timestamps (byte-identical for the same arrays)."""
    from bandscribe import atomic

    arrays: dict[str, np.ndarray] = {"hop_s": np.asarray(float(track.hop_s))}
    for name in track.names:
        arrays[f"{name}_hz"] = np.asarray(track.hz[name], dtype=np.float32)
        arrays[f"{name}_conf"] = np.asarray(track.conf[name], dtype=np.float32)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, arr in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, arr, allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, member.getvalue())
    atomic.write_bytes(Path(path), buf.getvalue())
    return Path(path).stat().st_size


def load_npz(path: Path) -> F0Track:
    with np.load(str(path), allow_pickle=False) as z:
        hop = float(z["hop_s"]) if "hop_s" in z.files else HOP_S
        names = sorted({k[: -len("_hz")] for k in z.files if k.endswith("_hz")})
        hz = {n: np.asarray(z[f"{n}_hz"], dtype=np.float32) for n in names}
        conf = {n: np.asarray(z[f"{n}_conf"], dtype=np.float32) for n in names if f"{n}_conf" in z.files}
    return F0Track(hz=hz, conf=conf, hop_s=hop)


def merge(*tracks: F0Track, n: int | None = None) -> F0Track:
    """One track with every tracker of ``tracks`` (later ones win), all cut/padded to ``n`` frames (default: the
    longest)."""
    hz: dict[str, np.ndarray] = {}
    conf: dict[str, np.ndarray] = {}
    hop = tracks[0].hop_s if tracks else HOP_S
    for t in tracks:
        if abs(t.hop_s - hop) > 1e-9:
            raise ValueError(f"F0 tracks with different hops: {t.hop_s} vs {hop}")
        hz.update(t.hz)
        conf.update(t.conf)
    size = n if n is not None else max((len(v) for v in hz.values()), default=0)
    return F0Track(hz={k: fit_length(v, size) for k, v in hz.items()},
                   conf={k: fit_length(v, size) for k, v in conf.items()}, hop_s=hop)


# ------------------------------------------------------------------------------------------------ pyin


def pyin_track(mono: np.ndarray, sr: int, params: Mapping[str, Any] | None = None, *,
               duration_s: float | None = None) -> F0Track:
    """librosa.pyin on ``mono`` resampled to ``params["pyin"]["sr"]``; frames on the ``HOP_S`` grid."""
    import librosa

    p = {**DEFAULT_PARAMS["pyin"], **dict((params or {}).get("pyin") or {})}
    hop_s = float((params or {}).get("hop_s", HOP_S))
    target = int(p["sr"])
    y = np.asarray(mono, dtype=np.float32).reshape(-1)
    dur = float(duration_s if duration_s is not None else len(y) / float(sr))
    if int(sr) != target:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=target, res_type="soxr_hq")
    hop = int(round(hop_s * target))
    f, vflag, vprob = librosa.pyin(y, fmin=float(p["fmin_hz"]), fmax=float(p["fmax_hz"]), sr=target,
                                   frame_length=int(p["frame_length"]), hop_length=hop, center=True)
    hz = np.where(np.asarray(vflag, dtype=bool) & np.isfinite(f), f, 0.0)
    n = n_frames(dur, hop_s)
    return F0Track(hz={"pyin": fit_length(hz, n)}, conf={"pyin": fit_length(np.nan_to_num(vprob), n)}, hop_s=hop_s)


# --------------------------------------------------------------------------------------- note evidence


def evidence_span(onset: float, offset: float, params: Mapping[str, Any] | None = None) -> tuple[float, float]:
    """``onset + 30 ms .. min(onset + 120 ms, offset - 10 ms)``; when that is shorter than 30 ms (a note shorter than
    70 ms) ``onset + 10 ms .. offset``."""
    w = (params or DEFAULT_PARAMS).get("evidence_s", DEFAULT_PARAMS["evidence_s"])
    a, b = float(onset) + float(w[0]), min(float(onset) + float(w[1]), float(offset) - 0.01)
    if b - a < 0.03:
        a, b = float(onset) + 0.01, max(float(onset) + 0.03, float(offset))
    return a, b


def note_estimates(track: F0Track, onset: float, offset: float, params: Mapping[str, Any] | None = None
                   ) -> dict[str, float]:
    """{tracker: median MIDI of its valid frames in the evidence span} (trackers with too few frames left out)."""
    p = params or DEFAULT_PARAMS
    a, b = evidence_span(onset, offset, p)
    i0, i1 = track.span(a, b)
    out: dict[str, float] = {}
    for name in track.names:
        m = track.valid_midi(name, p.get("conf_min"))[i0:i1]
        m = m[np.isfinite(m)]
        if len(m) >= int(p.get("min_frames", 3)):
            out[name] = float(np.median(m))
    return out


@dataclass(frozen=True)
class Vote:
    action: str  # confirmed | octave | semitone | disagree | split | no_f0
    pitch: int  # the note's pitch after the vote
    target: int | None  # the pitch the agreeing trackers name (None: no two agree)
    voters: tuple[str, ...] = ()


VOTE_ACTIONS = ("confirmed", "octave", "semitone", "disagree", "split", "no_f0")


def vote(pitch: int, ests: Mapping[str, float], params: Mapping[str, Any] | None = None) -> Vote:
    """DESIGN 5.3: when two trackers agree (within ``agree_st``) on a pitch ``T``, an octave (12, 24, 36) away from
    ``pitch`` moves the note there, one semitone away moves it too ("±1반음 차이면 F0를 믿는다"); any other
    interval is left alone (``disagree``, a review case). ``split``: trackers answered but no two agree;
    ``no_f0``: fewer than two answered."""
    p = params or DEFAULT_PARAMS
    tol = float(p.get("agree_st", 0.5))
    need = int(p.get("min_votes", 2))
    vals = {k: float(v) for k, v in ests.items() if v is not None and math.isfinite(float(v))}
    if len(vals) < need:
        return Vote("no_f0", int(pitch), None)
    best: tuple[int, int, int] | None = None  # (support, -distance to pitch, target)
    for t in sorted({int(round(v)) for v in vals.values()}):
        sup = [k for k, v in vals.items() if abs(v - t) <= tol]
        if len(sup) >= need:
            key = (len(sup), -abs(t - int(pitch)), t)
            if best is None or key > best:
                best = key
    if best is None:
        return Vote("split", int(pitch), None)
    t = best[2]
    voters = tuple(sorted(k for k, v in vals.items() if abs(v - t) <= tol))
    d = t - int(pitch)
    if d == 0:
        return Vote("confirmed", int(pitch), t, voters)
    if d % 12 == 0 and abs(d) <= 36:
        return Vote("octave", t, t, voters)
    if abs(d) == 1:
        return Vote("semitone", t, t, voters)
    return Vote("disagree", int(pitch), t, voters)


# ------------------------------------------------------------------------------------------- contours


def consensus(track: F0Track, params: Mapping[str, Any] | None = None, *, ref_midi: np.ndarray | None = None,
              min_trackers: int = 2) -> np.ndarray:
    """Per frame: the median of the valid trackers' MIDI pitch, NaN where fewer than ``min_trackers`` are valid
    (or, without ``ref_midi``, where they do not agree within ``agree_st``). With ``ref_midi`` (e.g. the pitch of
    the note sounding at that frame) every tracker's value is first moved by whole octaves to the one nearest the
    reference, so octave slips of single trackers do not break a contour."""
    p = params or DEFAULT_PARAMS
    n = track.n
    vals = np.stack([track.valid_midi(name, p.get("conf_min")) for name in track.names]) if track.names else \
        np.full((0, n), np.nan)
    if ref_midi is not None:
        ref = np.asarray(ref_midi, dtype=np.float64)[:n]
        k = np.round((ref[None, :] - vals) / 12.0)
        vals = np.where(np.isfinite(ref)[None, :], vals + 12.0 * k, np.nan)
    cnt = np.sum(np.isfinite(vals), axis=0)
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames give NaN, which is what we want
        med = np.nanmedian(np.where(np.isfinite(vals), vals, np.nan), axis=0) if len(vals) else np.full(n, np.nan)
        if ref_midi is None and len(vals):
            near = np.sum(np.abs(vals - med[None, :]) <= float(p.get("agree_st", 0.5)), axis=0)
            cnt = np.minimum(cnt, near)
    out = np.where(cnt >= int(min_trackers), med, np.nan)
    return out.astype(np.float64)


def running_median(x: np.ndarray, width: int = 5, min_count: int = 3) -> np.ndarray:
    """NaN-aware running median (odd ``width``): NaN where fewer than ``min_count`` of the window are finite. Smooths
    the frame-to-frame switching between trackers that answer at slightly different pitches."""
    x = np.asarray(x, dtype=np.float64)
    h = width // 2
    pad = np.concatenate([np.full(h, np.nan), x, np.full(h, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, width)
    cnt = np.sum(np.isfinite(win), axis=1)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(win, axis=1)
    return np.where(cnt >= min_count, med, np.nan)


# ----------------------------------------------------------------------------------- tracker accuracy


def frame_reference(notes: Iterable[Mapping[str, Any]], n: int, hop_s: float = HOP_S, *, margin_s: float = 0.03,
                    skip: Iterable[int] = ()) -> np.ndarray:
    """GT MIDI per frame from monophonic GT notes (NaN outside notes, within ``margin_s`` of a note's ends, where
    two notes overlap, and for the indices in ``skip`` - dead notes, harmonics)."""
    ref = np.full(n, np.nan)
    count = np.zeros(n, dtype=np.int32)
    skip = set(skip)
    for i, x in enumerate(notes):
        if i in skip:
            continue
        a = int(math.ceil((float(x["onset_s"]) + margin_s) / hop_s))
        b = int(math.floor((float(x["offset_s"]) - margin_s) / hop_s)) + 1
        a, b = max(0, a), min(n, b)
        if b > a:
            ref[a:b] = float(x["pitch"])
            count[a:b] += 1
    ref[count > 1] = np.nan
    return ref


BANDS_HZ: tuple[tuple[float, float], ...] = ((30.0, 100.0), (100.0, 400.0))


def accuracy(track: F0Track, ref_midi: np.ndarray, params: Mapping[str, Any] | None = None, *,
             bands: Sequence[tuple[float, float]] = BANDS_HZ) -> dict[str, dict[str, Any]]:
    """Per tracker and GT band ("30-100", "100-400", "all"): frames (GT voiced), ``rpa`` (valid and within 50 cents),
    ``rca`` (the same modulo octaves), ``vr`` (valid at all), ``rpa_any`` (within 50 cents ignoring the confidence),
    ``octave`` (valid, wrong by whole octaves) - each a fraction of the band's GT frames."""
    p = params or DEFAULT_PARAMS
    n = min(track.n, len(ref_midi))
    ref = np.asarray(ref_midi[:n], dtype=np.float64)
    ref_hz = midi_to_hz(ref)
    out: dict[str, dict[str, Any]] = {}
    for name in track.names:
        m_any = track.midi(name)[:n]
        ok = track.valid(name, p.get("conf_min"))[:n]
        err = m_any - ref
        fold = np.abs(err - 12.0 * np.round(err / 12.0))
        res: dict[str, Any] = {}
        for label, sel in [(f"{int(lo)}-{int(hi)}", np.isfinite(ref) & (ref_hz >= lo) & (ref_hz < hi))
                           for lo, hi in bands] + [("all", np.isfinite(ref))]:
            k = int(np.sum(sel))
            if k == 0:
                res[label] = {"frames": 0}
                continue
            e, f, v = err[sel], fold[sel], ok[sel]
            close = np.isfinite(e) & (np.abs(e) <= 0.5)
            res[label] = {"frames": k, "rpa": float(np.mean(close & v)), "rca": float(np.mean(np.isfinite(f) & (f <= 0.5) & v)),
                          "vr": float(np.mean(v)), "rpa_any": float(np.mean(close)),
                          "octave": float(np.mean(v & np.isfinite(e) & (np.abs(e) > 0.5) & (f <= 0.5)))}
        out[name] = res
    return out
