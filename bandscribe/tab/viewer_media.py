"""What the M4b viewer plays and draws besides the score (``bandscribe.tab.viewer``):

- ``stage_dirs``: the stage dirs behind the job's export (the score stage's dependencies, followed transitively);
- ``sources``: the recordings the A/B mixer can play - the original, the separated stems, the practice mixes;
- ``cqt``: a constant-Q image of one source for the piano roll (uint8, ``CQT_BINS_PER_SEMITONE`` bins per semitone
  from ``CQT_MIN_MIDI``, ``CQT_HOP / CQT_SR`` s per column), cached in the job's ``cache/viewer``;
- ``candidates``: the transcribers' notes the pipeline did not keep (outlined in the roll: the guitar pass's notes
  the guitar mask or the silence gate dropped, Basic Pitch notes no kept note matches);
- ``router_windows``: the router's windows with their labels (the part timeline band).
Nothing here runs a pipeline stage; everything reads what a run left.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic

CQT_SR = 22050
CQT_HOP = 512
CQT_MIN_MIDI = 28           # E1, the bass's low string
CQT_SEMITONES = 72          # up to E7
CQT_BINS_PER_SEMITONE = 3
CQT_RANGE_DB = 70.0
CQT_VERSION = "1"
MATCH_S = 0.05              # a Basic Pitch note is "kept" when a kept note of its pitch starts this close
STEM_LABELS = {"guitar": "기타 스템", "bass": "베이스 스템", "drums": "드럼 스템", "vocals": "보컬 스템",
               "piano": "피아노 스템", "other": "기타 악기 스템"}
STEM_ORDER = ("guitar", "bass", "drums", "vocals", "piano", "other")
PRACTICE = (("mix_minus_guitar", "기타 뺀 믹스"), ("mix_minus_bass", "베이스 뺀 믹스"))
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class MediaError(RuntimeError):
    """User-facing (Korean) problem of a viewer media request."""


def stage_dirs(job_dir: Path, score_key: str | None) -> dict[str, Path]:
    """name -> stage dir behind the export's score stage (breadth first; the first dir found for a name wins)."""
    stages = Path(job_dir) / "stages"
    out: dict[str, Path] = {}
    if not score_key:
        return out
    todo: deque[tuple[str, str]] = deque([("score", str(score_key)[:12])])
    while todo:
        name, key = todo.popleft()
        if name in out:
            continue
        d = stages / name / key[:12]
        if not d.is_dir():
            continue
        out[name] = d
        try:
            deps = atomic.read_json(d / "manifest.json").get("deps") or {}
        except (OSError, ValueError):
            continue
        for k, v in deps.items():
            if k not in out:
                todo.append((str(k), str(v)))
    return out


def original(job_dir: Path, fallback: Path | None) -> Path | None:
    """The recording the mixer plays as the original: the decoded ``input/mix_44k_f32.wav`` the stems were separated
    from (same first sample - a compressed input's decoder delay would shift it against the stems), else
    ``fallback`` (the input file)."""
    wav = Path(job_dir) / "input" / "mix_44k_f32.wav"
    return wav if wav.is_file() else fallback


def sources(job_dir: Path, dirs: Mapping[str, Path], original: Path | None) -> list[dict[str, Any]]:
    """The recordings of the mixer: id, label, file (only those on disk), ``on`` = playing by default."""
    out: list[dict[str, Any]] = []
    if original is not None and Path(original).is_file():
        out.append({"id": "mix", "label": "원곡", "file": Path(original), "on": True})
    sep = dirs.get("sep")
    for s in STEM_ORDER:
        f = sep / "stems" / f"{s}.wav" if sep else None
        if f is not None and f.is_file():
            out.append({"id": f"stem_{s}", "label": STEM_LABELS[s], "file": f, "on": False})
    stems = dirs.get("stems")
    for name, label in PRACTICE:
        f = stems / "practice" / f"{name}.wav" if stems else None
        if f is not None and f.is_file():
            out.append({"id": name, "label": label, "file": f, "on": False})
    return out


def _lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def cqt_image(x: np.ndarray, sr: int) -> np.ndarray:
    """uint8 [bins, frames] (row 0 = lowest bin) of a mono signal."""
    import librosa

    if sr != CQT_SR:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=CQT_SR)
    fmin = float(librosa.midi_to_hz(CQT_MIN_MIDI)) * 2.0 ** (-1.0 / (12 * CQT_BINS_PER_SEMITONE))
    n_bins = CQT_SEMITONES * CQT_BINS_PER_SEMITONE
    c = np.abs(librosa.cqt(x.astype(np.float32), sr=CQT_SR, hop_length=CQT_HOP, fmin=fmin, n_bins=n_bins,
                           bins_per_octave=12 * CQT_BINS_PER_SEMITONE))
    db = 20.0 * np.log10(np.maximum(c, 1e-9))
    top = float(np.percentile(db, 99.9)) if db.size else 0.0
    img = np.clip((db - (top - CQT_RANGE_DB)) / CQT_RANGE_DB, 0.0, 1.0)
    return np.ascontiguousarray((img * 255.0 + 0.5).astype(np.uint8))


def cqt(job_dir: Path, src: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """(raw uint8 bytes, row-major [bins, frames], and its header) of ``src``; computed once per file content."""
    import soundfile as sf

    f = Path(src["file"])
    try:
        st = f.stat()
    except OSError as e:
        raise MediaError(f"{src['label']} 파일이 없습니다: {f}") from e
    key = hashlib.sha256(f"{CQT_VERSION}|{f.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    cache = Path(job_dir) / "cache" / "viewer"
    raw, meta = cache / f"cqt_{src['id']}_{key}.u8", cache / f"cqt_{src['id']}_{key}.json"
    with _lock(str(raw)):
        if raw.is_file() and meta.is_file():
            return raw.read_bytes(), json.loads(meta.read_text(encoding="utf-8"))
        try:
            x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001 - mp3/m4a originals: librosa's loader (audioread) reads those
            try:
                import librosa

                x1, sr = librosa.load(str(f), sr=CQT_SR, mono=True)
                x = x1[:, None]
            except Exception as e2:  # noqa: BLE001
                raise MediaError(f"{src['label']} 을(를) 읽지 못했습니다: {e2}") from e
        img = cqt_image(x.mean(axis=1), int(sr))
        head = {"bins": int(img.shape[0]), "frames": int(img.shape[1]), "fps": CQT_SR / CQT_HOP,
                "min_midi": CQT_MIN_MIDI, "bins_per_semitone": CQT_BINS_PER_SEMITONE, "source": src["id"]}
        cache.mkdir(parents=True, exist_ok=True)
        atomic.write_bytes(raw, img.tobytes())
        atomic.write_json(meta, head)
        return img.tobytes(), head


def _raw_notes(d: Path | None, name: str) -> list[dict[str, Any]]:
    if d is None or not (d / "raw" / name).is_file():
        return []
    try:
        return list(atomic.read_json(d / "raw" / name).get("notes") or [])
    except (OSError, ValueError):
        return []


def _unkept(cands: Sequence[Mapping[str, Any]], kept: Sequence[Mapping[str, Any]], tol: float) -> list[dict[str, Any]]:
    by_pitch: dict[int, list[float]] = {}
    for n in kept:
        by_pitch.setdefault(int(n["pitch"]), []).append(float(n["onset_s"]))
    for v in by_pitch.values():
        v.sort()
    out = []
    for n in cands:
        ts = by_pitch.get(int(n["pitch"])) or []
        t = float(n["onset_s"])
        j = int(np.searchsorted(ts, t)) if ts else 0
        near = [abs(ts[k] - t) for k in (j - 1, j) if 0 <= k < len(ts)]
        if not near or min(near) > tol:
            out.append(n)
    return out


def candidates(dirs: Mapping[str, Path], kept: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """[{track, pitch, on, off, src, why}] of the transcribers' notes no kept note of ``kept`` (track -> the
    base's raw notes) stands for."""
    out: list[dict[str, Any]] = []
    plans = (("guitar_all", dirs.get("amt_gtr"), "guitar_mono__muscriptor.json", "muscriptor", 1e-3,
              "기타 마스크·무음 게이트가 뺀 음"),
             ("guitar_all", dirs.get("amt_bp"), "guitar_mono__basicpitch.json", "basicpitch", MATCH_S,
              "Basic Pitch 만 들은 음"),
             ("bass", dirs.get("amt_ms1"), "bass_mono__muscriptor.json", "muscriptor", 1e-3, "베이스 정리가 뺀 음"),
             ("bass", dirs.get("amt_bp"), "bass_mono__basicpitch.json", "basicpitch", MATCH_S, "Basic Pitch 만 들은 음"))
    for tid, d, name, src, tol, why in plans:
        for n in _unkept(_raw_notes(d, name), kept.get(tid) or [], tol):
            out.append({"track": tid, "pitch": int(n["pitch"]), "on": round(float(n["onset_s"]), 4),
                        "off": round(float(n["offset_s"]), 4), "src": src, "why": why})
    return out


def router_windows(dirs: Mapping[str, Path]) -> list[dict[str, Any]]:
    d = dirs.get("router")
    if d is None or not (d / "router.json").is_file():
        return []
    try:
        ws = atomic.read_json(d / "router.json").get("windows") or []
    except (OSError, ValueError):
        return []
    return [{"start_s": float(w["start_s"]), "end_s": float(w["end_s"]), "label": str(w.get("label") or ""),
             "section": w.get("section")} for w in ws if "start_s" in w and "end_s" in w]
