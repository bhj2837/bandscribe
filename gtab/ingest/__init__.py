"""Ingest: a local audio file or a YouTube URL -> ``jobs/<song_key>/input/`` (DESIGN S0, M0_SPEC 12).

``input/`` holds ``source.<ext>`` (byte copy of the original, never re-encoded), ``mix_44k_f32.wav``
(stereo float32, soxr, float gain so the true peak is under the ceiling) and ``meta.json``. It is
assembled in a ``.tmp-ingest-*`` scratch dir under the store root and published with one directory
rename, so an interrupted ingest never leaves a half-built ``input/``; ``JobStore.gc_temp`` removes
the scratch dirs of dead processes. Input is complete iff ``input/meta.json`` exists.

song_key: ``yt-<videoId>`` for YouTube, ``f-<first 16 hex of the PCM sha256>`` for files, so the same
audio in another container or under another name lands in the same job. The PCM hash is always taken
from a 44.1 kHz decode (``ingest.sample_rate`` is fixed), so the key never depends on settings.

Settings: ``meta.json["params"]`` records every ingest setting that shapes ``input/`` (ceiling, flag
thresholds, rate, ingest code version). A cached input is served only when its params equal the
current ones; otherwise it is rebuilt from the job's own stored source copy (no network) and replaced,
so an explicit ``--set ingest.true_peak_ceiling_dbfs=...`` is never silently dropped.

The index key for files is ``file:<content key>`` (gtab.ingest.filekey: a chunked, multi-threaded
sha256 of every byte), which keeps "same file under another name -> cache hit" under 1 s for large
WAV bounces.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf

from gtab import __version__, atomic, paths
from gtab.ingest import decode, flags, loudness, youtube
from gtab.ingest.decode import DecodeError, IngestError
from gtab.ingest.filekey import content_key
from gtab.ingest.youtube import YoutubeError
from gtab.jobs.store import JobStore, make_temp_dir, rmtree_quiet

__all__ = [
    "DecodeError", "IngestError", "IngestResult", "YoutubeError", "content_key", "ingest", "ingest_params", "is_url",
]

log = logging.getLogger(__name__)

META = "meta.json"
MIX = "mix_44k_f32.wav"
_DECODED = "decoded_f32.wav"
# Bump when input/ would come out different for the same source and settings; old inputs are rebuilt.
INGEST_VERSION = 2
# A freshly written WAV is often still open in Defender/the indexer for a moment, and Windows refuses
# to rename a directory while any handle inside it is open. Retry for up to ~4 s.
_RENAME_RETRIES = 12
_RENAME_BACKOFF_S = 0.05

YOUTUBE_DISABLED_MSG = (
    "YouTube 입력이 꺼져 있습니다 (youtube.enabled = false). "
    r"켜려면 D:\gtab\data\config.toml 의 [youtube] 에 enabled = true 를 쓰거나 --set youtube.enabled=true 를 주세요. "
    "아니면 오디오 파일을 직접 넣어 주세요."
)
YOUTUBE_RISK_NOTICE = (
    "YouTube 경로 사용 중: 약관은 허락 없는 다운로드를 금지하고, JS 챌린지 해결은 해외 판례에서 기술적 보호조치 "
    "우회로 본 적이 있습니다(한국법 미검증). 쿠키·PO 토큰은 쓰지 않습니다. 동의 기록: data/config.toml [consent]."
)


@dataclass(frozen=True)
class IngestResult:
    song_key: str
    job_dir: Path
    meta: dict[str, Any]
    cache_hit: bool


@dataclass(frozen=True)
class _Source:
    """What gets copied into input/ and how meta.json describes it."""

    path: Path
    key: str  # content_key(path)
    kind: str  # "file" | "youtube"
    ref: str
    filename: str
    youtube: dict[str, Any] | None


def is_url(source: str) -> bool:
    return bool(re.match(r"^https?://", str(source).strip(), re.IGNORECASE))


def ingest_params(cfg: Any) -> dict[str, Any]:
    """Every setting that shapes input/ (stored as meta["params"]; a cache hit requires equality)."""
    ic = cfg.ingest
    return {
        "version": INGEST_VERSION,
        "sample_rate": int(ic.sample_rate),
        "true_peak_ceiling_dbfs": float(ic.true_peak_ceiling_dbfs),
        "quasi_mono_side_mid_db": float(ic.quasi_mono_side_mid_db),
        "lowpass_detect_hz": int(ic.lowpass_detect_hz),
    }


def ingest(source: str | Path, *, store: JobStore, cfg: Any) -> IngestResult:
    """Build (or find in the cache) the job input for ``source``: a file path or an http(s) URL."""
    text = str(source).strip()
    if is_url(text):
        return _ingest_youtube(text, store, cfg)
    return _ingest_file(Path(text), store, cfg)


# ------------------------------------------------------------------------------------------ sources


def _ingest_file(src: Path, store: JobStore, cfg: Any) -> IngestResult:
    if not src.is_file():
        raise IngestError(f"파일을 찾을 수 없습니다: {src}")
    params = ingest_params(cfg)
    # Fast path (< 1 s): hashing the file bytes needs no ffmpeg. A copy under another name hits here.
    key = content_key(src)
    song_key = store.index_get(f"file:{key}")
    if song_key:
        meta = _load_meta(store, song_key)
        if meta is not None:
            if meta.get("params") == params:
                log.info("cache hit: %s -> %s", src.name, song_key)
                return IngestResult(song_key, store.job_dir(song_key), meta, True)
            rebuilt = _rebuild(store, cfg, song_key, meta, params)
            if rebuilt is not None:
                return rebuilt
        else:
            log.warning("index maps %s to %s but its input is missing; rebuilding", src.name, song_key)

    source = _Source(src, key, "file", str(src.resolve()), src.name, None)
    song_key, meta, cache_hit = _build_input(store, cfg, source, song_key=None)
    store.index_put(f"file:{key}", song_key)
    return IngestResult(song_key, store.job_dir(song_key), meta, cache_hit)


def _ingest_youtube(url: str, store: JobStore, cfg: Any) -> IngestResult:
    if not cfg.youtube.enabled:
        raise IngestError(YOUTUBE_DISABLED_MSG)
    vid = youtube.parse_video_id(url)
    if vid is None:
        raise IngestError(f"YouTube 동영상 주소로 인식하지 못했습니다: {url}\n(재생목록·채널 주소는 받지 않습니다. 동영상 하나의 주소를 넣어 주세요.)")
    song_key = f"yt-{vid}"
    params = ingest_params(cfg)

    # Cache check without any network access: the index, then the deterministic job dir.
    for key in dict.fromkeys(k for k in (store.index_get(f"yt:{vid}"), song_key) if k):
        meta = _load_meta(store, key)
        if meta is None:
            continue
        _check_same_video(meta, vid, key)
        store.index_put(f"yt:{vid}", key)  # heals a lost index entry; no-op otherwise
        if meta.get("params") == params:
            log.info("cache hit: %s -> %s", vid, key)
            return IngestResult(key, store.job_dir(key), meta, True)
        rebuilt = _rebuild(store, cfg, key, meta, params)  # from the stored copy: no second download
        if rebuilt is not None:
            return rebuilt

    log.info(YOUTUBE_RISK_NOTICE)
    audio, info = youtube.download(url, paths.DOWNLOADS / vid, cfg)
    if info.get("id") not in (None, vid):
        raise YoutubeError(f"요청한 동영상({vid})과 받은 동영상({info.get('id')})이 다릅니다.")
    key = content_key(audio)
    source = _Source(audio, key, "youtube", youtube.canonical_url(vid), audio.name, info)
    song_key, meta, cache_hit = _build_input(store, cfg, source, song_key=song_key)
    store.index_put(f"yt:{vid}", song_key)
    # The downloaded file itself, if later given as a local file, maps to the same job.
    store.index_put(f"file:{key}", song_key)
    return IngestResult(song_key, store.job_dir(song_key), meta, cache_hit)


def _check_same_video(meta: dict[str, Any], vid: str, key: str) -> None:
    # NTFS is case-insensitive but video ids are not: yt-abc... and yt-ABC... share one folder.
    cached = (meta.get("youtube") or {}).get("id")
    if cached is not None and cached != vid:
        raise IngestError(
            f"작업 폴더 {key} 는 다른 동영상({cached})의 것입니다. 대소문자만 다른 ID 충돌이라 자동으로 처리하지 않습니다."
        )


def _describe_param_change(old: Any, new: dict[str, Any]) -> str:
    if not isinstance(old, dict):
        return "설정 기록 없음(이전 버전 입력)"
    return ", ".join(f"{k} {old.get(k)!r} -> {v!r}" for k, v in new.items() if old.get(k) != v)


def _rebuild(store: JobStore, cfg: Any, song_key: str, meta: dict[str, Any], params: dict[str, Any]) -> IngestResult | None:
    """Rebuild a job's input/ with the current settings from its own stored source copy (no network).

    The job keeps its identity (kind, ref, original filename, YouTube info). Returns None when the
    stored copy is missing, so the caller falls back to building from what the user gave.
    """
    log.info("입력 설정이 캐시와 달라 %s 의 input/ 을 다시 만듭니다: %s", song_key,
             _describe_param_change(meta.get("params"), params))
    src_meta = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    stored_as = src_meta.get("stored_as")
    stored = store.input_dir(song_key) / stored_as if isinstance(stored_as, str) and stored_as else None
    if stored is None or not stored.is_file():
        log.warning("%s has no stored source copy; building from the given input instead", song_key)
        return None
    source = _Source(
        stored,
        content_key(stored),
        str(src_meta.get("kind") or "file"),
        str(src_meta.get("ref") or stored),
        str(src_meta.get("filename") or stored.name),
        meta.get("youtube") if isinstance(meta.get("youtube"), dict) else None,
    )
    song_key, new_meta, cache_hit = _build_input(store, cfg, source, song_key=song_key)
    return IngestResult(song_key, store.job_dir(song_key), new_meta, cache_hit)


# ------------------------------------------------------------------------------------------- build


def _load_meta(store: JobStore, song_key: str) -> dict[str, Any] | None:
    path = store.input_dir(song_key) / META
    try:
        meta = atomic.read_json(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        log.warning("unreadable %s: %s", path, e)
        return None
    return meta if isinstance(meta, dict) else None


def _safe_ext(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return ext if re.fullmatch(r"[a-z0-9]{1,8}", ext) else "bin"


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(float(x), nd)


def _file_entry(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": atomic.sha256_file(path)}


def _build_input(
    store: JobStore, cfg: Any, source: _Source, *, song_key: str | None
) -> tuple[str, dict[str, Any], bool]:
    """Decode + measure ``source`` in a scratch dir and publish it as ``input/``.

    ``song_key`` None means "derive from the PCM hash" (files). Returns (song_key, meta, cache_hit);
    cache_hit is True when that job's input, built with the same settings, turned out to exist already
    (same audio in another container, or a concurrent ingest finished first).
    """
    params = ingest_params(cfg)
    sr = params["sample_rate"]
    ceiling = params["true_peak_ceiling_dbfs"]
    stage = make_temp_dir(store.root, "ingest")
    try:
        # Work only from our own copy, so probe/decode/hash all describe the bytes we keep.
        stored_name = f"source.{_safe_ext(source.path)}"
        stored = stage / stored_name
        shutil.copyfile(source.path, stored)
        if content_key(stored) != source.key:
            raise IngestError(f"원본 파일이 처리 도중 바뀌었습니다. 파일 쓰기가 끝난 뒤 다시 시도해 주세요: {source.path}")
        probe_info = decode.probe(stored)
        decoded = stage / _DECODED
        decode.decode_to_f32(stored, decoded, sr)
        pcm = decode.pcm_sha256(decoded)

        if song_key is None:
            song_key = f"f-{pcm[:16]}"
            existing = _load_meta(store, song_key)
            if existing is not None:
                if existing.get("params") == params:
                    log.info("cache hit: %s has the same audio as %s", source.path.name, song_key)
                    return song_key, existing, True
                log.info("입력 설정이 캐시와 달라 %s 의 input/ 을 다시 만듭니다: %s", song_key,
                         _describe_param_change(existing.get("params"), params))

        measured = loudness.measure(decoded)
        mix = stage / MIX
        gain_db = loudness.apply_ceiling(decoded, mix, ceiling, measured=measured)
        fl = flags.compute_flags(mix, probe_info, cfg, format_id=(source.youtube or {}).get("format_id"))
        decoded.unlink()
        info = sf.info(str(mix))

        tp, integrated = measured["true_peak_dbfs"], measured["integrated_lufs"]
        meta: dict[str, Any] = {
            "gtab_version": __version__,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "song_key": song_key,
            "duration_s": round(info.frames / info.samplerate, 3),
            "params": params,
            "source": {
                "kind": source.kind,
                "ref": source.ref,
                "filename": source.filename,
                "stored_as": stored_name,
                "sha256": atomic.sha256_file(stored),
                "content_key": source.key,  # the index key (file:<content_key>)
                "bytes": stored.stat().st_size,
                "probe": probe_info,
            },
            "youtube": source.youtube,
            "decode": {
                "sr": sr,
                "channels": info.channels,
                "sample_format": "float32",
                "resampler": "soxr",
                "frames": info.frames,
                "pcm_sha256": pcm,  # of the decoded audio before gain; defines f-<...> keys
            },
            "loudness": {
                **{k: _r(v) for k, v in measured.items()},
                "gain_db": gain_db,
                "ceiling_dbfs": ceiling,
                "output_true_peak_dbfs": _r(None if tp is None else tp + gain_db),
                "output_integrated_lufs": _r(None if integrated is None else integrated + gain_db),
            },
            "flags": fl,
            "files": {stored_name: _file_entry(stored), MIX: _file_entry(mix)},
        }
        atomic.write_json(stage / META, meta)
        final_meta, cache_hit = _publish(stage, store, song_key, meta)
        return song_key, final_meta, cache_hit
    finally:
        rmtree_quiet(stage)  # already gone after a successful publish


def _rename_dir(src: Path, dst: Path) -> None:
    """os.rename (fails if ``dst`` exists, which is the signal we want) with retries for transient
    Windows sharing violations."""
    for attempt in range(_RENAME_RETRIES):
        try:
            os.rename(src, dst)
            return
        except FileExistsError:
            raise
        except PermissionError:
            if dst.exists():
                raise FileExistsError(str(dst)) from None
            if attempt == _RENAME_RETRIES - 1:
                raise
            time.sleep(_RENAME_BACKOFF_S * (attempt + 1))


def _publish(stage: Path, store: JobStore, song_key: str, meta: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Rename ``stage`` to ``input/``. An existing input/ built with the same params wins (cache hit);
    one that is damaged (no meta.json) or built with other params is moved aside and replaced."""
    final = store.input_dir(song_key)
    job_dir = store.job_dir(song_key)
    for _ in range(3):
        try:
            _rename_dir(stage, final)
            log.info("input ready: %s", final)
            return meta, False
        except FileExistsError:
            pass
        except PermissionError as e:
            raise IngestError(f"입력 폴더를 옮기지 못했습니다(안의 파일을 다른 프로그램이 쓰는 중): {final} ({e})") from e
        existing = _load_meta(store, song_key)
        if existing is not None and existing.get("params") == meta["params"]:
            log.info("%s was completed by a concurrent ingest; keeping that one", final)
            return existing, True
        if existing is None:
            # input/ without meta.json is damage (manual edits, partial deletes), never a normal state.
            log.warning("replacing incomplete %s", final)
        else:
            log.info("replacing %s (built with other ingest settings)", final)
        # The .tmp-<tag>-<pid>-<rand> name lets gc_temp finish the removal if rmtree is interrupted.
        aside = job_dir / f".tmp-old-input-{os.getpid()}-{os.urandom(4).hex()}"
        try:
            _rename_dir(final, aside)
        except FileNotFoundError:
            continue
        except PermissionError as e:
            raise IngestError(
                f"기존 입력 폴더를 바꾸지 못했습니다(안의 파일을 다른 프로그램이 쓰는 중): {final} ({e})"
            ) from e
        rmtree_quiet(aside)
    raise IngestError(f"입력 폴더를 만들지 못했습니다(다른 프로세스와 계속 충돌): {final}")
