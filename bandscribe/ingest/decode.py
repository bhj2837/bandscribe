"""Probe and decode any audio container to float32 stereo WAV (DESIGN S0, M0_SPEC 12).

Why float32 end to end: loud commercial masters overshoot full scale once they have been through a
lossy codec (Opus +1.02 dBFS, AAC +1.58 dBFS in docs/research/experiments/codec_exp3.py). An int16
decode would clip those peaks irreversibly, so we keep floats here and bring the level down later
with a plain gain (loudness.apply_ceiling). Downmixing to mono is never done.

Also home of IngestError, the base class of every ingest failure: decode is the lowest module of the
package, so youtube/loudness/__init__ can all import it without cycles.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from bandscribe import paths

log = logging.getLogger(__name__)

# ffmpeg is capped only to avoid hanging forever on a broken pipe or a stuck network share;
# a full album side decodes in well under a minute on this PC.
FFPROBE_TIMEOUT_S = 120
FFMPEG_TIMEOUT_S = 1800
STDERR_TAIL_CHARS = 1500
# Long embedded tags (lyrics, cue sheets) would bloat meta.json without helping anything downstream.
MAX_TAG_CHARS = 1000
_BLOCK_FRAMES = 1 << 18


class IngestError(RuntimeError):
    """Any ingest failure. The message is user-facing (Korean) and printed by the CLI."""


class DecodeError(IngestError):
    """ffprobe/ffmpeg could not read or convert the input."""


def stderr_tail(text: str | None, limit: int = STDERR_TAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def tool_path(name: str) -> str:
    """Resolve ffmpeg/ffprobe on PATH once per call, with an actionable message when missing."""
    exe = shutil.which(name)
    if exe is None:
        raise DecodeError(f"{name} 를 PATH 에서 찾지 못했습니다. ffmpeg(gyan full 빌드)를 설치하고 PATH 에 넣어 주세요.")
    return exe


def run_tool(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg/ffprobe: no shell, UTF-8 text, stdin closed so ffmpeg never waits for a keypress."""
    try:
        return subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=paths.child_env(),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise DecodeError(f"{Path(args[0]).stem} 가 {timeout:.0f} 초 안에 끝나지 않았습니다.") from e
    except OSError as e:
        raise DecodeError(f"{args[0]} 실행 실패: {e}") from e


def _to_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _trim_tags(tags: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in tags.items():
        s = str(v)
        out[str(k)] = s if len(s) <= MAX_TAG_CHARS else s[:MAX_TAG_CHARS] + "..."
    return out


def probe(path: Path) -> dict[str, Any]:
    """ffprobe summary of the first audio stream (the one decode_to_f32 uses) plus container info."""
    path = Path(path)
    proc = run_tool(
        [tool_path("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        timeout=FFPROBE_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise DecodeError(f"오디오 파일로 읽지 못했습니다: {path.name}\n{stderr_tail(proc.stderr)}")
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as e:
        raise DecodeError(f"ffprobe 출력을 해석하지 못했습니다: {path.name} ({e})") from e

    streams = data.get("streams") or []
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise DecodeError(f"오디오 스트림이 없습니다: {path.name}")
    a = audio[0]
    fmt = data.get("format") or {}
    # Stream tags win over container tags: for Ogg/Opus the Vorbis comments live on the stream.
    tags = {**(fmt.get("tags") or {}), **(a.get("tags") or {})}
    return {
        "codec_name": a.get("codec_name"),
        "profile": a.get("profile"),
        "sample_rate": _to_int(a.get("sample_rate")),
        "channels": _to_int(a.get("channels")),
        "channel_layout": a.get("channel_layout"),
        "sample_fmt": a.get("sample_fmt"),
        "bits_per_raw_sample": _to_int(a.get("bits_per_raw_sample")),
        # WebM/Matroska often only carry the container bit rate; for audio-only files that is the audio's.
        "bit_rate": _to_int(a.get("bit_rate")) or _to_int(fmt.get("bit_rate")),
        "duration": _to_float(a.get("duration")) or _to_float(fmt.get("duration")),
        "format_name": fmt.get("format_name"),
        "size": _to_int(fmt.get("size")),
        "n_audio_streams": len(audio),
        "has_video": any(
            s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic") for s in streams
        ),
        "tags": _trim_tags(tags),
    }


def decode_to_f32(src: Path, dst: Path, sr: int = 44100) -> None:
    """Decode the first audio stream to stereo float32 WAV at ``sr`` using soxr.

    pcm_f32le keeps inter-sample overs (> 1.0) intact. A mono source is duplicated to both channels
    (L == R, flagged later as mono_source/quasi_mono); a multichannel source is folded to stereo by
    ffmpeg's standard matrix, since everything downstream is stereo-in.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = [tool_path("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", str(src)]
    args += ["-map", "0:a:0", "-vn", "-af", f"aresample=resampler=soxr:osr={int(sr)}", "-ac", "2"]
    args += ["-c:a", "pcm_f32le", "-f", "wav", str(dst)]
    proc = run_tool(args, timeout=FFMPEG_TIMEOUT_S)
    if proc.returncode != 0 or not dst.is_file():
        raise DecodeError(f"디코딩에 실패했습니다: {src.name}\n{stderr_tail(proc.stderr)}")
    try:
        info = sf.info(str(dst))
    except RuntimeError as e:  # soundfile's errors subclass RuntimeError
        raise DecodeError(f"디코딩 결과를 읽지 못했습니다: {dst.name} ({e})") from e
    if info.frames == 0:
        raise DecodeError(f"디코딩 결과가 비어 있습니다(오디오 길이 0): {src.name}")
    if info.channels != 2 or info.samplerate != sr or info.subtype != "FLOAT":
        raise DecodeError(
            f"디코딩 결과 형식이 예상과 다릅니다: {info.channels}ch {info.samplerate} Hz {info.subtype}"
        )
    log.debug("decoded %s -> %s (%d frames)", src.name, dst.name, info.frames)


def pcm_sha256(wav: Path) -> str:
    """sha256 of the interleaved float32 samples only.

    Independent of file name, container and header/metadata, so the same audio delivered as WAV and
    as FLAC maps to the same song_key. Read in blocks so a long file never sits in RAM twice.
    """
    h = hashlib.sha256()
    with sf.SoundFile(str(wav)) as f:
        for block in f.blocks(blocksize=_BLOCK_FRAMES, dtype="float32", always_2d=True):
            h.update(np.ascontiguousarray(block, dtype="<f4").tobytes())
    return h.hexdigest()
