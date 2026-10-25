"""Tab videos on disk and their download (docs/TAB_VIDEO.md): ``data/eval/tabvid/`` holds ``videos/<id>.mp4``,
``frames/<id>/`` and ``read/<id>_n<strings>.json``. The download is the YouTube ingest's (our yt-dlp config only, no
cookies, 5 s between downloads, ``youtube.enabled``) with a video format instead of audio."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

from bandscribe import paths

VIDEO_FORMAT = "bv*[height<=1080][ext=mp4]/bv*[height<=1080]"
VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv")


def root() -> Path:
    return paths.EVAL / "tabvid"  # looked up at call time (tests repoint EVAL)


def video_id(source: str) -> str | None:
    """The YouTube id of a URL or a bare 11-character id, else None."""
    from bandscribe.ingest import youtube as Y

    s = (source or "").strip()
    if len(s) == 11 and Y.parse_video_id(f"https://youtu.be/{s}") == s:
        return s
    return Y.parse_video_id(s)


def find_video(vid: str) -> Path | None:
    d = root() / "videos"
    found = [p for p in d.glob(f"{vid}.*") if p.suffix.lower() in VIDEO_SUFFIXES] if d.is_dir() else []
    return max(found, key=lambda p: p.stat().st_size) if found else None


def download(vid: str, cfg: Any) -> Path:
    """The video of ``vid`` (1080p at most, no audio), downloaded once into ``videos/``."""
    from bandscribe.ingest import youtube as Y

    have = find_video(vid)
    if have is not None:
        return have
    if not cfg.youtube.enabled:
        raise Y.YoutubeError("YouTube 입력이 꺼져 있습니다 (youtube.enabled = false). 영상 파일을 직접 주세요.")
    y = cfg.youtube
    vcfg = types.SimpleNamespace(youtube=types.SimpleNamespace(format=VIDEO_FORMAT, cookies=y.cookies,
                                                               po_token=y.po_token, timeout_s=y.timeout_s))
    path, _info = Y.download(Y.canonical_url(vid), root() / "videos", vcfg)
    return path
