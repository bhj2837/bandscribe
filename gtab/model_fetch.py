"""`gtab models fetch|verify|list` backend (DATA owner, M1_M2_SPEC §9.5 item 7; DESIGN §9.4).

Sources come from ``gtab/model_sources.toml`` (committed). A fetch shows URL, size (HTTP HEAD), license and
destination, asks y/N (``yes`` skips), downloads with :func:`gtab.datasets.fetch.download_file` (resume,
server-announced SHA1 checked) and records the file through P0's ``gtab.models.record`` in
``data/models/models.lock.json``. A pinned ``sha256`` in the source must match (else the file is deleted
and nothing is recorded); without one, the first download pins it (trust on first use).

This module is **never** used by ``gtab run`` or a worker: stages only look models up
(``gtab.models.local_path``) and fail at plan time when one is missing — so nothing downloads under the
GPU lock and stage keys never change because a model appeared mid-run (§0, A19).
"""

from __future__ import annotations

import difflib
import logging
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gtab import paths
from gtab.datasets.registry import is_allowed_url

log = logging.getLogger(__name__)

SOURCES_PATH: Path = Path(__file__).with_name("model_sources.toml")
SOURCES_FORMAT = "gtab.model_sources/1"


class ModelFetchError(RuntimeError):
    """User-facing (Korean) model fetch/verify error."""


@dataclass(frozen=True)
class ModelSource:
    name: str
    url: str
    dest: str            # POSIX, relative to GTAB_ROOT
    bytes: int = 0
    sha256: str = ""
    license: str = ""
    source: str = ""
    notes: str = ""

    def dest_path(self) -> Path:
        return paths.GTAB_ROOT / self.dest


def load_sources(path: Path | None = None) -> dict[str, ModelSource]:
    p = Path(path or SOURCES_PATH)
    with open(p, "rb") as f:
        data = tomllib.load(f)
    if data.get("format") != SOURCES_FORMAT:
        raise ModelFetchError(f"{p.name}: format 이 {SOURCES_FORMAT} 가 아닙니다")
    out = {}
    for name, d in data.items():
        if name == "format":
            continue
        if not isinstance(d, dict) or not is_allowed_url(str(d.get("url", ""))) or not d.get("dest"):
            raise ModelFetchError(f"{p.name} [{name}]: url(https://) 와 dest 가 필요합니다")
        if Path(d["dest"]).is_absolute() or ".." in Path(d["dest"]).parts:
            raise ModelFetchError(f"{p.name} [{name}]: dest 는 GTAB_ROOT 기준 상대 경로여야 합니다")
        if not str(d.get("license", "")).strip():
            raise ModelFetchError(f"{p.name} [{name}]: license 가 비어 있습니다")
        out[name] = ModelSource(name=name, **d)
    return out


def _lock_entries() -> dict[str, Any]:
    """Recorded models by name. A corrupt lock raises (``gtab.models.ModelRegistryError``), never reads as empty:
    treating it as empty would let a fetch rewrite the lock without the hand-seeded local-only entries."""
    from gtab import models

    return models.entries()


def head_size(url: str, timeout_s: float = 20.0) -> int | None:
    from gtab.datasets.fetch import USER_AGENT

    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n and n.isdigit() else None
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def describe_ko(src: ModelSource, size: int | None) -> str:
    sz = size or src.bytes
    return "\n".join([
        f"모델: {src.name}",
        f"URL: {src.url}",
        f"크기: {sz / 1e6:,.1f} MB" if sz else "크기: 알 수 없음",
        f"라이선스: {src.license}",
        f"출처: {src.source}" if src.source else "",
        f"저장 위치: {src.dest_path()}",
        f"SHA256: {src.sha256 or '공개된 값 없음 - 처음 받은 파일의 값을 고정합니다(TOFU)'}",
    ]).replace("\n\n", "\n")


def _get_source(name: str, sources: dict[str, ModelSource]) -> ModelSource:
    if name in sources:
        return sources[name]
    if name in _lock_entries():
        raise ModelFetchError(f"'{name}' 은(는) 로컬 전용 모델이라 받을 URL 이 없습니다 (models.lock.json 에만 있음).")
    close = difflib.get_close_matches(name, list(sources), n=1)
    hint = f" '{close[0]}' 을(를) 말씀하신 건가요?" if close else ""
    raise ModelFetchError(f"모르는 모델입니다: '{name}'.{hint} (받을 수 있는 모델: {', '.join(sources) or '없음'})")


_NOTE_TAIL = {
    "md5": " [download checked against the publisher's md5]",
    "sha1": " [download checked against the server-announced SHA1]",
    "adopted+sha256": " [existing file adopted; matches the pinned sha256]",
    "adopted+sha1": " [existing file adopted; matches the server-announced SHA1]",
    "adopted": " [existing file adopted without a server checksum; sha256 pinned on first record]",
    None: " [sha256 pinned on first download]",
}


def _entry(src: ModelSource, *, nbytes: int, sha256: str, verified: str | None) -> Any:
    from gtab import models
    from gtab.datasets.store import utc_now

    notes = (src.notes + _NOTE_TAIL.get(verified, _NOTE_TAIL[None])).strip()
    return models.ModelEntry(name=src.name, path=src.dest, url=src.url, bytes=nbytes, sha256=sha256,
                             license=src.license, source=src.source or "gtab models fetch", notes=notes,
                             recorded_utc=utc_now())


def _file_sha1(path: Path) -> str:
    import hashlib

    h = hashlib.sha1(usedforsecurity=False)
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, *, yes: bool = False, confirm: Callable[[str], bool] | None = None,
          progress: bool | Callable[[int, int | None], None] = True, sources_path: Path | None = None) -> Any:
    """Download (if needed) and record one model; returns the recorded ``gtab.models.ModelEntry``."""
    from gtab import models
    from gtab.datasets.fetch import ChecksumMismatch, FetchCancelled, download_file, head_sha1

    src = _get_source(name, load_sources(sources_path))
    dest = src.dest_path()
    rec = models.get(name)
    pin = src.sha256.lower()
    if (rec is not None and dest.is_file() and dest.stat().st_size == rec.bytes
            and models.sha256_cached(dest) == rec.sha256 and (not pin or pin == rec.sha256)):
        log.info("%s already present and recorded", name)
        return rec

    if dest.is_file():
        # A complete file only appears via download_file's atomic rename (e.g. the lock write was lost) or was
        # put there by hand: accept it if it matches the pin, instead of downloading 80+ MB again.
        sha = models.sha256_cached(dest)
        if pin and sha != pin:
            log.warning("%s: existing file does not match the pinned sha256, downloading again", dest)
        elif rec is not None and rec.sha256 and sha != rec.sha256:
            log.warning("%s: existing file differs from the recorded sha256, downloading again", dest)
        else:
            # Without a pin, ask the server for its checksum (one HEAD request) so trust-on-first-use is not blind.
            server_sha1 = None if pin else head_sha1(src.url, 20.0)
            if server_sha1 is not None and _file_sha1(dest) != server_sha1:
                log.warning("%s: existing file does not match the server SHA1, downloading again", dest)
            else:
                how = "adopted+sha256" if pin else ("adopted+sha1" if server_sha1 else "adopted")
                entry = _entry(src, nbytes=dest.stat().st_size, sha256=sha, verified=how)
                models.record(entry)
                return entry

    if not yes:
        if confirm is None or not confirm(describe_ko(src, head_size(src.url))):
            raise FetchCancelled("취소했습니다.")
    res = download_file(src.url, dest, expected_bytes=src.bytes or None, progress=progress)
    if pin and res["sha256"] != pin:
        dest.unlink(missing_ok=True)
        raise ChecksumMismatch(f"{name}: SHA256 이 고정값과 다릅니다 (받음 {res['sha256']}). 파일을 지웠고 기록하지 않았습니다.")
    entry = _entry(src, nbytes=res["bytes"], sha256=res["sha256"], verified=res.get("verified_against"))
    models.record(entry)
    return entry


def verify(name: str | None = None) -> list[tuple[str, bool]]:
    """Recompute sha256 via ``gtab.models.verify`` for one or every recorded model."""
    from gtab import models

    names = [name] if name else sorted(_lock_entries())
    out = []
    for n in names:
        if models.get(n) is None:
            raise ModelFetchError(f"models.lock.json 에 없는 모델입니다: {n}")
        try:
            ok = bool(models.verify(n))
        except FileNotFoundError:
            ok = False
        out.append((n, ok))
    return out


def list_models(*, sources_path: Path | None = None) -> list[dict[str, Any]]:
    """Sources merged with the lock: name, bytes, license, status, path, url, local_only."""
    from gtab import models

    sources = load_sources(sources_path)
    lock = _lock_entries()
    rows = []
    for name in list(sources) + sorted(n for n in lock if n not in sources):
        src, rec = sources.get(name), lock.get(name)
        if rec is not None and rec.path:
            p: Path | None = models.resolve(rec)
        else:
            p = src.dest_path() if src else None
        if rec is None:
            st = "unrecorded" if p is not None and p.is_file() else "absent"
        elif p is None or not p.exists():
            st = "missing_file"
        elif p.is_file() and rec.bytes and p.stat().st_size != rec.bytes:
            st = "size_mismatch"
        else:
            st = "recorded"
        rows.append({"name": name, "bytes": (rec.bytes if rec else 0) or (src.bytes if src else 0),
                     "license": (rec.license if rec else "") or (src.license if src else ""), "status": st,
                     "path": str(p) if p else "", "url": src.url if src else (rec.url if rec else ""),
                     "local_only": src is None})
    return rows
