"""Downloads: one resumable, checksum-verified file transfer, and whole-dataset fetches (M1_M2_SPEC §9.5 item 2).

Rules (§0): exactly one plain HTTP request per URL, with an honest User-Agent. A refusal (403/429/503, a
Cloudflare challenge, an HTML page where a file was expected — e.g. a Google Drive quota or virus-scan
interstitial) is **not** worked around: it becomes :class:`ManualDownloadRequired` with Korean manual steps
(page, files, destination, ``gtab data import-local`` command). Nothing here runs during ``gtab run`` or
under the GPU lock.

``download_file`` streams to ``<dest>.part`` (HTTP Range resume after a cut), hashes while streaming
(sha256 always; md5 when the publisher states one; the SHA1 a Nextcloud/ownCloud server announces in
``OC-Checksum``), checks size and checksums, then renames atomically. A checksum mismatch deletes the part
file and leaves no ``dest``. It is reused by ``gtab.model_fetch``.

Progress goes to stderr (ASCII only: the Windows console is cp949) when ``progress=True``; library callers
pass ``False`` or a callback.
"""

from __future__ import annotations

import hashlib
import html
import http.client
import logging
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gtab import __version__, atomic
from gtab.datasets import registry, store

log = logging.getLogger(__name__)

USER_AGENT = f"gtab/{__version__} (personal research tool; Python-urllib)"
CHUNK = 1 << 20
TIMEOUT_S = 60.0
GDRIVE_LIST_URL = "https://drive.google.com/embeddedfolderview?id={id}"
GDRIVE_FILE_URL = "https://drive.google.com/uc?export=download&id={id}"
# Pause between Drive requests: hundreds of small files; stay a polite client.
GDRIVE_PACE_S = 0.2
# Drive answers an occasional HTTP 500 for one file of hundreds (seen 2026-10-03; the same URL worked seconds
# later). Such server errors are retried a bounded number of times after these pauses (s); refusals
# (403/429/challenge) are never retried.
GDRIVE_RETRY_S = (5.0, 30.0)
# Record Drive progress in the lock every N files (the lock is rewritten as a whole each time).
GDRIVE_RECORD_EVERY = 40
EXTRACT_MARKER = store.EXTRACT_MARKER
_REFUSED = (401, 403, 429, 503)

ProgressFn = Callable[[int, int | None], None]


class FetchError(RuntimeError):
    """Download/extraction failure. The message is user-facing (Korean)."""


class ServerError(FetchError):
    """HTTP 500/502/504: a server-side failure that may be transient (not a refusal)."""

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


class ChecksumMismatch(FetchError):
    pass


class ManualDownloadRequired(FetchError):
    """The server refused a plain download (or needs a login/browser). Carries the manual steps."""

    def __init__(self, message: str, *, url: str = "", status: int | None = None, steps_ko: str = "") -> None:
        super().__init__(message + (f"\n{steps_ko}" if steps_ko else ""))
        self.url = url
        self.status = status
        self.steps_ko = steps_ko


class FetchCancelled(FetchError):
    pass


# ------------------------------------------------------------------------------------------- progress


class _StderrProgress:
    def __init__(self, label: str) -> None:
        self.label = label if label.isascii() else label.encode("ascii", "replace").decode()
        self.t0 = time.monotonic()
        self.last = 0.0

    def __call__(self, done: int, total: int | None) -> None:
        now = time.monotonic()
        if now - self.last < 0.5 and (total is None or done < total):
            return
        self.last = now
        rate = done / max(now - self.t0, 1e-6) / 1e6
        tot = f"/{total / 1e6:.1f}" if total else ""
        pct = f" {100.0 * done / total:5.1f}%" if total else ""
        sys.stderr.write(f"\r  {self.label}  {done / 1e6:.1f}{tot} MB{pct}  {rate:.1f} MB/s   ")
        if total is not None and done >= total:
            sys.stderr.write("\n")
        sys.stderr.flush()


def _progress_fn(progress: bool | ProgressFn, label: str) -> ProgressFn | None:
    if callable(progress):
        return progress
    return _StderrProgress(label) if progress else None


# ------------------------------------------------------------------------------------------ one file


def _request(url: str, start: int) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.Request(url, headers=headers)


def _is_challenge(headers: Any) -> bool:
    return (headers.get("cf-mitigated") or "").lower() == "challenge"


def _refused(url: str, status: int | None, why: str) -> ManualDownloadRequired:
    return ManualDownloadRequired(
        f"서버가 자동 다운로드를 거절했습니다 ({why}): {url}\n"
        "봇 차단을 우회하지 않습니다. 브라우저로 직접 받아 주세요.",
        url=url, status=status,
    )


def _open(url: str, start: int, timeout_s: float) -> Any:
    try:
        return urllib.request.urlopen(_request(url, start), timeout=timeout_s)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            return e  # handled by the caller (range past the end)
        if e.code in _REFUSED or _is_challenge(e.headers):
            raise _refused(url, e.code, f"HTTP {e.code}") from e
        if e.code in (500, 502, 504):
            raise ServerError(f"서버 오류로 받지 못했습니다 (HTTP {e.code}): {url}. 다시 실행하면 이어 받습니다.",
                              status=e.code) from e
        if e.code == 404:
            raise FetchError(f"파일을 찾지 못했습니다 (HTTP 404): {url}") from e
        raise FetchError(f"다운로드 실패 (HTTP {e.code}): {url}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"네트워크 오류로 받지 못했습니다: {url} ({e.reason}). 연결을 확인하고 다시 실행하면 이어 받습니다.") from e
    except TimeoutError as e:
        raise FetchError(f"응답이 없어 중단했습니다: {url}. 다시 실행하면 이어 받습니다.") from e


def _replace_retry(src: Path, dst: Path) -> None:
    for attempt in range(20):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01 * (attempt + 1))


def _oc_sha1(headers: Any) -> str | None:
    m = re.search(r"SHA1:([0-9a-fA-F]{40})", headers.get("OC-Checksum") or "")
    return m.group(1).lower() if m else None


def head_sha1(url: str, timeout_s: float) -> str | None:
    """Server-announced SHA1 (OC-Checksum) for a file finished from an existing complete part file."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return _oc_sha1(r.headers)
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
        return None


def download_file(url: str, dest: Path, *, expected_bytes: int | None = None, md5: str | None = None,
                  progress: bool | ProgressFn = True, timeout_s: float = TIMEOUT_S,
                  allow_html: bool = False) -> dict[str, Any]:
    """Download ``url`` to ``dest`` via ``dest.part`` with resume; verify size/md5; atomic rename.

    Returns ``{"url", "bytes", "sha256", "md5", "verified_against"}`` (``md5`` = computed hex;
    ``verified_against`` = "md5" / "sha1" (server) / None).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    expected_bytes = expected_bytes or None
    md5 = (md5 or "").lower() or None
    fresh_restart_done = False

    while True:
        start = part.stat().st_size if part.exists() else 0
        if expected_bytes is not None and start > expected_bytes:
            part.unlink()
            start = 0
        h_sha, h_md5, h_sha1 = hashlib.sha256(), hashlib.md5(usedforsecurity=False), hashlib.sha1(usedforsecurity=False)
        if start:
            with open(part, "rb") as f:
                while block := f.read(CHUNK):
                    h_sha.update(block)
                    h_md5.update(block)
                    h_sha1.update(block)
        resp = _open(url, start, timeout_s)
        if isinstance(resp, urllib.error.HTTPError):  # 416: nothing left to send
            resp.close()
            if expected_bytes is None or start == expected_bytes:
                total, server_sha1 = start, head_sha1(url, timeout_s)
                break
            if fresh_restart_done:
                raise FetchError(f"서버가 이어 받기를 거절했습니다 (HTTP 416): {url}")
            part.unlink(missing_ok=True)  # stale part of another size: start over once
            fresh_restart_done = True
            continue
        with resp:
            status = resp.status
            if _is_challenge(resp.headers):
                raise _refused(url, status, "Cloudflare challenge")
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype.startswith("text/html") and not allow_html:
                raise _refused(url, status, "파일 대신 HTML 페이지가 왔습니다(로그인·용량 제한·경고 페이지)")
            if status == 200 and start:
                # Server ignored the Range header: restart from zero.
                start = 0
                h_sha, h_md5, h_sha1 = hashlib.sha256(), hashlib.md5(usedforsecurity=False), hashlib.sha1(usedforsecurity=False)
            length = resp.headers.get("Content-Length")
            total_hint = (start + int(length)) if length and length.isdigit() else expected_bytes
            server_sha1 = _oc_sha1(resp.headers) if start == 0 else None
            show = _progress_fn(progress, dest.name)
            done = start
            with open(part, "ab" if start else "wb") as f:
                while True:
                    try:
                        block = resp.read(CHUNK)
                    except (TimeoutError, OSError, http.client.HTTPException) as e:
                        raise FetchError(f"받는 중 연결이 끊겼습니다: {url} ({e}). 다시 실행하면 이어 받습니다.") from e
                    if not block:
                        break
                    f.write(block)
                    h_sha.update(block)
                    h_md5.update(block)
                    h_sha1.update(block)
                    done += len(block)
                    if show:
                        show(done, total_hint)
            if isinstance(show, _StderrProgress) and (total_hint is None or done < total_hint):
                sys.stderr.write("\n")  # the size check below reports a short read
            total = done
        break

    verified = None
    if expected_bytes is not None and total != expected_bytes:
        if total < expected_bytes:
            raise FetchError(f"받은 크기가 모자랍니다 ({total} / {expected_bytes} B): {dest.name}. 다시 실행하면 이어 받습니다.")
        part.unlink(missing_ok=True)
        raise ChecksumMismatch(f"받은 크기가 기록과 다릅니다 ({total} != {expected_bytes} B): {dest.name}")
    got_md5 = h_md5.hexdigest()
    if md5 is not None:
        if got_md5 != md5:
            part.unlink(missing_ok=True)
            raise ChecksumMismatch(f"MD5 가 맞지 않습니다: {dest.name} (받음 {got_md5}, 기대 {md5}). 파일을 지웠습니다.")
        verified = "md5"
    elif server_sha1 is not None:
        if h_sha1.hexdigest() != server_sha1:
            part.unlink(missing_ok=True)
            raise ChecksumMismatch(f"서버가 알려 준 SHA1 과 맞지 않습니다: {dest.name}. 파일을 지웠습니다.")
        verified = "sha1"
    _replace_retry(part, dest)
    log.debug("downloaded %s (%d B, sha256 %s)", dest, total, h_sha.hexdigest())
    return {"url": url, "bytes": total, "sha256": h_sha.hexdigest(), "md5": got_md5, "verified_against": verified}


# ------------------------------------------------------------------------------------------ extraction


def _safe_member(name: str) -> bool:
    p = Path(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts and not re.match(r"^[A-Za-z]:", name)


def extract_archive(archive: Path, dest_dir: Path, *, sha256: str = "") -> Path:
    """Unpack a zip/tar into ``dest_dir`` atomically (temp dir + rename); idempotent via a marker file."""
    archive, dest_dir = Path(archive), Path(dest_dir)
    marker = dest_dir / EXTRACT_MARKER
    if marker.is_file():
        try:
            m = atomic.read_json(marker)
            # same archive and nothing deleted since -> done (a deleted file triggers a fresh unpack)
            if m.get("sha256") == sha256 and not store.extraction_missing(dest_dir, m):
                return dest_dir
        except (OSError, ValueError):
            pass
    tmp = store.make_scratch_dir(dest_dir.parent, f"extract-{dest_dir.name}")
    members: list[str] = []
    try:
        name = archive.name.lower()
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                bad = [m for m in z.namelist() if not _safe_member(m)]
                if bad:
                    raise FetchError(f"압축 파일에 안전하지 않은 경로가 있습니다: {bad[:3]}")
                for info in z.infolist():
                    if info.filename.startswith("__MACOSX/") or Path(info.filename).name.startswith("._"):
                        continue
                    target = z.extract(info, tmp)  # the sanitised path actually written (Windows-illegal chars)
                    if not info.is_dir():
                        members.append(Path(target).relative_to(tmp).as_posix())
        elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            with tarfile.open(archive) as t:
                t.extractall(tmp, filter="data")
                members = [m.name for m in t.getmembers() if m.isfile()]
        else:
            raise FetchError(f"압축 형식을 모릅니다: {archive.name}")
        n = len(members)
        # the member list lets verify / a re-run notice a deleted file even when the user added others
        atomic.write_json(tmp / EXTRACT_MARKER, {"archive": archive.name, "sha256": sha256, "files": n,
                                                 "members": sorted(members)})
        aside = None
        if dest_dir.exists():  # stale/foreign extraction: move aside first so the swap is one rename
            _carry_over(dest_dir, tmp)
            aside = store.make_scratch_dir(dest_dir.parent, f"old-{dest_dir.name}")
            aside.rmdir()
            _replace_retry(dest_dir, aside)
        _replace_retry(tmp, dest_dir)
        if aside is not None:
            store.remove_tree(aside)
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as e:
        store.remove_tree(tmp)
        raise FetchError(f"압축을 풀지 못했습니다: {archive.name} ({e})") from e
    except BaseException:
        store.remove_tree(tmp)
        raise
    log.info("extracted %s -> %s (%d files)", archive.name, dest_dir, n)
    return dest_dir


def _carry_over(old: Path, new: Path) -> None:
    """Copy files that the archive does not contain from an old extraction into the new one.

    They were added after unpacking — above all a Cambridge-MT ``lines.yaml`` holding the user's by-ear labels,
    which a re-extraction must never wipe. Files the archive does contain are restored from the archive.
    """
    for dirpath, _, filenames in os.walk(old):
        for fn in filenames:
            src = Path(dirpath) / fn
            rel = src.relative_to(old)
            if fn == EXTRACT_MARKER or (new / rel).exists():
                continue
            (new / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, new / rel)
            log.info("kept %s from the previous extraction", rel.as_posix())


def _archive_stem(name: str) -> str:
    return store.archive_stem(name)


# ------------------------------------------------------------------------------------- Google Drive


_GDRIVE_ENTRY_RE = re.compile(
    r'<div class="flip-entry" id="entry-(?P<id>[A-Za-z0-9_-]+)".*?<a href="(?P<href>[^"]+)".*?'
    r'<div class="flip-entry-title">(?P<title>[^<]*)</div>', re.S)


def parse_gdrive_listing(page: str) -> list[dict[str, str]]:
    """Entries of a public Drive folder's ``embeddedfolderview`` page: ``[{"id", "name", "type"}]``."""
    out = []
    for m in _GDRIVE_ENTRY_RE.finditer(page):
        kind = "folder" if "/drive/folders/" in m.group("href") else "file"
        out.append({"id": m.group("id"), "name": html.unescape(m.group("title")).strip(), "type": kind})
    return out


def _http_text(url: str, timeout_s: float = TIMEOUT_S) -> str:
    resp = _open(url, 0, timeout_s)
    with resp:
        if _is_challenge(resp.headers):
            raise _refused(url, resp.status, "Cloudflare challenge")
        return resp.read().decode("utf-8", "replace")


def list_gdrive_folder(folder_id: str, *, prefix: str = "", depth: int = 3,
                       fetch_text: Callable[[str], str] = _http_text) -> list[dict[str, str]]:
    """Recursive listing ``[{"id", "path"}]`` (path relative to the folder, POSIX). Hidden files skipped."""
    items = parse_gdrive_listing(fetch_text(GDRIVE_LIST_URL.format(id=folder_id)))
    files = []
    for it in sorted(items, key=lambda d: d["name"]):
        if it["name"].startswith(".") or not it["name"]:
            continue
        path = f"{prefix}{it['name']}"
        if it["type"] == "folder":
            if depth > 0:
                files += list_gdrive_folder(it["id"], prefix=path + "/", depth=depth - 1, fetch_text=fetch_text)
        else:
            files.append({"id": it["id"], "path": path})
    return files


# ------------------------------------------------------------------------------------------- datasets


@dataclass
class PlanItem:
    rel: str                       # path relative to the dataset dir, e.g. "raw/annotation.zip"
    url: str
    bytes: int = 0
    md5: str = ""
    extract: bool = False
    song: str = ""
    title: str = ""


@dataclass
class FetchPlan:
    spec: registry.DatasetSpec
    root: Path
    items: list[PlanItem] = field(default_factory=list)
    folders: list[registry.RegistryFolder] = field(default_factory=list)

    @property
    def known_bytes(self) -> int:
        return sum(i.bytes for i in self.items)

    def describe_ko(self) -> str:
        spec = self.spec
        size = self.known_bytes
        if self.folders:
            size = max(size, spec.approx_bytes)
            size_txt = f"약 {size / 1e9:.2f} GB (Drive 폴더라 받기 전에는 정확한 크기를 모릅니다)"
        else:
            size_txt = f"{size / 1e6:,.1f} MB" if size < 1e9 else f"{size / 1e9:.2f} GB"
        lines = [
            f"데이터셋: {spec.name} - {spec.title} (Tier {spec.tier})",
            f"출처: {spec.page}",
            f"라이선스: {spec.license}",
            f"크기: {size_txt}",
            f"저장 위치: {store.dataset_dir(spec.name, self.root)}",
            "받을 파일:",
        ]
        for it in self.items:
            lines.append(f"  - {it.rel}  ({it.bytes / 1e6:,.1f} MB)  {it.url}" if it.bytes else f"  - {it.rel}  {it.url}")
        for d in self.folders:
            lines.append(f"  - raw/{d.name}/ (Google Drive 폴더, 파일 약 {d.expected_files}개)")
        lines.append("개인 연구용으로 이 PC 안에만 두며 재배포하지 않습니다.")
        return "\n".join(lines)


def plan_fetch(name: str, *, parts: list[str] | None = None, root: Path | None = None) -> FetchPlan:
    """Which files/folders a fetch would get. ``parts`` = labels, or ``["all"]``; default = required."""
    spec = registry.get(name)
    root = Path(root or store.default_root())
    if spec.access == "manual":
        raise ManualDownloadRequired(
            f"{spec.name} 는 자동으로 받을 수 없습니다 (접근 승인·약관 동의가 필요).", url=spec.page, steps_ko=spec.manual_ko)
    known = spec.parts()
    if parts:
        want = set(known) if "all" in parts else set(parts)
        unknown = sorted(want - set(known))
        if unknown:
            raise FetchError(f"{spec.name} 에 없는 part 입니다: {unknown}. 가능한 값: {', '.join(known)} 또는 all")
    else:
        want = set(spec.required_parts())
    plan = FetchPlan(spec=spec, root=root)
    for f in spec.files:
        if f.label in want:
            plan.items.append(PlanItem(rel=f"raw/{f.name}", url=f.url, bytes=f.bytes, md5=f.md5, extract=f.extract,
                                       song=f.song, title=f.title))
    plan.folders = [d for d in spec.folders if d.name in want]
    return plan


def _already_have(base: Path, rel: str, rec: dict[str, Any] | None, item: PlanItem) -> dict[str, Any] | None:
    """Existing complete file -> its (possibly new) lock record; None -> must download.

    ``dest`` only ever appears by the atomic rename after a finished, checked download, so a present file
    without a lock record (crash between rename and lock update) is re-hashed and accepted if it matches the
    registry (md5 when published, else size).
    """
    p = base / rel
    if not p.is_file():
        return None
    size = p.stat().st_size
    if item.bytes and size != item.bytes:
        return None
    if rec and rec.get("bytes") == size and rec.get("sha256"):
        return rec
    sha, md5 = store.hash_file(p, md5=True)
    if item.md5 and md5 != item.md5.lower():
        return None
    return {"url": item.url, "bytes": size, "sha256": sha, "md5": md5, "fetched_utc": store.utc_now(),
            "verified_against": "md5" if item.md5 else None}


def execute(plan: FetchPlan, *, progress: bool | ProgressFn = True,
            notify: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run a plan: download (resume), record each file in the lock, extract archives. Returns a summary."""
    spec, root = plan.spec, plan.root
    base = store.dataset_dir(spec.name, root)
    say = notify or (lambda s: log.info("%s", s))
    have = (store.entry(spec.name, root) or {}).get("files", {})
    got, skipped, extracted = 0, 0, []
    for item in plan.items:
        rec = _already_have(base, item.rel, have.get(item.rel), item)
        if rec is None:
            say(f"받는 중: {item.rel}")
            try:
                res = download_file(item.url, base / item.rel, expected_bytes=item.bytes or None, md5=item.md5 or None,
                                    progress=progress)
            except ManualDownloadRequired as e:
                raise ManualDownloadRequired(str(e).split("\n")[0], url=e.url, status=e.status,
                                             steps_ko=spec.manual_ko) from e
            rec = {**res, "fetched_utc": store.utc_now()}
            got += 1
        else:
            skipped += 1
        if have.get(item.rel) != rec:
            store.record_files(spec.name, {item.rel: rec}, status="fetched", root=root)
        if item.extract:
            stem = _archive_stem(Path(item.rel).name)
            out = extract_archive(base / item.rel, base / "extracted" / stem, sha256=rec["sha256"])
            extracted.append(out)
            if spec.name == "cambridge_mt" and item.song:
                _cmt_after_extract(out, item)

    for folder in plan.folders:
        g, s = _fetch_gdrive_folder(spec, folder, base, root, have, progress=progress, say=say)
        got += g
        skipped += s
    return {"name": spec.name, "downloaded": got, "skipped": skipped, "extracted": [str(p) for p in extracted],
            "status": store.status(spec.name, root)}


def _download_retrying(url: str, dest: Path) -> dict[str, Any]:
    """``download_file`` with bounded retries of server errors only (see :data:`GDRIVE_RETRY_S`)."""
    for attempt in range(len(GDRIVE_RETRY_S) + 1):
        try:
            return download_file(url, dest, progress=False)
        except ServerError as e:
            if attempt == len(GDRIVE_RETRY_S):
                raise
            log.warning("HTTP %s for %s, retrying in %.0f s", e.status, url, GDRIVE_RETRY_S[attempt])
            time.sleep(GDRIVE_RETRY_S[attempt])
    raise AssertionError("unreachable")


def _fetch_gdrive_folder(spec: registry.DatasetSpec, folder: registry.RegistryFolder, base: Path, root: Path,
                         have: dict[str, Any], *, progress: bool | ProgressFn,
                         say: Callable[[str], None]) -> tuple[int, int]:
    say(f"Google Drive 폴더 목록 읽는 중: {folder.name}")
    try:
        listing = list_gdrive_folder(folder.folder_id, prefix=f"{folder.name}/")
    except ManualDownloadRequired as e:
        raise ManualDownloadRequired(str(e).split("\n")[0], url=e.url, status=e.status, steps_ko=spec.manual_ko) from e
    if folder.expected_files and len(listing) < folder.expected_files:
        log.warning("%s: listed %d files, expected %d", folder.name, len(listing), folder.expected_files)
    pending: dict[str, dict[str, Any]] = {}
    got = skipped = 0

    def flush() -> None:
        if pending:
            store.record_files(spec.name, dict(pending), status="fetched", root=root)
            pending.clear()

    try:
        for i, it in enumerate(listing, 1):
            rel = f"raw/{it['path']}"
            item = PlanItem(rel=rel, url=GDRIVE_FILE_URL.format(id=it["id"]))
            rec = _already_have(base, rel, have.get(rel), item)
            if rec is None:
                if got:
                    time.sleep(GDRIVE_PACE_S)
                res = _download_retrying(item.url, base / rel)
                rec = {**res, "fetched_utc": store.utc_now()}
                got += 1
                if progress is True and (got % 20 == 0 or i == len(listing)):
                    sys.stderr.write(f"\r  {folder.name}: {i}/{len(listing)}   ")
                    sys.stderr.flush()
            else:
                skipped += 1
            if have.get(rel) != rec:
                pending[rel] = rec
            if len(pending) >= GDRIVE_RECORD_EVERY:
                flush()
    except ManualDownloadRequired as e:
        flush()
        raise ManualDownloadRequired(
            f"Google Drive 가 자동 다운로드를 멈췄습니다 ({folder.name}, {got}개 받음). 잠시 뒤 같은 명령을 다시 실행하면 이어 받습니다.",
            url=e.url, status=e.status, steps_ko=spec.manual_ko) from e
    finally:
        flush()
    if progress is True and got:
        sys.stderr.write("\n")
    return got, skipped


def _cmt_after_extract(out: Path, item: PlanItem) -> None:
    """Cambridge-MT: write the lines.yaml draft into the song folder of a freshly extracted archive."""
    song_dir = cmt_song_dir(out)
    wavs = sorted(p.relative_to(song_dir).as_posix() for p in song_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in store.AUDIO_EXTS)
    store.write_lines_skeleton(song_dir, wavs, song=item.song, title=item.title)


def cmt_song_dir(extracted: Path) -> Path:
    """The folder holding the stems: archives usually wrap them in one top folder."""
    entries = [p for p in extracted.iterdir() if p.name != EXTRACT_MARKER and not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted


def fetch(name: str, *, parts: list[str] | None = None, yes: bool = False,
          confirm: Callable[[str], bool] | None = None, root: Path | None = None,
          progress: bool | ProgressFn = True, notify: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Plan, confirm (unless ``yes``), execute. ``confirm`` gets the Korean plan text and returns y/N."""
    plan = plan_fetch(name, parts=parts, root=root)
    if not yes:
        if confirm is None or not confirm(plan.describe_ko()):
            raise FetchCancelled("취소했습니다.")
    return execute(plan, progress=progress, notify=notify)


def free_space_ok(root: Path, need_bytes: int) -> bool:
    try:
        return shutil.disk_usage(root if root.exists() else root.anchor).free > need_bytes * 2.2
    except OSError:
        return True
