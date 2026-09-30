"""Content-addressed job/artifact store (DESIGN §9.3, M0_SPEC §6).

Layout under ``root`` (default ``paths.JOBS``)::

    index.json / index.lock            "file:<sha256>" | "yt:<videoId>" -> song_key
    <song_key>/input/                  built atomically by gtab.ingest
    <song_key>/stages/<stage>/<key12>/ one directory per (stage, key); manifest.json marks it complete
    **/.tmp-<tag>-<pid>-<rand>/        scratch dirs; renamed into place or removed by gc_temp()

Invariant: a final stage directory only ever appears through a single directory rename of a
fully written temp dir (manifest included), so readers never observe a half-written result.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import secrets
import shutil
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from gtab import atomic, paths

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
TMP_PREFIX = ".tmp-"
KEY_PREFIX_LEN = 12
INDEX_LOCK_TIMEOUT_S = 30.0

# Fields stage_writer owns in manifest.json; callers' `meta` may not override them.
_RESERVED_META = frozenset({"stage", "key", "song_key", "status", "created_utc", "duration_s", "outputs"})

# ASCII only (DESIGN §9.8). No trailing dot: Windows silently strips it, which would alias two names.
_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9_-])?$")
_KEY_RE = re.compile(r"^[0-9a-f]{12,}$")
# The tag may itself contain dashes (e.g. "old-<key12>"); pid and rand are the last two fields.
_TMP_RE = re.compile(r"^\.tmp-(?P<tag>.+)-(?P<pid>\d+)-(?P<rand>[0-9a-z]+)$")
_WIN_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
)

# Windows: a directory rename fails with ERROR_ACCESS_DENIED while any handle inside it is open,
# including short-lived ones from Defender or the search indexer. Retry those briefly.
_RENAME_RETRIES = 12
_RENAME_BACKOFF_S = 0.05


class StoreError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# helpers


def _check_name(kind: str, name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"invalid {kind} {name!r}: use ASCII letters, digits, '_', '-', '.'")
    if name.split(".")[0].upper() in _WIN_RESERVED:
        raise ValueError(f"invalid {kind} {name!r}: reserved device name on Windows")
    return name


def _check_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise ValueError(f"invalid stage key {key!r}: expected lowercase hex (>= 12 chars)")
    return key


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing. Strict on purpose: a Path or set in params would make keys
    machine- or run-dependent, so those raise TypeError instead of being coerced."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _scratch_path(parent: Path, tag: str) -> Path:
    return Path(parent) / f"{TMP_PREFIX}{tag}-{os.getpid()}-{secrets.token_hex(4)}"


def make_temp_dir(parent: Path, tag: str) -> Path:
    """Create ``parent/.tmp-<tag>-<pid>-<rand>``. Shared naming lets gc_temp() find scratch dirs
    of dead processes (also usable by ingest when it builds ``input/``)."""
    Path(parent).mkdir(parents=True, exist_ok=True)
    while True:
        p = _scratch_path(parent, tag)
        try:
            p.mkdir()
            return p
        except FileExistsError:
            continue


def _on_rm_error(func: Any, path: str, exc: BaseException) -> None:
    # Read-only files (e.g. copied from a CD rip or git objects) block unlink on Windows.
    if isinstance(exc, PermissionError):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
            return
        except OSError:
            pass
    raise exc


def rmtree_quiet(path: Path) -> bool:
    """Best-effort recursive delete. Returns False (and logs) instead of raising; whatever is
    left behind keeps its .tmp- name and is retried by gc_temp()."""
    for attempt in range(3):
        try:
            shutil.rmtree(path, onexc=_on_rm_error)
            return True
        except FileNotFoundError:
            return True
        except OSError as e:
            if attempt == 2:
                log.warning("could not remove %s: %s", path, e)
                return False
            time.sleep(0.1 * (attempt + 1))
    return False


def _rename_dir(src: Path, dst: Path) -> None:
    """os.rename with retries on transient Windows sharing violations.

    os.rename (not os.replace): on Windows it fails with FileExistsError when ``dst`` exists,
    which is exactly the "someone else already finished" signal we need. FileExistsError is
    raised immediately; PermissionError is retried, then re-raised."""
    for attempt in range(_RENAME_RETRIES):
        try:
            os.rename(src, dst)
            return
        except FileExistsError:
            raise
        except PermissionError:
            if dst.exists():
                # POSIX-style ENOTEMPTY/EACCES for an existing target: same meaning as above.
                raise FileExistsError(str(dst)) from None
            if attempt == _RENAME_RETRIES - 1:
                raise
            time.sleep(_RENAME_BACKOFF_S * (attempt + 1))
        except OSError as e:
            if dst.exists():
                raise FileExistsError(str(dst)) from e
            raise


def _hash_outputs(root: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == MANIFEST:
            continue
        outputs[rel] = {"bytes": p.stat().st_size, "sha256": atomic.sha256_file(p)}
    return outputs


# --------------------------------------------------------------------------------------------
# process liveness (for gc_temp)

_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87
_FILETIME_UNIX_EPOCH = 116444736000000000  # 100 ns ticks between 1601-01-01 and 1970-01-01


@functools.cache
def _kernel32() -> Any:
    import ctypes
    from ctypes import wintypes

    # Private WinDLL instance so these prototypes don't leak into other modules' ctypes use.
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetExitCodeProcess.restype = wintypes.BOOL
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetProcessTimes.restype = wintypes.BOOL
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k32.CloseHandle.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    return k32


def process_start_time(pid: int) -> float | None:
    """Unix time at which ``pid`` started, or None if the process does not exist.

    Raises PermissionError when the process exists but cannot be queried (other session /
    protected), so callers can treat it as alive."""
    if sys.platform != "win32":  # pragma: no cover - gtab targets Windows; minimal fallback
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        return 0.0

    import ctypes
    from ctypes import wintypes

    k32 = _kernel32()
    h = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        err = ctypes.get_last_error()
        if err == _ERROR_INVALID_PARAMETER:
            return None
        raise PermissionError(err, f"OpenProcess({pid}) failed")
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            raise PermissionError(ctypes.get_last_error(), f"GetExitCodeProcess({pid}) failed")
        # A process object can outlive the process while someone holds a handle to it.
        if code.value != _STILL_ACTIVE:
            return None
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not k32.GetProcessTimes(h, *(ctypes.byref(t) for t in (created, exited, kernel, user))):
            return 0.0  # alive, start time unknown
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return (ticks - _FILETIME_UNIX_EPOCH) / 1e7
    finally:
        k32.CloseHandle(h)


def _owner_alive(pid: int, dir_created: float, slack_s: float = 2.0) -> bool:
    """True if the process that created a temp dir may still be running.

    Windows recycles pids quickly, so "pid exists" is not enough: if the process currently
    holding that pid started *after* the directory was created, it cannot be the writer."""
    try:
        started = process_start_time(pid)
    except PermissionError:
        return True  # exists but opaque: be conservative
    if started is None:
        return False
    return started <= dir_created + slack_s


def _dir_created(p: Path) -> float:
    st = p.stat()
    return getattr(st, "st_birthtime", st.st_ctime)


# --------------------------------------------------------------------------------------------


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        # Resolved at call time (not as a default arg) so GTAB_ROOT / monkeypatched paths apply.
        self.root = Path(root) if root is not None else paths.JOBS

    # ---- layout ------------------------------------------------------------------------------

    def _job_path(self, song_key: str) -> Path:
        return self.root / _check_name("song_key", song_key)

    def job_dir(self, song_key: str) -> Path:
        p = self._job_path(song_key)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def input_dir(self, song_key: str) -> Path:
        # Not created: ingest renames a finished temp dir onto this path.
        return self._job_path(song_key) / "input"

    def list_jobs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and _NAME_RE.match(p.name))

    @staticmethod
    def stage_key(
        stage: str,
        code_version: str,
        params: dict,
        inputs: dict[str, str],
        models: dict[str, str] | None = None,
    ) -> str:
        payload = {
            "stage": stage,
            "code_version": code_version,
            "params": params,
            "inputs": inputs,
            "models": models or {},  # None and {} mean the same thing
        }
        return atomic.sha256_bytes(canonical_json(payload).encode("utf-8"))

    def stage_dir(self, song_key: str, stage: str, key: str) -> Path:
        return self._job_path(song_key) / "stages" / _check_name("stage", stage) / _check_key(key)[:KEY_PREFIX_LEN]

    # ---- completion ----------------------------------------------------------------------------

    def load_manifest(self, song_key: str, stage: str, key: str) -> dict:
        return atomic.read_json(self.stage_dir(song_key, stage, key) / MANIFEST)

    def is_complete(self, song_key: str, stage: str, key: str) -> bool:
        return self._complete_at(self.stage_dir(song_key, stage, key), key)

    @staticmethod
    def _manifest_key(final: Path) -> str | None:
        """Key recorded by a *complete* manifest in ``final``, else None."""
        try:
            m = atomic.read_json(final / MANIFEST)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            log.warning("unreadable manifest in %s: %s", final, e)
            return None
        if isinstance(m, dict) and m.get("status") == "complete" and isinstance(m.get("key"), str):
            return m["key"]
        return None

    @classmethod
    def _complete_at(cls, final: Path, key: str) -> bool:
        # Full-key comparison also guards against a (astronomically unlikely) key12 prefix collision.
        return cls._manifest_key(final) == key

    # ---- writing -------------------------------------------------------------------------------

    @contextmanager
    def stage_writer(
        self, song_key: str, stage: str, key: str, meta: dict | None = None, *, replace: bool = False
    ) -> Iterator[Path]:
        """Yield a temp dir to write a stage's outputs into; publish it atomically on success.

        - normal exit: hash outputs, write manifest.json, rename temp -> final.
          If a complete final dir for the same key already exists (a concurrent writer won),
          our temp is discarded and the call still succeeds.
        - exception: temp removed, exception re-raised; the final dir is never touched.
        - replace=True (used by `force`): an existing final dir is swapped out for ours.
        Close every file you opened inside the temp dir before leaving the block: Windows
        refuses to rename a directory that has open handles.
        """
        meta = dict(meta or {})
        clash = _RESERVED_META & meta.keys()
        if clash:
            raise ValueError(f"meta may not set reserved manifest fields: {sorted(clash)}")
        final = self.stage_dir(song_key, stage, key)
        tmp = make_temp_dir(final.parent, key[:KEY_PREFIX_LEN])
        t0 = time.monotonic()
        try:
            yield tmp
            if (tmp / MANIFEST).exists():
                raise ValueError(f"stage {stage!r} wrote its own {MANIFEST}; that name is reserved")
            manifest = {
                "stage": stage,
                "key": key,
                "song_key": song_key,
                "status": "complete",
                "created_utc": _utc_now(),
                "duration_s": round(time.monotonic() - t0, 3),
                "outputs": _hash_outputs(tmp),
                **meta,
            }
            atomic.write_json(tmp / MANIFEST, manifest)
            self._publish(tmp, final, key, replace=replace)
        except BaseException:
            rmtree_quiet(tmp)
            raise

    def _publish(self, tmp: Path, final: Path, key: str, *, replace: bool) -> None:
        for _ in range(5):
            if replace and final.exists():
                aside = _scratch_path(final.parent, f"old-{key[:KEY_PREFIX_LEN]}")
                try:
                    _rename_dir(final, aside)
                except FileNotFoundError:
                    pass  # someone else moved it first
                except PermissionError as e:
                    raise StoreError(f"cannot replace {final}: a file inside is in use ({e})") from e
                else:
                    rmtree_quiet(aside)
                replace = False  # only ever displace the version that existed when we started
            try:
                _rename_dir(tmp, final)
                log.debug("published %s", final)
                return
            except FileExistsError:
                pass
            except PermissionError as e:
                raise StoreError(
                    f"cannot publish {final}: a file in the temp dir is still open "
                    f"(close all writers before leaving stage_writer) ({e})"
                ) from e
            existing = self._manifest_key(final)
            if existing == key:
                log.info("%s already completed by another writer; discarding ours", final)
                rmtree_quiet(tmp)
                return
            if existing is not None:
                raise StoreError(f"stage dir {final} holds key {existing}, not {key} (key12 prefix collision)")
            # A final dir without a valid manifest is damage (manual edits, partial deletes),
            # never a normal state; move it out of the way and retry.
            broken = _scratch_path(final.parent, f"broken-{key[:KEY_PREFIX_LEN]}")
            log.warning("replacing incomplete stage dir %s", final)
            try:
                _rename_dir(final, broken)
            except FileNotFoundError:
                continue
            rmtree_quiet(broken)
        raise StoreError(f"could not publish {final} after repeated conflicts")

    # ---- garbage collection -----------------------------------------------------------------------

    def _temp_candidates(self) -> Iterator[Path]:
        """Scratch dirs at the levels we create them: root, job dir, job subdirs, stage dirs."""
        if not self.root.is_dir():
            return

        def scan(d: Path, depth: int) -> Iterator[Path]:
            try:
                children = [c for c in d.iterdir() if c.is_dir()]
            except OSError:
                return
            for c in children:
                if c.name.startswith(TMP_PREFIX):
                    yield c
                elif depth < 3 and not c.name.startswith("."):
                    # stop before descending into final stage dirs (root/job/stages/<stage>/<key12>)
                    yield from scan(c, depth + 1)

        yield from scan(self.root, 0)

    def gc_temp(self) -> int:
        """Remove .tmp-* dirs whose creating process is gone. Returns how many were removed."""
        removed = 0
        for p in self._temp_candidates():
            m = _TMP_RE.match(p.name)
            if not m:
                log.debug("skip unrecognised temp name %s", p)
                continue
            try:
                created = _dir_created(p)
            except FileNotFoundError:
                continue
            if _owner_alive(int(m["pid"]), created):
                continue
            if rmtree_quiet(p):
                log.info("removed stale temp dir %s", p)
                removed += 1
        return removed

    # ---- index -------------------------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def _index_lock(self) -> FileLock:
        self.root.mkdir(parents=True, exist_ok=True)
        return FileLock(self.root / "index.lock", timeout=INDEX_LOCK_TIMEOUT_S)

    def _read_index_unlocked(self) -> dict[str, str]:
        try:
            data = atomic.read_json(self.index_path)
        except FileNotFoundError:
            return {}
        except ValueError:
            # The index is only a cache (song keys are derivable), so keep the bad file for
            # inspection and start over rather than failing every ingest.
            bad = self.index_path.with_name(f"index.corrupt-{int(time.time())}.json")
            log.error("index.json is corrupt; moved to %s", bad.name)
            os.replace(self.index_path, bad)
            return {}
        if not isinstance(data, dict):
            log.error("index.json is not an object; ignoring it")
            return {}
        return data

    # Reads take the lock too: on Windows os.replace() fails with ACCESS_DENIED while any reader
    # has the target open, so unlocked readers could make a concurrent index_put fail.
    def index_get(self, key: str) -> str | None:
        with self._index_lock():
            return self._read_index_unlocked().get(key)

    def index_put(self, key: str, song_key: str) -> None:
        _check_name("song_key", song_key)
        with self._index_lock():
            data = self._read_index_unlocked()
            if data.get(key) == song_key:
                return
            if key in data:
                log.warning("index %s: %s -> %s", key, data[key], song_key)
            data[key] = song_key
            atomic.write_json(self.index_path, data)

    def index_all(self) -> dict[str, str]:
        with self._index_lock():
            return dict(self._read_index_unlocked())
