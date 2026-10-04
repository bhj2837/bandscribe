"""Model registry: ``data/models/models.lock.json`` (M1_M2_SPEC 1.5, DESIGN 9.4).

Every model file a stage or worker loads is recorded here with url, size, sha256 and license (personal-use
licensing must stay traceable: MuScriptor weights CC BY-NC 4.0, SW weights of unknown provenance). Downloads
happen only through ``bandscribe models fetch`` (bandscribe.model_fetch), never during ``bandscribe run`` or under the GPU lock;
stages call :func:`local_path` from their ``models(cfg)`` hook, so a missing model surfaces as
:class:`ModelMissing` at plan time, before anything runs (M1_M2_SPEC 0, 3.1).

``models.lock.json`` = ``{"format": "bandscribe.models/1", "models": {name: ModelEntry-as-dict}}``. Entry paths are
relative to ``BANDSCRIBE_ROOT`` (POSIX style); an absolute path is accepted as-is.

:func:`sha256_cached` memoises hashes in ``data/models/.sha_cache.json`` keyed by (absolute path, size,
mtime_ns): a stage key needs the model sha on every ``bandscribe run``, and hashing the 1.2 GB MuScriptor file each
time would cost seconds. The core and the workers both update the cache, so read-modify-write happens under a
file lock; a corrupt cache is discarded, never fatal.

Paths are resolved from ``bandscribe.paths`` at call time (tests repoint them). Imports stdlib + filelock (+ the
stdlib-only ``bandscribe.paths`` / ``bandscribe.atomic``); importable in all three envs, Python-3.10-safe.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths

log = logging.getLogger(__name__)

FORMAT = "bandscribe.models/1"
SHA_CACHE_FORMAT = "bandscribe.sha_cache/1"
LOCK_TIMEOUT_S = 60.0

# Friendly Korean names for hints (others are shown by their registry name).
_LABELS = {
    "beat_this.final0": "Beat This! 체크포인트",
    "muscriptor-medium": "MuScriptor medium 가중치",
    "muscriptor-small": "MuScriptor small 가중치",
    "bs_roformer_sw.ckpt": "BS-RoFormer SW 체크포인트",
    "bs_roformer_sw.yaml": "BS-RoFormer SW 설정(yaml)",
    "basic_pitch.icassp2022": "Basic Pitch 모델(nmp.onnx)",
}


@dataclasses.dataclass
class ModelEntry:
    name: str
    path: str  # relative to BANDSCRIBE_ROOT (POSIX), or absolute
    url: str
    bytes: int
    sha256: str
    license: str
    source: str
    notes: str = ""
    recorded_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> ModelEntry:
        # Lenient: hand-edited or older entries may lack fields; unknown keys are ignored.
        return cls(
            name=str(d.get("name") or name),
            path=str(d.get("path") or ""),
            url=str(d.get("url") or ""),
            bytes=int(d.get("bytes") or 0),
            sha256=str(d.get("sha256") or ""),
            license=str(d.get("license") or ""),
            source=str(d.get("source") or ""),
            notes=str(d.get("notes") or ""),
            recorded_utc=str(d.get("recorded_utc") or ""),
        )


class ModelMissing(RuntimeError):
    """A stage's model is not recorded or its file is absent. ``str(e)`` is the Korean hint."""

    def __init__(self, name: str, hint_ko: str | None = None) -> None:
        self.name = name
        self.hint_ko = hint_ko or _default_hint(name)
        super().__init__(self.hint_ko)


class ModelRegistryError(RuntimeError):
    """models.lock.json exists but cannot be read (user-facing, Korean)."""


def _subject(label: str) -> str:
    """Label + the right Korean subject particle (이/가) when the label ends in a Hangul syllable."""
    last = label[-1:]
    if "가" <= last <= "힣":
        return label + ("이" if (ord(last) - 0xAC00) % 28 else "가")
    return label + "이(가)"


def _label(name: str) -> str:
    return _LABELS.get(name, f"모델 '{name}'")


def _default_hint(name: str) -> str:
    what = f"{_subject(_label(name))} 없습니다."
    # Local-only models have no download source in model_sources.toml; `bandscribe models fetch` would refuse them.
    if name.startswith("bs_roformer_sw"):
        return (f"{what} 원 업로더 계정이 삭제되어 받을 곳이 없습니다: 보관해 둔 사본을 "
                "data/models/bs_roformer_sw 에 두고 models.lock.json 에 기록하세요.")
    if name.startswith("muscriptor-"):
        return (f"{what} Hugging Face 캐시(data/hf)에 있어야 합니다: tools\\hf-login.cmd 로 로그인하고 "
                "모델 페이지의 사용 조건을 수락한 뒤 받으세요.")
    if name.startswith("basic_pitch"):
        # Bundled in the basic-pitch wheel of the bp310 env (pinned by its uv.lock); never downloaded by bandscribe.
        return (f"{what} bp310 환경에 basic-pitch 휠과 함께 설치됩니다: {Path(paths.ROOT) / 'envs' / 'bp310'} 폴더에서 "
                f"`{Path(paths.TOOLS) / 'uvw.cmd'} sync` 를 실행하세요.")
    return f"{what} 먼저 `bandscribe models fetch {name}` 를 실행하세요."


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lock_path() -> Path:
    return Path(paths.MODELS_LOCK)


def sha_cache_path() -> Path:
    return Path(paths.MODELS) / ".sha_cache.json"


def __getattr__(name: str) -> Any:
    # bandscribe.model_fetch reads `<models api>.MODELS_LOCK`; keep it live (call-time) like every other path here.
    if name == "MODELS_LOCK":
        return lock_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _file_lock(target: Path) -> Any:
    from filelock import FileLock

    target.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(target) + ".lock", timeout=LOCK_TIMEOUT_S)


# ------------------------------------------------------------------------------------------------ lock


def _read_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"format": FORMAT, "models": {}}
    try:
        data = atomic.read_json(path)
    except (OSError, ValueError) as e:
        raise ModelRegistryError(f"모델 기록 파일을 읽지 못했습니다: {path} ({e})") from e
    if not isinstance(data, dict) or data.get("format") != FORMAT or not isinstance(data.get("models"), dict):
        raise ModelRegistryError(f"모델 기록 파일 형식이 올바르지 않습니다: {path} (format {FORMAT} 이어야 함)")
    return data


def entries() -> dict[str, ModelEntry]:
    """Every recorded model, by name (empty when the lock file does not exist yet)."""
    data = _read_lock(lock_path())
    return {n: ModelEntry.from_dict(n, d) for n, d in data["models"].items() if isinstance(d, dict)}


def get(name: str) -> ModelEntry | None:
    return entries().get(name)


def record(entry: ModelEntry) -> None:
    """Add or replace ``entry.name``; read-modify-write under a file lock, atomic write, names sorted."""
    path = lock_path()
    with _file_lock(path):
        data = _read_lock(path)
        models = dict(data["models"])
        models[entry.name] = entry.to_dict()
        atomic.write_json(path, {"format": FORMAT, "models": {k: models[k] for k in sorted(models)}})
    log.debug("model recorded: %s -> %s", entry.name, entry.path)


def resolve(entry: ModelEntry) -> Path:
    """Absolute path of an entry (``BANDSCRIBE_ROOT / path``; absolute paths stay as they are)."""
    return Path(paths.ROOT) / entry.path


def local_path(name: str) -> Path:
    """The recorded file of ``name``; raises :class:`ModelMissing` if unrecorded or absent on disk."""
    entry = get(name)
    if entry is None or not entry.path:
        raise ModelMissing(name)
    p = resolve(entry)
    if not p.is_file():
        raise ModelMissing(
            name,
            f"{_label(name)} 파일이 없습니다: {p} (models.lock.json 에는 기록됨). "
            f"`bandscribe models fetch {name}` 로 다시 받거나, 받을 곳이 없는 모델이면 보관해 둔 사본을 그 위치에 두세요.",
        )
    return p


def verify(name: str) -> bool:
    """Recompute the file's sha256 (no cache) and compare with the record. Refreshes the memo.

    Raises :class:`ModelMissing` for an unrecorded name and ``FileNotFoundError`` when the file is gone.
    """
    entry = get(name)
    if entry is None:
        raise ModelMissing(name)
    p = resolve(entry)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    sha = atomic.sha256_file(p)
    _remember(p, sha)
    return sha == entry.sha256


# ------------------------------------------------------------------------------------------- sha cache


def _cache_key(p: Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        data = atomic.read_json(path)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        log.warning("sha cache unreadable, discarded: %s", path)
        return {}
    if not isinstance(data, dict) or data.get("format") != SHA_CACHE_FORMAT or not isinstance(data.get("entries"), dict):
        log.warning("sha cache has an unexpected format, discarded: %s", path)
        return {}
    return dict(data["entries"])


def _remember(p: Path, sha: str) -> None:
    try:
        st = p.stat()
        path = sha_cache_path()
        with _file_lock(path):
            ents = _read_cache(path)
            ents[_cache_key(p)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
            atomic.write_json(path, {"format": SHA_CACHE_FORMAT, "entries": {k: ents[k] for k in sorted(ents)}})
    except Exception as e:  # the cache is an optimisation: a busy lock or read-only dir must not fail a run
        log.warning("sha cache not updated for %s: %s", p, e)


def sha256_cached(path: str | os.PathLike[str]) -> str:
    """sha256 of a file, memoised by (absolute path, size, mtime_ns)."""
    p = Path(path)
    st = p.stat()  # FileNotFoundError for a missing file, like sha256_file
    try:
        hit = _read_cache(sha_cache_path()).get(_cache_key(p))
    except Exception:  # pragma: no cover - _read_cache already swallows the expected errors
        hit = None
    if (isinstance(hit, dict) and hit.get("size") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns
            and isinstance(hit.get("sha256"), str) and len(hit["sha256"]) == 64):
        return str(hit["sha256"])
    # Hash outside the lock (1.2 GB takes seconds); only the read-modify-write of the memo is locked.
    sha = atomic.sha256_file(p)
    _remember(p, sha)
    return sha
