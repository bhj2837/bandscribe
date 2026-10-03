"""Dataset registry: typed view of ``gtab/datasets/registry.toml`` (M1_M2_SPEC §9.5 item 1).

The TOML file is committed and is the only place that knows where a dataset comes from, how big it is,
which checksums its publisher states and under which license it may be used. Everything else (fetch,
store, CLI, loaders) reads it through :func:`load_registry` / :func:`get`.

stdlib only (tomllib), so the module is cheap to import from the CLI.
"""

from __future__ import annotations

import difflib
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_PATH: Path = Path(__file__).with_name("registry.toml")
FORMAT = "gtab.dataset_registry/1"

ACCESS_KINDS = ("zenodo", "direct", "gdrive_folder", "manual")
TIERS = ("B", "C")
# FAMILIES keys of gtab.schema.instrumentation (duplicated here on purpose: the registry must load without
# the schema package, and tests/test_datasets_registry.py checks the two stay equal once P0's schema exists).
FAMILY_KEYS = ("guitar", "bass", "keys", "synth", "strings", "brass", "winds", "vocals", "drums")


class RegistryError(ValueError):
    """Malformed registry.toml, or an unknown dataset name (message is user-facing Korean)."""


@dataclass(frozen=True)
class RegistryFile:
    name: str                 # file name under data/datasets/<dataset>/raw/
    url: str
    bytes: int = 0            # 0 = unknown before download
    md5: str = ""             # publisher's md5 ("" = none published -> sha256 is pinned on first download)
    required: bool = True
    extract: bool = False     # unpack a zip/tar into extracted/<stem>/
    part: str = ""            # group label for --part (defaults to the file name)
    song: str = ""            # Cambridge-MT: song id (ASCII) the archive holds
    title: str = ""

    @property
    def label(self) -> str:
        return self.part or self.name


@dataclass(frozen=True)
class RegistryFolder:
    """A public Google Drive folder whose files are listed at fetch time (access = gdrive_folder)."""

    name: str
    folder_id: str
    required: bool = True
    expected_files: int = 0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    tier: str
    access: str
    license: str
    page: str
    approx_bytes: int
    loader: str
    families: tuple[str, ...]
    notes: str = ""
    manual_ko: str = ""
    license_url: str = ""
    doi: str = ""
    zenodo_record: int | None = None
    contamination: str = ""
    files: tuple[RegistryFile, ...] = field(default_factory=tuple)
    folders: tuple[RegistryFolder, ...] = field(default_factory=tuple)

    def parts(self) -> list[str]:
        """Every selectable part label (files and Drive folders), registry order."""
        return [f.label for f in self.files] + [d.name for d in self.folders]

    def required_parts(self) -> list[str]:
        return [f.label for f in self.files if f.required] + [d.name for d in self.folders if d.required]


_DATASET_KEYS = {
    "title", "tier", "access", "license", "page", "approx_bytes", "loader", "families", "notes", "manual_ko",
    "license_url", "doi", "zenodo_record", "contamination", "files", "folders",
}
_FILE_KEYS = {"name", "url", "bytes", "md5", "required", "extract", "part", "song", "title"}
_FOLDER_KEYS = {"name", "folder_id", "required", "expected_files"}


def is_allowed_url(url: str) -> bool:
    """https only; plain http is accepted for a loopback test server."""
    return url.startswith("https://") or url.startswith(("http://127.0.0.1:", "http://localhost:"))


def _is_ascii_name(s: str) -> bool:
    return bool(s) and s.isascii() and "/" not in s and "\\" not in s and s not in (".", "..") and ":" not in s


def _check_keys(where: str, d: dict[str, Any], allowed: set[str]) -> None:
    extra = set(d) - allowed
    if extra:
        raise RegistryError(f"registry.toml [{where}]: 알 수 없는 키 {sorted(extra)}")


def _parse_dataset(name: str, d: dict[str, Any]) -> DatasetSpec:
    if not isinstance(d, dict):
        raise RegistryError(f"registry.toml: [{name}] 는 표(table)여야 합니다")
    _check_keys(name, d, _DATASET_KEYS)
    for key in ("title", "tier", "access", "license", "page", "approx_bytes", "loader", "families"):
        if key not in d:
            raise RegistryError(f"registry.toml [{name}]: '{key}' 가 없습니다")
    if d["tier"] not in TIERS:
        raise RegistryError(f"registry.toml [{name}]: tier 는 {TIERS} 중 하나여야 합니다")
    if d["access"] not in ACCESS_KINDS:
        raise RegistryError(f"registry.toml [{name}]: access 는 {ACCESS_KINDS} 중 하나여야 합니다")
    if not str(d["license"]).strip():
        raise RegistryError(f"registry.toml [{name}]: license 가 비어 있습니다")
    unknown_fam = [f for f in d["families"] if f not in FAMILY_KEYS]
    if unknown_fam:
        raise RegistryError(f"registry.toml [{name}]: 알 수 없는 families {unknown_fam}")

    files = []
    for i, f in enumerate(d.get("files", [])):
        _check_keys(f"{name}.files[{i}]", f, _FILE_KEYS)
        if not _is_ascii_name(str(f.get("name", ""))):
            raise RegistryError(f"registry.toml [{name}.files[{i}]]: name 은 ASCII 파일 이름이어야 합니다")
        if not is_allowed_url(str(f.get("url", ""))):
            raise RegistryError(f"registry.toml [{name}.files[{i}]]: url 은 https:// 여야 합니다")
        files.append(RegistryFile(**f))
    folders = []
    for i, f in enumerate(d.get("folders", [])):
        _check_keys(f"{name}.folders[{i}]", f, _FOLDER_KEYS)
        if not _is_ascii_name(str(f.get("name", ""))):
            raise RegistryError(f"registry.toml [{name}.folders[{i}]]: name 은 ASCII 이름이어야 합니다")
        folders.append(RegistryFolder(**f))
    labels = [f.label for f in files] + [f.name for f in folders]
    if len(labels) != len(set(labels)):
        raise RegistryError(f"registry.toml [{name}]: part 이름이 중복됩니다")
    if d["access"] == "gdrive_folder" and not folders:
        raise RegistryError(f"registry.toml [{name}]: gdrive_folder 인데 folders 가 없습니다")
    if d["access"] in ("zenodo", "direct") and not files:
        raise RegistryError(f"registry.toml [{name}]: {d['access']} 인데 files 가 없습니다")

    rest = {k: v for k, v in d.items() if k not in ("files", "folders", "families")}
    return DatasetSpec(name=name, families=tuple(d["families"]), files=tuple(files), folders=tuple(folders), **rest)


def parse_registry(data: dict[str, Any]) -> dict[str, DatasetSpec]:
    if data.get("format") != FORMAT:
        raise RegistryError(f"registry.toml: format 이 '{FORMAT}' 가 아닙니다")
    out: dict[str, DatasetSpec] = {}
    for name, d in data.items():
        if name == "format":
            continue
        if not (name.isascii() and name.replace("_", "").isalnum() and name.islower()):
            raise RegistryError(f"registry.toml: 데이터셋 이름 '{name}' 은 소문자 ASCII 여야 합니다")
        out[name] = _parse_dataset(name, d)
    return out


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime_ns: int) -> dict[str, DatasetSpec]:
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise RegistryError(f"registry.toml 을 읽지 못했습니다: {e}") from e
    return parse_registry(data)


def load_registry(path: Path | None = None) -> dict[str, DatasetSpec]:
    p = Path(path or REGISTRY_PATH)
    return dict(_load_cached(str(p), p.stat().st_mtime_ns))


def get(name: str, *, path: Path | None = None) -> DatasetSpec:
    reg = load_registry(path)
    try:
        return reg[name]
    except KeyError:
        close = difflib.get_close_matches(name, list(reg), n=1)
        hint = f" '{close[0]}' 을(를) 말씀하신 건가요?" if close else ""
        raise RegistryError(f"등록되지 않은 데이터셋입니다: '{name}'.{hint} (목록: {', '.join(reg)})") from None
