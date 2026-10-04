"""Local dataset store: ``data/datasets/datasets.lock.json`` and the files it describes (M1_M2_SPEC §9.5 item 3).

Layout per dataset ``data/datasets/<name>/``:
- ``raw/<file>``           downloads (``bandscribe data fetch``), never modified after the atomic rename
- ``extracted/<stem>/``    unpacked archives (one directory per archive, marker ``.bandscribe_extracted.json``)
- ``local/...``            files registered by ``bandscribe data import-local`` (hardlink, or copy across drives)

``datasets.lock.json`` (§6.11) records every file we hold, keyed by its POSIX path relative to the dataset
directory: ``{"format": "bandscribe.datasets/1", "datasets": {name: {"files": {rel: {"url" | "local", "bytes",
"sha256", "md5", "fetched_utc", "verified_against"}}, "extracted": "extracted", "status":
"fetched|verified|manual", "license"}}}``. Additions to §6.11: ``license`` (copied from the registry, so the
lock alone says under which terms we hold the files) and ``verified_against`` = the publisher checksum checked
at download ("md5" from Zenodo, "sha1" announced by the server) or null (trust on first use: the recorded
sha256 becomes the pin).
Every read-modify-write happens under a file lock and ends in an atomic replace, so a crashed fetch never
leaves a half-written lock and two processes never lose each other's records.

Core env only (filelock + stdlib); loaders and the public API import this lazily.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from filelock import FileLock

from bandscribe import atomic, paths
from bandscribe.datasets import registry
from bandscribe.jobs.store import make_temp_dir, rmtree_quiet

log = logging.getLogger(__name__)

LOCK_NAME = "datasets.lock.json"
LOCK_FORMAT = "bandscribe.datasets/1"
STATUSES = ("fetched", "verified", "manual")
LOCK_TIMEOUT_S = 60.0
_HASH_CHUNK = 1 << 20
# Written by bandscribe.datasets.fetch.extract_archive into every extracted/<stem>/: {"archive", "sha256", "files"}.
EXTRACT_MARKER = ".bandscribe_extracted.json"
# Name written before the 2026-10-04 rename (docs/RENAME.md). Still honoured, so existing extractions are not
# redone; a re-extraction writes the new name.
LEGACY_EXTRACT_MARKERS: tuple[str, ...] = (".gtab_extracted.json",)
MARKER_NAMES: tuple[str, ...] = (EXTRACT_MARKER, *LEGACY_EXTRACT_MARKERS)


def find_marker(d: Path) -> Path:
    """The extraction marker in ``d``: the current name, else a legacy one, else the current name (absent)."""
    for name in MARKER_NAMES:
        if (Path(d) / name).is_file():
            return Path(d) / name
    return Path(d) / EXTRACT_MARKER


class StoreError(RuntimeError):
    """Lock file or local-copy problem. The message is user-facing (Korean)."""


# ------------------------------------------------------------------------------------------ locations


def default_root() -> Path:
    """``data/datasets`` (``bandscribe.paths.DATASETS``), read at call time so tests can repoint it."""
    return Path(paths.DATASETS)


def dataset_dir(name: str, root: Path | None = None) -> Path:
    return Path(root or default_root()) / name


def lock_path(root: Path | None = None) -> Path:
    return Path(root or default_root()) / LOCK_NAME


def archive_stem(name: str) -> str:
    """``extracted/<stem>/`` name of an archive: the file name without its archive extension."""
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip", ".tar"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def make_scratch_dir(parent: Path, tag: str) -> Path:
    """``parent/.tmp-<tag>-<pid>-<rand>`` (M0 naming, so ``bandscribe gc``-style cleanup recognises leftovers)."""
    return make_temp_dir(Path(parent), tag)


def remove_tree(path: Path) -> bool:
    return rmtree_quiet(Path(path))


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(s: str) -> str:
    """ASCII directory name ``[a-z0-9_-]`` for anything DATA generates (song dirs of import-local).

    Same rules as EVAL's ``bandscribe.eval.runs.slug`` (M1_M2_SPEC §0: ``:`` -> ``-``, other characters -> ``_``,
    repeats collapsed, empty -> error); duplicated because DATA may not import EVAL (§2 dependency table).
    Leading/trailing separators are stripped so "Woodfire - Haunted House" gives "woodfire_-_haunted_house".
    """
    out = []
    for ch in s.strip().lower():
        if ch.isascii() and (ch.isalnum() or ch in "_-"):
            out.append(ch)
        elif ch == ":":
            out.append("-")
        else:
            out.append("_")
    text = re.sub(r"_+", "_", re.sub(r"-+", "-", "".join(out))).strip("_-")
    if not text:
        raise ValueError(f"이름에서 ASCII 폴더 이름을 만들 수 없습니다: {s!r}")
    return text


# ------------------------------------------------------------------------------------------- lock I/O


def _empty_lock() -> dict[str, Any]:
    return {"format": LOCK_FORMAT, "datasets": {}}


def _read_lock_unlocked(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_lock()
    try:
        data = atomic.read_json(path)
    except (OSError, json.JSONDecodeError) as e:
        # Never silently start over: the lock is the only record of which files were checksum-verified.
        raise StoreError(f"{path} 를 읽지 못했습니다 ({e}). 파일을 고치거나, 지운 뒤 `bandscribe data fetch <이름>` 을 "
                         "다시 실행하세요(이미 받은 파일은 다시 받지 않고 기록만 새로 합니다).") from e
    if not isinstance(data, dict) or data.get("format") != LOCK_FORMAT or not isinstance(data.get("datasets"), dict):
        raise StoreError(f"{path} 의 형식이 '{LOCK_FORMAT}' 가 아닙니다.")
    return data


def read_lock(root: Path | None = None) -> dict[str, Any]:
    return _read_lock_unlocked(lock_path(root))


@contextmanager
def locked(root: Path | None = None) -> Iterator[dict[str, Any]]:
    """Read-modify-write the lock under a file lock; written atomically when the block exits normally."""
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=LOCK_TIMEOUT_S):
        data = _read_lock_unlocked(path)
        before = json.dumps(data, sort_keys=True)
        yield data
        if json.dumps(data, sort_keys=True) != before:
            atomic.write_json(path, data)


def _ds(data: dict[str, Any], name: str) -> dict[str, Any]:
    return data["datasets"].setdefault(name, {"files": {}, "extracted": "extracted", "status": "fetched"})


def entry(name: str, root: Path | None = None) -> dict[str, Any] | None:
    return read_lock(root)["datasets"].get(name)


def record_files(name: str, files: dict[str, dict[str, Any]], *, status: str | None = None,
                 root: Path | None = None) -> None:
    """Add/replace file records (keys = POSIX paths relative to the dataset dir) and optionally set the status."""
    if status is not None and status not in STATUSES:
        raise ValueError(status)
    try:
        lic = registry.get(name).license
    except registry.RegistryError:
        lic = None
    with locked(root) as data:
        ds = _ds(data, name)
        if lic:
            ds["license"] = lic  # the lock stands alone as the record of what we hold and under which terms
        ds["files"].update(files)
        ds["files"] = dict(sorted(ds["files"].items()))
        if status is not None:
            ds["status"] = status


def set_status(name: str, status: str, root: Path | None = None) -> None:
    if status not in STATUSES:
        raise ValueError(status)
    with locked(root) as data:
        if name in data["datasets"]:
            data["datasets"][name]["status"] = status


def forget_files(name: str, rels: list[str], root: Path | None = None) -> None:
    with locked(root) as data:
        ds = data["datasets"].get(name)
        if ds:
            for rel in rels:
                ds["files"].pop(rel, None)


# ---------------------------------------------------------------------------------------------- status


def _required_complete(spec: registry.DatasetSpec, files: dict[str, Any]) -> bool:
    for f in spec.files:
        if f.required and f"raw/{f.name}" not in files:
            return False
    for d in spec.folders:
        if d.required:
            n = sum(1 for rel in files if rel.startswith(f"raw/{d.name}/"))
            if n < max(1, d.expected_files):
                return False
    return True


def status(name: str, root: Path | None = None) -> str:
    """``absent`` | ``partial`` | ``fetched`` | ``verified`` | ``manual`` (import-local)."""
    ds = entry(name, root)
    if not ds or not ds.get("files"):
        return "absent"
    st = ds.get("status", "fetched")
    if st == "manual":
        return "manual"
    try:
        spec = registry.get(name)
    except registry.RegistryError:
        return st
    if spec.access != "manual" and not _required_complete(spec, ds["files"]):
        return "partial"
    return st


def is_available(name: str, root: Path | None = None) -> bool:
    return status(name, root) in ("fetched", "verified", "manual")


def total_bytes(name: str, root: Path | None = None) -> int:
    ds = entry(name, root) or {}
    return sum(int(f.get("bytes") or 0) for f in ds.get("files", {}).values())


# ---------------------------------------------------------------------------------------------- hashing


def hash_file(path: Path, *, md5: bool = False) -> tuple[str, str | None]:
    """(sha256, md5 or None) in one pass."""
    h = hashlib.sha256()
    m = hashlib.md5(usedforsecurity=False) if md5 else None
    with open(path, "rb") as f:
        while block := f.read(_HASH_CHUNK):
            h.update(block)
            if m is not None:
                m.update(block)
    return h.hexdigest(), (m.hexdigest() if m is not None else None)


def count_files(d: Path) -> int:
    return sum(1 for _, _, files in os.walk(d) for fn in files if fn not in MARKER_NAMES)


def extraction_missing(d: Path, marker: dict[str, Any]) -> int:
    """How many unpacked files are gone from ``d``: by the marker's member list, or (older markers without
    one) by comparing the file count, which a file added later can mask."""
    members = marker.get("members")
    if isinstance(members, list):
        return sum(1 for m in members if not (d / m).is_file())
    return max(0, int(marker.get("files") or 0) - count_files(d))


def check_extracted(name: str, files: dict[str, Any], root: Path | None = None) -> list[str]:
    """Problems with the unpacked copies of recorded archives (the loaders read ``extracted/``, not ``raw/``).

    For every registry file with ``extract = true`` that the lock records: ``extracted/<stem>/`` must carry the
    marker of *this* archive (same sha256) and still hold every file that was unpacked. A problem is
    fixed by running ``bandscribe data fetch <name>`` again (extraction is idempotent and atomic). Korean text per item.
    """
    try:
        spec = registry.get(name)
    except registry.RegistryError:
        return []
    base = dataset_dir(name, root)
    out = []
    for f in spec.files:
        rel = f"raw/{f.name}"
        rec = files.get(rel)
        if not f.extract or rec is None:
            continue
        d = base / "extracted" / archive_stem(f.name)
        marker = find_marker(d)
        try:
            m = atomic.read_json(marker)
        except (OSError, ValueError):
            out.append(f"{rel}: 압축 해제본이 없습니다 ({d.relative_to(base).as_posix()}/)")
            continue
        if m.get("sha256") != rec.get("sha256"):
            out.append(f"{rel}: 압축 해제본이 다른 파일에서 나왔습니다")
            continue
        gone = extraction_missing(d, m)
        if gone:
            out.append(f"{rel}: 압축 해제본에 파일이 모자랍니다 ({gone}개 없음 / {m.get('files')}개)")
    return out


def verify(name: str, root: Path | None = None, *,
           progress: Callable[[str, int, int], None] | None = None) -> dict[str, Any]:
    """Recompute the sha256 of every recorded file and compare with the lock; check extracted archives.

    Returns ``{"name", "ok", "checked", "missing": [rel], "mismatch": [rel], "extracted": [problem], "status"}``.
    A clean result promotes ``fetched`` to ``verified`` (``manual`` stays ``manual`` and gets ``verified_utc``).
    """
    ds = entry(name, root)
    if not ds or not ds.get("files"):
        return {"name": name, "ok": False, "checked": 0, "missing": [], "mismatch": [], "extracted": [],
                "status": "absent"}
    base = dataset_dir(name, root)
    missing, mismatch = [], []
    items = sorted(ds["files"].items())
    for i, (rel, rec) in enumerate(items):
        if progress:
            progress(rel, i, len(items))
        p = base / rel
        if not p.is_file():
            missing.append(rel)
            continue
        sha, _ = hash_file(p)
        if sha != rec.get("sha256"):
            mismatch.append(rel)
    extracted = check_extracted(name, ds["files"], root)
    ok = not missing and not mismatch and not extracted
    with locked(root) as data:
        d = data["datasets"].get(name)
        if d is not None:
            d["verified_utc" if ok else "verify_failed_utc"] = utc_now()
            if ok and d.get("status") == "fetched":
                d["status"] = "verified"
            elif not ok and d.get("status") == "verified":
                d["status"] = "fetched"
    return {"name": name, "ok": ok, "checked": len(items), "missing": missing, "mismatch": mismatch,
            "extracted": extracted, "status": status(name, root)}


# ------------------------------------------------------------------------------------------ import-local

_SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


def _iter_source_files(src: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__MACOSX")
        for fn in sorted(filenames):
            if fn in _SKIP_NAMES or fn.startswith("._") or fn.endswith(".part"):
                continue
            yield Path(dirpath) / fn


def _place(src: Path, dst: Path, link: bool) -> str:
    """Hardlink (same volume, no extra space) or copy. Returns "link" | "copy" | "same"."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return "same"
        except OSError:
            pass
        dst.unlink()
    if link:
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass  # other volume / FAT / permission -> copy
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return "copy"


def import_local(name: str, src: Path, *, song: str | None = None, root: Path | None = None,
                 link: bool = True) -> dict[str, Any]:
    """Register a manually downloaded copy under ``data/datasets/<name>/local/`` and record sha256s.

    ``cambridge_mt`` needs ``song``: files go to ``local/<slug(song)>/`` and a ``lines.yaml`` skeleton is
    written next to them (kept if it already exists: it holds the user's by-ear labels).
    """
    spec = registry.get(name)
    src = Path(src)
    if not src.is_dir():
        raise StoreError(f"폴더가 아닙니다: {src}")
    if name == "cambridge_mt" and not song:
        raise StoreError("cambridge_mt 는 --song <곡 이름> 이 필요합니다 (곡마다 폴더 하나).")
    base = dataset_dir(name, root)
    dest = base / "local" / (slug(song) if song else "")
    files = list(_iter_source_files(src))
    if not files:
        raise StoreError(f"가져올 파일이 없습니다: {src}")
    if dest.resolve().is_relative_to(src.resolve()) or src.resolve().is_relative_to(base.resolve()):
        raise StoreError("원본 폴더와 데이터셋 폴더가 겹칩니다. 다른 곳에 있는 폴더를 지정하세요.")

    records: dict[str, dict[str, Any]] = {}
    counts = {"link": 0, "copy": 0, "same": 0}
    now = utc_now()
    for f in files:
        rel_in_src = f.relative_to(src)
        dst = dest / rel_in_src
        counts[_place(f, dst, link)] += 1
        sha, _ = hash_file(dst)
        rel = dst.relative_to(base).as_posix()
        records[rel] = {"local": str(f), "bytes": dst.stat().st_size, "sha256": sha, "md5": None,
                        "fetched_utc": now, "verified_against": None}

    prev = entry(name, root)
    new_status = "manual" if not prev or prev.get("status", "manual") == "manual" or not prev.get("files") else None
    record_files(name, records, status=new_status, root=root)

    lines_yaml = None
    if name == "cambridge_mt":
        wavs = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*")
                      if p.suffix.lower() in AUDIO_EXTS and p.is_file())
        lines_yaml = write_lines_skeleton(dest, wavs, song=slug(song or ""), title=song or "")
    log.info("import-local %s: %d files (%s) -> %s", name, len(records), counts, dest)
    return {"name": name, "dest": dest, "files": len(records), "bytes": sum(r["bytes"] for r in records.values()),
            "linked": counts["link"], "copied": counts["copy"], "lines_yaml": lines_yaml, "license": spec.license}


# -------------------------------------------------------------------------- Cambridge-MT lines.yaml skeleton

AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff"}
LINES_FORMAT = "bandscribe.cmt_lines/1"

_TOKEN_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
# (family, keywords) in priority order: guitar before vocals ("ElecGtr1a_VoxM160" is a Vox amp, not a voice),
# drums before bass ("BassDrum" is a kick).
_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("guitar", ("gtr", "gtrs", "guitar", "guitars", "guit", "gt")),
    ("drums", ("kick", "kik", "snare", "snr", "hat", "hats", "hihat", "hh", "tom", "toms", "overhead", "overheads",
               "oh", "ohs", "room", "rooms", "cymbal", "cymbals", "crash", "ride", "drum", "drums", "kit", "perc",
               "percussion", "tamb", "tambourine", "shaker", "clap", "claps", "cowbell", "conga", "bongo", "trigger")),
    ("bass", ("bass", "sub")),
    ("vocals", ("vox", "vocal", "vocals", "voc", "voice", "bv", "bvs", "bvox", "choir", "harmony", "harmonies",
                "adlib", "adlibs", "sing", "singer")),
    ("keys", ("piano", "pno", "rhodes", "wurli", "wurlitzer", "organ", "hammond", "keys", "key", "keyboard",
              "clav", "clavinet", "epiano", "ep", "celeste", "celesta", "glock", "vibes", "marimba", "harpsichord")),
    ("synth", ("synth", "synths", "pad", "pads", "stylophone", "moog", "arp", "arpeggio", "seq", "sequence",
               "lead", "pluck")),
    ("strings", ("violin", "violins", "viola", "cello", "cellos", "string", "strings", "fiddle", "harp")),
    ("brass", ("trumpet", "trumpets", "trombone", "trombones", "horn", "horns", "brass", "tuba", "flugelhorn")),
    ("winds", ("sax", "saxophone", "flute", "clarinet", "oboe", "bassoon", "harmonica", "whistle", "recorder")),
)
# Marks a microphone / amp / DI variant of one performance (not a separate take).
_MIC_WORDS = {"mic", "mics", "amp", "amps", "di", "close", "far", "sm", "md", "ribbon", "dyn", "condenser", "front",
              "back", "top", "bottom", "vox", "orange", "marshall", "mesa", "fender", "cab", "direct"}
# Part.kind (schema §6.8) per family; families without their own kind are "other".
_KIND_OF_FAMILY = {"guitar": "guitar", "bass": "bass", "vocals": "vocals", "drums": "drums", "keys": "keys"}


def _tokens(stem: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(stem)]


def guess_part(filename: str) -> dict[str, Any]:
    """Guess ``{"kind", "family", "di", "acoustic", "number", "mic"}`` from a stem file name (by keywords).

    ``number`` is the digit token right after the guitar keyword ("ElecGtr2b" -> 2), which separates guitar
    parts from mics of the same part ("ElecGtrMic1/Mic2" -> no number -> same line).
    """
    stem = Path(filename).stem
    toks = _tokens(stem)
    joined = "".join(toks)
    family = None
    if "bassdrum" in joined or "bassdrm" in joined:
        family = "drums"
    elif "bassoon" in toks:
        family = "winds"
    elif "doublebass" in joined or "upright" in toks or "contrabass" in toks:
        family = "bass"
    else:
        for fam, words in _FAMILY_KEYWORDS:
            if any(t in words for t in toks):
                family = fam
                break
    if family == "synth" and "lead" in toks and not any(t in ("synth", "synths") for t in toks):
        family = None  # "LeadVox" is caught by vocals first; a bare "Lead..." stays unknown
    is_di = "di" in toks
    number = None
    if family == "guitar":
        for i, t in enumerate(toks):
            if t in ("gtr", "gtrs", "guitar", "guitars", "guit", "gt") and i + 1 < len(toks) and toks[i + 1].isdigit():
                number = int(toks[i + 1])
                break
    kind = "di" if is_di and family in ("guitar", "bass") else _KIND_OF_FAMILY.get(family or "", "other")
    # only meaningful for guitar/bass lines ("LeadVox" is not a Vox amp)
    mic = family in ("guitar", "bass") and (is_di or any(t in _MIC_WORDS for t in toks))
    if "mix" in toks and family is None:
        kind = "mix"
    acoustic = family == "guitar" and any(t in ("ac", "acoustic", "acc", "acous", "nylon", "classical") for t in toks)
    return {"kind": kind, "family": family, "di": is_di, "acoustic": acoustic, "number": number, "mic": mic}


def _q(s: Any) -> str:
    """YAML scalar via JSON (a JSON string is a valid YAML double-quoted scalar)."""
    if s is None:
        return "null"
    if isinstance(s, bool):
        return "true" if s else "false"
    if isinstance(s, (int, float)):
        return json.dumps(s)
    return json.dumps(str(s), ensure_ascii=False)


def lines_skeleton_text(files: list[str], *, song: str, title: str = "", group: str | None = None) -> str:
    """Korean-commented ``lines.yaml`` draft for one multitrack song (files relative to the song dir)."""
    parts = []
    lines: dict[str, dict[str, Any]] = {}
    families: set[str] = set()
    for i, rel in enumerate(files, 1):
        g = guess_part(rel)
        line = None
        if g["family"] == "guitar":
            line = f"{'AG' if g['acoustic'] else 'G'}{g['number'] or 1}"
            lines.setdefault(line, {"name": f"{'Acoustic ' if g['acoustic'] else ''}Guitar {g['number'] or 1}",
                                    "kind": "guitar"})
        elif g["family"] == "bass":
            line = "B1"
            lines.setdefault(line, {"name": "Bass", "kind": "bass"})
        if g["family"]:
            families.add(g["family"])
        parts.append({"id": f"p{i:02d}", "file": rel, "kind": g["kind"], "line": line, "take": None,
                      "mic": g["mic"]})
    # takes: several mics/DI of one performance share take 1; otherwise (ElecGtr1a/1b ...) guess separate takes
    for lid in lines:
        members = [p for p in parts if p["line"] == lid]
        same_performance = all(p["mic"] for p in members)
        for k, p in enumerate(members, 1):
            p["take"] = 1 if same_performance else k
    fam_order = [f for f in registry.FAMILY_KEYS if f in families]
    out = [
        "# bandscribe Cambridge-MT 곡 라벨 (lines.yaml) - 파일 이름으로 자동 추정한 초안입니다. 반드시 귀로 확인해 고치세요.",
        "# kind: guitar | bass | vocals | drums | keys | other | di | mix  (di = 앰프를 거치지 않은 직결 트랙)",
        "# line: 같은 Line(더블링, 같은 기타의 마이크 여러 개, DI+앰프)은 같은 id. 기타·베이스가 아니면 null.",
        "#       기타 Line 끼리 번호가 다르면 다른 파트입니다(G1, G2 ...). 파일 이름만 보고 나눈 것이라 틀릴 수 있습니다.",
        "# take: 같은 Line 안의 테이크 번호(1부터). 같은 연주를 마이크 여러 개·DI 로 받았으면 모두 같은 번호,",
        "#       같은 파트를 따로 두 번 연주한 더블이면 1, 2. (파일 이름으로 추정: 마이크·앰프 표시가 있으면 같은 연주로 봄)",
        "# families_present: 이 곡에서 실제로 소리 나는 악기군(b13 정답). guitar, bass, keys, synth, strings, brass,",
        "#       winds, vocals, drums 중에서 고릅니다. 건반/신스 유무를 특히 확인하세요.",
        "# lines[].role: lead | rhythm | both | null (선택), relation: 더블/유니즌/트윈/오버더브 관계(선택).",
        "# reviewed: 다 확인했으면 true. false 이면 평가 리포트에 '미확인 라벨'로 표시됩니다.",
        f"format: {LINES_FORMAT}",
        f"song: {_q(song)}",
        f"title: {_q(title)}",
        f"group: {_q(group or song.split('_')[0])}          # 곡 블록 부트스트랩 단위(보통 아티스트)",
        "split: null              # dev | test | null",
        "reviewed: false",
        f"families_present: [{', '.join(fam_order)}]",
        "mix: null                # 제공된 믹스 파일(곡 폴더 기준 경로). 없으면 스템 합을 믹스로 씁니다.",
        "parts:",
    ]
    for p in parts:
        out.append(f"  - {{id: {p['id']}, file: {_q(p['file'])}, kind: {p['kind']}, line: {_q(p['line'])}, "
                   f"take: {_q(p['take'])}}}")
    out.append("lines:" + ("" if lines else " []"))
    for lid in sorted(lines, key=lambda s: (s[0] != "G", s)):
        info = lines[lid]
        out.append(f"  - id: {lid}")
        out.append(f"    name: {_q(info['name'])}")
        out.append(f"    kind: {info['kind']}")
        out.append("    role: null")
        out.append("    relation: {double: false, unison_with: [], twin_with: null, overdub: false}")
    return "\n".join(out) + "\n"


def write_lines_skeleton(song_dir: Path, files: list[str], *, song: str, title: str = "") -> Path:
    """Write ``lines.yaml`` unless one exists (it may already hold the user's labels)."""
    path = Path(song_dir) / "lines.yaml"
    if path.exists():
        log.info("keeping existing %s", path)
        return path
    atomic.write_text(path, lines_skeleton_text(files, song=song, title=title))
    return path
