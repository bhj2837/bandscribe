"""Cambridge-MT (Mixing Secrets) multitrack loader (Tier B).

A song is any folder under ``data/datasets/cambridge_mt/{extracted,local}/`` that holds a ``lines.yaml``
(written as a draft by ``bandscribe data fetch cambridge_mt`` after extraction, or by ``bandscribe data import-local
cambridge_mt <폴더> --song <이름>``; the user corrects it by ear). The stems are the WAV files the yaml lists
(paths relative to the yaml). There is no note ground truth: Tier B references come from MuScriptor run on the
true solo tracks (AMT's reftx). ``mix`` = the yaml's ``mix`` file if given, else None (EVAL sums the stems).

``lines.yaml`` (format ``bandscribe.cmt_lines/1``, see ``bandscribe.datasets.store.lines_skeleton_text``):
``song, title, group, split, reviewed, families_present, mix, parts: [{id, file, kind, line, take}],
lines: [{id, name, kind, role, relation}]``. ``GtLine.takes`` = 1-based indices into ``parts`` of the parts
on that line (the dataset analogue of GP ref_track numbers).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bandscribe import formats, paths
from bandscribe.datasets.loaders import _common as C

log = logging.getLogger(__name__)

NAME = "cambridge_mt"
LICENSE = "Cambridge-MT library terms (educational/non-commercial; personal local use only)"
LINES_FORMAT = "bandscribe.cmt_lines/1"
_KINDS = {"guitar", "bass", "vocals", "drums", "keys", "other", "mix", "di"}
_FAMILIES = C.FAMILY_ORDER


class LinesYamlError(ValueError):
    pass


def _yaml() -> Any:
    try:
        import yaml
    except ImportError as e:  # pyyaml is a core dependency (P0); keep the error actionable
        raise LinesYamlError(f"pyyaml 이 없습니다. {paths.ROOT} 에서 `tools\\uvw.cmd sync --extra core --group dev` 를 "
                             "실행하세요.") from e
    return yaml


def song_dirs(root: Path) -> list[Path]:
    root = Path(root)
    out = []
    for sub in ("extracted", "local"):
        base = root / sub
        if base.is_dir():
            out += [p.parent for p in C.walk_files(base, ("lines.yaml",))]
    return sorted(set(out))


def read_lines(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        # a lines.yaml labelled before the 2026-10-04 rename says gtab.cmt_lines/1 (bandscribe.formats)
        d = formats.normalize_doc(_yaml().safe_load(f) or {})
    if d.get("format") != LINES_FORMAT:
        raise LinesYamlError(f"{path}: format 이 {LINES_FORMAT} 가 아닙니다")
    for i, p in enumerate(d.get("parts") or []):
        if p.get("kind") not in _KINDS:
            raise LinesYamlError(f"{path}: parts[{i}].kind '{p.get('kind')}' 는 {sorted(_KINDS)} 중 하나여야 합니다")
        if not p.get("file"):
            raise LinesYamlError(f"{path}: parts[{i}].file 이 없습니다")
    bad = [f for f in d.get("families_present") or [] if f not in _FAMILIES]
    if bad:
        raise LinesYamlError(f"{path}: families_present 에 모르는 값 {bad}")
    line_ids = {ln.get("id") for ln in d.get("lines") or []}
    for p in d.get("parts") or []:
        if p.get("line") and p["line"] not in line_ids:
            raise LinesYamlError(f"{path}: part {p.get('id')} 의 line '{p['line']}' 가 lines 에 없습니다")
    return d


def _track(root: Path, song_dir: Path) -> Any:
    s = C.schema()
    y = read_lines(song_dir / "lines.yaml")
    song = str(y.get("song") or song_dir.name)
    parts, missing = [], []
    for i, p in enumerate(y.get("parts") or [], 1):
        f = song_dir / p["file"]
        if not f.is_file():
            missing.append(p["file"])
            continue
        _, _, ch = C.audio_info(f)
        parts.append(s.Part(id=str(p.get("id") or f"p{i:02d}"), kind=p["kind"], audio=C.rel(f, root), channels=ch,
                            line=p.get("line"), take=p.get("take")))
    idx_of = {part.id: n for n, part in enumerate(parts, 1)}
    lines = []
    roles = {}
    for ln in y.get("lines") or []:
        if ln.get("kind") not in ("guitar", "bass"):
            continue
        takes = [idx_of[part.id] for part in parts if part.line == ln["id"]]
        lines.append(s.GtLine(id=ln["id"], name=str(ln.get("name") or ln["id"]), kind=ln["kind"], takes=takes,
                              relation=ln.get("relation") or {}))
        if ln.get("role"):
            roles[ln["id"]] = ln["role"]
    mix = y.get("mix")
    mix_rel = C.rel(song_dir / mix, root) if mix and (song_dir / mix).is_file() else None
    return s.DatasetTrack(
        format="bandscribe.dataset_track/1", dataset=NAME, track_id=song, split=y.get("split"),
        group=str(y.get("group") or song.split("_")[0]), tier="B", license=LICENSE, mix=mix_rel, parts=parts,
        lines=lines, families_present=list(y.get("families_present") or []), notes={}, string_fret=None,
        beats_s=None, downbeats_s=None, tuning=None,
        meta={"title": y.get("title"), "reviewed": bool(y.get("reviewed")), "roles": roles,
              "lines_yaml": C.rel(song_dir / "lines.yaml", root), "missing_files": missing,
              "reference": "no note GT: use MuScriptor reference transcription of the true tracks (reftx)"})


def track_ids(root: Path, *, split: str | None = None) -> list[str]:
    """Song ids from the lines.yaml files (no audio is opened)."""
    out = []
    for d in song_dirs(Path(root)):
        y = read_lines(d / "lines.yaml")
        if split is None or y.get("split") == split:
            out.append(str(y.get("song") or d.name))
    return out


def iter_tracks(root: Path, *, split: str | None = None) -> Iterator[Any]:
    root = Path(root)
    for d in song_dirs(root):
        t = _track(root, d)
        if split is None or t.split == split:
            yield t


def load_track(root: Path, track_id: str) -> Any:
    root = Path(root)
    for d in song_dirs(root):
        y = read_lines(d / "lines.yaml")
        if str(y.get("song") or d.name) == track_id:
            return _track(root, d)
    raise KeyError(f"Cambridge-MT 에 없는 곡입니다: {track_id}")
