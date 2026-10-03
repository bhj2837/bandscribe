"""MedleyDB loader for a local copy (Tier B; Zenodo 1649325 is access-restricted -> ``import-local`` only).

Not verified against real files (no copy on this PC yet, 2026-09-30): written against the documented MedleyDB
v1 layout — ``<Song>/<Song>_METADATA.yaml`` with ``mix_filename``, ``stem_dir``, ``has_bleed``, ``genre``,
``instrumental`` and ``stems: {S01: {filename, instrument, component, raw: {...}}}``; stems in ``stem_dir``;
optional ``Annotations/Melody_Annotations/MELODY{1,2,3}/<Song>_MELODY{n}.csv`` anywhere under the root.
Guitar stems become one line each (``G<n>`` in stem order), bass stems line ``B1``. No note GT (melody f0
paths go to ``meta``). MedleyDB-Pitch is in Basic Pitch's training data (contamination.md).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gtab.datasets.loaders import _common as C

log = logging.getLogger(__name__)

NAME = "medleydb"
LICENSE = "CC BY-NC-SA 4.0 (MedleyDB; verify)"

# instrument label (lower-case substring) -> family; first match wins (order matters: "bass drum" is drums).
_LABEL_FAMILY: tuple[tuple[str, str], ...] = (
    ("bass drum", "drums"), ("drum", "drums"), ("cymbal", "drums"), ("percussion", "drums"), ("tabla", "drums"),
    ("timpani", "drums"), ("tambourine", "drums"), ("shaker", "drums"), ("bongo", "drums"), ("conga", "drums"),
    ("snare", "drums"), ("kick", "drums"), ("hi-hat", "drums"), ("gong", "drums"), ("cabasa", "drums"),
    ("bassoon", "winds"), ("guitar", "guitar"), ("banjo", "guitar"), ("mandolin", "guitar"), ("ukulele", "guitar"),
    ("electric bass", "bass"), ("double bass", "bass"), ("bass", "bass"),
    ("singer", "vocals"), ("vocal", "vocals"), ("choir", "vocals"), ("rapper", "vocals"), ("speaker", "vocals"),
    ("beatboxing", "vocals"),
    ("piano", "keys"), ("organ", "keys"), ("harpsichord", "keys"), ("clavinet", "keys"), ("vibraphone", "keys"),
    ("glockenspiel", "keys"), ("marimba", "keys"), ("xylophone", "keys"), ("celesta", "keys"), ("accordion", "keys"),
    ("synthesizer", "synth"), ("fx/processed", "synth"), ("sampler", "synth"),
    ("violin", "strings"), ("viola", "strings"), ("cello", "strings"), ("string", "strings"), ("harp", "strings"),
    ("erhu", "strings"), ("dizi", "winds"),
    ("trumpet", "brass"), ("trombone", "brass"), ("horn", "brass"), ("tuba", "brass"), ("brass", "brass"),
    ("cornet", "brass"), ("euphonium", "brass"),
    ("sax", "winds"), ("flute", "winds"), ("clarinet", "winds"), ("oboe", "winds"), ("piccolo", "winds"),
    ("harmonica", "winds"), ("whistle", "winds"), ("recorder", "winds"),
)
_KIND = {"guitar": "guitar", "bass": "bass", "vocals": "vocals", "drums": "drums", "keys": "keys"}


def family_of(label: str) -> str | None:
    lab = (label or "").lower()
    for key, fam in _LABEL_FAMILY:
        if key in lab:
            return fam
    return None


def _yaml() -> Any:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("pyyaml 이 없습니다. `tools\\uvw.cmd sync --extra core --group dev` 를 실행하세요.") from e
    return yaml


def index(root: Path) -> dict[str, Path]:
    out = {}
    for p in C.walk_files(Path(root), ("_metadata.yaml",)):
        out[p.name[: -len("_METADATA.yaml")]] = p
    return out


def _melody_files(root: Path, song: str) -> list[str]:
    return [C.rel(p, root) for p in C.walk_files(root, (".csv",)) if re.match(rf"^{re.escape(song)}_MELODY\d\.csv$", p.name)]


def _track(root: Path, song: str, meta_path: Path) -> Any:
    s = C.schema()
    with open(meta_path, encoding="utf-8") as f:
        y = _yaml().safe_load(f) or {}
    song_dir = meta_path.parent
    stem_dir = song_dir / str(y.get("stem_dir") or f"{song}_STEMS")
    parts, lines, fams = [], [], set()
    n_gtr = 0
    stems = y.get("stems") or {}
    for sid in sorted(stems):
        st = stems[sid] or {}
        f = stem_dir / str(st.get("filename", ""))
        label = str(st.get("instrument", ""))
        fam = family_of(label)
        if fam:
            fams.add(fam)
        line = None
        if fam == "guitar":
            n_gtr += 1
            line = f"G{n_gtr}"
            lines.append(s.GtLine(id=line, name=f"{label} ({sid})", kind="guitar", takes=[]))
        elif fam == "bass":
            line = "B1"
            if not any(ln.id == "B1" for ln in lines):
                lines.append(s.GtLine(id="B1", name="Bass", kind="bass", takes=[]))
        if not f.is_file():
            continue
        _, _, ch = C.audio_info(f)
        parts.append(s.Part(id=sid, kind=_KIND.get(fam or "", "other"), audio=C.rel(f, root), channels=ch, line=line,
                            take=1 if line else None))
    idx_of = {p.id: n for n, p in enumerate(parts, 1)}
    lines = [ln.model_copy(update={"takes": [idx_of[p.id] for p in parts if p.line == ln.id]}) for ln in lines]
    mix = song_dir / str(y.get("mix_filename") or f"{song}_MIX.wav")
    labels = {sid: (stems[sid] or {}).get("instrument") for sid in sorted(stems)}
    return s.DatasetTrack(
        format="gtab.dataset_track/1", dataset=NAME, track_id=song, split=None,
        group=str(y.get("artist") or song.split("_")[0]), tier="B", license=LICENSE,
        mix=C.rel(mix, root) if mix.is_file() else None, parts=parts, lines=lines,
        families_present=[f for f in C.FAMILY_ORDER if f in fams],
        notes={}, string_fret=None, beats_s=None, downbeats_s=None, tuning=None,
        meta={"artist": y.get("artist"), "title": y.get("title"), "genre": y.get("genre"),
              "has_bleed": str(y.get("has_bleed", "")).lower() == "yes", "instrumental": y.get("instrumental"),
              "stem_labels": labels, "melody_f0": _melody_files(root, song),
              "contamination": "MedleyDB-Pitch in Basic Pitch training data"})


def track_ids(root: Path, *, split: str | None = None) -> list[str]:
    return [] if split is not None else sorted(index(Path(root)))


def iter_tracks(root: Path, *, split: str | None = None) -> Iterator[Any]:
    if split is not None:
        return
    root = Path(root)
    for song, meta in sorted(index(root).items()):
        yield _track(root, song, meta)


def load_track(root: Path, track_id: str) -> Any:
    meta = index(Path(root)).get(track_id)
    if meta is None:
        raise KeyError(f"MedleyDB 에 없는 곡입니다: {track_id}")
    return _track(Path(root), track_id, meta)
