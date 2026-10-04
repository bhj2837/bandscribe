"""FiloBass loader (Tier C; Zenodo 10069709, CC BY 4.0) — jazz upright bass.

Verified against the real archive (2026-09-30): ``FiloBass ISMIR Publication/`` with 48 songs in
``audio_bass_stems/<Song>.mp3`` (stereo 44.1 kHz MP3, the separated bass stem), ``midi_fully_aligned/<Song>.mid``
(performance-aligned notes; 1 tempo, 4/4, notes in track 1), ``midi_downbeat_aligned`` / ``midi_from_score``
(score timing), ``musicxml*``, and ``syncpoints/<Song>.json`` = ``[[bar_index, time_s], ...]`` (bar 0 is the
bar before the first full bar, e.g. ``[0, 1]`` then ``[1, 3.0539]``). GT notes come from
``midi_fully_aligned``; downbeats = the sync point times; beats are interpolated 4 per bar (all songs are
4/4 per the MIDI time signatures). No string/fret labels -> ``string_fret`` None. Standard upright tuning
(E1 A1 D2 G2) is recorded as the nominal tuning. No official split.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bandscribe.datasets.loaders import _common as C

log = logging.getLogger(__name__)

NAME = "filobass"
LICENSE = "CC BY 4.0"
LINE_ID = "bass"
BASS_TUNING = [43, 38, 33, 28]  # string 1 = G2 .. string 4 = E1


def index(root: Path) -> dict[str, dict[str, Path]]:
    idx: dict[str, dict[str, Path]] = {}
    for p in C.walk_files(Path(root), (".mp3", ".wav", ".mid", ".json")):
        folder = p.parent.name.lower()
        key = {"audio_bass_stems": "audio", "midi_fully_aligned": "midi", "syncpoints": "sync"}.get(folder)
        if key:
            idx.setdefault(p.stem, {})[key] = p
    return {k: v for k, v in idx.items() if "midi" in v}


def _beats(sync: Path | None, beats_per_bar: int = 4) -> tuple[list[float], list[float]]:
    if sync is None:
        return [], []
    with open(sync, encoding="utf-8") as f:
        raw = json.load(f)
    # [bar, time] = bar start; a few files add [bar, time, tick] sub-bar points (Milestones) -> not downbeats
    pts = sorted({int(p[0]): float(p[1]) for p in reversed(raw) if len(p) == 2}.items())
    downs = [C.r6(t) for _, t in pts]
    beats: list[float] = []
    for (b0, t0), (b1, t1) in zip(pts, pts[1:]):
        step = (t1 - t0) / (beats_per_bar * max(1, b1 - b0))
        n = beats_per_bar * max(1, b1 - b0)
        beats += [C.r6(t0 + k * step) for k in range(n)]
    if downs:
        beats.append(downs[-1])
    return beats, downs


def _track(root: Path, song: str, files: dict[str, Path]) -> Any:
    s = C.schema()
    mid = C.read_midi(files["midi"])
    raw = [C.RawNote(n.onset, n.offset, n.pitch, "acoustic_bass", None, None, n.velocity) for n in mid.notes]
    audio = files.get("audio")
    ns, _ = C.build_noteset("bass_stem" if audio else "annotation", raw,
                           duration=C.audio_duration(audio) if audio else None,
                           source=f"FiloBass midi_fully_aligned ({C.rel(files['midi'], root)})")
    bpb = mid.time_signatures[0][1] if mid.time_signatures else 4
    beats, downs = _beats(files.get("sync"), bpb)
    parts = []
    if audio:
        _, _, ch = C.audio_info(audio)
        parts.append(s.Part(id="bass_stem", kind="bass", audio=C.rel(audio, root), channels=ch, line=LINE_ID, take=1))
    return s.DatasetTrack(
        format="bandscribe.dataset_track/1", dataset=NAME, track_id=song, split=None, group=song, tier="C",
        license=LICENSE, mix=C.rel(audio, root) if audio else None, parts=parts,
        lines=[s.GtLine(id=LINE_ID, name="Upright bass", kind="bass", takes=[1])], families_present=["bass"],
        notes={LINE_ID: ns}, string_fret=None, beats_s=beats or None, downbeats_s=downs or None,
        tuning=list(BASS_TUNING),
        meta={"midi": C.rel(files["midi"], root), "syncpoints": C.rel(files["sync"], root) if "sync" in files else None,
              "time_signature": f"{bpb}/4", "audio_missing": audio is None,
              "note": "audio is a separated bass stem (MP3); jazz upright bass"})


def track_ids(root: Path, *, split: str | None = None) -> list[str]:
    return [] if split is not None else sorted(index(Path(root)))


def iter_tracks(root: Path, *, split: str | None = None) -> Iterator[Any]:
    if split is not None:
        return
    root = Path(root)
    for song, files in sorted(index(root).items()):
        yield _track(root, song, files)


def load_track(root: Path, track_id: str) -> Any:
    files = index(Path(root)).get(track_id)
    if files is None:
        raise KeyError(f"FiloBass 에 없는 곡입니다: {track_id}")
    return _track(Path(root), track_id, files)
