"""GuitarSet 1.1.0 loader (Tier C; Zenodo 3371780, CC BY 4.0).

Verified against the real files (2026-09-30): ``annotation.zip`` holds 360 ``<id>.jams`` at its top level;
JAMS ``note_midi`` annotations come one per string with ``annotation_metadata.data_source`` "0".."5", and
source "0" is the **low E** string (its notes sit around MIDI 40-50) -> §6 string = 6 - source. ``note_midi``
values are float MIDI (pitch-tracked) -> rounded. ``beat_position`` gives beats with ``position`` 1 on the
downbeat. Mono mic audio: ``audio_mono-mic/<id>_mic.wav``; pickup mix: ``audio_mono-pickup_mix/<id>_mix.wav``.

Track id ``PP_StyleN-TEMPO-KEY_mode`` (e.g. ``00_BN1-129-Eb_comp``): PP = player (group for the block
bootstrap), mode comp|solo (the other mode of the same excerpt is ``meta.pair``). The key in the id is the
song key, not a tuning: GuitarSet is recorded in standard tuning.

Split (pre-registered for the latency experiment, M1_M2_SPEC §10): ``dev`` = players 00-02, ``test`` =
players 03-05. GuitarSet is in Basic Pitch's training data -> never a transcription verdict (DESIGN 8.2).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gtab.datasets.loaders import _common as C

log = logging.getLogger(__name__)

NAME = "guitarset"
LICENSE = "CC BY 4.0"
TRACK_RE = re.compile(r"^(?P<player>\d{2})_(?P<style>[A-Za-z]+)(?P<prog>\d)-(?P<tempo>\d+)-(?P<key>[A-G][#b]?)_(?P<mode>comp|solo)$")
DEV_PLAYERS = ("00", "01", "02")
TEST_PLAYERS = ("03", "04", "05")
LINE_ID = "gtr"


def split_of(player: str) -> str:
    return "dev" if player in DEV_PLAYERS else "test"


def index(root: Path) -> dict[str, dict[str, Path]]:
    """track id -> {"jams", "mic"?, "pickup_mix"?}; found anywhere under root (extracted/ or local/)."""
    idx: dict[str, dict[str, Path]] = {}
    for p in C.walk_files(Path(root), (".jams", ".wav")):
        stem = p.stem
        if p.suffix.lower() == ".jams":
            idx.setdefault(stem, {})["jams"] = p
        elif stem.endswith("_mic"):
            idx.setdefault(stem[:-4], {})["mic"] = p
        elif stem.endswith("_mix"):
            idx.setdefault(stem[:-4], {})["pickup_mix"] = p
    return {k: v for k, v in idx.items() if "jams" in v and TRACK_RE.match(k)}


def _data_rows(ann: dict[str, Any]) -> list[dict[str, Any]]:
    """JAMS data as a list of observations (both the list and the columnar dict layouts exist)."""
    d = ann.get("data", [])
    if isinstance(d, list):
        return d
    keys = list(d)
    n = len(d[keys[0]]) if keys else 0
    return [{k: d[k][i] for k in keys} for i in range(n)]


def parse_jams(path: Path) -> dict[str, Any]:
    """Notes per string (§6 numbering), beats, downbeats, tempo, duration from one GuitarSet JAMS file."""
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    notes: list[C.RawNote] = []
    beats: list[float] = []
    downbeats: list[float] = []
    tempo = None
    for ann in j.get("annotations", []):
        ns = ann.get("namespace")
        if ns == "note_midi":
            src = str((ann.get("annotation_metadata") or {}).get("data_source", "")).strip()
            if not src.isdigit() or not 0 <= int(src) <= 5:
                continue
            string = 6 - int(src)
            for row in _data_rows(ann):
                pitch = int(round(float(row["value"])))
                on = float(row["time"])
                notes.append(C.RawNote(on, on + float(row["duration"]), pitch, "acoustic_guitar", string,
                                       C.fret_from_pitch(pitch, string, C.GUITAR_STD_TUNING)))
        elif ns == "beat_position":
            for row in _data_rows(ann):
                t = C.r6(row["time"])
                beats.append(t)
                if int((row.get("value") or {}).get("position", 0)) == 1:
                    downbeats.append(t)
        elif ns == "tempo" and tempo is None:
            rows = _data_rows(ann)
            tempo = float(rows[0]["value"]) if rows else None
    dur = (j.get("file_metadata") or {}).get("duration")
    return {"notes": notes, "beats": sorted(beats), "downbeats": sorted(downbeats), "tempo": tempo,
            "duration": float(dur) if dur else None}


def _track(root: Path, track_id: str, files: dict[str, Path]) -> Any:
    s = C.schema()
    m = TRACK_RE.match(track_id)
    assert m is not None
    info = parse_jams(files["jams"])
    mic = files.get("mic")
    duration = C.audio_duration(mic) if mic else None
    ns, ordered = C.build_noteset(
        "mic" if mic else "annotation", info["notes"], duration=duration or C.r6(info["duration"] or 0) or None,
        source=f"GuitarSet JAMS note_midi ({C.rel(files['jams'], root)})")
    parts = []
    if mic:
        parts.append(s.Part(id="mic", kind="guitar", audio=C.rel(mic, root), channels=1, line=LINE_ID, take=1))
    if files.get("pickup_mix"):
        parts.append(s.Part(id="pickup_mix", kind="di", audio=C.rel(files["pickup_mix"], root), channels=1,
                            line=LINE_ID, take=1))
    other_mode = "solo" if m["mode"] == "comp" else "comp"
    sf = C.string_fret_or_none(ordered)
    return s.DatasetTrack(
        format="gtab.dataset_track/1", dataset=NAME, track_id=track_id, split=split_of(m["player"]),
        group=m["player"], tier="C", license=LICENSE, mix=C.rel(mic, root) if mic else None, parts=parts,
        lines=[s.GtLine(id=LINE_ID, name="Guitar", kind="guitar", takes=[1])],
        families_present=["guitar"], notes={LINE_ID: ns}, string_fret={LINE_ID: sf} if sf is not None else None,
        beats_s=info["beats"] or None, downbeats_s=info["downbeats"] or None, tuning=list(C.GUITAR_STD_TUNING),
        meta={"player": m["player"], "style": m["style"], "progression": int(m["prog"]),
              "tempo_bpm": int(m["tempo"]), "key": m["key"], "mode": m["mode"],
              "pair": f"{m['player']}_{m['style']}{m['prog']}-{m['tempo']}-{m['key']}_{other_mode}",
              "jams": C.rel(files["jams"], root), "audio_missing": mic is None,
              "split_rule": "latency: dev = players 00-02, test = 03-05 (M1_M2_SPEC §10)",
              "contamination": "in Basic Pitch training data - not for transcription verdicts"})


def track_ids(root: Path, *, split: str | None = None) -> list[str]:
    """Track ids (sorted) without parsing any JAMS file."""
    return [k for k in sorted(index(Path(root))) if split is None or split_of(k[:2]) == split]


def iter_tracks(root: Path, *, split: str | None = None) -> Iterator[Any]:
    root = Path(root)
    for track_id, files in sorted(index(root).items()):
        m = TRACK_RE.match(track_id)
        if split is not None and m and split_of(m["player"]) != split:
            continue
        yield _track(root, track_id, files)


def load_track(root: Path, track_id: str) -> Any:
    files = index(Path(root)).get(track_id)
    if files is None:
        raise KeyError(f"GuitarSet 에 없는 트랙입니다: {track_id}")
    return _track(Path(root), track_id, files)
