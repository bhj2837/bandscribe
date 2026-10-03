"""IDMT-SMT-Bass-Single-Track loader (Tier C; Zenodo 7544099, CC BY-NC-ND 4.0).

Verified against the real archive (2026-09-30): ``annotation/NNN.xml`` (``instrumentRecording`` ->
``globalParameter`` {audioFileName, instrument, instrumentModel, pickUpSetting, instrumentTuning "28,33,38,43"
low -> high, recordingArtist} and ``transcription/event`` {pitch, onsetSec, offsetSec, fretNumber,
stringNumber, excitationStyle, expressionStyle, modulationFrequencyRange, modulationFrequency}),
``audio/NNN.wav``, ``misc/beats_csv/NNN_beats.csv`` ("time,bar.beat"; beat 1 = downbeat), plus GPX / MusicXML /
PDF / Sonic Visualiser files. IDMT's ``stringNumber`` 1 is the **lowest** string (event pitch 36 = fret 3 on
string 2 = A1 33) -> §6 string = n_strings - idmt + 1, and the tuning list is reversed (highest first).
Notes with ``expressionStyle`` "HA" are natural harmonics (track 017): ``pitch`` is the sounding pitch and the
fret is where the string is touched (5th fret -> +24 semitones), so ``tuning + fret == pitch`` does not hold
for them; ``meta.styles`` (same order as the notes) marks them.
"""

from __future__ import annotations

import csv
import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gtab.datasets.loaders import _common as C

log = logging.getLogger(__name__)

NAME = "idmt_bass_st"
LICENSE = "CC BY-NC-ND 4.0"
LINE_ID = "bass"


def index(root: Path) -> dict[str, dict[str, Path]]:
    idx: dict[str, dict[str, Path]] = {}
    for p in C.walk_files(Path(root), (".xml", ".wav", ".csv")):
        parent = p.parent.name.lower()
        if p.suffix.lower() == ".xml" and parent == "annotation":
            idx.setdefault(p.stem, {})["xml"] = p
        elif p.suffix.lower() == ".wav" and parent == "audio":
            idx.setdefault(p.stem, {})["wav"] = p
        elif p.name.lower().endswith("_beats.csv"):
            idx.setdefault(p.name[: -len("_beats.csv")], {})["beats"] = p
    return {k: v for k, v in idx.items() if "xml" in v}


def _text(el: ET.Element | None, tag: str, default: str = "") -> str:
    if el is None:
        return default
    x = el.find(tag)
    return (x.text or "").strip() if x is not None and x.text is not None else default


def parse_xml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    g = root.find("globalParameter")
    tuning_low_high = [int(x) for x in _text(g, "instrumentTuning").replace(" ", "").split(",") if x]
    n_str = len(tuning_low_high)
    tuning = list(reversed(tuning_low_high))  # §6: string 1 = highest
    notes, styles = [], []
    for ev in root.iter("event"):
        pitch = int(float(_text(ev, "pitch")))
        idmt_string = int(_text(ev, "stringNumber", "0") or 0)
        fret = _text(ev, "fretNumber")
        string = n_str - idmt_string + 1 if n_str and 1 <= idmt_string <= n_str else None
        notes.append(C.RawNote(float(_text(ev, "onsetSec")), float(_text(ev, "offsetSec")), pitch, "electric_bass",
                               string, int(float(fret)) if fret else None))
        styles.append({"excitation": _text(ev, "excitationStyle"), "expression": _text(ev, "expressionStyle"),
                       "mod_freq_hz": float(_text(ev, "modulationFrequency", "0") or 0),
                       "mod_range": float(_text(ev, "modulationFrequencyRange", "0") or 0)})
    meta = {k: _text(g, k) for k in ("audioFileName", "instrument", "instrumentModel", "pickUpSetting",
                                      "recordingArtist")}
    return {"notes": notes, "styles": styles, "tuning": tuning, "meta": meta}


def parse_beats(path: Path) -> tuple[list[float], list[float]]:
    beats, downs = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip():
                continue
            t = C.r6(float(row[0]))
            beats.append(t)
            beat_no = row[1].strip().split(".")[-1]
            if beat_no == "1":
                downs.append(t)
    return beats, downs


def _track(root: Path, track_id: str, files: dict[str, Path]) -> Any:
    s = C.schema()
    info = parse_xml(files["xml"])
    wav = files.get("wav")
    ns, ordered = C.build_noteset("bass" if wav else "annotation", info["notes"],
                                  duration=C.audio_duration(wav) if wav else None,
                                  source=f"IDMT XML ({C.rel(files['xml'], root)})")
    # styles follow the notes' canonical order (for M5 technique checks)
    order = sorted(range(len(info["notes"])), key=lambda i: (C.r6(info["notes"][i].onset), info["notes"][i].pitch,
                                                               C.r6(info["notes"][i].offset)))
    beats, downs = parse_beats(files["beats"]) if "beats" in files else ([], [])
    sf = C.string_fret_or_none(ordered)
    parts = []
    if wav:
        _, _, ch = C.audio_info(wav)
        parts.append(s.Part(id="bass", kind="bass", audio=C.rel(wav, root), channels=ch, line=LINE_ID, take=1))
    return s.DatasetTrack(
        format="gtab.dataset_track/1", dataset=NAME, track_id=track_id, split=None, group=track_id, tier="C",
        license=LICENSE, mix=C.rel(wav, root) if wav else None, parts=parts,
        lines=[s.GtLine(id=LINE_ID, name="Bass", kind="bass", takes=[1])], families_present=["bass"],
        notes={LINE_ID: ns}, string_fret={LINE_ID: sf} if sf is not None else None,
        beats_s=beats or None, downbeats_s=downs or None, tuning=info["tuning"] or None,
        meta={**info["meta"], "styles": [info["styles"][i] for i in order], "xml": C.rel(files["xml"], root),
              "harmonics": sum(1 for st in info["styles"] if st["expression"] == "HA"),
              "string_rule": "IDMT stringNumber 1 = lowest -> converted to 1 = highest (§6)"})


def track_ids(root: Path, *, split: str | None = None) -> list[str]:
    return [] if split is not None else sorted(index(Path(root)))


def iter_tracks(root: Path, *, split: str | None = None) -> Iterator[Any]:
    if split is not None:
        return  # no official split
    root = Path(root)
    for track_id, files in sorted(index(root).items()):
        yield _track(root, track_id, files)


def load_track(root: Path, track_id: str) -> Any:
    files = index(Path(root)).get(track_id)
    if files is None:
        raise KeyError(f"IDMT-SMT-Bass-Single-Track 에 없는 트랙입니다: {track_id}")
    return _track(Path(root), track_id, files)
