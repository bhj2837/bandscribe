"""EGDB loader (Tier C; public Google Drive folder, license not stated -> personal research use only).

Verified against the real files (2026-09-30): folders ``audio_DI``, ``audio_Ftwin``, ``audio_JCjazz``,
``audio_Marshall``, ``audio_Mesa``, ``audio_Plexi`` with ``<clip>.wav`` (DI: mono 16-bit; amp renders: stereo
24-bit; all 44.1 kHz) and
``audio_label/<clip>.midi`` (SMF format 1, 480 tpq, one tempo, 7 tracks: track 0 = conductor with chord
markers, tracks 1..6 = strings; channel = track - 1; track names are "1".."6" or empty). Clip ids 1..240
(241 is absent although the README says 216-241). Track index 1 holds the highest notes (median E4..) and
track 6 the lowest (min MIDI 40) -> track index = §6 string number directly.
Tuning: EGDB states none; standard tuning is assumed for every clip. Checked 2026-10-03 on all 240 labels:
84 of 35,704 string-labelled notes (0.24 %, in 42 clips) lie below the open pitch of their labelled string,
almost always one isolated note on a string whose other notes reach the standard open pitch exactly - label
noise (a wrong string, or an octave slip: 83 of the 84 become playable at +12), not a down-tuned clip. (An
earlier version inferred a uniform down-tuning from the lowest such note and shifted every fret of 95 renders.)
Those notes keep their pitch (the dataset's GT); the track's ``string_fret`` is None and
``meta.string_label_errors`` counts them. Clips 33 and 40 put every note in the conductor track 0 (no string
labels): ``string_fret`` None, ``meta.notes_without_string``.

Official split (README.txt): clips 216-241 = test ("9/1 splitting"), others train; timbres Marshall, Ftwin,
Mesa = training timbres, JCjazz and Plexi = test timbres (``meta.timbre_split``); DI = "di".
Timing check (2026-09-30, 9 DI clips incl. test ids): label onsets vs a spectral-flux onset envelope of the DI
audio peak at 0 to -2 frames of 10 ms, i.e. aligned within the envelope's own framing bias - no offset applied.
One DatasetTrack per (timbre, clip): ``track_id = "<timbre>_<clip:03d>"`` (e.g. ``plexi_216``), ``group`` =
the clip (all renders of one clip form one bootstrap block). ``meta.drive_guess`` (clean/distorted) comes
from the amp model names only (Fender Twin, Roland JC-120 clean; Marshall, Mesa, Plexi driven) — not verified
by listening. ``meta.scenario`` = "distortion" | "clean" | "di" (same guess) for EVAL's per-scenario rows.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gtab.datasets.loaders import _common as C

log = logging.getLogger(__name__)

NAME = "egdb"
LICENSE = "not stated - personal research use only"
TIMBRES = ("DI", "Ftwin", "JCjazz", "Marshall", "Mesa", "Plexi")
TRAIN_TIMBRES = ("Marshall", "Ftwin", "Mesa")
TEST_TIMBRES = ("JCjazz", "Plexi")
DRIVE_GUESS = {"DI": "clean", "Ftwin": "clean", "JCjazz": "clean", "Marshall": "distorted", "Mesa": "distorted",
               "Plexi": "distorted"}
TEST_CLIPS = range(216, 242)
LINE_ID = "gtr"
_ID_RE = re.compile(r"^(?P<timbre>[a-z]+)_(?P<clip>\d{1,3})$")


def scenario_of(timbre: str) -> str:
    """Evaluation subset label (``meta.scenario``): "di", or "distortion" / "clean" from :data:`DRIVE_GUESS`
    (the subsets of the gate-m2 distortion check, M1_M2_SPEC §10)."""
    if timbre == "DI":
        return "di"
    return "distortion" if DRIVE_GUESS[timbre] == "distorted" else "clean"


def clip_split(clip: int) -> str:
    return "test" if clip in TEST_CLIPS else "train"


def timbre_split(timbre: str) -> str:
    return "test" if timbre in TEST_TIMBRES else ("train" if timbre in TRAIN_TIMBRES else "di")


def index(root: Path) -> dict[str, Any]:
    """{"labels": {clip: path}, "audio": {(timbre, clip): path}, "readme": path|None}."""
    labels: dict[int, Path] = {}
    audio: dict[tuple[str, int], Path] = {}
    readme = None
    timbre_by_folder = {f"audio_{t}".lower(): t for t in TIMBRES}
    for p in C.walk_files(Path(root), (".midi", ".mid", ".wav", ".txt")):
        folder = p.parent.name.lower()
        if p.suffix.lower() == ".txt":
            if p.name.lower() == "readme.txt" and readme is None:
                readme = p
            continue
        if not p.stem.isdigit():
            continue
        clip = int(p.stem)
        if folder == "audio_label" and p.suffix.lower() in (".midi", ".mid"):
            labels[clip] = p
        elif folder in timbre_by_folder and p.suffix.lower() == ".wav":
            audio[(timbre_by_folder[folder], clip)] = p
    return {"labels": labels, "audio": audio, "readme": readme}


def _notes(label: Path) -> tuple[list[C.RawNote], int, int]:
    """Notes with (string, fret) in standard tuning; returns (notes, impossible string labels, unlabelled notes)."""
    mid = C.read_midi(label)
    raw, impossible, unlabelled = [], 0, 0
    for n in mid.notes:
        string = n.track if 1 <= n.track <= 6 else None
        fret = C.fret_from_pitch(n.pitch, string, C.GUITAR_STD_TUNING) if string else None
        if string is None:
            unlabelled += 1
        elif fret is None:
            impossible += 1
        raw.append(C.RawNote(n.onset, n.offset, n.pitch, None, string, fret, n.velocity))
    return raw, impossible, unlabelled


def _track(root: Path, idx: dict[str, Any], timbre: str, clip: int) -> Any:
    s = C.schema()
    label = idx["labels"][clip]
    wav = idx["audio"].get((timbre, clip))
    raw, impossible, unlabelled = _notes(label)
    view = f"audio_{timbre}"
    ns, ordered = C.build_noteset(view, raw, duration=C.audio_duration(wav) if wav else None,
                                  source=f"EGDB label MIDI ({C.rel(label, root)})")
    sf = C.string_fret_or_none(ordered)
    parts = []
    if wav:
        _, _, ch = C.audio_info(wav)
        parts.append(s.Part(id=timbre.lower(), kind="di" if timbre == "DI" else "guitar", audio=C.rel(wav, root),
                            channels=ch, line=LINE_ID, take=1))
    return s.DatasetTrack(
        format="gtab.dataset_track/1", dataset=NAME, track_id=f"{timbre.lower()}_{clip:03d}",
        split=clip_split(clip), group=f"{clip:03d}", tier="C", license=LICENSE,
        mix=C.rel(wav, root) if wav else None, parts=parts,
        lines=[s.GtLine(id=LINE_ID, name="Guitar", kind="guitar", takes=[1])], families_present=["guitar"],
        notes={LINE_ID: ns}, string_fret={LINE_ID: sf} if sf is not None else None, beats_s=None, downbeats_s=None,
        tuning=list(C.GUITAR_STD_TUNING),
        meta={"clip": clip, "timbre": timbre, "timbre_split": timbre_split(timbre),
              "drive_guess": DRIVE_GUESS[timbre], "scenario": scenario_of(timbre),
              "tuning_assumed": "standard (EGDB states no tuning)",
              "string_label_errors": impossible, "notes_without_string": unlabelled,
              "label": C.rel(label, root), "audio_missing": wav is None,
              "split_rule": "README: clips 216-241 test; timbres JCjazz/Plexi test"})


def _keys(idx: dict[str, Any], split: str | None) -> list[tuple[str, int]]:
    return [(timbre, clip) for (timbre, clip) in sorted(idx["audio"], key=lambda k: (TIMBRES.index(k[0]), k[1]))
            if clip in idx["labels"] and (split is None or clip_split(clip) == split)]


def track_ids(root: Path, *, split: str | None = None) -> list[str]:
    """``<timbre>_<clip:03d>`` ids in iteration order (timbre order of :data:`TIMBRES`, then clip)."""
    return [f"{t.lower()}_{c:03d}" for t, c in _keys(index(Path(root)), split)]


def iter_tracks(root: Path, *, split: str | None = None) -> Iterator[Any]:
    """Every available (timbre, clip) render with a label; ``split`` = "train" | "test" (clip split)."""
    root = Path(root)
    idx = index(root)
    for timbre, clip in _keys(idx, split):
        yield _track(root, idx, timbre, clip)


def load_track(root: Path, track_id: str) -> Any:
    m = _ID_RE.match(track_id)
    by_lower = {t.lower(): t for t in TIMBRES}
    if not m or m["timbre"] not in by_lower:
        raise KeyError(f"EGDB 트랙 이름은 '<음색>_<번호>' 입니다 (예: plexi_216): {track_id}")
    root = Path(root)
    idx = index(root)
    clip = int(m["clip"])
    if clip not in idx["labels"]:
        raise KeyError(f"EGDB 에 없는 클립입니다: {clip}")
    return _track(root, idx, by_lower[m["timbre"]], clip)
