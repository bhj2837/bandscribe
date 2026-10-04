"""Shared helpers for the dataset loaders: schema construction, file discovery, a small MIDI reader.

The loaders turn each dataset's own layout into ``bandscribe.schema.dataset.DatasetTrack`` (M1_M2_SPEC §6.8). The
schema modules (P0) are imported inside :func:`schema` so importing a loader stays cheap.

Conventions enforced here (§6): times in seconds rounded to 1e-6; MIDI pitch ints; notes sorted by
(onset, pitch, offset, instrument or ""); string 1 = highest-pitched string, ``tuning[i]`` = open pitch of
string ``i+1``; paths in a DatasetTrack are POSIX and relative to the dataset root (``data/datasets/<name>``).

The MIDI reader is a minimal Standard MIDI File parser (formats 0/1, running status, tempo map). It keeps the
loaders independent of mido's version and handles the two label layouts we need (EGDB: one track per
string; FiloBass: one bass track).
"""

from __future__ import annotations

import logging
import os
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bandscribe.datasets.registry import FAMILY_KEYS

log = logging.getLogger(__name__)

GUITAR_STD_TUNING = [64, 59, 55, 50, 45, 40]  # string 1 (high E) .. string 6 (low E)
# FAMILIES key order of bandscribe.schema.instrumentation (families_present lists are written in this order); the one
# stdlib copy lives in the registry (tests/test_datasets_registry.py checks it against the schema).
FAMILY_ORDER = FAMILY_KEYS
_SKIP_DIR_PREFIXES = (".", "__MACOSX")


def schema() -> SimpleNamespace:
    """The §6.8 pydantic models (P0's ``bandscribe.schema``), imported lazily."""
    from bandscribe.schema.dataset import DatasetTrack, Part
    from bandscribe.schema.gt import GtLine
    from bandscribe.schema.notes import BackendInfo, NoteEvent, NoteSet

    return SimpleNamespace(DatasetTrack=DatasetTrack, Part=Part, GtLine=GtLine, NoteSet=NoteSet,
                           NoteEvent=NoteEvent, BackendInfo=BackendInfo)


def r6(x: float) -> float:
    return round(float(x), 6)


def rel(path: Path, root: Path) -> str:
    return Path(path).relative_to(root).as_posix()


def audio_info(path: Path) -> tuple[int, int, int]:
    """(frames, samplerate, channels) via soundfile (no decode)."""
    import soundfile as sf

    info = sf.info(str(path))
    return int(info.frames), int(info.samplerate), int(info.channels)


def audio_duration(path: Path) -> float | None:
    try:
        frames, sr, _ = audio_info(path)
        return r6(frames / sr)
    except Exception:  # unreadable/missing audio: the track can still carry annotations
        log.debug("cannot read %s", path, exc_info=True)
        return None


def walk_files(root: Path, suffixes: Iterable[str]) -> Iterator[Path]:
    """Every file under ``root`` with one of ``suffixes`` (lower-case), skipping hidden/temp/partial files."""
    suf = tuple(s.lower() for s in suffixes)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(_SKIP_DIR_PREFIXES) and not d.startswith(".tmp-"))
        for fn in sorted(filenames):
            if fn.startswith(".") or fn.endswith(".part"):
                continue
            if fn.lower().endswith(suf):
                yield Path(dirpath) / fn


def find_dirs(root: Path, name: str) -> list[Path]:
    """Directories called ``name`` (case-insensitive) anywhere under ``root``."""
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(_SKIP_DIR_PREFIXES) and not d.startswith(".tmp-"))
        for d in dirnames:
            if d.lower() == name.lower():
                out.append(Path(dirpath) / d)
    return out


@dataclass
class RawNote:
    onset: float
    offset: float
    pitch: int
    instrument: str | None = None
    string: int | None = None      # §6 numbering (1 = highest-pitched)
    fret: int | None = None
    velocity: int | None = None


def build_noteset(view: str, notes: list[RawNote], *, duration: float | None, source: str,
                  params: dict[str, Any] | None = None) -> tuple[Any, list[RawNote]]:
    """NoteSet with GT notes in canonical order; also returns the notes in that same order (for string_fret).

    Zero/negative-length notes get offset = onset + 1 ms (annotation artefacts; counted in params).
    """
    s = schema()
    fixed = 0
    clean = []
    for n in notes:
        on, off = max(0.0, r6(n.onset)), r6(n.offset)
        if off <= on:
            off = r6(on + 0.001)
            fixed += 1
        clean.append(RawNote(on, off, int(n.pitch), n.instrument, n.string, n.fret, n.velocity))
    clean.sort(key=lambda n: (n.onset, n.pitch, n.offset, n.instrument or ""))
    events = [s.NoteEvent(onset_s=n.onset, offset_s=n.offset, pitch=n.pitch, instrument=n.instrument,
                          velocity=n.velocity) for n in clean]
    p = dict(params or {})
    p["source"] = source
    if fixed:
        p["zero_length_fixed"] = fixed
    ns = s.NoteSet(view=view, backend=s.BackendInfo(name="gt", params=p), audio_duration_s=duration, notes=events)
    return ns, clean


def string_fret_or_none(notes: list[RawNote]) -> list[tuple[int, int]] | None:
    if not notes or any(n.string is None or n.fret is None or n.fret < 0 for n in notes):
        return None
    return [(int(n.string), int(n.fret)) for n in notes]


def fret_from_pitch(pitch: int, string: int, tuning: list[int]) -> int | None:
    if not 1 <= string <= len(tuning):
        return None
    f = int(pitch) - int(tuning[string - 1])
    return f if 0 <= f <= 30 else None


# ---------------------------------------------------------------------------------------------- MIDI


@dataclass
class MidiNote:
    track: int
    track_name: str
    channel: int
    pitch: int
    velocity: int
    onset: float
    offset: float


@dataclass
class MidiFile:
    ticks_per_beat: int
    notes: list[MidiNote]
    tempos: list[tuple[int, int]]            # (tick, microseconds per quarter)
    time_signatures: list[tuple[float, int, int]]  # (seconds, numerator, denominator)
    track_names: list[str]
    markers: list[tuple[float, str]]


class MidiError(ValueError):
    pass


def _vlq(data: bytes, i: int) -> tuple[int, int]:
    v = 0
    for _ in range(4):
        b = data[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i
    raise MidiError("variable-length quantity too long")


def read_midi(path: Path) -> MidiFile:
    """Parse a Standard MIDI File into seconds (tempo map from all tracks, as format-1 players do)."""
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise MidiError(f"not a MIDI file: {path}")
    hlen = struct.unpack(">I", data[4:8])[0]
    fmt, ntrk, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise MidiError("SMPTE time division is not supported")
    tpb = division
    pos = 8 + hlen
    raw_tracks: list[list[tuple[int, str, tuple]]] = []
    names: list[str] = []
    for _ in range(ntrk):
        if data[pos:pos + 4] != b"MTrk":
            raise MidiError("missing MTrk chunk")
        tlen = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        chunk = data[pos + 8:pos + 8 + tlen]
        pos += 8 + tlen
        events: list[tuple[int, str, tuple]] = []
        i, tick, status, name = 0, 0, 0, ""
        while i < len(chunk):
            delta, i = _vlq(chunk, i)
            tick += delta
            b = chunk[i]
            if b == 0xFF:
                mtype = chunk[i + 1]
                ln, j = _vlq(chunk, i + 2)
                payload = chunk[j:j + ln]
                i = j + ln
                if mtype == 0x51 and ln == 3:
                    events.append((tick, "tempo", (int.from_bytes(payload, "big"),)))
                elif mtype == 0x58 and ln >= 2:
                    events.append((tick, "timesig", (payload[0], 2 ** payload[1])))
                elif mtype == 0x03 and not name:
                    name = payload.decode("latin-1", "replace")
                elif mtype in (0x06, 0x01):
                    events.append((tick, "marker", (payload.decode("latin-1", "replace"),)))
                elif mtype == 0x2F:
                    break
                continue
            if b in (0xF0, 0xF7):
                ln, j = _vlq(chunk, i + 1)
                i = j + ln
                continue
            if b & 0x80:
                status = b
                i += 1
            elif not status:
                raise MidiError("running status without a status byte")
            kind = status & 0xF0
            ch = status & 0x0F
            if kind in (0xC0, 0xD0):
                i += 1
                continue
            d1, d2 = chunk[i], chunk[i + 1]
            i += 2
            if kind == 0x90 and d2 > 0:
                events.append((tick, "on", (ch, d1, d2)))
            elif kind == 0x80 or (kind == 0x90 and d2 == 0):
                events.append((tick, "off", (ch, d1)))
        raw_tracks.append(events)
        names.append(name)

    tempos = sorted({(t, ev[0]) for tr in raw_tracks for (t, k, ev) in tr if k == "tempo"})
    if not tempos or tempos[0][0] != 0:
        tempos = [(0, 500000)] + tempos
    # collapse same-tick tempo changes (keep the last one listed)
    dedup: dict[int, int] = {}
    for t, us in tempos:
        dedup[t] = us
    tempos = sorted(dedup.items())

    seg_ticks = [t for t, _ in tempos]
    seg_secs = [0.0]
    for (t0, us0), (t1, _) in zip(tempos, tempos[1:]):
        seg_secs.append(seg_secs[-1] + (t1 - t0) * us0 / 1e6 / tpb)

    import bisect

    def tick_to_s(tick: int) -> float:
        k = bisect.bisect_right(seg_ticks, tick) - 1
        return seg_secs[k] + (tick - tempos[k][0]) * tempos[k][1] / 1e6 / tpb

    notes: list[MidiNote] = []
    timesigs: list[tuple[float, int, int]] = []
    markers: list[tuple[float, str]] = []
    for ti, tr in enumerate(raw_tracks):
        open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for tick, kind, ev in tr:
            if kind == "on":
                open_notes.setdefault((ev[0], ev[1]), []).append((tick, ev[2]))
            elif kind == "off":
                stack = open_notes.get((ev[0], ev[1]))
                if stack:
                    t_on, vel = stack.pop(0)  # FIFO: overlapping same-pitch notes close in order
                    notes.append(MidiNote(ti, names[ti], ev[0], ev[1], vel, tick_to_s(t_on), tick_to_s(tick)))
            elif kind == "timesig":
                timesigs.append((tick_to_s(tick), ev[0], ev[1]))
            elif kind == "marker":
                markers.append((tick_to_s(tick), ev[0]))
        last_tick = tr[-1][0] if tr else 0
        for (ch, pitch), stack in open_notes.items():
            for t_on, vel in stack:  # unterminated notes end at the track's last event
                notes.append(MidiNote(ti, names[ti], ch, pitch, vel, tick_to_s(t_on), tick_to_s(max(last_tick, t_on))))
    notes.sort(key=lambda n: (n.onset, n.pitch, n.offset, n.track))
    return MidiFile(tpb, notes, tempos, sorted(timesigs), names, sorted(markers))
