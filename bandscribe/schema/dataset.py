"""Dataset tracks (Tier B/C, M1_M2_SPEC 6.8): what every ``bandscribe.datasets`` loader yields.

Paths are POSIX, relative to the dataset root (``data/datasets/<name>``). GT notes are seconds-based
:class:`~bandscribe.schema.notes.NoteSet` objects per line; ``string_fret`` lists (string, fret) per note in the same
order, with strings numbered from the highest-pitched string (loaders convert dataset indices on import).
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from bandscribe.schema._common import _Model
from bandscribe.schema.gt import GtLine
from bandscribe.schema.instrumentation import check_families
from bandscribe.schema.notes import NoteSet


class Part(_Model):
    id: str
    kind: Literal["guitar", "bass", "vocals", "drums", "keys", "other", "mix", "di"]
    audio: str  # path relative to the dataset root
    channels: int
    line: str | None = None
    take: int | None = None


class DatasetTrack(_Model):
    format: Literal["bandscribe.dataset_track/1"]
    dataset: str
    track_id: str
    split: str | None
    group: str | None  # song/performer id for the block bootstrap and comp/solo pairing
    tier: Literal["B", "C"]
    license: str
    mix: str | None
    parts: list[Part]
    lines: list[GtLine]
    families_present: list[str]  # FAMILIES keys playing in this track (GuitarSet: ["guitar"])
    notes: dict[str, NoteSet]  # line id -> GT notes (seconds); NoteEvent.instrument may be None
    string_fret: dict[str, list[tuple[int, int]]] | None  # line id -> per-note (string, fret), notes' order
    beats_s: list[float] | None
    downbeats_s: list[float] | None
    tuning: list[int] | None
    meta: dict

    @model_validator(mode="after")
    def _check(self) -> DatasetTrack:
        check_families(self.families_present, "families_present")
        if self.string_fret is not None:
            for line, pairs in self.string_fret.items():
                ns = self.notes.get(line)
                if ns is None:
                    raise ValueError(f"string_fret[{line!r}] has no notes[{line!r}]")
                if len(pairs) != len(ns.notes):
                    raise ValueError(
                        f"string_fret[{line!r}] has {len(pairs)} entries for {len(ns.notes)} notes (same order)"
                    )
        return self
