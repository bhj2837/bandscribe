"""Note sets: the raw output of every transcription backend and of ground-truth loaders (M1_M2_SPEC 6.1).

File naming: ``<view>__<backend>.json`` (e.g. ``raw/guitar_mono__muscriptor.json``). No timestamps: CPU stage
outputs are byte-deterministic (DESIGN 8.7), so notes are kept in one canonical order,
``(onset_s, pitch, offset_s, instrument or "")``, which :class:`NoteSet` enforces.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from bandscribe.schema._common import Midi, Prob, Seconds, _Model


class PitchBend(_Model):
    t_s: list[float]  # absolute times (s)
    cents: list[float]  # relative to the note's `pitch`; same length as t_s

    @model_validator(mode="after")
    def _same_length(self) -> PitchBend:
        if len(self.t_s) != len(self.cents):
            raise ValueError(f"t_s ({len(self.t_s)}) and cents ({len(self.cents)}) must have the same length")
        return self


class NoteEvent(_Model):
    onset_s: Seconds
    offset_s: Seconds
    pitch: Midi
    instrument: str | None = None  # MT3_FULL_PLUS group name (bandscribe.schema.instrumentation); None = class-agnostic
    velocity: int | None = Field(None, ge=0, le=127)  # MuScriptor never sets it
    amplitude: float | None = None  # Basic Pitch note amplitude
    confidence: Prob | None = None
    bend: PitchBend | None = None

    @model_validator(mode="after")
    def _check(self) -> NoteEvent:
        if self.offset_s < self.onset_s:
            raise ValueError(f"offset_s ({self.offset_s}) < onset_s ({self.onset_s})")
        return self


def note_sort_key(n: NoteEvent) -> tuple[float, int, float, str]:
    """The canonical NoteSet order (M1_M2_SPEC 6.1)."""
    return (n.onset_s, n.pitch, n.offset_s, n.instrument or "")


class BackendInfo(_Model):
    name: str  # "muscriptor" | "basicpitch" | "gt" | "external:<system>" | "synthetic"
    version: str | None = None
    model: str | None = None
    model_sha256: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    instruments: list[str] | None = None  # the hard mask actually used (None = all classes)
    device: str | None = None
    dtype: str | None = None


class NoteSet(_Model):
    format: Literal["bandscribe.notes/1"] = "bandscribe.notes/1"
    view: str  # view id: mix_mono | guitar_mono | bass_mono | piano_other_mono | nonvox_mono | ...
    view_sha256: str | None = None  # bandscribe.audio.pcm_sha256 of the input view
    backend: BackendInfo
    time_offset_applied_s: float = 0.0  # already added to onset/offset (MuScriptor: -latency)
    audio_duration_s: Seconds | None = None
    # Only parts of the view were transcribed (S3.5 presence windows): [[start_s, end_s], ...] in view time,
    # sorted and disjoint. Note times stay in view time. None = the whole view.
    segments: list[tuple[Seconds, Seconds]] | None = None
    notes: list[NoteEvent]

    @model_validator(mode="after")
    def _sorted(self) -> NoteSet:
        prev_end = 0.0
        for a, b in self.segments or []:
            if b <= a or a < prev_end:
                raise ValueError(f"segments must be sorted, disjoint and non-empty: [{a}, {b}] after {prev_end}")
            prev_end = b
        keys = [note_sort_key(n) for n in self.notes]
        for i in range(1, len(keys)):
            if keys[i] < keys[i - 1]:
                raise ValueError(
                    f"notes must be sorted by (onset_s, pitch, offset_s, instrument or ''); "
                    f"note {i} {keys[i]} comes after {keys[i - 1]}"
                )
        return self


def sort_notes(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Notes in the canonical NoteSet order (stable)."""
    return sorted(notes, key=note_sort_key)


def notes_to_arrays(ns: NoteSet, *, instruments: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(intervals (n, 2) seconds float64, pitches (n,) MIDI float64)`` - the mir_eval-style note arrays.

    ``instruments`` keeps only notes of those classes (notes without a class are dropped then). Pitches stay
    MIDI numbers: callers convert to Hz (``mir_eval.util.midi_to_hz``) before any mir_eval call.
    """
    sel = [n for n in ns.notes if instruments is None or n.instrument in instruments]
    iv = np.array([[n.onset_s, n.offset_s] for n in sel], dtype=np.float64).reshape(-1, 2)
    p = np.array([n.pitch for n in sel], dtype=np.float64)
    return iv, p
