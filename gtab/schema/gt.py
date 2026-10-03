"""Tier A ground truth (M1_M2_SPEC 6.7): ``gt.yaml`` (:class:`GtSpec`) and ``gt_notes.json`` (:class:`GtNotes`).

Per song directory ``data/eval/tierA/<song_id>/`` (song_id ASCII ``[a-z0-9_-]+``): ``gt.yaml``, ``ref.<ext>``,
``refscore.json``, ``gt_notes.json``, ``tempo_map.json``, ``render.wav``, ``taps.json``, ``align_report.json``,
``align_check.wav``. ``tempo_map.json`` and ``taps.json`` are plain documents::

    tempo_map.json: {"format":"gtab.tempomap/1","source":"dtw|dtw+taps|synthetic","tpb":48,
                     "beats":[{"t_s","bar","beat"}],"beats_per_bar":{"<bar>": n}}   (gtab.schema.grid.TempoMap)
    taps.json:      {"format":"gtab.taps/1","source_file","taps":[{"t_s","bar"|null,"label"}]}

Bars are expanded musical bars (repeats and alternate endings unrolled; 1 = first full bar, 0 = pickup), ranges
half-open (``end_bar`` exclusive: "17~24마디면 start_bar: 17, end_bar: 25"). Strings: 1 = highest-pitched;
``tuning[i]`` = open pitch of string ``i+1``; pitch = tuning + capo + fret.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from gtab.schema._common import Midi, Seconds, _Model, check_half_open
from gtab.schema.instrumentation import check_families

SONG_ID_RE = re.compile(r"^[a-z0-9_-]+$")


class GtTrack(_Model):
    ref_track: int  # track number in the reference file
    name: str
    kind: Literal["guitar", "bass", "other"]
    line: str | None  # the Line this track belongs to
    tuning: list[Midi] | None  # string 1 (highest) first
    capo: int = Field(0, ge=0, le=24)
    player: str | None = None


class GtLine(_Model):
    id: str
    name: str
    kind: Literal["guitar", "bass"]
    takes: list[int]  # ref_track numbers (several = doubled takes of one Line, DESIGN R1)
    relation: dict = Field(default_factory=dict)  # {"double", "unison_with": [ids], "twin_with": id|None, "overdub"}


class GtSection(_Model):
    id: str
    start_bar: int  # expanded musical bars
    end_bar: int  # exclusive
    scenario: str
    roles: dict[str, Literal["lead", "rhythm", "both", "silent"]]
    split: Literal["dev", "test"]

    @model_validator(mode="after")
    def _range(self) -> GtSection:
        check_half_open(self.start_bar, self.end_bar, f"section {self.id}")
        return self


class GtSpec(_Model):
    """``gt.yaml``."""

    format: Literal["gtab.gt/1"]
    song_id: str
    title: str | None
    artist: str | None
    audio: dict  # {"song_key": "f-...", "note": "CD rip"}
    version_flags: dict  # {"version": "album|mv|live", "trimmed": bool, "note": str}
    # {"file": "ref.gp5", "format": "gp5", "expanded_bars": int, "bar_order": list[int] | None,
    #  "encoding": str | None}   (GP3-5 string encoding override; default: try cp949, then cp1252)
    reference: dict
    families_present: list[str]  # b13 ground truth: FAMILIES keys actually playing
    tracks: list[GtTrack]
    lines: list[GtLine]
    players: list[dict]
    sections: list[GtSection]
    alignment: dict  # {"method", "approved": bool, "approved_utc", "minutes_spent"}

    @field_validator("song_id")
    @classmethod
    def _song_id(cls, v: str) -> str:
        if not SONG_ID_RE.fullmatch(v):
            raise ValueError(f"song_id {v!r}: only [a-z0-9_-] (ASCII; it names a folder)")
        return v

    @model_validator(mode="after")
    def _check(self) -> GtSpec:
        check_families(self.families_present, "families_present")
        ids = [ln.id for ln in self.lines]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate line ids: {sorted(i for i in set(ids) if ids.count(i) > 1)}")
        return self


class GtNote(_Model):
    line: str
    ref_track: int
    bar: int  # expanded musical bar
    tick: float = Field(ge=0.0)  # from the bar's first beat, tpb per beat unit
    dur_ticks: float = Field(ge=0.0)
    pitch: Midi
    string: int | None = Field(ge=1)  # 1 = highest-pitched string
    fret: int | None = Field(ge=0)
    tied: bool = False
    shared: bool = False  # the same (pitch, bar, tick) also occurs in another line (unison, DESIGN R3)
    onset_s: Seconds | None = None  # filled after `gtab gt align`
    offset_s: Seconds | None = None


class GtNotes(_Model):
    """``gt_notes.json`` (expanded)."""

    format: Literal["gtab.gtnotes/1"]
    song_id: str
    tpb: int = Field(ge=1)
    bars: list[dict]  # [{"bar","numerator","denominator","beat_unit","nominal_bpm"}]
    notes: list[GtNote]
