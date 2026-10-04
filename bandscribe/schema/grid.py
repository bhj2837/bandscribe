"""Beats and the bar/beat grid (M1_M2_SPEC 6.2), plus :class:`TempoMap`.

``beats.json`` (:class:`BeatsRaw`) is what the Beat This! worker reports; ``grid.json`` (:class:`Grid`) is the
refined grid of the ``grid`` stage. :class:`TempoMap` is the one time <-> (bar, tick) mapping used by the grid
*and* by ground truth (``tempo_map.json`` of a Tier A song), so metrics, MIDI export and GT tools share it.

Bar numbers are musical (1 = first full bar, 0 = pickup); ranges are half-open; ticks are ``tpb`` (48) per beat
of the bar's beat unit; ``beat`` is the 1-based position in its bar.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from bandscribe.formats import normalize
from bandscribe.schema._common import Prob, Seconds, _Model, check_half_open

BeatUnit = Literal["quarter", "dotted_quarter", "eighth"]


class BeatsRaw(_Model):
    """``beats.json``: the beat tracker's raw output."""

    format: Literal["bandscribe.beats/1"]
    tracker: str  # "beat_this/final0"
    checkpoint_sha256: str | None
    dbn: bool
    fps: float = Field(gt=0.0)  # 50
    beats_s: list[float]
    downbeats_s: list[float]
    duration_s: Seconds


class GridBeat(_Model):
    t_s: Seconds
    t_raw_s: Seconds  # tracker time before refinement
    bar: int = Field(ge=0)
    beat: int = Field(ge=1)  # 1-based position in the bar
    is_downbeat: bool
    shift_ms: float  # t_s - t_raw_s
    activation: Prob | None


class GridBar(_Model):
    bar: int = Field(ge=0)
    start_s: Seconds
    end_s: Seconds
    n_beats: int = Field(ge=1)
    numerator: int = Field(ge=1)
    denominator: int = Field(ge=1)
    beat_unit: BeatUnit
    pickup: bool = False


class TempoSegment(_Model):
    start_bar: int
    end_bar: int  # exclusive
    bpm: float = Field(gt=0.0)  # bpm of the beat unit

    @model_validator(mode="after")
    def _range(self) -> TempoSegment:
        check_half_open(self.start_bar, self.end_bar, "TempoSegment")
        return self


class Grid(_Model):
    """``grid.json``."""

    format: Literal["bandscribe.grid/1"]
    tpb: int = Field(48, ge=1)
    duration_s: Seconds
    beats: list[GridBeat]
    bars: list[GridBar]
    tempo: list[TempoSegment]
    meter: list[dict]  # [{"start_bar","end_bar","numerator","denominator","beat_unit"}] (half-open)
    beat_unit: BeatUnit  # dominant
    pickup: dict  # {"present": bool, "beats": int}
    hypotheses: dict  # {"tempo_octave": {...}, "compound": {...}} (M1_M2_SPEC 6.2)
    confidence: dict  # {"beats": p, "downbeats": p, "meter": p}
    params: dict

    @model_validator(mode="after")
    def _monotonic(self) -> Grid:
        # A beat grid is strictly increasing in time and non-decreasing in bar number; TempoMap and the
        # MIDI tempo map (one set_tempo per beat interval) divide by beat intervals.
        for a, b in zip(self.beats, self.beats[1:]):
            if not b.t_s > a.t_s:
                raise ValueError(f"beat times must be strictly increasing ({a.t_s} then {b.t_s})")
            if b.bar < a.bar:
                raise ValueError(f"bar numbers must not decrease ({a.bar} then {b.bar})")
        for a, b in zip(self.bars, self.bars[1:]):
            if not b.bar > a.bar:
                raise ValueError(f"bars must be strictly increasing ({a.bar} then {b.bar})")
        return self


# --------------------------------------------------------------------------------------------- TempoMap


class TempoMap:
    """Piecewise-linear time <-> beat mapping with musical bars and ticks.

    ``beat_times[i]`` is global beat ``i`` (a fractional global beat index interpolates linearly between
    beats; outside the listed beats the first / last interval is extrapolated). ``bars[i]`` / ``beats[i]`` give
    the bar of beat ``i`` and its 1-based position in that bar; ``beats_per_bar`` gives each bar's length in
    beats (used to extrapolate bars before the first / after the last listed beat). A bar whose first listed
    beat is not beat 1 (e.g. a grid that starts mid-bar) starts ``beats[i] - 1`` beats before it.

    Ticks are ``tpb`` per beat of the bar's beat unit, measured from the bar's first beat; a tick past the end
    of its bar simply continues into the following beats (a note's end tick may cross the bar line).
    Every method accepts scalars or array-likes and returns numpy arrays of the broadcast shape.
    """

    def __init__(
        self,
        beat_times: Sequence[float],
        bars: Sequence[int],
        beats: Sequence[int],
        beats_per_bar: Mapping[int, int],
        tpb: int = 48,
    ) -> None:
        t = np.asarray(beat_times, dtype=np.float64).reshape(-1)
        b = [int(x) for x in bars]
        k = [int(x) for x in beats]
        if len(t) < 2:
            raise ValueError("TempoMap needs at least two beats")
        if not (len(t) == len(b) == len(k)):
            raise ValueError(f"beat_times ({len(t)}), bars ({len(b)}) and beats ({len(k)}) differ in length")
        if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
            raise ValueError("beat times must be finite and strictly increasing")
        if any(y < x for x, y in zip(b, b[1:])):
            raise ValueError("bar numbers must not decrease along the beats")
        if any(x < 1 for x in k):
            raise ValueError("beat positions are 1-based")
        if int(tpb) < 1:
            raise ValueError("tpb must be >= 1")
        self.beat_times = t
        self.bars = np.asarray(b, dtype=np.int64)
        self.beats = np.asarray(k, dtype=np.int64)
        self.beats_per_bar: dict[int, int] = {int(kk): int(v) for kk, v in beats_per_bar.items()}
        self.tpb = int(tpb)
        for bar_no, n in self.beats_per_bar.items():
            if n < 1:
                raise ValueError(f"beats_per_bar[{bar_no}] must be >= 1")
        # Global beat index (float) of each listed bar's first beat.
        starts: dict[int, float] = {}
        for i, (bar_no, pos) in enumerate(zip(b, k)):
            if bar_no not in starts:
                starts[bar_no] = float(i - (pos - 1))
        self._bar_list = sorted(starts)
        self._bar_start = [starts[x] for x in self._bar_list]
        for x, y in zip(self._bar_start, self._bar_start[1:]):
            if not y > x:
                raise ValueError("bars overlap: a bar starts before the previous one")

    # ---------------------------------------------------------------------------------- constructors

    @classmethod
    def from_grid(cls, grid: Grid) -> TempoMap:
        return cls(
            [gb.t_s for gb in grid.beats],
            [gb.bar for gb in grid.beats],
            [gb.beat for gb in grid.beats],
            {gb.bar: gb.n_beats for gb in grid.bars},
            grid.tpb,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TempoMap:
        """From a ``tempo_map.json`` document (``bandscribe.tempomap/1``) or a ``grid.json`` document."""
        fmt = normalize(d.get("format"))  # legacy "gtab.*" ids too (bandscribe.formats)
        if fmt == "bandscribe.grid/1":
            return cls.from_grid(Grid.model_validate(d))
        if fmt != "bandscribe.tempomap/1":
            raise ValueError(f"not a tempo map or grid document: format {fmt!r}")
        beats = d["beats"]
        return cls(
            [x["t_s"] for x in beats],
            [x["bar"] for x in beats],
            [x["beat"] for x in beats],
            {int(kk): int(v) for kk, v in d["beats_per_bar"].items()},
            int(d.get("tpb", 48)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> TempoMap:
        """``tempo_map.json`` (M1_M2_SPEC 6.7) or ``grid.json``."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ----------------------------------------------------------------------------------- bar lengths

    def _bpb(self, bar: int) -> int:
        """Beats in ``bar``: from beats_per_bar, else the nearest known bar's (previous first), else 4."""
        n = self.beats_per_bar.get(bar)
        if n is not None:
            return n
        if self.beats_per_bar:
            keys = sorted(self.beats_per_bar)
            i = bisect.bisect_left(keys, bar)
            near = keys[i - 1] if i > 0 else keys[0]
            return self.beats_per_bar[near]
        return 4

    def _bar_start_beat(self, bar: int) -> float:
        """Global beat index of ``bar``'s first beat (extrapolated with bar lengths outside the listed bars)."""
        i = bisect.bisect_left(self._bar_list, bar)
        if i < len(self._bar_list) and self._bar_list[i] == bar:
            return self._bar_start[i]
        if i == 0:  # before the first listed bar: walk backwards
            first = self._bar_list[0]
            s = self._bar_start[0]
            for x in range(bar, first):
                s -= self._bpb(x)
            return s
        prev = self._bar_list[i - 1]  # after (or between) listed bars: walk forward from the previous one
        s = self._bar_start[i - 1]
        for x in range(prev, bar):
            s += self._bpb(x)
        return s

    def _bar_of_beat(self, g: float) -> tuple[int, float]:
        """(bar, start beat of that bar) containing global beat ``g``."""
        i = bisect.bisect_right(self._bar_start, g) - 1
        if i < 0:  # before the first listed bar
            bar = self._bar_list[0]
            s = self._bar_start[0]
            while s > g:
                bar -= 1
                s -= self._bpb(bar)
            return bar, s
        bar, s = self._bar_list[i], self._bar_start[i]
        if i == len(self._bar_list) - 1:  # at or after the last listed bar: walk forward
            while g >= s + self._bpb(bar):
                s += self._bpb(bar)
                bar += 1
        return bar, s

    # ------------------------------------------------------------------------------------- mappings

    def time_to_beat(self, t: Any) -> np.ndarray:
        """Fractional global beat index; linear between beats, first/last interval extrapolated outside."""
        ta = np.asarray(t, dtype=np.float64)
        bt = self.beat_times
        i = np.clip(np.searchsorted(bt, ta, side="right") - 1, 0, len(bt) - 2)
        return i + (ta - bt[i]) / (bt[i + 1] - bt[i])

    def beat_to_time(self, b: Any) -> np.ndarray:
        ba = np.asarray(b, dtype=np.float64)
        bt = self.beat_times
        i = np.clip(np.floor(ba), 0, len(bt) - 2).astype(np.int64)
        return bt[i] + (ba - i) * (bt[i + 1] - bt[i])

    def time_to_bar_tick(self, t: Any) -> tuple[np.ndarray, np.ndarray]:
        """(bar int, tick float within the bar) for each time."""
        g = self.time_to_beat(t)
        flat = np.atleast_1d(g).reshape(-1)
        bars = np.empty(flat.shape, dtype=np.int64)
        ticks = np.empty(flat.shape, dtype=np.float64)
        for j, gv in enumerate(flat):
            bar, s = self._bar_of_beat(float(gv))
            bars[j] = bar
            ticks[j] = (float(gv) - s) * self.tpb
        shape = np.shape(g)
        return bars.reshape(shape), ticks.reshape(shape)

    def bar_tick_to_time(self, bar: Any, tick: Any) -> np.ndarray:
        bar_a, tick_a = np.broadcast_arrays(np.asarray(bar), np.asarray(tick, dtype=np.float64))
        starts = {int(x): self._bar_start_beat(int(x)) for x in np.unique(bar_a)} if bar_a.size else {}
        g = np.array([starts[int(x)] for x in bar_a.reshape(-1)], dtype=np.float64).reshape(bar_a.shape)
        return self.beat_to_time(g + tick_a / self.tpb)

    def ticks_per_second(self, t: Any) -> np.ndarray:
        """Local ticks per second at ``t`` (tpb / duration of the beat interval containing ``t``)."""
        ta = np.asarray(t, dtype=np.float64)
        bt = self.beat_times
        i = np.clip(np.searchsorted(bt, ta, side="right") - 1, 0, len(bt) - 2)
        return self.tpb / (bt[i + 1] - bt[i])

    # -------------------------------------------------------------------------------------- export

    def to_dict(self, source: str = "synthetic") -> dict[str, Any]:
        """A ``tempo_map.json`` document (M1_M2_SPEC 6.7) of this map."""
        return {
            "format": "bandscribe.tempomap/1",
            "source": source,
            "tpb": self.tpb,
            "beats": [
                {"t_s": round(float(t), 6), "bar": int(b), "beat": int(k)}
                for t, b, k in zip(self.beat_times, self.bars, self.beats)
            ],
            "beats_per_bar": {str(kk): int(v) for kk, v in sorted(self.beats_per_bar.items())},
        }

    def __repr__(self) -> str:
        span = f"{self.beat_times[0]:.3f}-{self.beat_times[-1]:.3f} s"
        return f"TempoMap({len(self.beat_times)} beats, bars {self._bar_list[0]}-{self._bar_list[-1]}, {span})"
