"""Synthetic guitar scenes with ground truth (DESIGN §8.2 "합성 리믹스"), Tier B remixes and a mastering limiter.

Generators ``tone``, ``amp_cab``, ``rhythm_take``, ``lead_line``, ``twin_lead``, ``pan`` and ``stereo_reverb``
are ported from ``docs/research/experiments/synth_exp.py`` / ``router_exp.py`` **with identical RNG call
order**, so ``parity_scenes()`` reproduces the 14 rows of ``router_exp.main()`` (checked against the original
script by ``tests/test_eval_remix_parity.py``). The ports additionally return the notes they play (MIDI pitch,
onset incl. jitter / strum offsets, duration) as ground truth.

Scenario line maps follow the output contract R1–R9 (DESIGN §4.2): doubles / ADT / quad = one line; P = two
lines; twin = Lead 1 / Lead 2; twin_double = rhythm + Lead 1 + Lead 2; same_part_double = one line with two
takes (a doubled *lead* melody; the doubled rhythm case is ``A_rhythm_only``); centre_unison = rhythm + lead
where the lead doubles the riff's root notes for the first half (those notes are ``shared`` in both lines).

GT of a line lists each played note once: for a doubled line the first take is the reference performance;
the other takes' notes stay available in ``Take.notes`` (self-agreement data, R1 calibration).

Stereo arrays are channel-first ``(2, n)`` float64, like the research code; ``write_scene`` writes WAVs via
``bandscribe.audio.write_f32`` (byte-deterministic).

``master`` is a deterministic look-ahead brickwall limiter (float64, no threads, no randomness) that turns a
stem sum into a commercial-loudness master reproducibly (E1 inputs).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from bandscribe import atomic

log = logging.getLogger(__name__)

SR = 44100
DUR_S = 8.0
BPM = 120.0
BEAT_S = 60.0 / BPM
BAR_S = 4 * BEAT_S
TPB = 48

SCENARIOS: tuple[str, ...] = ("A", "A_rhythm_only", "B", "F", "ADT", "quad", "P70", "P_hard", "twin",
                              "twin_double", "same_part_double", "centre_unison")

# router_exp.main() seeds: rhythm takes 11..14, lead 21, twin lead 22, reverb rng 1 (shared, in row order).
_RHYTHM_SEEDS = (11, 12, 13, 14)
_LEAD_SEED = 21
_TWIN_SEED = 22
_REVERB_SEED = 1
_SEED_STRIDE = 1000

# (onset s, frequency Hz, duration s) of the lead phrase (synth_exp.lead_line / router_exp.twin_lead).
LEAD_PHRASE: tuple[tuple[float, float, float], ...] = (
    (0.5, 329.63, 0.5), (1.0, 392.0, 0.25), (1.25, 440.0, 0.75), (2.0, 493.88, 1.0),
    (3.0, 587.33, 0.5), (3.5, 659.26, 1.0), (4.5, 587.33, 0.25), (4.75, 493.88, 0.25),
    (5.0, 440.0, 0.5), (5.5, 392.0, 0.5), (6.0, 329.63, 1.5))
RHYTHM_ROOTS: tuple[float, ...] = (82.41, 98.00, 110.00, 130.81)  # E2 G2 A2 C3, one per 2-s bar


def hz_to_midi(f: float) -> int:
    return int(round(69 + 12 * math.log2(f / 440.0)))


@dataclass(frozen=True)
class SynthNote:
    """A played note: ``nominal_s`` is the grid position before jitter (used to tag unison notes)."""

    onset_s: float
    offset_s: float
    pitch: int
    nominal_s: float


# ---------------------------------------------------------------------------------------- generators


def tone(f: float, dur: float, decay: float, rng: np.random.Generator, vib_depth_cents: float = 0.0,
         vib_rate: float = 5.5, nharm: int = 40, *, render: bool = True) -> np.ndarray | None:
    """Additive guitar-ish tone (synth_exp.tone). ``render=False`` consumes the RNG identically, no DSP."""
    n = int(dur * SR)
    if not render:
        for k in range(1, nharm + 1):
            if k * f > 0.45 * SR:
                break
            rng.uniform(0, 2 * np.pi)
        rng.normal(0, 0.3, int(0.004 * SR))
        return None
    t = np.arange(n) / SR
    if vib_depth_cents > 0:
        ramp = np.clip((t - 0.15) / 0.2, 0, 1)  # vibrato ramps in after 150 ms
        inst_f = f * 2 ** (vib_depth_cents / 1200 * ramp * np.sin(2 * np.pi * vib_rate * t))
    else:
        inst_f = np.full(n, f)
    phase = 2 * np.pi * np.cumsum(inst_f) / SR
    y = np.zeros(n)
    for k in range(1, nharm + 1):
        if k * f > 0.45 * SR:
            break
        amp = (1.0 / k) * np.exp(-t * decay * (1 + 0.15 * k))
        y += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    y[: int(0.004 * SR)] += rng.normal(0, 0.3, int(0.004 * SR)) * np.linspace(1, 0, int(0.004 * SR))
    return y


def amp_cab(x: np.ndarray, gain: float, lpf_hz: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """tanh drive + 4th-order cab low-pass + 90 Hz high-pass (synth_exp.amp_cab; rng unused, kept for parity)."""
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
    y = np.tanh(gain * x)
    b, a = signal.butter(4, lpf_hz / (SR / 2), "low")
    y = signal.lfilter(b, a, y)
    b, a = signal.butter(2, 90 / (SR / 2), "high")
    return signal.lfilter(b, a, y)


def place(buf: np.ndarray, x: np.ndarray, start: float) -> None:
    s = int(start * SR)
    if s < 0:
        x = x[-s:]
        s = 0
    e = min(len(buf), s + len(x))
    buf[s:e] += x[: e - s]


def _finish(buf: np.ndarray, n: int, gain: float, lpf: float, rng: np.random.Generator,
            render: bool) -> np.ndarray:
    if not render:
        return np.zeros(n)
    y = amp_cab(buf[:n], gain, lpf, rng)
    return y / np.sqrt(np.mean(y ** 2)) * 0.1


def _clip_notes(notes: list[SynthNote], dur_s: float) -> list[SynthNote]:
    out = []
    for nt in notes:
        on = max(0.0, nt.onset_s)
        off = min(dur_s, nt.offset_s)
        if on < dur_s and off > on:
            out.append(SynthNote(on, off, nt.pitch, nt.nominal_s))
    return out


def rhythm_take(rng: np.random.Generator, jitter_ms: float = 8.0, detune_cents: float = 4.0, gain: float = 6.0,
                lpf: float = 5000.0, *, dur_s: float = DUR_S, render: bool = True,
                roots_only_bars: int = 0) -> tuple[np.ndarray, list[SynthNote]]:
    """Power-chord eighths E5-G5-A5-C5, 2 s per bar at 120 BPM (synth_exp.rhythm_take).

    ``roots_only_bars`` > 0 plays only the root string for that many bars and nothing after (the unison riff of
    ``centre_unison``); 0 = the original generator.
    """
    n = int(SR * dur_s)
    n_bars = int(round(dur_s / BAR_S))
    buf = np.zeros(n + SR)
    det = 2 ** (rng.normal(0, detune_cents) / 1200)
    notes: list[SynthNote] = []
    bars = range(roots_only_bars) if roots_only_bars else range(n_bars)
    for bar in bars:
        r = RHYTHM_ROOTS[bar % len(RHYTHM_ROOTS)]
        for e in range(8):
            nominal = bar * BAR_S + e * 0.25
            t0 = nominal + rng.normal(0, jitter_ms / 1000)
            mults = (1.0,) if roots_only_bars else (1.0, 1.5, 2.0)
            for s_i, mult in enumerate(mults):
                note = tone(r * mult * det, 0.26, decay=9.0, rng=rng, render=render)
                if render:
                    place(buf, note * (0.9 if s_i else 1.0), t0 + 0.004 * s_i)
                on = t0 + 0.004 * s_i
                notes.append(SynthNote(on, on + 0.26, hz_to_midi(r * mult), nominal))
    return _finish(buf, n, gain, lpf, rng, render), _clip_notes(notes, dur_s)


def _phrase(dur_s: float, start_at: float = 0.0) -> list[tuple[float, float, float]]:
    """Lead phrase repeated every 8 s to fill ``dur_s`` (8 s = the original single phrase)."""
    out = []
    reps = max(1, int(math.ceil(dur_s / DUR_S)))
    for k in range(reps):
        for t0, f, d in LEAD_PHRASE:
            t = t0 + k * DUR_S
            if start_at <= t < dur_s:
                out.append((t, f, d))
    return out


def lead_line(rng: np.random.Generator, gain: float = 9.0, lpf: float = 5500.0, *, dur_s: float = DUR_S,
              render: bool = True, start_at: float = 0.0) -> tuple[np.ndarray, list[SynthNote]]:
    """Single-note lead with vibrato on long notes (synth_exp.lead_line)."""
    n = int(SR * dur_s)
    buf = np.zeros(n + SR)
    notes: list[SynthNote] = []
    for t0, f, d in _phrase(dur_s, start_at):
        x = tone(f, d + 0.05, decay=1.2, rng=rng, vib_depth_cents=25 if d >= 0.75 else 0, render=render)
        if render:
            place(buf, x, t0)
        notes.append(SynthNote(t0, t0 + d + 0.05, hz_to_midi(f), t0))
    return _finish(buf, n, gain, lpf, rng, render), _clip_notes(notes, dur_s)


def twin_lead(rng: np.random.Generator, semis: int = 4, gain: float = 9.0, *, dur_s: float = DUR_S,
              render: bool = True) -> tuple[np.ndarray, list[SynthNote]]:
    """Harmony lead ``semis`` above the lead phrase, 8 ms onset jitter (router_exp.twin_lead)."""
    n = int(SR * dur_s)
    buf = np.zeros(n + SR)
    notes: list[SynthNote] = []
    for t0, f, d in _phrase(dur_s):
        nominal = t0
        t0 = t0 + rng.normal(0, 0.008)
        fs = f * 2 ** (semis / 12)
        x = tone(fs, d + 0.05, decay=1.2, rng=rng, vib_depth_cents=25 if d >= 0.75 else 0, render=render)
        if render:
            place(buf, x, t0)
        notes.append(SynthNote(t0, t0 + d + 0.05, hz_to_midi(fs), nominal))
    return _finish(buf, n, gain, 5500.0, rng, render), _clip_notes(notes, dur_s)


def pan(x: np.ndarray, p: float) -> np.ndarray:
    """Constant-power pan, p in [-1 (L), 1 (R)] -> (2, n)."""
    th = (p + 1) * np.pi / 4
    return np.stack([np.cos(th) * x, np.sin(th) * x])


def hard_left(x: np.ndarray) -> np.ndarray:
    return np.stack([x, np.zeros_like(x)])


def hard_right(x: np.ndarray) -> np.ndarray:
    return np.stack([np.zeros_like(x), x])


def centre(x: np.ndarray) -> np.ndarray:
    return pan(x, 0.0)


@dataclass(frozen=True)
class Reverb:
    """Stereo reverb IRs drawn once (synth_exp.stereo_reverb order: irL then irR) so the same reverb can be
    applied to several takes separately; reverb is linear, so the per-take results sum to the reverb of the sum."""

    ir_l: np.ndarray
    ir_r: np.ndarray
    wet_db: float

    @staticmethod
    def draw(rng: np.random.Generator, rt60: float = 1.2, wet_db: float = -12.0, *, render: bool = True) -> Reverb:
        n = int(rt60 * SR)
        t = np.arange(n) / SR
        env = np.exp(-6.9 * t / rt60)
        ir_l = rng.normal(0, 1, n) * env
        ir_r = rng.normal(0, 1, n) * env
        ir_l /= np.sqrt(np.sum(ir_l ** 2))
        ir_r /= np.sqrt(np.sum(ir_r ** 2))
        return Reverb(ir_l, ir_r, wet_db)

    def apply(self, x_stereo: np.ndarray) -> np.ndarray:
        n = x_stereo.shape[1]
        mono = x_stereo.mean(axis=0)
        wet = np.stack([signal.fftconvolve(mono, self.ir_l)[:n], signal.fftconvolve(mono, self.ir_r)[:n]])
        return x_stereo + 10 ** (self.wet_db / 20) * wet


def stereo_reverb(x_stereo: np.ndarray, rt60: float = 1.2, wet_db: float = -12.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    return Reverb.draw(rng if rng is not None else np.random.default_rng(), rt60, wet_db).apply(x_stereo)


# ------------------------------------------------------------------------------------------- full band


def _band_parts(rng: np.random.Generator, dur_s: float) -> dict[str, tuple[np.ndarray, list[SynthNote]]]:
    """Synthetic bass / kick / hat / vocal-like / pad so a scene can go through a 6-stem separator.

    Deliberately simple and deterministic (own RNG, so the guitar takes are unchanged by ``full_band``)."""
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    parts: dict[str, tuple[np.ndarray, list[SynthNote]]] = {}
    # bass: root eighths one octave below the riff, filtered saw
    bass = np.zeros(n + SR)
    bnotes: list[SynthNote] = []
    for bar in range(int(round(dur_s / BAR_S))):
        f = RHYTHM_ROOTS[bar % len(RHYTHM_ROOTS)] / 2
        for e in range(8):
            t0 = bar * BAR_S + e * 0.25 + rng.normal(0, 0.004)
            m = int(0.24 * SR)
            tt = np.arange(m) / SR
            saw = 2 * ((f * tt) % 1.0) - 1
            place(bass, saw * np.exp(-tt * 6.0), t0)
            bnotes.append(SynthNote(t0, t0 + 0.24, hz_to_midi(f), bar * BAR_S + e * 0.25))
    b, a = signal.butter(4, 900 / (SR / 2), "low")
    y = signal.lfilter(b, a, bass[:n])
    parts["bass"] = (y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * 0.12, _clip_notes(bnotes, dur_s))
    # kick on every beat: pitch-swept sine; hat on eighths: high-passed noise bursts
    kick = np.zeros(n + SR)
    hat = np.zeros(n + SR)
    m = int(0.15 * SR)
    tt = np.arange(m) / SR
    k_wave = np.sin(2 * np.pi * (50 * tt + 60 * (1 - np.exp(-tt * 30)) / 30)) * np.exp(-tt * 25)
    hb, ha = signal.butter(4, 7000 / (SR / 2), "high")
    for i in range(int(dur_s / BEAT_S)):
        place(kick, k_wave, i * BEAT_S)
    for i in range(int(dur_s / (BEAT_S / 2))):
        burst = signal.lfilter(hb, ha, rng.normal(0, 1, int(0.05 * SR))) * np.exp(-np.arange(int(0.05 * SR)) / SR * 60)
        place(hat, burst * 0.3, i * BEAT_S / 2)
    parts["kick"] = (kick[:n] * 0.3, [])
    parts["hat"] = (hat[:n] * 0.2, [])
    # vocal-like: vowel formants on a vibrato pulse train following the lead phrase an octave lower
    voc = np.zeros(n + SR)
    for t0, f, d in _phrase(dur_s):
        m = int((d + 0.05) * SR)
        tt = np.arange(m) / SR
        ph = 2 * np.pi * np.cumsum(f / 2 * 2 ** (30 / 1200 * np.sin(2 * np.pi * 5.0 * tt))) / SR
        src = sum(np.sin(k * ph) / k for k in range(1, 25))
        env = np.minimum(1, tt / 0.03) * np.minimum(1, (tt[-1] - tt) / 0.05 + 1e-9)
        place(voc, src * env, t0 + 0.25)
    for fc, bw in ((700, 110), (1220, 120), (2600, 160)):
        bb, aa = signal.iirpeak(fc / (SR / 2), fc / bw)
        voc[:n] = voc[:n] + 0.5 * signal.lfilter(bb, aa, voc[:n])
    parts["vocals"] = (voc[:n] / (np.sqrt(np.mean(voc[:n] ** 2)) + 1e-12) * 0.08, [])
    # pad: sustained root-fifth-octave sines, slow tremolo (becomes stereo-wide below)
    pad = np.zeros(n)
    for bar in range(int(round(dur_s / BAR_S))):
        f = RHYTHM_ROOTS[bar % len(RHYTHM_ROOTS)] * 2
        s, e = int(bar * BAR_S * SR), min(n, int((bar + 1) * BAR_S * SR))
        tt = t[s:e]
        pad[s:e] = sum(np.sin(2 * np.pi * f * m_ * tt) for m_ in (1.0, 1.5, 2.0)) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * tt))
    parts["pad"] = (pad / (np.sqrt(np.mean(pad ** 2)) + 1e-12) * 0.05, [])
    return parts


# ------------------------------------------------------------------------------------------------ scenes


@dataclass
class Take:
    id: str
    line: str | None                  # GT line id; None = not a transcribed part (drums, vocals, pad)
    stereo: np.ndarray                # (2, n) contribution of this take to the mix
    notes: list[Any]                  # list[bandscribe.schema.notes.NoteEvent]
    kind: str = "guitar"              # guitar | bass | drums | vocals | keys


@dataclass
class Scene:
    scenario: str
    seed: int
    sr: int
    mix: np.ndarray                   # (2, n)
    takes: list[Take]
    lines: list[Any]                  # list[bandscribe.schema.gt.GtLine]
    gt: Any                           # bandscribe.schema.gt.GtNotes
    meta: dict = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        return scene_item_id(self.scenario, self.seed, self.meta)

    def guitar_stereo(self) -> np.ndarray:
        """True guitar-only stem (sum of guitar takes), for separation checks."""
        parts = [t.stereo for t in self.takes if t.kind == "guitar"]
        return np.sum(parts, axis=0) if parts else np.zeros_like(self.mix)

    def stem(self, kind: str) -> np.ndarray:
        parts = [t.stereo for t in self.takes if t.kind == kind]
        return np.sum(parts, axis=0) if parts else np.zeros_like(self.mix)


def scene_item_id(scenario: str, seed: int, meta: dict | None = None) -> str:
    meta = meta or {}
    tags = []
    if meta.get("lead_gain_db"):
        tags.append(f"lead{meta['lead_gain_db']:+g}db")
    if meta.get("reverb"):
        tags.append("rev")
    if meta.get("full_band"):
        tags.append("band")
    return "synth:" + "_".join([scenario, f"s{seed}", *tags])


def _note_events(notes: Sequence[SynthNote], instrument: str | None = None) -> list[Any]:
    from bandscribe.schema.notes import NoteEvent

    evs = [NoteEvent(onset_s=round(n.onset_s, 6), offset_s=round(n.offset_s, 6), pitch=n.pitch,
                     instrument=instrument) for n in notes]
    return sorted(evs, key=lambda e: (e.onset_s, e.pitch, e.offset_s))


def _bars_meta(dur_s: float) -> list[dict]:
    return [{"bar": b, "numerator": 4, "denominator": 4, "beat_unit": "quarter", "nominal_bpm": BPM}
            for b in range(1, int(math.ceil(dur_s / BAR_S)) + 1)]


def tempo_map_dict(dur_s: float) -> dict:
    """tempo_map.json of a synthetic scene: constant 120 BPM 4/4, bar 1 starts at 0 s (spec §6.7)."""
    n_beats = int(round(dur_s / BEAT_S)) + 1
    beats = [{"t_s": round(i * BEAT_S, 6), "bar": 1 + i // 4, "beat": 1 + i % 4} for i in range(n_beats)]
    bpb = {str(b): 4 for b in range(1, 2 + (n_beats - 1) // 4)}
    return {"format": "bandscribe.tempomap/1", "source": "synthetic", "tpb": TPB, "beats": beats, "beats_per_bar": bpb}


def scene_tempo_map(scene_or_dur: Scene | float):
    from bandscribe.schema.grid import TempoMap

    dur = scene_or_dur.meta["dur_s"] if isinstance(scene_or_dur, Scene) else float(scene_or_dur)
    d = tempo_map_dict(dur)
    return TempoMap([b["t_s"] for b in d["beats"]], [b["bar"] for b in d["beats"]], [b["beat"] for b in d["beats"]],
                    {int(k): v for k, v in d["beats_per_bar"].items()}, TPB)


def _gt_notes(song_id: str, dur_s: float, line_notes: list[tuple[str, int, list[SynthNote]]]) -> Any:
    """GtNotes at 120 BPM 4/4 (bar 1 at 0 s); unison notes (same pitch, same nominal grid time in two lines)
    are flagged ``shared`` in every line that has them (spec §6.7 ``shared``)."""
    from bandscribe.schema.gt import GtNote, GtNotes

    keyset: dict[tuple[int, float], set[str]] = {}
    for line, _ref, notes in line_notes:
        for n in notes:
            keyset.setdefault((n.pitch, round(n.nominal_s, 6)), set()).add(line)
    out = []
    for line, ref, notes in line_notes:
        for n in notes:
            bar = 1 + int(math.floor(n.onset_s / BAR_S + 1e-9))
            tick = (n.onset_s - (bar - 1) * BAR_S) / BEAT_S * TPB
            shared = len(keyset[(n.pitch, round(n.nominal_s, 6))]) > 1
            out.append(GtNote(line=line, ref_track=ref, bar=bar, tick=round(tick, 4),
                              dur_ticks=round((n.offset_s - n.onset_s) / BEAT_S * TPB, 4), pitch=n.pitch,
                              string=None, fret=None, tied=False, shared=shared,
                              onset_s=round(n.onset_s, 6), offset_s=round(n.offset_s, 6)))
    out.sort(key=lambda g: (g.onset_s, g.pitch, g.line))
    return GtNotes(format="bandscribe.gtnotes/1", song_id=song_id, tpb=TPB, bars=_bars_meta(dur_s), notes=out)


def _line(id_: str, name: str, takes: list[int], kind: str = "guitar", **relation: Any) -> Any:
    from bandscribe.schema.gt import GtLine

    rel = {"double": False, "unison_with": [], "twin_with": None, "overdub": False}
    rel.update(relation)
    return GtLine(id=id_, name=name, kind=kind, takes=takes, relation=rel)


class _Sources:
    """Lazily generated takes with router_exp seeds (offset by ``seed * 1000``)."""

    def __init__(self, seed: int, dur_s: float, render: bool) -> None:
        self.seed, self.dur_s, self.render = seed, dur_s, render
        self._cache: dict[str, tuple[np.ndarray, list[SynthNote]]] = {}

    def _rng(self, base: int) -> np.random.Generator:
        return np.random.default_rng(base + _SEED_STRIDE * self.seed)

    def rhythm(self, k: int) -> tuple[np.ndarray, list[SynthNote]]:
        key = f"r{k}"
        if key not in self._cache:
            self._cache[key] = rhythm_take(self._rng(_RHYTHM_SEEDS[k - 1]), dur_s=self.dur_s, render=self.render)
        return self._cache[key]

    def lead(self) -> tuple[np.ndarray, list[SynthNote]]:
        if "lead" not in self._cache:
            self._cache["lead"] = lead_line(self._rng(_LEAD_SEED), dur_s=self.dur_s, render=self.render)
        return self._cache["lead"]

    def lead2(self) -> tuple[np.ndarray, list[SynthNote]]:
        if "lead2" not in self._cache:
            self._cache["lead2"] = twin_lead(self._rng(_TWIN_SEED), dur_s=self.dur_s, render=self.render)
        return self._cache["lead2"]

    def lead_double(self, k: int) -> tuple[np.ndarray, list[SynthNote]]:
        key = f"ld{k}"
        if key not in self._cache:
            self._cache[key] = twin_lead(self._rng(22 + k), semis=0, dur_s=self.dur_s, render=self.render)
        return self._cache[key]

    def unison_lead(self) -> tuple[np.ndarray, list[SynthNote]]:
        """Lead guitar doubling the riff roots for the first half, then the lead phrase (lead amp settings)."""
        if "ulead" not in self._cache:
            half_bars = max(1, int(round(self.dur_s / BAR_S)) // 2)
            y1, n1 = rhythm_take(self._rng(31), gain=9.0, lpf=5500.0, dur_s=self.dur_s, render=self.render,
                                 roots_only_bars=half_bars)
            y2, n2 = lead_line(self._rng(32), dur_s=self.dur_s, render=self.render, start_at=half_bars * BAR_S)
            self._cache["ulead"] = (y1 + y2, n1 + n2)
        return self._cache["ulead"]


def _mix_rows(scenario: str, src: _Sources, lead_gain: float) -> tuple[list[tuple[str, str, np.ndarray, list[SynthNote]]], list[Any], list[tuple[str, int, list[SynthNote]]]]:
    """(takes as (id, line, stereo, notes), GtLines, GT (line, ref_track, notes)) of one scenario."""
    r = src.rhythm
    lead = lambda: (src.lead()[0] * lead_gain, src.lead()[1])  # noqa: E731
    d15 = int(0.015 * SR)
    if scenario == "A":
        takes = [("r1", "rhythm", hard_left(r(1)[0]), r(1)[1]), ("r2", "rhythm", hard_right(r(2)[0]), r(2)[1]),
                 ("lead", "lead", centre(lead()[0]), lead()[1])]
        lines = [_line("rhythm", "Rhythm (L/R double)", [1, 2], double=True), _line("lead", "Lead", [3])]
        gt = [("rhythm", 1, r(1)[1]), ("lead", 3, lead()[1])]
    elif scenario == "A_rhythm_only":
        takes = [("r1", "rhythm", hard_left(r(1)[0]), r(1)[1]), ("r2", "rhythm", hard_right(r(2)[0]), r(2)[1])]
        lines = [_line("rhythm", "Rhythm (L/R double)", [1, 2], double=True)]
        gt = [("rhythm", 1, r(1)[1])]
    elif scenario == "B":
        takes = [("r1", "rhythm", centre(r(1)[0]), r(1)[1]), ("lead", "lead", centre(lead()[0]), lead()[1])]
        lines = [_line("rhythm", "Rhythm", [1]), _line("lead", "Lead", [2])]
        gt = [("rhythm", 1, r(1)[1]), ("lead", 2, lead()[1])]
    elif scenario == "F":
        takes = [("r1", "rhythm", centre(r(1)[0] / np.sqrt(2)), r(1)[1]),
                 ("r2", "rhythm", centre(r(2)[0] / np.sqrt(2)), r(2)[1]),
                 ("lead", "lead", centre(lead()[0]), lead()[1])]
        lines = [_line("rhythm", "Rhythm (mono double)", [1, 2], double=True), _line("lead", "Lead", [3])]
        gt = [("rhythm", 1, r(1)[1]), ("lead", 3, lead()[1])]
    elif scenario == "ADT":
        y1, n1 = r(1)
        adt = np.concatenate([np.zeros(d15), y1[:-d15]])
        adt_notes = [SynthNote(n.onset_s + 0.015, n.offset_s + 0.015, n.pitch, n.nominal_s) for n in n1]
        takes = [("r1", "rhythm", hard_left(y1), n1), ("adt", "rhythm", hard_right(adt), _clip_notes(adt_notes, src.dur_s)),
                 ("lead", "lead", centre(lead()[0]), lead()[1])]
        lines = [_line("rhythm", "Rhythm (ADT)", [1, 2], adt=True), _line("lead", "Lead", [3])]
        gt = [("rhythm", 1, n1), ("lead", 3, lead()[1])]
    elif scenario == "quad":
        takes = [("r1", "rhythm", hard_left(r(1)[0]), r(1)[1]), ("r2", "rhythm", hard_right(r(2)[0]), r(2)[1]),
                 ("r3", "rhythm", pan(r(3)[0], -0.6), r(3)[1]), ("r4", "rhythm", pan(r(4)[0], 0.6), r(4)[1]),
                 ("lead", "lead", centre(lead()[0]), lead()[1])]
        lines = [_line("rhythm", "Rhythm (quad)", [1, 2, 3, 4], double=True), _line("lead", "Lead", [5])]
        gt = [("rhythm", 1, r(1)[1]), ("lead", 5, lead()[1])]
    elif scenario == "P70":
        takes = [("r1", "rhythm", pan(r(1)[0], -0.7), r(1)[1]), ("lead", "lead", pan(lead()[0], 0.7), lead()[1])]
        lines = [_line("rhythm", "Rhythm (70% L)", [1]), _line("lead", "Lead (70% R)", [2])]
        gt = [("rhythm", 1, r(1)[1]), ("lead", 2, lead()[1])]
    elif scenario == "P_hard":
        takes = [("r1", "rhythm", hard_left(r(1)[0]), r(1)[1]), ("lead", "lead", hard_right(lead()[0]), lead()[1])]
        lines = [_line("rhythm", "Rhythm (hard L)", [1]), _line("lead", "Lead (hard R)", [2])]
        gt = [("rhythm", 1, r(1)[1]), ("lead", 2, lead()[1])]
    elif scenario == "twin":
        l2 = src.lead2()
        takes = [("lead1", "lead1", hard_left(lead()[0]), lead()[1]), ("lead2", "lead2", hard_right(l2[0] * lead_gain), l2[1])]
        lines = [_line("lead1", "Lead 1", [1], twin_with="lead2"), _line("lead2", "Lead 2", [2], twin_with="lead1")]
        gt = [("lead1", 1, lead()[1]), ("lead2", 2, l2[1])]
    elif scenario == "twin_double":
        l2 = src.lead2()
        takes = [("r1", "rhythm", hard_left(r(1)[0]), r(1)[1]), ("lead1", "lead1", hard_left(lead()[0]), lead()[1]),
                 ("r2", "rhythm", hard_right(r(2)[0]), r(2)[1]), ("lead2", "lead2", hard_right(l2[0] * lead_gain), l2[1])]
        lines = [_line("rhythm", "Rhythm (L/R double)", [1, 3], double=True),
                 _line("lead1", "Lead 1", [2], twin_with="lead2"), _line("lead2", "Lead 2", [4], twin_with="lead1")]
        gt = [("rhythm", 1, r(1)[1]), ("lead1", 2, lead()[1]), ("lead2", 4, l2[1])]
    elif scenario == "same_part_double":
        d1, d2 = src.lead_double(1), src.lead_double(2)
        takes = [("lead_t1", "lead", hard_left(d1[0] * lead_gain), d1[1]), ("lead_t2", "lead", hard_right(d2[0] * lead_gain), d2[1])]
        lines = [_line("lead", "Lead (two takes)", [1, 2], double=True)]
        gt = [("lead", 1, d1[1])]
    elif scenario == "centre_unison":
        ul = src.unison_lead()
        takes = [("r1", "rhythm", hard_left(r(1)[0]), r(1)[1]), ("r2", "rhythm", hard_right(r(2)[0]), r(2)[1]),
                 ("lead", "lead", centre(ul[0] * lead_gain), ul[1])]
        lines = [_line("rhythm", "Rhythm (L/R double)", [1, 2], double=True, unison_with=["lead"]),
                 _line("lead", "Lead (unison, then solo)", [3], unison_with=["rhythm"])]
        gt = [("rhythm", 1, r(1)[1]), ("lead", 3, ul[1])]
    else:
        raise ValueError(f"unknown scenario {scenario!r}; known: {', '.join(SCENARIOS)}")
    return takes, lines, gt


def make_scene(scenario: str, seed: int = 0, *, dur_s: float = DUR_S, lead_gain_db: float = 0.0,
               reverb: bool = False, full_band: bool = False, render: bool = True) -> Scene:
    """One synthetic scene. ``seed`` 0 uses router_exp's seeds; ``render=False`` skips all DSP (GT only, same notes).

    ``reverb`` adds a decorrelated stereo reverb (RT60 1.2 s, −12 dB wet) to every guitar take (one IR pair).
    ``full_band`` adds bass/kick/hat/vocal-like/pad takes (own RNG) so the scene can go through SW.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; known: {', '.join(SCENARIOS)}")
    src = _Sources(seed, dur_s, render)
    lead_gain = 10 ** (lead_gain_db / 20)
    rows, lines, gt_rows = _mix_rows(scenario, src, lead_gain)
    n = int(SR * dur_s)
    rev = Reverb.draw(np.random.default_rng(_REVERB_SEED + _SEED_STRIDE * seed)) if reverb else None
    takes: list[Take] = []
    for tid, line, st, notes in rows:
        if rev is not None and render:
            st = rev.apply(st)
        takes.append(Take(tid, line, st if render else np.zeros((2, n)), _note_events(notes, None), "guitar"))
    if full_band:
        parts = _band_parts(np.random.default_rng(500 + _SEED_STRIDE * seed), dur_s)
        ref = len(takes) + 1
        by = {"bass": centre, "kick": centre, "hat": lambda x: pan(x, 0.3), "vocals": centre,
              "pad": lambda x: np.stack([x, np.roll(x, int(0.011 * SR))])}
        kinds = {"bass": "bass", "kick": "drums", "hat": "drums", "vocals": "vocals", "pad": "keys"}
        for name, (y, notes) in parts.items():
            line = "bass" if name == "bass" else None
            takes.append(Take(name, line, by[name](y) if render else np.zeros((2, n)),
                              _note_events(notes, None), kinds[name]))
            if name == "bass":
                lines.append(_line("bass", "Bass", [ref], kind="bass"))
                gt_rows.append(("bass", ref, notes))
            ref += 1
    mix = np.sum([t.stereo for t in takes], axis=0) if render else np.zeros((2, n))
    meta = {"generator": "bandscribe.eval.remix/1", "scenario": scenario, "seed": seed, "dur_s": dur_s,
            "lead_gain_db": lead_gain_db, "reverb": reverb, "full_band": full_band, "bpm": BPM,
            "rendered": render}
    song_id = scene_item_id(scenario, seed, meta)
    return Scene(scenario, seed, SR, mix, takes, lines, _gt_notes(song_id, dur_s, gt_rows), meta)


def parity_scenes() -> dict[str, np.ndarray]:
    """The 14 stereo scenes of ``router_exp.main()`` in row order, same seeds and RNG order ((2, n) arrays).

    Row names match the original script. The shared reverb RNG (seed 1) is consumed by the
    'lead w/ stereo reverb' row first and by 'B both centre + stereo reverb' second, as in the original.
    """
    rng = np.random.default_rng(_REVERB_SEED)
    r1, r2, r3, r4 = (rhythm_take(np.random.default_rng(s))[0] for s in _RHYTHM_SEEDS)
    lead = lead_line(np.random.default_rng(_LEAD_SEED))[0]
    lead2 = twin_lead(np.random.default_rng(_TWIN_SEED))[0]
    n = int(SR * DUR_S)
    d15 = int(0.015 * SR)
    adt = np.concatenate([np.zeros(d15), r1[:-d15]])
    assert len(adt) == n
    sc: dict[str, np.ndarray] = {}
    sc["A  doubles L/R + centre lead 0dB"] = hard_left(r1) + hard_right(r2) + centre(lead)
    sc["A  doubles L/R, rhythm-only section"] = hard_left(r1) + hard_right(r2)
    sc["A  doubles L/R + centre lead -6dB"] = hard_left(r1) + hard_right(r2) + centre(0.5 * lead)
    sc["A  doubles L/R + lead w/ stereo reverb"] = hard_left(r1) + hard_right(r2) + stereo_reverb(centre(lead), rng=rng)
    sc["A  quad (2 hard + 2 at 60%) + centre lead"] = (hard_left(r1) + hard_right(r2) + pan(r3, -0.6) + pan(r4, 0.6)
                                                       + centre(lead))
    sc["ADT fake double (15ms) L/R + centre lead"] = hard_left(r1) + hard_right(adt) + centre(lead)
    sc["2 players: rhythm 70%L, lead 70%R"] = pan(r1, -0.7) + pan(lead, 0.7)
    sc["2 players: rhythm hard L, lead hard R"] = hard_left(r1) + hard_right(lead)
    sc["B  rhythm centre + lead centre"] = centre(r1) + centre(lead)
    sc["B  rhythm centre + lead 20%R"] = centre(r1) + pan(lead, 0.2)
    sc["B  both centre + stereo reverb"] = stereo_reverb(centre(r1) + centre(lead), rng=rng)
    sc["F  mono sum of A"] = centre((r1 + r2) / np.sqrt(2) + lead)
    sc["TRAP twin leads L/R (harm.) + doubles L/R"] = hard_left(r1 + lead) + hard_right(r2 + lead2)
    sc["TRAP twin leads L/R, no rhythm"] = hard_left(lead) + hard_right(lead2)
    return sc


def write_scene(scene: Scene, out_dir: Path) -> Path:
    """mix.wav, takes/<id>.wav (float32 stereo), gt_notes.json, tempo_map.json, scene.json -> ``out_dir``."""
    from bandscribe.audio import write_f32
    from bandscribe.eval.runs import slug

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_f32(out / "mix.wav", scene.mix.T.astype(np.float32), scene.sr)
    takes_meta = []
    for t in scene.takes:
        rel = f"takes/{slug(t.id)}.wav"
        write_f32(out / rel, t.stereo.T.astype(np.float32), scene.sr)
        takes_meta.append({"id": t.id, "line": t.line, "kind": t.kind, "path": rel, "n_notes": len(t.notes)})
    atomic.write_text(out / "gt_notes.json", scene.gt.model_dump_json(indent=1) + "\n")
    atomic.write_json(out / "tempo_map.json", tempo_map_dict(scene.meta["dur_s"]))
    atomic.write_json(out / "scene.json", {
        "format": "bandscribe.scene/1", "item": scene.item_id, "sr": scene.sr, "meta": scene.meta,
        "lines": [ln.model_dump() for ln in scene.lines], "takes": takes_meta})
    return out


# ---------------------------------------------------------------------------------------- mastering

_TP_BLOCK = 1 << 17
_TP_MARGIN = 64  # input samples of context on each side of a block (resample_poly FIR spans ~10)


def true_peak_per_sample(x: np.ndarray, oversample: int = 4) -> np.ndarray:
    """Max |x| over channels and the ``oversample`` phases of the oversampled signal, per input sample.

    Computed in blocks with context so a whole song never needs a 4× copy in RAM."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, _TP_BLOCK):
        e = min(n, s + _TP_BLOCK)
        a, b = max(0, s - _TP_MARGIN), min(n, e + _TP_MARGIN)
        y = signal.resample_poly(x[a:b], oversample, 1, axis=0)
        y = np.abs(y).max(axis=1).reshape(-1, oversample).max(axis=1)
        out[s:e] = np.maximum(y[s - a: s - a + (e - s)], np.abs(x[s:e]).max(axis=1))
    return out


def _forward_min(g: np.ndarray, length: int) -> np.ndarray:
    """out[i] = min(g[i : i + length]) (look-ahead window), padding past the end with 1.0."""
    from scipy.ndimage import minimum_filter1d

    length = max(1, int(length)) | 1  # odd, so the origin shift below is exact
    return minimum_filter1d(g, size=length, mode="constant", cval=1.0, origin=-(length // 2))


def _release(gmin: np.ndarray, a: float) -> np.ndarray:
    """env[i] = min(gmin[i], 1 − (1 − env[i−1])·a): instantaneous attack, one-pole release towards 1.

    With u = 1 − env and v = 1 − gmin: u[i] = max(v[i], a·u[i−1]) = a^i · max_{k≤i}(v[k]·a^−k); evaluated in
    the log domain block by block (exact up to float rounding, vectorised, deterministic)."""
    v = 1.0 - gmin
    out = np.empty_like(v)
    la = math.log(a)
    carry = 0.0
    block = 1 << 16
    for s in range(0, len(v), block):
        vb = v[s: s + block]
        k = np.arange(len(vb), dtype=np.float64)
        with np.errstate(divide="ignore"):
            w = np.where(vb > 0, np.log(np.where(vb > 0, vb, 1.0)), -np.inf) - k * la
        cm = np.maximum.accumulate(w)
        u = np.exp(cm + k * la)
        if carry > 0:
            u = np.maximum(u, carry * np.exp((k + 1) * la))
        out[s: s + len(vb)] = u
        carry = float(u[-1])
    return 1.0 - out


def limiter_envelope(x: np.ndarray, sr: int, *, ceiling_dbtp: float = -1.0, lookahead_s: float = 0.005,
                     release_s: float = 0.08) -> np.ndarray:
    """Per-sample gain (n,) of one ``limit`` pass on ``x`` (float64)."""
    x = np.asarray(x, dtype=np.float64)
    ceiling = 10 ** (ceiling_dbtp / 20)
    peak = true_peak_per_sample(x)
    g = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    gmin = _forward_min(g, int(round(lookahead_s * sr)))
    return _release(gmin, math.exp(-1.0 / (release_s * sr)))


def limit(x: np.ndarray, sr: int, *, ceiling_dbtp: float = -1.0, lookahead_s: float = 0.005,
          release_s: float = 0.08) -> np.ndarray:
    """One look-ahead brickwall pass (float64). Gain from the 4× true-peak estimate, sliding minimum over the
    look-ahead window (so the gain is already down when the peak arrives: equivalent to delaying the signal by
    the look-ahead and trimming the delay), instantaneous attack, one-pole release (τ = ``release_s``)."""
    x = np.asarray(x, dtype=np.float64)
    env = limiter_envelope(x, sr, ceiling_dbtp=ceiling_dbtp, lookahead_s=lookahead_s, release_s=release_s)
    return x * (env[:, None] if x.ndim == 2 else env)


def _loudness(x: np.ndarray, sr: int) -> float:
    import pyloudnorm

    with np.errstate(divide="ignore"):
        return float(pyloudnorm.Meter(sr, filter_class="DeMan").integrated_loudness(x))


def master(x: np.ndarray, sr: int, *, lufs: float = -8.0, ceiling_dbtp: float = -1.0,
           tol_lu: float = 0.2, max_repeats: int = 3) -> np.ndarray:
    """Deterministic master: static gain to ``lufs`` (BS.1770, DeMan filter like M0 ingest), look-ahead limiter
    to ``ceiling_dbtp`` (4× true peak), re-measure and repeat at most ``max_repeats`` times until within
    ±``tol_lu``, then a sample-domain clip at the ceiling as a safety net. (n,) or (n, ch) in, float32 out."""
    x64 = np.asarray(x, dtype=np.float64)
    loud = _loudness(x64, sr)
    if not math.isfinite(loud):
        return np.asarray(x, dtype=np.float32)
    gain_db = lufs - loud
    ceiling = 10 ** (ceiling_dbtp / 20)
    y = x64
    for _ in range(1 + max_repeats):
        y = limit(x64 * 10 ** (gain_db / 20), sr, ceiling_dbtp=ceiling_dbtp)
        got = _loudness(y, sr)
        if not math.isfinite(got) or abs(got - lufs) <= tol_lu:
            break
        gain_db += lufs - got
    return np.clip(y, -ceiling, ceiling).astype(np.float32)


def master_gain(x: np.ndarray, sr: int, *, lufs: float = -8.0, ceiling_dbtp: float = -1.0,
                tol_lu: float = 0.2, max_repeats: int = 3) -> np.ndarray:
    """The per-sample gain (n,) float64 that ``master`` applies to ``x`` before its final safety clip.

    Same loop as ``master`` (static gain to ``lufs``, limiter, re-measure). The distractor suite computes it on the
    full mix of an excerpt and multiplies **every** condition (and every stem) by it, so the guitar signal is the
    same, sample for sample, in all conditions and only the presence of the other sounds changes; the ``full``
    condition then equals ``master(full)`` up to float rounding. Silence (no loudness) -> all ones."""
    x64 = np.asarray(x, dtype=np.float64)
    n = x64.shape[0]
    loud = _loudness(x64, sr)
    if not math.isfinite(loud):
        return np.ones(n, dtype=np.float64)
    gain_db = lufs - loud
    g = env = None
    for _ in range(1 + max_repeats):
        g = 10 ** (gain_db / 20)
        env = limiter_envelope(x64 * g, sr, ceiling_dbtp=ceiling_dbtp)
        got = _loudness(x64 * g * (env[:, None] if x64.ndim == 2 else env), sr)
        if not math.isfinite(got) or abs(got - lufs) <= tol_lu:
            break
        gain_db += lufs - got
    assert g is not None and env is not None
    return g * env


# ---------------------------------------------------------------------------------------- Tier B mix


def mix_parts(track: Any) -> list[Any]:
    """Parts that go into the ``original`` Tier B remix (see ``tierb_mix``): no ``mix`` part, and no ``di``
    part whose line also has a non-DI part."""
    amped_lines = {p.line for p in track.parts if p.kind not in ("mix", "di") and p.line}
    return [p for p in track.parts if p.kind != "mix" and not (p.kind == "di" and p.line in amped_lines)]


def tierb_mix(track: Any, scenario: str = "original", master_lufs: float = -8.0, *,
              root: Path | None = None) -> np.ndarray:
    """Stem sum of a Tier B ``DatasetTrack`` mastered to ``master_lufs`` -> (n, 2) float32 at 44.1 kHz.

    ``original`` = every non-mix part summed at unity (mono parts duplicated to both channels), so the mix is
    consistent with the true solo tracks used as references (reftx). A ``di`` part is left out when its line
    also has an amped/miked part: multitracks ship the DI for re-amping next to the recorded amp, and adding
    both would put a clean copy of the guitar under the real tone (no engineer's mix does that). A DI that is
    its line's only source stays in. Other remix scenarios (A/B/F rebalance) come later (E6). ``root`` defaults
    to ``bandscribe.datasets.dataset_root(track.dataset)``.
    """
    import soundfile as sf

    if scenario != "original":
        raise NotImplementedError(f"tierb_mix scenario {scenario!r} is not implemented yet (M2 needs 'original')")
    if root is None:
        from bandscribe import datasets

        root = Path(datasets.dataset_root(track.dataset))
    parts = mix_parts(track)
    paths_ = [Path(root) / p.audio for p in parts] if parts else ([Path(root) / track.mix] if track.mix else [])
    if not paths_:
        raise ValueError(f"{track.dataset}:{track.track_id} has no audio parts")
    total: np.ndarray | None = None
    for p in paths_:
        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
        if sr != SR:
            import soxr

            y = soxr.resample(y, sr, SR).astype(np.float32)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        elif y.shape[1] > 2:
            y = y[:, :2]
        if total is None:
            total = y.astype(np.float64)
        else:
            if len(y) > len(total):
                total = np.pad(total, ((0, len(y) - len(total)), (0, 0)))
            total[: len(y)] += y
        del y
    assert total is not None
    return master(total, SR, lufs=master_lufs)


def iter_scenes(scenarios: Sequence[str] = SCENARIOS, seeds: Sequence[int] = (0,), *,
                render: bool = True, factory: Callable[..., Scene] = make_scene, **kw: Any):
    for sc in scenarios:
        for s in seeds:
            yield factory(sc, s, render=render, **kw)
