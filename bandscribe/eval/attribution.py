"""Cause attribution of guitar transcription errors and separation leakage (distractor suite, DESIGN §8.8).

Everything here is numpy on audio that the caller already has: the true stems of an excerpt grouped by sound
category (``bandscribe.eval.stem_taxonomy``), the separated guitar stem, and note lists as (onset, offset, pitch).

**Pitch-band energy.** ``band_energy`` turns a mono signal into two (frames x 88) matrices, MIDI 21-108: the power of
the STFT bins within +-50 cents of the pitch's f0 and of its 2nd harmonic (Hann, n_fft 8192 = 186 ms, hop 1024 =
23 ms, centred frames; when no bin centre falls inside a narrow low band the nearest bin stands in). A note's
energy for a category is the mean over the frames of ``[onset, onset + clip(duration, 50 ms, 500 ms)]`` of the f0
band, plus the 2nd-harmonic band for notes below MIDI 52 (E3, 165 Hz) only: there the bins (5.4 Hz apart; a
semitone at E2 is 4.9 Hz) cannot resolve the fundamental, which is also weak on guitar and bass. Above it the 2nd
harmonic band would credit a sound playing one octave **higher** (a pad note at 76 lands in the 2nd-harmonic band
of a false note at 64 that the guitar's own harmonic made).

**False positive (estimated note with no reference match), first rule that applies:**

1. ``guitar_timing``: a reference note of the same pitch overlaps it in time or starts within 200 ms (the onset
   missed the 50 ms window, or the note was split / doubled);
2. the category whose true stem has the most energy in the note's bands, **if that is not a guitar**: that sound
   is the cause. This comes before the harmonic rules on purpose: keys, pads and strings play the guitar's harmony,
   so a distractor note very often sits an octave or a fifth from a guitar note (first smoke run, Hammond organ:
   70 of 104 "octave" false notes had the organ, not the guitar, as the loudest sound at their pitch);
3. ``guitar_octave``: a reference note at +-12, +-19 or +-24 semitones overlaps it (harmonics: octave, octave +
   fifth, two octaves);
4. ``guitar_near``: a reference note at +-1 or +-2 semitones overlaps it (bend, vibrato, pitch slip);
5. ``guitar_unref`` when the guitar is the loudest sound there (the true guitar sounds at that pitch but the
   reference has no note: a reference miss or a sympathetic / harmonic note), else ``unknown`` (nothing above the
   floor).

**Missed reference note (false negative):** ``est_timing`` / ``est_octave`` / ``est_near`` as above with the
roles swapped; otherwise the strongest **non-guitar** category in the note's bands is the masker if its energy is
at least the guitar's minus 6 dB (cause = that category), else ``guitar_clear`` (the guitar dominated its own
band: the loss is not a distractor's doing).

Nothing is attributed below a floor: the larger of ``FLOOR_DB`` (-60 dB) under the loud frames of the excerpt's full
mix (95th percentile of the frame power summed over all bins) and ``LOCAL_FLOOR_DB`` (-40 dB) under the full mix's
power in the note's own frames (so onset splatter of a loud chord does not "explain" an unrelated pitch).

**Leakage** (``leakage``): STFT (n_fft 4096, hop 1024) of every category present in a condition and of the
separated guitar stem. A time-frequency bin is *dominated* by a category when that category holds >= 80 % of the
summed true power there (>= 6 dB over all others together). For each category: ``leak`` = separated-guitar power /
true power over its dominated bins (how much of that sound the guitar stem passes on; for the guitar categories
this is the retention), and ``share`` = the fraction of the separated guitar stem's power that lies in that
category's dominated bins (what the guitar stem is made of). A band-energy ratio, not a least-squares projection:
it needs no phase alignment and reads directly as "dB of synth pad left in the guitar stem".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

SR = 44100
N_FFT = 8192
HOP = 1024
MIDI_LO, MIDI_HI = 21, 108
N_PITCH = MIDI_HI - MIDI_LO + 1
BAND_CENTS = 50.0
FLOOR_DB = -60.0
LOCAL_FLOOR_DB = -40.0
DUR_MIN_S, DUR_MAX_S = 0.05, 0.5
NEAR_ONSET_S = 0.2
OVERLAP_SLACK_S = 0.05
H2_BELOW_MIDI = 52
OCTAVE_STEPS = (12, 19, 24)
NEAR_STEPS = (1, 2)
MASK_MARGIN_DB = 6.0
LEAK_N_FFT = 4096
LEAK_DOMINANCE = 0.8

FP_CAUSES_GUITAR = ("guitar_timing", "guitar_octave", "guitar_near", "guitar_unref")
FN_CAUSES_EST = ("est_timing", "est_octave", "est_near")
UNKNOWN = "unknown"
GUITAR_CLEAR = "guitar_clear"


def _frames_power(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """|STFT|^2 (frames, n_fft/2+1) of mono ``x`` with a Hann window and centred frames (zero padding)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    pad = n_fft // 2
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = 1 + max(0, (len(xp) - n_fft) // hop)
    win = np.hanning(n_fft).astype(np.float64)
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    block = 256
    for s in range(0, n_frames, block):
        e = min(n_frames, s + block)
        idx = (np.arange(s, e) * hop)[:, None] + np.arange(n_fft)[None, :]
        spec = np.fft.rfft(xp[idx] * win[None, :], axis=1)
        out[s:e] = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return out


def pitch_band_matrix(harmonic: int = 1, n_fft: int = N_FFT, sr: int = SR) -> np.ndarray:
    """(bins, 88) 0/1 weights: bins whose centre lies within +-50 cents of ``harmonic`` x f0 (nearest bin if none)."""
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    m = np.zeros((len(freqs), N_PITCH), dtype=np.float32)
    r = 2 ** (BAND_CENTS / 1200.0)
    for k, p in enumerate(range(MIDI_LO, MIDI_HI + 1)):
        f = harmonic * 440.0 * 2 ** ((p - 69) / 12.0)
        if f >= sr / 2:
            continue
        sel = (freqs >= f / r) & (freqs <= f * r)
        if not sel.any():
            sel = np.zeros(len(freqs), dtype=bool)
            sel[int(np.argmin(np.abs(freqs - f)))] = True
        m[sel, k] = 1.0
    return m


_BANDS: dict[tuple[int, int, int], np.ndarray] = {}


def _band(harmonic: int, n_fft: int, sr: int) -> np.ndarray:
    key = (harmonic, n_fft, sr)
    if key not in _BANDS:
        _BANDS[key] = pitch_band_matrix(harmonic, n_fft, sr)
    return _BANDS[key]


@dataclass
class BandEnergy:
    """Pitch-band power of one signal: ``f0`` and ``h2`` are (frames, 88); ``frame_s`` = hop / sr."""

    f0: np.ndarray
    h2: np.ndarray
    frame_s: float
    frame_power_p95: float  # 95th percentile over frames of the power summed over all bins (the floor reference)
    total: np.ndarray | None = None  # (frames,) power summed over all bins

    def _span(self, onset: float, offset: float) -> tuple[int, int]:
        dur = min(DUR_MAX_S, max(DUR_MIN_S, float(offset) - float(onset)))
        a = int(np.floor(float(onset) / self.frame_s))
        z = int(np.ceil((float(onset) + dur) / self.frame_s))
        return max(0, a), min(len(self.f0), max(z, a + 1))

    def note(self, onset: float, offset: float, pitch: int) -> float:
        """Mean power of the f0 band (+ the 2nd-harmonic band below MIDI 52) over the note's frames (0.0 outside
        the pitch range)."""
        k = int(round(pitch)) - MIDI_LO
        if not 0 <= k < N_PITCH or not len(self.f0):
            return 0.0
        a, z = self._span(onset, offset)
        if z <= a:
            return 0.0
        if int(round(pitch)) < H2_BELOW_MIDI:
            return float(np.mean(self.f0[a:z, k] + self.h2[a:z, k]))
        return float(np.mean(self.f0[a:z, k]))

    def local_total(self, onset: float, offset: float) -> float:
        """Mean all-bin power over the note's frames (the local floor reference)."""
        if self.total is None or not len(self.total):
            return 0.0
        a, z = self._span(onset, offset)
        return float(np.mean(self.total[a:z])) if z > a else 0.0


def band_energy(x: np.ndarray, sr: int = SR, *, n_fft: int = N_FFT, hop: int = HOP) -> BandEnergy:
    p = _frames_power(x, n_fft, hop)
    f0 = p @ _band(1, n_fft, sr)
    h2 = p @ _band(2, n_fft, sr)
    tot = p.sum(axis=1).astype(np.float64)
    p95 = float(np.percentile(tot, 95)) if len(p) else 0.0
    return BandEnergy(f0=f0, h2=h2, frame_s=hop / float(sr), frame_power_p95=p95, total=tot)


# --------------------------------------------------------------------------------------- note relations


def _overlaps(a0: float, a1: float, b0: float, b1: float, slack: float = OVERLAP_SLACK_S) -> bool:
    return a0 < b1 + slack and b0 < a1 + slack


def _relation(note: tuple[float, float, int], others: np.ndarray) -> str | None:
    """'timing' | 'octave' | 'near' | None for one note against an (n, 3) array of (onset, offset, pitch)."""
    if not len(others):
        return None
    on, off, p = float(note[0]), max(float(note[1]), float(note[0])), int(note[2])
    o_on, o_off, o_p = others[:, 0], np.maximum(others[:, 1], others[:, 0]), others[:, 2].astype(int)
    ov = (o_on < off + OVERLAP_SLACK_S) & (on < o_off + OVERLAP_SLACK_S)
    same = o_p == p
    if np.any(same & (ov | (np.abs(o_on - on) <= NEAR_ONSET_S))):
        return "timing"
    d = np.abs(o_p - p)
    if np.any(ov & np.isin(d, OCTAVE_STEPS)):
        return "octave"
    if np.any(ov & np.isin(d, NEAR_STEPS)):
        return "near"
    return None


def _as_rows(notes: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    a = np.asarray(notes, dtype=float)
    return a.reshape(-1, 3) if a.size else np.zeros((0, 3))


@dataclass
class Cause:
    label: str          # the cause label (see module docstring)
    dominant: str       # the category with the most band energy (any category; "" if none above the floor)
    dominant_share: float  # its share of the summed band energy of all categories present


def _energies(bands: Mapping[str, BandEnergy], note: tuple[float, float, int]) -> dict[str, float]:
    return {c: be.note(note[0], note[1], int(note[2])) for c, be in bands.items()}


def floor_power(bands: Mapping[str, BandEnergy], mix_p95: float | None = None) -> float:
    """Band-energy floor: ``FLOOR_DB`` below the mix's loud-frame power (default: the loudest category's)."""
    ref = mix_p95 if mix_p95 is not None else max((b.frame_power_p95 for b in bands.values()), default=0.0)
    return float(ref) * 10 ** (FLOOR_DB / 10.0)


def _note_floor(note: tuple[float, float, int], floor: float, mix: BandEnergy | None) -> float:
    if mix is None:
        return floor
    return max(floor, mix.local_total(note[0], note[1]) * 10 ** (LOCAL_FLOOR_DB / 10.0))


def attribute_fp(note: tuple[float, float, int], ref: np.ndarray, bands: Mapping[str, BandEnergy], *,
                 guitar: Sequence[str], floor: float, mix: BandEnergy | None = None) -> Cause:
    """Cause of one false-positive note (see the module docstring); ``bands`` = the condition's categories,
    ``mix`` = the condition's mix (local floor)."""
    rel = _relation(note, _as_rows(ref))
    floor = _note_floor(note, floor, mix)
    e = _energies(bands, note)
    tot = sum(e.values())
    dom = max(sorted(e), key=lambda c: e[c]) if e else ""
    share = (e[dom] / tot) if tot > 0 and dom else 0.0
    if dom and e[dom] < floor:
        dom_out = ""
    else:
        dom_out = dom
    if rel == "timing":
        return Cause("guitar_timing", dom_out, share)
    if dom_out and dom_out not in guitar:
        return Cause(dom_out, dom_out, share)
    if rel is not None:
        return Cause(f"guitar_{rel}", dom_out, share)
    if dom_out in guitar:
        return Cause("guitar_unref", dom_out, share)
    return Cause(UNKNOWN, "", 0.0)


def attribute_fn(note: tuple[float, float, int], est: np.ndarray, bands: Mapping[str, BandEnergy], *,
                 guitar: Sequence[str], floor: float, mix: BandEnergy | None = None) -> Cause:
    """Cause of one missed reference note: est relation, else the strongest non-guitar masker (or guitar_clear)."""
    rel = _relation(note, _as_rows(est))
    floor = _note_floor(note, floor, mix)
    e = _energies(bands, note)
    others = {c: v for c, v in e.items() if c not in guitar}
    g = sum(v for c, v in e.items() if c in guitar)
    tot = sum(e.values())
    dom = max(sorted(others), key=lambda c: others[c]) if others else ""
    share = (others[dom] / tot) if dom and tot > 0 else 0.0
    if dom and others[dom] < floor:
        dom = ""
        share = 0.0
    if rel is not None:
        return Cause(f"est_{rel}", dom, share)
    if dom and others[dom] >= g * 10 ** (-MASK_MARGIN_DB / 10.0):
        return Cause(dom, dom, share)
    return Cause(GUITAR_CLEAR, dom, share)


# --------------------------------------------------------------------------------------------- leakage


def leakage(categories: Mapping[str, np.ndarray], separated_guitar: np.ndarray, *, sr: int = SR,
            n_fft: int = LEAK_N_FFT, hop: int = HOP, dominance: float = LEAK_DOMINANCE,
            group: Mapping[str, str] | None = None) -> dict[str, dict[str, float]]:
    """{category: {"leak": ratio, "share": fraction, "bins": n}} (see the module docstring).

    ``categories`` = mono true signals as they enter the condition mix; ``group`` maps categories to a reported
    name (e.g. both guitar categories -> ``guitar``) before dominance is computed.
    """
    group = dict(group or {})
    merged: dict[str, np.ndarray] = {}
    n = len(np.asarray(separated_guitar).reshape(-1))
    for c, x in categories.items():
        name = group.get(c, c)
        x = np.asarray(x, dtype=np.float64).reshape(-1)[:n]
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        merged[name] = merged[name] + x if name in merged else x
    pg = _frames_power(separated_guitar, n_fft, hop).astype(np.float64)
    powers = {c: _frames_power(x, n_fft, hop).astype(np.float64) for c, x in sorted(merged.items())}
    if not powers:
        return {}
    tot = np.sum(list(powers.values()), axis=0)
    alive = tot > 1e-12
    g_all = float(pg[alive].sum()) if alive.any() else 0.0
    out: dict[str, dict[str, float]] = {}
    for c, p in powers.items():
        dom = alive & (p >= dominance * tot)
        num = float(pg[dom].sum())
        den = float(tot[dom].sum())
        out[c] = {"leak": (num / den) if den > 0 else float("nan"),
                  "share": (num / g_all) if g_all > 0 else float("nan"), "bins": int(dom.sum())}
    return out


def ratio_db(x: float) -> float:
    return float(10.0 * np.log10(x)) if x is not None and np.isfinite(x) and x > 0 else float("nan")
