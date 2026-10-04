"""Synthetic songs and fakes shared by the GRID tests (test_grid_*, test_sections_*, test_s35_*, test_pipeline_*).

Not a test module (no ``test_`` prefix). Everything is deterministic (seeded numpy Generators).
"""
from __future__ import annotations

import secrets
from pathlib import Path

import numpy as np

from bandscribe import atomic
from bandscribe.pipeline import graph

SR = 22050
FPS_ENV = 22050 / 256
ENV_LAG_S = 0.008


def note_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


CHORDS = {
    "A": [[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]],
    "B": [[62, 65, 69], [58, 62, 65], [53, 57, 60], [60, 64, 67]],
    "C": [[76, 80, 83], [69, 73, 76], [71, 75, 78], [76, 80, 83]],
}
TIMBRE = {"A": ("saw", 8), "B": ("sine", 1), "C": ("square", 6)}


def tone(freq: float, dur: float, kind: str, nh: int, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    for k in range(1, nh + 1):
        if kind == "square" and k % 2 == 0:
            continue
        if k * freq > 0.45 * sr:
            break
        y += (1.0 / k) * np.sin(2 * np.pi * k * freq * t + rng.uniform(0, 2 * np.pi))
    return y * np.exp(-t * 1.5)


def _place(buf: np.ndarray, x: np.ndarray, start: int) -> None:
    if start >= len(buf):
        return
    e = min(len(buf), start + len(x))
    buf[start:e] += x[:e - start]


def drum_hits(label: str, n_beats: int, beat_s: float) -> list[tuple[float, str]]:
    hits = []
    for b in range(n_beats):
        hits.append((b * beat_s, "kick" if b % 2 == 0 else "snare"))
        if label in "BC":
            hits.append(((b + 0.5) * beat_s, "hat"))
    return hits


def render_hit(kind: str, rng: np.random.Generator, label: str, sr: int = SR) -> np.ndarray:
    if kind == "kick":
        tt = np.arange(int(0.2 * sr)) / sr
        return 0.8 * np.sin(2 * np.pi * (50 + 80 * np.exp(-tt / 0.03)) * tt) * np.exp(-tt / 0.1)
    if kind == "snare":
        tt = np.arange(int(0.15 * sr)) / sr
        return rng.normal(0, 0.3, tt.size) * np.exp(-tt / 0.04)
    L = int(0.03 * sr)
    s = rng.normal(0, 0.08 if label == "B" else 0.15, L) * np.exp(-np.arange(L) / (0.005 * sr))
    return np.diff(s, prepend=0.0) * 3 if label == "C" else s


def make_grid(n_bars: int, bpm: float = 120.0, lead_s: float = 1.0, beats_per_bar: int = 4) -> dict:
    beat = 60.0 / bpm
    bar_s = beats_per_bar * beat
    beats, bars = [], []
    for b in range(n_bars):
        st = lead_s + b * bar_s
        bars.append({"bar": b + 1, "start_s": round(st, 6), "end_s": round(st + bar_s, 6), "n_beats": beats_per_bar,
                     "numerator": beats_per_bar, "denominator": 4, "beat_unit": "quarter", "pickup": False})
        for k in range(beats_per_bar):
            t = round(st + k * beat, 6)
            beats.append({"t_s": t, "t_raw_s": t, "bar": b + 1, "beat": k + 1, "is_downbeat": k == 0,
                          "shift_ms": 0.0, "activation": None})
    dur = lead_s + n_bars * bar_s + 1.0
    return {"format": "bandscribe.grid/1", "tpb": 48, "duration_s": round(dur, 6), "beats": beats, "bars": bars,
            "tempo": [{"start_bar": 1, "end_bar": n_bars + 1, "bpm": bpm}],
            "meter": [{"start_bar": 1, "end_bar": n_bars + 1, "numerator": beats_per_bar, "denominator": 4,
                       "beat_unit": "quarter"}],
            "beat_unit": "quarter", "pickup": {"present": False, "beats": 0}, "hypotheses": {},
            "confidence": {"beats": 1.0, "downbeats": 1.0, "meter": 1.0}, "params": {}}


def synth_song(form: str = "ABABCA", n_bars: int = 8, bpm: float = 120.0, lead_s: float = 1.0,
               shift: dict[int, float] | None = None, seed: int = 0, sr: int = SR
               ) -> tuple[np.ndarray, dict, list[float]]:
    """Sections from distinct chord progressions, timbres and drum patterns.

    shift: {section index: seconds} plays that section late relative to the (nominal) grid.
    Returns (mono float32 signal at sr, nominal grid, percussive hit times incl. shifts).
    """
    rng = np.random.default_rng(seed)
    shift = shift or {}
    beat = 60.0 / bpm
    bar_s = 4 * beat
    total = lead_s + len(form) * n_bars * bar_s + 1.0
    y = np.zeros(int(total * sr))
    hits: list[float] = []
    for n, lbl in enumerate(form):
        t0 = lead_s + n * n_bars * bar_s + shift.get(n, 0.0)
        kind, nh = TIMBRE[lbl]
        for b in range(n_bars):
            for m in CHORDS[lbl][b % 4]:
                _place(y, tone(note_hz(m), bar_s, kind, nh, rng, sr) * 0.15, int(round((t0 + b * bar_s) * sr)))
        for t, k in drum_hits(lbl, n_bars * 4, beat):
            _place(y, render_hit(k, rng, lbl, sr), int(round((t0 + t) * sr)))
            hits.append(t0 + t)
    return y.astype(np.float32), make_grid(len(form) * n_bars, bpm, lead_s), sorted(hits)


def onsets_from(times, strengths=None, duration: float | None = None) -> dict:
    """An onsets dict (as bandscribe.analysis.onsets.compute returns) with spikes at the given times."""
    times = np.asarray(times, dtype=float)
    strengths = np.ones(times.size) if strengths is None else np.asarray(strengths, dtype=float)
    duration = duration if duration is not None else (float(times.max()) + 2.0 if times.size else 10.0)
    env = np.zeros(int(duration * FPS_ENV) + 2)
    for t, s in zip(times, strengths):
        i = int(round((t + ENV_LAG_S) * FPS_ENV))
        if 0 <= i < env.size:
            env[i] = max(env[i], s)
    order = np.argsort(times)
    return {"env": env.astype(np.float32), "fps": FPS_ENV, "times": times[order], "strength": strengths[order]}


# ------------------------------------------------------------------------------------------ riff + lead


def riff_take(rng: np.random.Generator, n_bars: int, bpm: float, sr: int = 44100, jitter_ms: float = 8.0
              ) -> np.ndarray:
    """Distorted power-chord eighths (E5 G5 A5 C5, one chord per bar), one independent take."""
    from scipy import signal

    beat = 60.0 / bpm
    roots = [82.41, 98.0, 110.0, 130.81]
    n = int(round(n_bars * 4 * beat * sr))
    buf = np.zeros(n + sr)
    for b in range(n_bars):
        f = roots[b % 4]
        for k in range(8):
            t0 = (b * 4 + k * 0.5) * beat + rng.normal(0, jitter_ms / 1000)
            s = tone(f, 0.5 * beat * 0.95, "saw", 20, rng, sr) + 0.7 * tone(f * 1.5, 0.5 * beat * 0.95, "saw", 20, rng, sr)
            _place(buf, s, max(0, int(round(t0 * sr))))
    y = np.tanh(6.0 * buf[:n] / (np.sqrt(np.mean(buf[:n] ** 2)) + 1e-9))
    b_, a_ = signal.butter(4, 5000 / (sr / 2), "low")
    return signal.lfilter(b_, a_, y) * 0.1


def lead_line(rng: np.random.Generator, dur_s: float, sr: int = 44100) -> np.ndarray:
    scale = [329.63, 392.0, 440.0, 493.88, 587.33, 659.26, 783.99, 880.0]
    n = int(dur_s * sr)
    buf = np.zeros(n + sr)
    t = 0.1
    while t < dur_s - 0.3:
        d = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
        _place(buf, tone(float(rng.choice(scale)), d + 0.05, "saw", 10, rng, sr), int(t * sr))
        t += d
    y = buf[:n]
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.1


def riff_song(n_occ: int = 3, lead_in=(1, 2), n_bars: int = 4, bpm: float = 120.0, lead_s: float = 1.0,
              sr: int = 44100, seed: int = 3) -> tuple[np.ndarray, dict, dict, dict, set[int]]:
    """n_occ repeats of a riff section; a lead plays over the listed occurrences.

    Returns (mono float32 guitar stem, grid, sections doc, repeat map doc, set of bars with lead).
    """
    rng = np.random.default_rng(seed)
    bar_s = 4 * 60.0 / bpm
    occ_len = int(round(n_bars * bar_s * sr))
    total = int(round((lead_s + n_occ * n_bars * bar_s + 1.0) * sr))
    y = np.zeros(total)
    lead_bars: set[int] = set()
    for i in range(n_occ):
        start = int(round((lead_s + i * n_bars * bar_s) * sr))
        take = riff_take(np.random.default_rng(seed * 100 + i), n_bars, bpm, sr)
        _place(y, take[:occ_len], start)
        if i in lead_in:
            _place(y, lead_line(np.random.default_rng(seed * 1000 + i), n_bars * bar_s, sr), start)
            lead_bars.update(range(1 + i * n_bars, 1 + (i + 1) * n_bars))
    grid = make_grid(n_occ * n_bars, bpm, lead_s)
    secs = [{"id": f"S{i + 1}", "label": "A", "start_bar": 1 + i * n_bars, "end_bar": 1 + (i + 1) * n_bars,
             "start_s": lead_s + i * n_bars * bar_s, "end_s": lead_s + (i + 1) * n_bars * bar_s, "confidence": 1.0}
            for i in range(n_occ)]
    sections_doc = {"format": "bandscribe.sections/1", "method": "synthetic", "sections": secs, "params": {}}
    repeat_doc = {"format": "bandscribe.repeat_map/1", "params": {}, "bars": [],
                  "groups": [{"label": "A", "reference": "S1",
                              "occurrences": [{"section": s["id"], "start_bar": s["start_bar"], "n_bars": n_bars,
                                               "offset_s": 0.0} for s in secs]}]}
    return y.astype(np.float32), grid, sections_doc, repeat_doc, lead_bars


# ------------------------------------------------------------------------------------------ fake stages

GPU = {"beats", "sep", "amt_ms1", "amt_gtr"}


def model_missing(name: str) -> BaseException:
    """The registry's own ``ModelMissing`` for ``name`` (Korean hint naming ``bandscribe models fetch``)."""
    from bandscribe.models import ModelMissing

    return ModelMissing(name, f"{name} 이(가) 없습니다. 먼저 `bandscribe models fetch {name}` 를 실행하세요.")


def write_gpu_run(out: Path, label: str, degraded: bool) -> None:
    atomic.write_json(out / "gpu_run.json", {"format": "bandscribe.gpu_run/1", "backend": "fake", "ladder": ["top"],
                                             "rung": {"index": 1 if degraded else 0, "label": label,
                                                      "device": "cuda"},
                                             "degraded": degraded, "waited_s": 0.0})


class Fakes:
    """Fake STAGES for every stage of the graph: deterministic data files + noisy _worker provenance."""

    def __init__(self) -> None:
        self.degraded: set[str] = set()
        self.missing: set[str] = set()
        self.calls: list[str] = []
        self.variant = "v1"

    def impl(self, name: str) -> dict:
        def run(ctx):
            self.calls.append(name)
            out = Path(ctx.out_dir)
            (out / "_worker").mkdir(exist_ok=True)
            (out / "_worker" / "result.json").write_text(secrets.token_hex(8))  # pids, timestamps, temp paths
            atomic.write_json(out / f"{name}.json", {"stage": name, "params": ctx.params,
                                                     "deps": sorted(ctx.dep_dirs)})
            if name in GPU:
                write_gpu_run(out, "low" if name in self.degraded else "top", name in self.degraded)
            if name == "sep":
                (out / "stems").mkdir()
                (out / "stems" / "bass.wav").write_bytes(b"RIFFbass")
            if name == "stems":
                (out / "practice").mkdir()
                (out / "practice" / "mix_minus_guitar.wav").write_bytes(b"RIFFmmg")
                (out / "practice" / "mix_minus_bass.wav").write_bytes(b"RIFFmmb")
                atomic.write_json(out / "stem_stats.json", {"leftover": {"flag": True, "rms_db_rel_mix": -15.0,
                                                                         "worst_windows": [{"start_s": 10.0,
                                                                                            "rel_db": -12.0}]}})
            if name == "instr":
                atomic.write_json(out / "instrumentation.json", {"format": "bandscribe.instrumentation/1"})
            if name == "notes":
                (out / "midi").mkdir()
                (out / "midi" / "guitar_all.mid").write_bytes(b"MThd-guitar-" + self.variant.encode())
                (out / "midi" / "bass_raw.mid").write_bytes(b"MThd-bass")

        def models(cfg):
            if name in self.missing:
                raise model_missing(f"model.{name}")
            return {f"model.{name}": "0" * 64} if name in GPU else {}

        return {"run": run, "code_version": "1", "params": lambda cfg: {"v": self.variant}, "models": models,
                "device": "gpu" if name in GPU else "cpu"}

    def install(self, monkeypatch) -> None:
        def provider(module):
            return {n: self.impl(n) for n in graph.NAMES if graph.PROVIDER[n] == module}

        monkeypatch.setattr(graph, "_provider", provider)
