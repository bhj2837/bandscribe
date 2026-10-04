"""The ``stems`` stage: everything derived from the mix and the six SW stems (M1_M2_SPEC 3.2 row 5, 6.4, 9.1 item 5).

Streams all seven files block by block (M0 RAM rule: never six whole stems in memory; a 4-min stem is ~85 MB)
and writes, with the deterministic float WAV writer (CPU stage => byte-identical data outputs, M1_M2_SPEC 0):

- ``nonvox.wav`` = mix - vocals - drums - bass; ``leftover.wav`` = mix - sum of the 6 stems (stereo);
- ``practice/mix_minus_guitar.wav`` = mix - guitar, ``practice/mix_minus_bass.wav`` = mix - bass (stereo);
- ``views/<id>_mono.wav`` = (L+R)/2 of mix, guitar, bass, piano+other, nonvox, guitar+other; ``views.json``
  (``source`` names files of the ``sep`` stage dir, ``input/mix_44k_f32.wav`` or this dir);
- ``stem_stats.json``: RMS / relative level / peak per stem, leftover level + flag + worst 10 s windows,
  nonvox level; ``energy.npz``: RMS (dB) at 10 Hz per stem incl. mix, nonvox and leftover.

Leftover is not a failure: above ``sep.leftover_warn_db`` (relative to the mix) it is flagged and the song /
windows are reported (b6). Note SW zeroes every stem's DC bin (``zero_dc``), so the leftover also carries the
mix's DC offset. Worst windows are ranked by leftover level relative to the *window's* mix level, ignoring
windows more than 30 dB below the song's mix level (silence would make any ratio meaningless).

All sums of squares are float64 over fixed blocks, and every float written to JSON is rounded, so two runs
produce identical bytes. ``energy.npz`` is written with fixed zip timestamps (``np.savez`` stamps the time).
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic

log = logging.getLogger(__name__)

STEMS = ("bass", "drums", "other", "vocals", "guitar", "piano")
VIEWS = ("mix", "guitar", "bass", "piano_other", "nonvox", "guitar_other")
ENERGY_STREAMS = ("mix",) + STEMS + ("nonvox", "leftover")
BLOCK_FRAMES = 1 << 18
ENERGY_HZ = 10.0
WINDOW_S = 10.0
WORST_WINDOWS = 5
SILENT_WINDOW_DB = -30.0  # windows this far below the song's mix level are not ranked
_EPS = 1e-20
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

VIEW_SOURCES = {
    "mix": "input/mix_44k_f32.wav",
    "guitar": "stems/guitar.wav",
    "bass": "stems/bass.wav",
    "piano_other": "stems/piano.wav + stems/other.wav",
    "nonvox": "nonvox.wav",
    "guitar_other": "stems/guitar.wav + stems/other.wav",
}


def _db(sum_sq: float, count: float) -> float:
    return 10.0 * math.log10(sum_sq / count + _EPS) if count > 0 else -200.0


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def write_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Like ``np.savez`` (uncompressed, same member names) but with fixed timestamps and member order."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in arrays:
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_DATE)
            info.external_attr = 0o600 << 16
            zf.writestr(info, member.getvalue())
    atomic.write_bytes(Path(path), buf.getvalue())


class _Acc:
    """Whole-file and per-frame sums of squares (float64) plus the peak of one stream."""

    def __init__(self, n_frames_energy: int, hop: int, n_windows: int = 0, win: int = 0) -> None:
        self.sum_sq = 0.0
        self.count = 0
        self.peak = 0.0
        self.hop = hop
        self.frames = np.zeros(n_frames_energy, dtype=np.float64)
        self.win = win
        self.windows = np.zeros(n_windows, dtype=np.float64) if n_windows else None

    @staticmethod
    def _binned(per_sample: np.ndarray, s0: int, size: int, out: np.ndarray) -> None:
        idx = (np.arange(len(per_sample), dtype=np.int64) + s0) // size
        f0 = int(idx[0])
        sums = np.bincount(idx - f0, weights=per_sample)
        out[f0:f0 + len(sums)] += sums

    def add(self, x: np.ndarray, s0: int) -> None:
        if not len(x):
            return
        x64 = x.astype(np.float64)
        per_sample = np.einsum("ij,ij->i", x64, x64)  # sum over channels
        self.sum_sq += float(per_sample.sum())
        self.count += x.size
        self.peak = max(self.peak, float(np.max(np.abs(x))))
        self._binned(per_sample, s0, self.hop, self.frames)
        if self.windows is not None:
            self._binned(per_sample, s0, self.win, self.windows)


def _frame_counts(n: int, size: int, channels: int) -> np.ndarray:
    k = -(-n // size)
    counts = np.full(k, size * channels, dtype=np.float64)
    if k and n % size:
        counts[-1] = (n % size) * channels
    return counts


def _db_series(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(sums / np.maximum(counts, 1.0) + _EPS)).astype(np.float32)


def derive(mix_wav: Path, stems_dir: Path, out_dir: Path, *, leftover_warn_db: float = -20.0,
           block_frames: int = BLOCK_FRAMES) -> dict[str, Any]:
    """Write every ``stems`` stage output into ``out_dir``; returns the stem_stats document."""
    import soundfile as sf

    from bandscribe.audio import F32Writer, to_mono

    mix_wav, stems_dir, out_dir = Path(mix_wav), Path(stems_dir), Path(out_dir)
    (out_dir / "practice").mkdir(parents=True, exist_ok=True)
    (out_dir / "views").mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        readers: dict[str, Any] = {"mix": stack.enter_context(sf.SoundFile(str(mix_wav)))}
        for s in STEMS:
            readers[s] = stack.enter_context(sf.SoundFile(str(stems_dir / f"{s}.wav")))
        sr = readers["mix"].samplerate
        n_total = readers["mix"].frames
        for name, r in readers.items():
            if r.samplerate != sr or r.frames != n_total or r.channels != 2:
                raise ValueError(f"{name}: {r.frames} frames / {r.samplerate} Hz / {r.channels} ch; expected "
                                 f"{n_total} frames / {sr} Hz / 2 ch like the mix")
        hop = int(round(sr / ENERGY_HZ))
        win = int(round(sr * WINDOW_S))
        n_energy = -(-n_total // hop)
        n_win = -(-n_total // win)
        acc = {k: _Acc(n_energy, hop, n_win if k in ("mix", "leftover") else 0, win) for k in ENERGY_STREAMS}

        stereo_out = {"nonvox": out_dir / "nonvox.wav", "leftover": out_dir / "leftover.wav",
                      "mix_minus_guitar": out_dir / "practice" / "mix_minus_guitar.wav",
                      "mix_minus_bass": out_dir / "practice" / "mix_minus_bass.wav"}
        writers = {k: F32Writer(p, sr, 2) for k, p in stereo_out.items()}
        view_paths = {v: out_dir / "views" / f"{v}_mono.wav" for v in VIEWS}
        writers.update({f"view:{v}": F32Writer(p, sr, 1) for v, p in view_paths.items()})
        view_sha = {v: hashlib.sha256() for v in VIEWS}

        s0 = 0
        try:
            while s0 < n_total:
                b = {k: r.read(block_frames, dtype="float32", always_2d=True) for k, r in readers.items()}
                n = len(b["mix"])
                if n == 0 or any(len(v) != n for v in b.values()):
                    raise ValueError(f"short read at frame {s0}")
                mix = b["mix"]
                nonvox = mix - b["vocals"] - b["drums"] - b["bass"]
                leftover = mix - b["bass"] - b["drums"] - b["other"] - b["vocals"] - b["guitar"] - b["piano"]
                writers["nonvox"].write(nonvox)
                writers["leftover"].write(leftover)
                writers["mix_minus_guitar"].write(mix - b["guitar"])
                writers["mix_minus_bass"].write(mix - b["bass"])
                views = {"mix": mix, "guitar": b["guitar"], "bass": b["bass"], "piano_other": b["piano"] + b["other"],
                         "nonvox": nonvox, "guitar_other": b["guitar"] + b["other"]}
                for v in VIEWS:
                    # bandscribe.audio.to_mono: the one (L+R)/2 definition, so views are bit-identical however made
                    mono = np.ascontiguousarray(to_mono(views[v]), dtype="<f4")
                    writers[f"view:{v}"].write(mono.reshape(-1, 1))
                    view_sha[v].update(mono.tobytes())  # == pcm_sha256 of the written file (interleaved f32)
                for k in STEMS + ("mix",):
                    acc[k].add(b[k], s0)
                acc["nonvox"].add(nonvox, s0)
                acc["leftover"].add(leftover, s0)
                s0 += n
        except BaseException:
            for w in writers.values():  # publish nothing half-written
                w.abort()
            raise
        for w in writers.values():
            w.close()

    mix_db = _db(acc["mix"].sum_sq, acc["mix"].count)
    stems_doc = {}
    for s in STEMS:
        a = acc[s]
        rms = _db(a.sum_sq, a.count)
        stems_doc[s] = {"rms_db": _r(rms), "rel_db": _r(rms - mix_db), "peak": _r(a.peak, 6)}
    lo = acc["leftover"]
    lo_rel = _db(lo.sum_sq, lo.count) - mix_db
    counts_w = _frame_counts(n_total, win, 2)
    mix_w = 10.0 * np.log10(acc["mix"].windows / counts_w + _EPS)
    lo_w = 10.0 * np.log10(lo.windows / counts_w + _EPS)
    ranked = [(float(lo_w[i] - mix_w[i]), i) for i in range(len(counts_w)) if mix_w[i] >= mix_db + SILENT_WINDOW_DB]
    ranked.sort(key=lambda t: (-round(t[0], 2), t[1]))
    worst = [{"start_s": _r(i * WINDOW_S, 3), "rel_db": _r(rel, 2)} for rel, i in ranked[:WORST_WINDOWS]]
    stats = {
        "format": "bandscribe.stem_stats/1",
        "sr": sr,
        "frames": n_total,
        "mix_rms_db": _r(mix_db),
        "stems": stems_doc,
        "leftover": {"rms_db_rel_mix": _r(lo_rel), "flag": bool(lo_rel > leftover_warn_db),
                     "threshold_db": float(leftover_warn_db), "window_s": WINDOW_S, "worst_windows": worst},
        "nonvox": {"rel_db": _r(_db(acc["nonvox"].sum_sq, acc["nonvox"].count) - mix_db)},
    }
    atomic.write_json(out_dir / "stem_stats.json", stats)

    counts_e = _frame_counts(n_total, hop, 2)
    energy = {k: _db_series(acc[k].frames, counts_e) for k in ENERGY_STREAMS}
    energy["fps"] = np.array([ENERGY_HZ], dtype=np.float32)
    write_npz_deterministic(out_dir / "energy.npz", energy)

    views_doc = {"format": "bandscribe.views/1", "sr": sr, "views": {
        f"{v}_mono": {"path": f"views/{v}_mono.wav", "pcm_sha256": view_sha[v].hexdigest(), "source": VIEW_SOURCES[v],
                      "mono": "(L+R)/2"} for v in VIEWS}}
    atomic.write_json(out_dir / "views.json", views_doc)
    if stats["leftover"]["flag"]:
        log.warning("%s", leftover_warning(stats))
    return stats


def leftover_warning(stats: dict[str, Any]) -> str | None:
    """Korean warning line for the run output when the leftover is above the threshold (b6), else None."""
    lo = (stats or {}).get("leftover") or {}
    if not lo.get("flag"):
        return None
    windows = ", ".join(f"{w['start_s']:.0f}초~ ({w['rel_db']:+.1f} dB)" for w in lo.get("worst_windows", []))
    return (f"leftover(믹스 - 6 스템 합)가 믹스 대비 {lo.get('rms_db_rel_mix'):+.1f} dB 로 기준 "
            f"{lo.get('threshold_db'):+.0f} dB 보다 큽니다. 분리가 덜 된 구간: {windows or '(없음)'}")
