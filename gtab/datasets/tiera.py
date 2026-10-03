"""Tier A candidate songs (the user's own cover repertoire) and the backing-track alignment diagnostic.

User request 2026-09-30: the songs the user actually covers live in ``D:\\backing`` together with the user's
practice backing tracks (the song with — probably — the guitar part removed, 1-2 s longer, other start
offset). This module keeps a local registry ``data/eval/tierA/songs.yaml`` (paths only: nothing is copied,
the originals are only ever read) and aligns each backing to its original:

1. decode both with ffmpeg (``gtab.ingest.decode.decode_to_f32``, float32 stereo 44.1 kHz) into a scratch dir;
2. coarse offset: cross-correlation of spectral-flux onset envelopes (100 frames/s) within ±``max_offset_s``;
3. fine offset: waveform cross-correlation in 12 windows spread over the song (±2 hops around the coarse
   lag) with parabolic sub-sample refinement; the median is used, the spread and a linear fit give the drift;
4. gain: least squares ``g = <o, b> / <b, b>`` over the overlap (sign = polarity);
5. residual = original − g · backing(shifted by a windowed-sinc fractional delay) -> ``residual_not_gt.wav``,
   with level/band/window statistics in ``align.json``;
6. codec floor: the original re-encoded with ffmpeg's AAC encoder at the backing's bit rate and aligned the
   same way. Its residual is what "nothing removed, only re-encoded" looks like, so ``excess_over_floor_db``
   (backing residual minus floor, overall and per band) separates a removed part from codec noise.

**The residual is NOT ground truth.** The backings are AAC (and three originals are MP3) and the way the
backing was made is unknown, so the residual mixes the removed part with codec noise and any processing
difference. It is only a diagnostic for "which instrument did the backing remove, and where". The label is on
the file itself (``residual_not_gt.wav``), in ``README_NOT_GROUND_TRUTH.txt`` next to it and in ``align.json``
(``ground_truth: false``). Copyrighted audio: every file stays under ``data/`` (git-ignored) on this PC.

Measured 2026-10-03 (``gtab data tier-a align``): offset 0 and drift 0 ppm for all five songs (the backings'
audio streams start with the original; only their video tracks are longer). AIZO / 絶対零度 / 青春コンプレックス:
residual +5.4 / +5.8 / +9.9 dB above the codec floor, mid-band heavy (a guitar-like part was removed). 怪獣の花唄 /
青と夏: +0.1 / +0.0 dB, i.e. indistinguishable from re-encoding the original - those two backings removed nothing
measurable (or the "original" already lacks the part).

Core env only (numpy, scipy, soundfile, ffmpeg via gtab.ingest). The songs registry is YAML (pyyaml).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gtab import atomic, paths

log = logging.getLogger(__name__)

SONGS_FORMAT = "gtab.tiera_songs/1"
ALIGN_FORMAT = "gtab.tiera_backing_align/1"
SR = 44100
HOP = 441                 # 100 onset frames per second
N_FFT = 1024
BANDS_HZ = ((20, 120), (120, 400), (400, 1000), (1000, 2500), (2500, 6000), (6000, 16000))
BAND_NAMES = ("sub_kick_20_120", "bass_120_400", "lowmid_400_1k", "mid_1k_2k5", "presence_2k5_6k", "air_6k_16k")
N_FINE_WINDOWS = 12
FINE_WIN_S = 8.0
LOSSLESS_CODECS = {"flac", "alac", "wavpack", "ape", "tta"}
NOT_GT_KO = ("이 잔차는 정답이 아닙니다: 반주(와 MP3 원곡)는 손실 압축이고 반주를 만든 방법도 모르므로, "
             "빠진 파트 + 코덱 잡음 + 처리 차이가 섞여 있습니다. 반주에서 무엇이 빠졌는지 확인하는 진단용입니다.")
NOT_GT_EN = ("NOT ground truth: the backing (and an MP3 original) is a lossy encode and the backing's production is "
             "unknown; the residual mixes the removed part with codec noise and processing differences. "
             "Diagnostic only.")
# The file name itself says what it is: a stray copy of the WAV must not pass for a clean guitar stem.
RESIDUAL_WAV = "residual_not_gt.wav"
_LEGACY_RESIDUAL = "residual.wav"  # name used before 2026-10-03
README_NAME = "README_NOT_GROUND_TRUTH.txt"
_ID_RE = re.compile(r"^[a-z0-9_-]+$")


class TierAError(RuntimeError):
    """User-facing (Korean) Tier A registry / alignment error."""


# ------------------------------------------------------------------------------------------- registry


def tier_a_dir() -> Path:
    """``data/eval/tierA`` (read at call time so tests can repoint ``gtab.paths.EVAL``)."""
    return Path(paths.EVAL) / "tierA"


def songs_path() -> Path:
    return tier_a_dir() / "songs.yaml"


@dataclass
class SongEntry:
    id: str
    title: str
    artist: str
    path: str
    duration_s: float | None = None
    lossless: bool = False
    codec: str = ""
    backing: str | None = None
    backing_duration_s: float | None = None
    backing_codec: str = ""
    backing_note: str = ""
    gt: str | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


_FIELDS = [f for f in SongEntry.__dataclass_fields__ if f != "extra"]


def _yaml() -> Any:
    try:
        import yaml
    except ImportError as e:
        raise TierAError("pyyaml 이 없습니다. D:\\gtab 에서 `tools\\uvw.cmd sync --extra core --group dev` 를 실행하세요.") from e
    return yaml


def load_songs(path: Path | None = None) -> dict[str, SongEntry]:
    p = Path(path or songs_path())
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        d = _yaml().safe_load(f) or {}
    if d.get("format") != SONGS_FORMAT:
        raise TierAError(f"{p}: format 이 {SONGS_FORMAT} 가 아닙니다")
    out: dict[str, SongEntry] = {}
    for s in d.get("songs") or []:
        known = {k: s[k] for k in _FIELDS if k in s}
        extra = {k: v for k, v in s.items() if k not in _FIELDS}
        e = SongEntry(**known, extra=extra)
        out[e.id] = e
    return out


def _q(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    return json.dumps(str(v), ensure_ascii=False)


def songs_yaml_text(songs: dict[str, SongEntry]) -> str:
    lines = [
        "# gtab Tier A 후보곡 목록 - 사용자가 실제로 커버하는 곡 (2026-09-30 사용자 요청).",
        "# 저작권 음원입니다: 이 PC 안에서만 쓰고 절대 커밋·공유하지 않습니다. 파일은 경로로만 가리킵니다(복사 안 함).",
        "# 원본 파일은 읽기만 합니다 - 수정·이동하지 마세요.",
        "# backing: 사용자가 만든 연습용 반주(원곡에서 아마 기타 파트를 뺀 것, 1-2초 길고 시작 위치가 다름).",
        "#   `gtab data tier-a align <id>` 로 원곡에 맞춰 잔차(원곡 - 반주)를 만들 수 있지만, 그 잔차는 정답이 아닙니다",
        "#   (둘 다 손실 압축). 무엇이 빠졌는지 확인하는 진단용입니다.",
        "# gt: 기계가 읽을 수 있는 정답 탭(GP 파일) 경로. 아직 없음(null). 생기면 `gtab gt import` 로 가져옵니다.",
        f"format: {SONGS_FORMAT}",
        "songs:",
    ]
    for e in songs.values():
        lines.append(f"  - id: {e.id}")
        for k in _FIELDS[1:]:
            lines.append(f"    {k}: {_q(getattr(e, k))}")
        for k, v in e.extra.items():
            lines.append(f"    {k}: {json.dumps(v, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def save_songs(songs: dict[str, SongEntry], path: Path | None = None) -> Path:
    p = Path(path or songs_path())
    atomic.write_text(p, songs_yaml_text(songs))
    return p


def _container_duration(path: Path) -> float | None:
    """Container duration (ffprobe format) — differs from the audio stream's when a video track is longer."""
    import json as _json

    from gtab import winjob
    from gtab.ingest.decode import tool_path

    run = winjob.run_captured([tool_path("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json",
                               str(path)], timeout_s=120, env=paths.child_env())
    try:
        return round(float(_json.loads(run.stdout)["format"]["duration"]), 3)
    except (ValueError, KeyError, TypeError):
        return None


def _describe(path: Path) -> tuple[float | None, bool, str]:
    """(audio duration_s, lossless, codec text) via ffprobe."""
    from gtab.ingest.decode import probe

    info = probe(path)
    codec = str(info.get("codec_name") or "?")
    lossless = codec.startswith("pcm_") or codec in LOSSLESS_CODECS
    br = info.get("bit_rate")
    text = codec if lossless or not br else f"{codec} {round(br / 1000)} kbps"
    if info.get("has_video"):
        text += " (+video)"
    dur = info.get("duration")
    return (round(float(dur), 3) if dur else None), lossless, text


def add_song(song_id: str, path: Path, *, title: str, artist: str, backing: Path | None = None,
             backing_note: str = "", notes: str = "", gt: Path | None = None,
             registry_path: Path | None = None) -> SongEntry:
    """Register (or update) one song; durations/codecs come from ffprobe. Files are never copied."""
    if not _ID_RE.match(song_id):
        raise TierAError(f"곡 ID 는 ASCII 소문자·숫자·_·- 만 됩니다: {song_id!r}")
    path = Path(path)
    if not path.is_file():
        raise TierAError(f"원곡 파일이 없습니다: {path}")
    dur, lossless, codec = _describe(path)
    e = SongEntry(id=song_id, title=title, artist=artist, path=path.as_posix(), duration_s=dur, lossless=lossless,
                  codec=codec, backing_note=backing_note, notes=notes, gt=Path(gt).as_posix() if gt else None)
    if backing is not None:
        backing = Path(backing)
        if not backing.is_file():
            raise TierAError(f"반주 파일이 없습니다: {backing}")
        bdur, _, bcodec = _describe(backing)
        e.backing, e.backing_duration_s, e.backing_codec = backing.as_posix(), bdur, bcodec
        cdur = _container_duration(backing)
        if cdur is not None and bdur is not None and abs(cdur - bdur) > 0.05:
            # e.g. the practice MP4s: audio as long as the song, video 1.7-2.4 s longer
            e.extra["backing_container_duration_s"] = cdur
    songs = load_songs(registry_path)
    old = songs.get(song_id)
    if old is not None:
        e.extra = {**old.extra, **e.extra}
        e.gt = e.gt or old.gt
        e.notes = e.notes or old.notes
        e.backing_note = e.backing_note or old.backing_note
    songs[song_id] = e
    save_songs(songs, registry_path)
    return e


# ------------------------------------------------------------------------------------------ signal DSP


def _decode(path: Path, tmp: Path, tag: str) -> np.ndarray:
    import soundfile as sf

    from gtab.ingest.decode import decode_to_f32

    wav = tmp / f"{tag}.wav"
    decode_to_f32(Path(path), wav, sr=SR)
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    wav.unlink(missing_ok=True)
    assert sr == SR
    return x


def _backing_kbps(path: Path) -> int:
    from gtab.ingest.decode import probe

    br = probe(path).get("bit_rate") or 128000
    return max(64, min(320, int(round(br / 1000))))


def _codec_floor(original: Path, kbps: int, tmp: Path) -> np.ndarray:
    """The original encoded to AAC at ``kbps`` (ffmpeg native encoder) and decoded again."""
    from gtab import winjob
    from gtab.ingest.decode import tool_path

    m4a = tmp / "floor.m4a"
    args = [tool_path("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", str(original),
            "-map", "0:a:0", "-vn", "-c:a", "aac", "-b:a", f"{kbps}k", str(m4a)]
    run = winjob.run_captured(args, timeout_s=900, env=paths.child_env())
    if run.returncode != 0 or not m4a.is_file():
        raise TierAError(f"코덱 바닥 기준용 AAC 인코딩에 실패했습니다: {run.stderr[-500:]}")
    try:
        return _decode(m4a, tmp, "floor")
    finally:
        m4a.unlink(missing_ok=True)


def onset_envelope(mono: np.ndarray, *, hop: int = HOP, n_fft: int = N_FFT) -> np.ndarray:
    """Spectral-flux onset envelope (log magnitude, half-wave rectified frame difference), 1 value per hop."""
    from scipy import fft as sfft

    x = np.asarray(mono, dtype=np.float32)
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    win = np.hanning(n_fft).astype(np.float32)
    env = np.zeros(n_frames, dtype=np.float64)
    prev = None
    block = 2048
    for f0 in range(0, n_frames, block):
        f1 = min(n_frames, f0 + block)
        idx = (np.arange(f0, f1)[:, None] * hop) + np.arange(n_fft)[None, :]
        frames = x[idx] * win
        mag = np.log1p(100.0 * np.abs(sfft.rfft(frames, axis=1, workers=1)))
        if prev is None:
            d = np.diff(mag, axis=0, prepend=mag[:1])
        else:
            d = np.diff(np.vstack([prev, mag]), axis=0)
        env[f0:f1] = np.maximum(d, 0.0).sum(axis=1)
        prev = mag[-1:]
    return env


def _zscore(e: np.ndarray) -> np.ndarray:
    # remove slow loudness trends (1 s moving average) so sections do not dominate the correlation
    k = 101
    trend = np.convolve(e, np.ones(k) / k, mode="same")
    y = e - trend
    s = y.std()
    return y / s if s > 0 else y


def coarse_offset(env_o: np.ndarray, env_b: np.ndarray, max_lag: int) -> dict[str, float]:
    """Lag L (frames) with env_b[n + L] ~ env_o[n], |L| <= max_lag; peak correlation and peak ratio."""
    from scipy import signal

    a, b = _zscore(env_o), _zscore(env_b)
    corr = signal.correlate(b, a, mode="full", method="fft")
    lags = signal.correlation_lags(len(b), len(a), mode="full")
    sel = np.abs(lags) <= max_lag
    corr, lags = corr[sel], lags[sel]
    i = int(np.argmax(corr))
    peak = float(corr[i])
    norm = math.sqrt(float((a * a).sum()) * float((b * b).sum())) or 1.0
    mask = np.abs(lags - lags[i]) > 10
    second = float(corr[mask].max()) if mask.any() else 0.0
    return {"lag_frames": int(lags[i]), "peak_corr": peak / norm,
            "peak_ratio": peak / second if second > 0 else float("inf")}


def _parabolic(y: np.ndarray, i: int) -> float:
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        den = a - 2 * b + c
        if den != 0:
            return float(0.5 * (a - c) / den)
    return 0.0


def fine_offset(o: np.ndarray, b: np.ndarray, coarse_samples: int, *, search: int = 2 * HOP,
                n_windows: int = N_FINE_WINDOWS, win_s: float = FINE_WIN_S) -> dict[str, Any]:
    """Sub-sample lag per window (b[n + lag] ~ o[n]); median, spread and linear drift over the song."""
    from scipy import signal

    usable = min(len(o), len(b) - coarse_samples) - max(0, -coarse_samples) - 2 * search
    win = min(int(win_s * SR), max(4096, usable // 2))  # short clips: shorter windows
    lo = max(0, -coarse_samples + search)
    hi = min(len(o), len(b) - coarse_samples - search) - win
    if hi <= lo:
        raise TierAError("원곡과 반주가 거의 겹치지 않습니다(정렬 불가).")
    starts = np.linspace(lo, hi, n_windows).astype(int)
    rows = []
    for s in starts:
        seg_o = o[s:s + win]
        if float(np.sqrt(np.mean(seg_o ** 2))) < 1e-4:
            continue  # silence: no information
        seg_b = b[s + coarse_samples - search: s + coarse_samples + win + search]
        c = signal.correlate(seg_b, seg_o, mode="valid", method="fft")  # len = 2*search + 1
        a = np.abs(c)  # an inverted backing correlates negatively: its peak is a minimum of c
        i = int(np.argmax(a))
        frac = _parabolic(a, i)
        norm = math.sqrt(float((seg_o ** 2).sum()) * float((seg_b[i:i + win] ** 2).sum())) or 1.0
        rows.append({"t_s": round(s / SR, 3), "lag_samples": round(coarse_samples - search + i + frac, 3),
                     "corr": round(float(c[i]) / norm, 4)})
    if not rows:
        raise TierAError("정렬할 소리가 있는 구간이 없습니다.")
    good = [r for r in rows if abs(r["corr"]) >= 0.3] or rows
    lags = np.array([r["lag_samples"] for r in good])
    ts = np.array([r["t_s"] for r in good])
    drift_ppm = float(np.polyfit(ts, lags, 1)[0] / SR * 1e6) if len(good) >= 3 and np.ptp(ts) > 0 else 0.0
    return {"lag_samples": float(np.median(lags)), "spread_samples": float(np.ptp(lags)), "drift_ppm": round(drift_ppm, 3),
            "windows": rows, "used": len(good)}


def fractional_shift(x: np.ndarray, lag: float, n_out: int, *, taps: int = 64) -> np.ndarray:
    """y[n] = x[n + lag] for n in [0, n_out) (zeros outside x), windowed-sinc for the fractional part."""
    from scipy import signal

    ilag = int(math.floor(lag))
    frac = lag - ilag
    y = np.zeros((n_out, x.shape[1]), dtype=np.float64)
    # integer part: y[n] = x[n + ilag]
    src0, dst0 = max(0, ilag), max(0, -ilag)
    n = min(len(x) - src0, n_out - dst0)
    if n <= 0:
        return y.astype(np.float32)
    y[dst0:dst0 + n] = x[src0:src0 + n]
    if frac > 1e-6:
        k = np.arange(-taps // 2 + 1, taps // 2 + 1)
        h = np.sinc(k - frac) * np.blackman(taps)
        h /= h.sum()
        # y[n] = sum_k h[k] * y0[n + k]  -> correlation with h == convolution with reversed h
        for ch in range(y.shape[1]):
            full = signal.oaconvolve(y[:, ch], h[::-1], mode="full")
            y[:, ch] = full[taps // 2: taps // 2 + n_out]
    return y.astype(np.float32)


def band_energies(x: np.ndarray, *, n_fft: int = 4096, hop: int = 2048) -> np.ndarray:
    """Energy per BANDS_HZ band of a mono signal (sum over frames)."""
    from scipy import fft as sfft

    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS_HZ]
    win = np.hanning(n_fft).astype(np.float32)
    out = np.zeros(len(BANDS_HZ))
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    for f0 in range(0, n_frames, 1024):
        f1 = min(n_frames, f0 + 1024)
        idx = (np.arange(f0, f1)[:, None] * hop) + np.arange(n_fft)[None, :]
        p = np.abs(sfft.rfft(x[idx] * win, axis=1, workers=1)) ** 2
        for j, m in enumerate(masks):
            out[j] += float(p[:, m].sum())
    return out


def _db(num: float, den: float) -> float | None:
    """Energy ratio in dB; None if the reference is silent; exact cancellation floors at -200 dB."""
    if den <= 0:
        return None
    return round(max(-200.0, 10 * math.log10(max(num, 1e-300) / den)), 2)


# ------------------------------------------------------------------------------------------ alignment


def align_backing(song_id: str, *, registry_path: Path | None = None, out_dir: Path | None = None,
                  max_offset_s: float = 30.0, codec_floor: bool = True,
                  progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Align the song's backing to the original; write ``residual_not_gt.wav`` + ``align.json`` + a README.

    Outputs go to ``data/eval/tierA/<id>/backing/`` (git-ignored ``data/``). Returns the report.
    """
    from gtab.audio import write_f32

    say = progress or (lambda s: log.info("%s", s))
    songs = load_songs(registry_path)
    if song_id not in songs:
        raise TierAError(f"등록되지 않은 곡입니다: {song_id} (목록: {', '.join(songs) or '없음'})")
    e = songs[song_id]
    if not e.backing:
        raise TierAError(f"{song_id}: 반주(backing)가 등록되어 있지 않습니다.")
    out = Path(out_dir or tier_a_dir() / song_id / "backing")
    out.mkdir(parents=True, exist_ok=True)
    # decoded float WAVs (~85 MB each) live next to the outputs, never in %TEMP% (app sandbox, M0 rule)
    with tempfile.TemporaryDirectory(prefix=".tmp-align-", dir=out) as td:
        say("디코딩: 원곡")
        o = _decode(Path(e.path), Path(td), "orig")
        say("디코딩: 반주")
        b = _decode(Path(e.backing), Path(td), "backing")
        floor = None
        if codec_floor:
            kbps = _backing_kbps(Path(e.backing))
            say(f"코덱 바닥 기준: 원곡을 AAC {kbps} kbps 로 다시 인코딩")
            floor_arr = _codec_floor(Path(e.path), kbps, Path(td))
            floor = align_arrays(o, floor_arr, max_offset_s=2.0)
            floor.pop("_residual")
            floor = {"encoder": f"ffmpeg aac {kbps}k", "offset_samples": floor["offset_samples"],
                     "levels": floor["levels"], "bands": floor["bands"],
                     "percentiles_db": floor["windows_5s"]["percentiles_db"], "windows": floor["windows_5s"]["all"]}
            del floor_arr
    report = align_arrays(o, b, max_offset_s=max_offset_s, progress=say)
    if floor is not None:
        report["codec_floor"] = floor
        report["excess_over_floor_db"] = _excess(report, floor)
        report["hint_ko"] = _hint_ko(report["bands"], 1.0, 1.0, report["windows_5s"]["percentiles_db"],
                                     level_db=report["levels"]["residual_rel_original_db"],
                                     excess=report["excess_over_floor_db"])
    say("잔차 쓰는 중")
    residual = report.pop("_residual")
    res_path = out / RESIDUAL_WAV
    write_f32(res_path, residual, SR)  # deterministic float WAV (no PEAK timestamp chunk), atomic
    (out / _LEGACY_RESIDUAL).unlink(missing_ok=True)  # same diagnostic under its old, unlabeled name
    atomic.write_text(out / README_NAME, _readme_text(song_id, e))
    report.update({
        "format": ALIGN_FORMAT, "song_id": song_id, "ground_truth": False, "warning_ko": NOT_GT_KO,
        "warning": NOT_GT_EN,
        "original": {"path": e.path, "codec": e.codec, "lossless": e.lossless, "duration_s": e.duration_s},
        "backing": {"path": e.backing, "codec": e.backing_codec, "duration_s": e.backing_duration_s},
        "residual_wav": res_path.name,
        "created_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    report = {k: report[k] for k in ["format", "song_id", "ground_truth", "warning_ko", "warning", "original",
                                     "backing"] + [k for k in report if k not in (
                                         "format", "song_id", "ground_truth", "warning_ko", "warning", "original",
                                         "backing")]}
    atomic.write_json(out / "align.json", report)
    _record_check(song_id, report, registry_path)
    return report


def _readme_text(song_id: str, e: SongEntry) -> str:
    orig = "무손실" if e.lossless else "손실 압축"
    return "\n".join([
        f"{RESIDUAL_WAV} 는 정답(ground truth)이 아닙니다 / NOT ground truth.",
        "",
        f"곡: {song_id} ({e.title} - {e.artist})",
        f"원곡: {e.path} ({orig}, {e.codec})",
        f"반주: {e.backing} ({e.backing_codec})",
        "",
        "잔차 = 원곡 - 게인 x 반주(시간 정렬 후). 반주를 만든 방법을 모르고 반주가 손실 압축이라,",
        "빠진 파트 + 코덱 잡음 + 처리 차이가 섞여 있습니다. 평가 정답이나 학습 데이터로 쓰지 마세요.",
        "반주에서 무엇이 빠졌는지(어느 악기, 어느 구간) 확인하는 진단용입니다. 수치는 align.json 에 있습니다.",
        "excess_over_floor_db 가 0 dB 근처면 반주는 원곡을 다시 인코딩한 것과 구별되지 않습니다(빠진 것 없음).",
        "",
        NOT_GT_EN,
        "Copyrighted audio: local use on this PC only; never commit or share.",
        "",
    ])


def _record_check(song_id: str, report: dict[str, Any], registry_path: Path | None) -> None:
    """Keep a one-line verdict in songs.yaml (``backing_check``) so `tier-a list` can show it."""
    songs = load_songs(registry_path)
    e = songs.get(song_id)
    if e is None:
        return
    ex = (report.get("excess_over_floor_db") or {}).get("overall")
    e.extra["backing_check"] = {
        "utc": report.get("created_utc"), "offset_s": report.get("offset_s"),
        "gain_db": (report.get("gain") or {}).get("db"),
        "residual_db": (report.get("levels") or {}).get("residual_rel_original_db"),
        "excess_over_codec_floor_db": ex,
        "removed_part_likely": None if ex is None else bool(ex >= 3.0),
        "report": f"{song_id}/backing/align.json"}
    save_songs(songs, registry_path)


def align_arrays(o: np.ndarray, b: np.ndarray, *, max_offset_s: float = 30.0,
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Core of :func:`align_backing` on (n, 2) float32 arrays at 44.1 kHz. Adds ``_residual`` (n_o, 2)."""
    say = progress or (lambda s: None)
    o = np.asarray(o, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    om, bm = o.mean(axis=1), b.mean(axis=1)
    say("onset 포락선 상호상관(대략 위치)")
    env_o, env_b = onset_envelope(om), onset_envelope(bm)
    coarse = coarse_offset(env_o, env_b, int(max_offset_s * SR / HOP))
    say("파형 상호상관(샘플 단위)")
    fine = fine_offset(om, bm, coarse["lag_frames"] * HOP)
    lag = fine["lag_samples"]

    say("게인 추정·잔차 계산")
    b_al = fractional_shift(b, lag, len(o))
    # overlap = where the shifted backing has real samples
    start = max(0, int(math.ceil(-lag)) + 64)
    end = min(len(o), int(math.floor(len(b) - lag)) - 64)
    ov = slice(start, max(start, end))
    oo, bb = o[ov].astype(np.float64), b_al[ov].astype(np.float64)
    g = float((oo * bb).sum() / max((bb * bb).sum(), 1e-20))
    g_ch = [float((oo[:, c] * bb[:, c]).sum() / max((bb[:, c] ** 2).sum(), 1e-20)) for c in range(o.shape[1])]
    residual = o.astype(np.float64) - g * b_al
    residual = residual.astype(np.float32)
    rr = residual[ov].astype(np.float64)

    e_o, e_r, e_b = float((oo ** 2).sum()), float((rr ** 2).sum()), float(((g * bb) ** 2).sum())
    mid_r, side_r = rr.mean(axis=1), (rr[:, 0] - rr[:, 1]) / 2 if rr.shape[1] > 1 else np.zeros(len(rr))
    mid_o = oo.mean(axis=1)
    bo = band_energies(mid_o.astype(np.float32))
    br = band_energies(mid_r.astype(np.float32))
    bands = [{"band": name, "lo_hz": lo, "hi_hz": hi, "residual_rel_db": _db(br[j], bo[j]),
              "share_of_residual": round(float(br[j] / br.sum()), 4) if br.sum() > 0 else None}
             for j, (name, (lo, hi)) in enumerate(zip(BAND_NAMES, BANDS_HZ))]
    # windowed residual level (5 s): where does the removed part play?
    w = 5 * SR
    wins = []
    for s0 in range(0, len(oo) - w + 1, w):
        eo = float((oo[s0:s0 + w] ** 2).sum())
        er = float((rr[s0:s0 + w] ** 2).sum())
        wins.append({"t_s": round((start + s0) / SR, 1), "residual_rel_db": _db(er, eo)})
    vals = np.array([x["residual_rel_db"] for x in wins if x["residual_rel_db"] is not None])
    pct = {f"p{q}": round(float(np.percentile(vals, q)), 2) for q in (10, 50, 90)} if len(vals) else {}
    hint = _hint_ko(bands, e_r, e_o, pct)
    return {
        "sr": SR,
        "offset_s": round(lag / SR, 6), "offset_samples": round(lag, 3),
        "offset_meaning": "backing time = original time + offset_s",
        "coarse": {"hop_s": HOP / SR, **{k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v)
                                         for k, v in coarse.items()}},
        "fine": fine,
        "gain": {"linear": round(g, 5), "db": round(20 * math.log10(abs(g)), 3) if g else None,
                 "per_channel_db": [round(20 * math.log10(abs(x)), 3) if x else None for x in g_ch],
                 "polarity": "same" if g >= 0 else "inverted"},
        "overlap_s": [round(start / SR, 3), round(end / SR, 3)],
        "levels": {"residual_rel_original_db": _db(e_r, e_o), "backing_rel_original_db": _db(e_b, e_o),
                   "residual_side_rel_mid_db": _db(float((side_r ** 2).sum()), float((mid_r ** 2).sum()))},
        "bands": bands,
        "windows_5s": {"percentiles_db": pct, "loudest": sorted(
            [x for x in wins if x["residual_rel_db"] is not None], key=lambda x: -x["residual_rel_db"])[:8],
            "all": wins},
        "hint_ko": hint,
        "_residual": residual,
    }


def _excess(rep: dict[str, Any], floor: dict[str, Any]) -> dict[str, Any]:
    def diff(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else round(a - b, 2)

    fb = {b["band"]: b["residual_rel_db"] for b in floor["bands"]}
    fw = {w["t_s"]: w["residual_rel_db"] for w in floor.get("windows", [])}
    wins = [{"t_s": w["t_s"], "excess_db": diff(w["residual_rel_db"], fw.get(w["t_s"]))}
            for w in rep["windows_5s"]["all"] if w["t_s"] in fw]
    vals = [w["excess_db"] for w in wins if w["excess_db"] is not None]
    return {"overall": diff(rep["levels"]["residual_rel_original_db"], floor["levels"]["residual_rel_original_db"]),
            "bands": {b["band"]: diff(b["residual_rel_db"], fb.get(b["band"])) for b in rep["bands"]},
            "active_ratio_3db": round(sum(v > 3.0 for v in vals) / len(vals), 3) if vals else None,
            "windows_5s": wins}


def _hint_ko(bands: list[dict[str, Any]], e_r: float, e_o: float, pct: dict[str, float], *,
             level_db: float | None = None, excess: dict[str, Any] | None = None) -> str:
    share = {b["band"]: (b["share_of_residual"] or 0.0) for b in bands}
    low = share.get("sub_kick_20_120", 0) + share.get("bass_120_400", 0)
    mid = share.get("lowmid_400_1k", 0) + share.get("mid_1k_2k5", 0) + share.get("presence_2k5_6k", 0)
    rel = level_db if level_db is not None else (10 * math.log10(e_r / e_o) if e_r > 0 and e_o > 0 else float("-inf"))
    parts = [f"잔차는 원곡 대비 {rel:.1f} dB."]
    if pct:
        parts.append(f"5초 창 잔차 수준 p10 {pct['p10']:.1f} / p50 {pct['p50']:.1f} / p90 {pct['p90']:.1f} dB.")
    if excess is not None and excess.get("overall") is not None:
        ex = excess["overall"]
        mids = [v for k, v in excess["bands"].items() if k in ("lowmid_400_1k", "mid_1k_2k5", "presence_2k5_6k")
                and v is not None]
        parts.append(f"같은 비트레이트로 다시 인코딩만 한 기준(코덱 바닥)보다 {ex:+.1f} dB"
                     + (f", 중역(400 Hz-6 kHz) {min(mids):+.1f}~{max(mids):+.1f} dB." if mids else "."))
        if excess.get("active_ratio_3db") is not None:
            parts.append(f"5초 창 중 코덱 바닥보다 3 dB 넘게 큰 곳 {100 * excess['active_ratio_3db']:.0f}% "
                         "(빠진 파트가 연주되는 구간 비율의 대략값).")
        if ex < 3:
            parts.append("코덱 잡음 수준과 비슷합니다: 이 반주는 원곡에서 거의 아무것도 빼지 않았을 수 있습니다.")
            return " ".join(parts + [f"{RESIDUAL_WAV} 를 귀로 들어 확인하세요."])
    if rel > -3:
        parts.append("잔차가 너무 커서 정렬이나 반주 제작 방식(다른 믹스·EQ)이 어긋났을 수 있습니다.")
    elif low > 0.6:
        parts.append("잔차 에너지가 400 Hz 아래에 몰려 있어 베이스·킥 쪽이 빠진 것으로 보입니다.")
    elif mid > 0.6:
        parts.append("잔차 에너지가 400 Hz-6 kHz 에 몰려 있어 기타(또는 건반·보컬) 같은 중역 파트가 빠진 것으로 보입니다.")
    parts.append(f"악기 판별은 {RESIDUAL_WAV} 를 귀로 들어 확인하세요.")
    return " ".join(parts)


def summary_row(report: dict[str, Any]) -> dict[str, Any]:
    return {"song": report.get("song_id"), "offset_s": report.get("offset_s"),
            "gain_db": (report.get("gain") or {}).get("db"),
            "residual_db": (report.get("levels") or {}).get("residual_rel_original_db"),
            "drift_ppm": (report.get("fine") or {}).get("drift_ppm"),
            "excess_db": (report.get("excess_over_floor_db") or {}).get("overall"),
            "spread": (report.get("fine") or {}).get("spread_samples"),
            "corr": (report.get("coarse") or {}).get("peak_corr")}


def entry_dict(e: SongEntry) -> dict[str, Any]:
    return asdict(e)
