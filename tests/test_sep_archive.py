"""bandscribe.sep.archive: 24-bit FLAC archive of float stems, restore, and the b7 error measure (relative to the mix)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bandscribe.sep import archive

SR = 44100


def _write(path: Path, x: np.ndarray) -> Path:
    sf.write(str(path), x.astype(np.float32), SR, subtype="FLOAT")
    return path


def _rms_db(x: np.ndarray) -> float:
    return 10 * math.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-20)


def _music(n: int, seed: int = 0, level: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    x = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t + 1)], axis=1) * level
    return x + rng.standard_normal((n, 2)) * level * 0.1


def test_roundtrip_below_minus_100_db_rel_mix(tmp_path: Path) -> None:
    mix = _music(SR * 3)
    stem = _write(tmp_path / "guitar.wav", mix * 0.5)
    flac = tmp_path / "a" / "guitar.flac"
    gain = archive.archive(stem, flac)
    assert flac.is_file() and archive.gain_path(flac).is_file()
    side = json.loads(archive.gain_path(flac).read_text(encoding="utf-8"))
    assert side["bits"] == 24 and float(side["gain"]) == gain
    info = sf.info(str(flac))
    assert info.subtype == "PCM_24" and info.format == "FLAC"
    rest = archive.restore(flac)
    orig, _ = sf.read(str(stem), dtype="float32", always_2d=True)
    assert rest.shape == orig.shape and rest.dtype == np.float32
    # peak normalised to -0.1 dBFS before quantising
    assert abs(float(np.max(np.abs(orig))) * gain - 10 ** (-0.1 / 20)) < 1e-9
    err = archive.roundtrip_error_db(stem, ref_rms_db=_rms_db(mix))
    assert err < -100.0, err
    assert err < -130.0  # 24-bit: ~-149 dBFS noise; well below the criterion


def test_peaks_above_full_scale_survive(tmp_path: Path) -> None:
    mix = _music(SR * 2, level=1.4)  # peaks > 1.0 (float stems of a loud master)
    stem = _write(tmp_path / "loud.wav", mix)
    assert float(np.max(np.abs(mix))) > 1.0
    archive.archive(stem, tmp_path / "loud.flac")
    rest = archive.restore(tmp_path / "loud.flac")
    assert float(np.max(np.abs(rest))) > 1.0
    assert archive.roundtrip_error_db(stem, ref_rms_db=_rms_db(mix)) < -100.0


def test_sparse_stem_with_high_crest_factor_passes(tmp_path: Path) -> None:
    """A stem 55 dB below the mix with a crest factor > 50 dB: self-relative it would fail, mix-relative it passes."""
    n = SR * 4
    mix = _music(n)
    mix_db = _rms_db(mix)
    x = np.zeros((n, 2))
    x[SR, :] = 1.0  # one click: huge crest factor
    x *= 10 ** ((mix_db - 55.0 - _rms_db(x)) / 20)  # stem RMS = mix - 55 dB
    crest = 20 * math.log10(float(np.max(np.abs(x)))) - _rms_db(x)
    assert crest > 50.0
    stem = _write(tmp_path / "sparse.wav", x)
    chk = archive.roundtrip_check(stem, mix_rms_db=mix_db)
    assert chk["status"] == "ok" and chk["error_db"] < -100.0, chk
    assert abs(chk["rel_rms_db"] + 55.0) < 0.01


def test_near_silent_stem_is_skipped_sparse(tmp_path: Path) -> None:
    mix = _music(SR * 2)
    stem = _write(tmp_path / "piano.wav", _music(SR * 2, seed=3) * 10 ** (-65 / 20))
    chk = archive.roundtrip_check(stem, mix_rms_db=_rms_db(mix))
    assert chk["status"] == "skipped_sparse" and chk["error_db"] is None and chk["rel_rms_db"] < -60


def test_silent_stem_archives(tmp_path: Path) -> None:
    stem = _write(tmp_path / "zero.wav", np.zeros((SR, 2)))
    assert archive.archive(stem, tmp_path / "zero.flac") == 1.0
    assert not np.any(archive.restore(tmp_path / "zero.flac"))


def test_failure_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mix = _music(SR)
    stem = _write(tmp_path / "g.wav", mix)
    monkeypatch.setattr(archive, "roundtrip_error_db", lambda *a, **k: -80.0)
    assert archive.roundtrip_check(stem, mix_rms_db=_rms_db(mix))["status"] == "fail"
