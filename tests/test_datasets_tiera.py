"""Tier A candidate registry (songs.yaml) and the backing-vs-original alignment diagnostic."""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf

from bandscribe.datasets import tiera

SR = tiera.SR


def _music(seconds: float, seed: int = 1) -> np.ndarray:
    """Stereo 'backing' with percussive onsets (clicks + decaying tones), so onset envelopes are informative."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = np.zeros((n, 2))
    t = np.arange(int(0.25 * SR)) / SR
    for k, start in enumerate(np.cumsum(rng.uniform(0.18, 0.45, size=int(seconds / 0.18)))):
        s = int(start * SR)
        if s + len(t) >= n:
            break
        f = rng.choice([110, 147, 196, 262, 330])
        burst = np.sin(2 * np.pi * f * t) * np.exp(-t * 12) + 0.3 * rng.standard_normal(len(t)) * np.exp(-t * 40)
        x[s:s + len(t), 0] += burst * rng.uniform(0.3, 0.8)
        x[s:s + len(t), 1] += burst * rng.uniform(0.3, 0.8)
    return (0.3 * x).astype(np.float32)


def _guitar(seconds: float, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    g = np.zeros((n, 2))
    t = np.arange(int(0.4 * SR)) / SR
    for start in np.arange(0.5, seconds - 0.5, 0.5):
        s = int(start * SR)
        f = rng.choice([392, 440, 494, 587, 659])
        tone = sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3)) * np.exp(-t * 4)
        g[s:s + len(t), 0] += 0.15 * tone
        g[s:s + len(t), 1] += 0.05 * tone  # panned left
    return g.astype(np.float32)


def test_align_arrays_offset_gain_and_residual():
    dur = 30.0
    bg, gtr = _music(dur), _guitar(dur)
    original = bg + gtr
    lead = 54443  # backing starts 1.2345 s earlier than the song (its time = original time + offset)
    backing = np.concatenate([np.zeros((lead, 2), np.float32), 0.8 * bg, np.zeros((SR, 2), np.float32)])
    rep = tiera.align_arrays(original, backing, max_offset_s=5.0)
    assert rep["offset_samples"] == pytest.approx(lead, abs=0.05)
    assert rep["offset_s"] == pytest.approx(lead / SR, abs=1e-5)
    assert rep["gain"]["db"] == pytest.approx(20 * math.log10(1 / 0.8), abs=0.05)
    assert rep["gain"]["polarity"] == "same" and abs(rep["fine"]["drift_ppm"]) < 1
    res = rep["_residual"]
    ov = slice(int(rep["overlap_s"][0] * SR), int(rep["overlap_s"][1] * SR))
    err = res[ov] - gtr[ov]
    assert 10 * math.log10((err ** 2).sum() / (gtr[ov] ** 2).sum()) < -30  # residual == the removed part
    # the removed 'guitar' lives at 392-2k Hz: the residual is mid-heavy, and panned (side > 0)
    share = {b["band"]: b["share_of_residual"] for b in rep["bands"]}
    assert share["lowmid_400_1k"] + share["mid_1k_2k5"] + share["presence_2k5_6k"] > 0.6


def test_align_negative_offset():
    bg = _music(20.0, seed=5)
    rep = tiera.align_arrays(bg, bg[SR:], max_offset_s=3.0)  # backing starts 1 s into the song
    assert rep["offset_samples"] == pytest.approx(-SR, abs=0.05)
    assert rep["gain"]["db"] == pytest.approx(0.0, abs=0.01)


def test_align_inverted_polarity():
    bg = _music(20.0, seed=6)
    rep = tiera.align_arrays(bg[SR:], -bg[: 15 * SR], max_offset_s=3.0)
    assert rep["offset_samples"] == pytest.approx(SR, abs=0.05)
    assert rep["gain"]["polarity"] == "inverted"
    assert rep["levels"]["residual_rel_original_db"] < -60


def test_fractional_shift_matches_analytic():
    t = np.arange(SR) / SR
    x = np.stack([np.sin(2 * np.pi * 1000 * t), np.cos(2 * np.pi * 250 * t)], axis=1).astype(np.float32)
    y = tiera.fractional_shift(x, 10.37, len(x) - 200)
    tt = (np.arange(len(y)) + 10.37) / SR
    ref = np.stack([np.sin(2 * np.pi * 1000 * tt), np.cos(2 * np.pi * 250 * tt)], axis=1)
    core = slice(100, len(y) - 100)
    assert np.max(np.abs(y[core] - ref[core])) < 2e-3


def _wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, SR, subtype="FLOAT")


def test_registry_roundtrip_and_end_to_end(tmp_path):
    pytest.importorskip("yaml")
    reg = tmp_path / "tierA" / "songs.yaml"
    bg, gtr = _music(12.0, seed=3), _guitar(12.0, seed=4)
    _wav(tmp_path / "src" / "노래.wav", bg + gtr)
    _wav(tmp_path / "src" / "노래_backing.wav", np.concatenate([np.zeros((777, 2), np.float32), bg]))
    e = tiera.add_song("song_a", tmp_path / "src" / "노래.wav", title="青と夏", artist="Mrs. GREEN APPLE",
                       backing=tmp_path / "src" / "노래_backing.wav", notes="무손실", registry_path=reg)
    assert e.lossless and e.duration_s == pytest.approx(12.0, abs=0.01) and e.backing_duration_s > 12.0
    text = reg.read_text(encoding="utf-8")
    assert "저작권" in text and "青と夏" in text
    songs = tiera.load_songs(reg)
    assert songs["song_a"].title == "青と夏" and songs["song_a"].gt is None
    # re-adding keeps user notes / extra keys
    reg.write_text(text.replace('notes: "무손실"', 'notes: "무손실"\n    my_field: "keep me"'), encoding="utf-8")
    tiera.add_song("song_a", tmp_path / "src" / "노래.wav", title="青と夏", artist="Mrs. GREEN APPLE",
                   backing=tmp_path / "src" / "노래_backing.wav", registry_path=reg)
    s2 = tiera.load_songs(reg)["song_a"]
    assert s2.extra == {"my_field": "keep me"} and s2.notes == "무손실"

    out = tmp_path / "tierA" / "song_a" / "backing"
    out.mkdir(parents=True)
    (out / "residual.wav").write_bytes(b"old")  # output of an earlier version
    rep = tiera.align_backing("song_a", registry_path=reg, out_dir=out, max_offset_s=2.0)
    assert rep["ground_truth"] is False and "정답이 아닙니다" in rep["warning_ko"]
    assert rep["offset_samples"] == pytest.approx(777, abs=0.05)
    res_wav = out / tiera.RESIDUAL_WAV
    assert res_wav.name == "residual_not_gt.wav" and (out / "align.json").is_file()
    assert not (out / "residual.wav").exists()  # the legacy unlabeled name is gone
    readme = (out / "README_NOT_GROUND_TRUTH.txt").read_text(encoding="utf-8")
    assert "정답(ground truth)이 아닙니다" in readme and "NOT ground truth" in readme and "무손실" in readme
    assert not [p for p in out.iterdir() if p.name.startswith((".tmp-", "."))]
    info = sf.info(str(res_wav))
    assert info.channels == 2 and info.frames == int(12.0 * SR) and info.subtype == "FLOAT"
    chk = tiera.load_songs(reg)["song_a"].extra["backing_check"]
    assert chk["removed_part_likely"] is True and chk["report"] == "song_a/backing/align.json"
    first = res_wav.read_bytes()
    assert b"PEAK" not in first[:200]  # bandscribe.audio.write_f32: no timestamped PEAK chunk
    rep2 = tiera.align_backing("song_a", registry_path=reg, out_dir=out, max_offset_s=2.0, codec_floor=False)
    assert res_wav.read_bytes() == first and rep2["offset_samples"] == rep["offset_samples"]  # deterministic
    assert rep["original"]["lossless"] is True
    # a lossless original re-encoded to AAC is the codec floor; the backing here removed a real part
    assert rep["codec_floor"]["encoder"].startswith("ffmpeg aac")
    assert rep["excess_over_floor_db"]["overall"] > 3


def test_bad_ids_and_missing(tmp_path):
    pytest.importorskip("yaml")
    with pytest.raises(tiera.TierAError, match="ASCII"):
        tiera.add_song("괴수", tmp_path / "x.wav", title="t", artist="a", registry_path=tmp_path / "s.yaml")
    with pytest.raises(tiera.TierAError, match="없습니다"):
        tiera.add_song("x", tmp_path / "x.wav", title="t", artist="a", registry_path=tmp_path / "s.yaml")
    with pytest.raises(tiera.TierAError, match="등록되지 않은"):
        tiera.align_backing("nope", registry_path=tmp_path / "s.yaml", out_dir=tmp_path / "o")
