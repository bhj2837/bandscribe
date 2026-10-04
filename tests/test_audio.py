"""bandscribe.audio: deterministic float32 WAV I/O (M1_M2_SPEC 1.4)."""
from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bandscribe import audio


def _stereo(n: int = 44100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, 2)) * 0.3).astype(np.float32)


def test_two_writes_are_byte_identical(tmp_path: Path) -> None:
    x = _stereo()
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    audio.write_f32(a, x)
    time.sleep(1.1)  # libsndfile's PEAK chunk carries a timestamp: this would differ with soundfile.write
    audio.write_f32(b, x)
    assert a.read_bytes() == b.read_bytes()


def test_header_is_fmt_fact_data_without_peak(tmp_path: Path) -> None:
    p = tmp_path / "x.wav"
    x = _stereo(1000)
    audio.write_f32(p, x, 48000)
    raw = p.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    assert struct.unpack("<I", raw[4:8])[0] == len(raw) - 8
    chunks, pos = [], 12
    while pos < len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        chunks.append((cid, size))
        pos += 8 + size
    assert chunks == [(b"fmt ", 18), (b"fact", 4), (b"data", 1000 * 2 * 4)]
    tag, ch, sr, byte_rate, align, bits, cb = struct.unpack("<HHIIHHH", raw[20:38])
    assert (tag, ch, sr, byte_rate, align, bits, cb) == (3, 2, 48000, 48000 * 8, 8, 32, 0)
    assert b"PEAK" not in raw[:100]


def test_soundfile_reads_back_bit_exact_including_overs(tmp_path: Path) -> None:
    x = _stereo()
    x[10] = [1.75, -2.5]  # loud masters decode above full scale; nothing may clip
    p = tmp_path / "x.wav"
    audio.write_f32(p, x)
    y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == audio.SR
    np.testing.assert_array_equal(y, x)
    assert sf.info(str(p)).subtype == "FLOAT"
    y2, sr2 = audio.read_f32(p)
    np.testing.assert_array_equal(y2, x)
    assert sr2 == audio.SR and y2.dtype == np.float32


def test_mono_input_and_float64_input(tmp_path: Path) -> None:
    m = np.linspace(-1, 1, 500)  # float64 (n,)
    p = tmp_path / "m.wav"
    audio.write_f32(p, m, 22050)
    assert audio.info(p) == (500, 22050, 1)
    y, _ = audio.read_f32(p)
    assert y.shape == (500, 1)
    np.testing.assert_array_equal(y[:, 0], m.astype(np.float32))


def test_nan_is_rejected_and_nothing_written(tmp_path: Path) -> None:
    x = _stereo(100)
    x[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        audio.write_f32(tmp_path / "bad.wav", x)
    assert list(tmp_path.iterdir()) == []


def test_f32writer_equals_write_f32(tmp_path: Path) -> None:
    x = _stereo(100_003)
    audio.write_f32(tmp_path / "a.wav", x)
    with audio.F32Writer(tmp_path / "b.wav", audio.SR, 2) as w:
        for s in range(0, len(x), 4096):
            w.write(x[s:s + 4096])
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()
    # mono writer accepts (n,) and (n, 1) blocks
    m = x[:, 0]
    audio.write_f32(tmp_path / "c.wav", m)
    w2 = audio.F32Writer(tmp_path / "d.wav", audio.SR, 1)
    w2.write(m[:500])
    w2.write(m[500:].reshape(-1, 1))
    w2.close()
    w2.close()  # idempotent
    assert (tmp_path / "c.wav").read_bytes() == (tmp_path / "d.wav").read_bytes()


def test_f32writer_is_atomic_and_aborts_on_error(tmp_path: Path) -> None:
    p = tmp_path / "x.wav"
    w = audio.F32Writer(p, audio.SR, 2)
    w.write(_stereo(10))
    assert not p.exists()  # nothing visible before close
    w.close()
    assert p.is_file()
    with pytest.raises(RuntimeError):
        with audio.F32Writer(tmp_path / "y.wav", audio.SR, 2) as w3:
            w3.write(_stereo(10))
            raise RuntimeError("boom")
    assert not (tmp_path / "y.wav").exists()
    assert sorted(q.name for q in tmp_path.iterdir()) == ["x.wav"]  # no temp files left
    w4 = audio.F32Writer(tmp_path / "z.wav", audio.SR, 2)
    with pytest.raises(ValueError, match="channel"):
        w4.write(np.zeros((5, 1), dtype=np.float32))
    w4.abort()


def test_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "e.wav"
    audio.write_f32(p, np.zeros((0, 2), dtype=np.float32))
    assert audio.info(p) == (0, audio.SR, 2)
    assert list(audio.iter_blocks(p)) == []


def test_iter_blocks_streams_2d_float32(tmp_path: Path) -> None:
    x = _stereo(10_000)
    p = tmp_path / "x.wav"
    audio.write_f32(p, x)
    blocks = list(audio.iter_blocks(p, block_frames=3000))
    assert [b.shape for b in blocks] == [(3000, 2), (3000, 2), (3000, 2), (1000, 2)]
    assert all(b.dtype == np.float32 for b in blocks)
    np.testing.assert_array_equal(np.concatenate(blocks), x)


def test_to_mono() -> None:
    x = np.array([[1.0, 0.0], [0.25, 0.75], [-1.0, 1.0]], dtype=np.float32)
    m = audio.to_mono(x)
    assert m.dtype == np.float32 and m.shape == (3,)
    np.testing.assert_array_equal(m, [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(audio.to_mono(x[:, :1]), x[:, 0])
    np.testing.assert_array_equal(audio.to_mono(x[:, 0]), x[:, 0])
    # block-wise == whole-file (the stems stage computes views block by block)
    y = _stereo(5000, seed=3)
    np.testing.assert_array_equal(np.concatenate([audio.to_mono(y[:1234]), audio.to_mono(y[1234:])]), audio.to_mono(y))


def test_pcm_sha256_matches_ingest_and_ignores_header(tmp_path: Path) -> None:
    from bandscribe.ingest.decode import pcm_sha256 as ingest_sha

    x = _stereo(70_000)
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    audio.write_f32(a, x)
    sf.write(str(b), x, audio.SR, subtype="FLOAT")  # different header (PEAK chunk), same samples
    assert a.read_bytes() != b.read_bytes()
    assert audio.pcm_sha256(a) == audio.pcm_sha256(b) == ingest_sha(a)
    assert audio.pcm_sha256(a, block_frames=1000) == audio.pcm_sha256(a)


def test_rms_db() -> None:
    assert audio.rms_db(np.ones(100)) == pytest.approx(0.0, abs=1e-9)
    assert audio.rms_db(np.full(10, 0.1)) == pytest.approx(-20.0, abs=1e-9)
    assert audio.rms_db(np.zeros(5)) == pytest.approx(-200.0)
    assert audio.rms_db(np.zeros(0)) == pytest.approx(-200.0)
