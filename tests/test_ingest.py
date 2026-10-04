"""bandscribe.ingest: decode, loudness, flags and the cached, atomic ingest of local files (M0_SPEC 12, 15).

Audio fixtures are synthesised and encoded with ffmpeg at test time (tests/fixtures/synth_audio.py).
Every test uses its own JobStore under tmp_path; nothing touches D:\\gtab\\data.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fixtures import synth_audio
from bandscribe import atomic, paths
from bandscribe.config import DEFAULTS_PATH, Config
from bandscribe.ingest import IngestError, IngestResult, content_key, decode, flags, ingest, ingest_params, loudness
from bandscribe.jobs.store import JobStore

LOSSY = ("mp3", "m4a", "opus")
ALL_FORMATS = ("wav", "flac", *LOSSY)
CEILING = -1.0


@pytest.fixture(scope="session")
def audio(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return synth_audio.make_fixture_set(tmp_path_factory.mktemp("audio"))


@pytest.fixture(scope="session")
def decoded(audio: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Every fixture decoded once with decode_to_f32 (shared by the read-only tests)."""
    d = tmp_path_factory.mktemp("decoded")
    out = {}
    for name, src in audio.items():
        dst = d / f"{name.replace('.', '_')}.wav"
        decode.decode_to_f32(src, dst)
        out[name] = dst
    return out


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs")


@pytest.fixture
def cfg() -> Config:
    return Config()


def _complete_inputs(root: Path) -> list[Path]:
    return sorted(root.glob("*/input/meta.json"))


def _temp_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob(".tmp-*") if p.is_dir())


# ---------------------------------------------------------------------------------------- decode


def test_probe_reports_source_format(audio: dict[str, Path]) -> None:
    p = decode.probe(audio["normal.wav"])
    assert p["codec_name"] == "pcm_s16le"
    assert p["sample_rate"] == 44100 and p["channels"] == 2
    assert p["duration"] == pytest.approx(synth_audio.SECONDS, abs=0.01)
    assert p["has_video"] is False and p["n_audio_streams"] == 1

    o = decode.probe(audio["normal.opus"])
    assert o["codec_name"] == "opus" and o["sample_rate"] == 48000
    assert o["bit_rate"] and 80_000 < o["bit_rate"] < 200_000
    assert decode.probe(audio["mono.wav"])["channels"] == 1


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_decode_gives_float32_stereo_44k(fmt: str, decoded: dict[str, Path]) -> None:
    info = sf.info(str(decoded[f"normal.{fmt}"]))
    assert info.samplerate == 44100
    assert info.channels == 2
    assert info.subtype == "FLOAT"
    # lossy codecs add a few ms of priming/padding at most
    assert info.duration == pytest.approx(synth_audio.SECONDS, abs=0.1)


def test_decode_mono_source_duplicates_channels(decoded: dict[str, Path]) -> None:
    y, sr = sf.read(str(decoded["mono.wav"]), dtype="float32", always_2d=True)
    assert sr == 44100 and y.shape[1] == 2
    assert np.array_equal(y[:, 0], y[:, 1])


def test_decode_resamples_nonstandard_rate(decoded: dict[str, Path]) -> None:
    info = sf.info(str(decoded["sr32k.wav"]))
    assert info.samplerate == 44100 and info.duration == pytest.approx(synth_audio.SECONDS, abs=0.01)


@pytest.mark.parametrize("fmt", ["opus", "m4a"])
def test_loud_master_overs_survive_decode(fmt: str, audio: dict[str, Path], decoded: dict[str, Path]) -> None:
    src, _ = sf.read(str(audio["loud.wav"]), dtype="float32")
    assert np.max(np.abs(src)) <= 1.0
    assert -9.5 < synth_audio.rms_dbfs(src) < -7.0  # "loud master" territory
    y, _ = sf.read(str(decoded[f"loud.{fmt}"]), dtype="float32")
    peak = float(np.max(np.abs(y)))
    assert peak > 1.0, f"{fmt}: decoded peak {peak:.4f} - overs were clipped"
    assert int(np.sum(np.abs(y) > 1.0)) > 10


def test_decode_rejects_non_audio(tmp_path: Path) -> None:
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not audio", encoding="utf-8")
    with pytest.raises(decode.DecodeError):
        decode.probe(bogus)
    with pytest.raises(decode.DecodeError):
        decode.decode_to_f32(bogus, tmp_path / "out.wav")


def test_pcm_sha256_ignores_container_and_name(audio: dict[str, Path], decoded: dict[str, Path], tmp_path: Path) -> None:
    h_wav = decode.pcm_sha256(decoded["normal.wav"])
    assert decode.pcm_sha256(decoded["normal.flac"]) == h_wav
    renamed = tmp_path / "other name.wav"
    shutil.copyfile(decoded["normal.wav"], renamed)
    assert decode.pcm_sha256(renamed) == h_wav
    assert decode.pcm_sha256(decoded["normal.mp3"]) != h_wav
    # equals a hash of the raw float32 sample bytes
    y, _ = sf.read(str(decoded["normal.wav"]), dtype="float32", always_2d=True)
    assert h_wav == atomic.sha256_bytes(np.ascontiguousarray(y).tobytes())


# -------------------------------------------------------------------------------------- loudness


@pytest.mark.parametrize("name", ["normal.wav", "normal.mp3", "normal.m4a", "normal.opus", "loud.opus", "loud.m4a", "mono.wav"])
def test_pyloudnorm_matches_ffmpeg_ebur128(name: str, decoded: dict[str, Path]) -> None:
    m = loudness.measure(decoded[name])
    assert synth_audio.SECONDS >= 10
    assert m["integrated_lufs"] is not None and m["ffmpeg_integrated_lufs"] is not None
    assert abs(m["integrated_lufs"] - m["ffmpeg_integrated_lufs"]) <= 0.1
    # a true peak is never below the sample peak (4x oversampling finds inter-sample peaks)
    assert m["true_peak_dbfs"] >= m["sample_peak_dbfs"] - 0.01
    assert m["true_peak_bound_dbfs"] >= m["true_peak_dbfs"]


def test_measure_silence_is_none(tmp_path: Path) -> None:
    wav = tmp_path / "silence.wav"
    sf.write(str(wav), np.zeros((44100 * 3, 2), dtype=np.float32), 44100, subtype="FLOAT")
    m = loudness.measure(wav)
    assert m["integrated_lufs"] is None and m["ffmpeg_integrated_lufs"] is None
    assert m["sample_peak_dbfs"] is None and m["true_peak_dbfs"] is None
    assert loudness.apply_ceiling(wav, tmp_path / "out.wav", CEILING) == 0.0


@pytest.mark.parametrize("fmt", ["opus", "m4a"])
def test_apply_ceiling_brings_true_peak_under_ceiling(fmt: str, decoded: dict[str, Path], tmp_path: Path) -> None:
    src = decoded[f"loud.{fmt}"]
    before = loudness.measure(src)
    assert before["true_peak_dbfs"] > 0.0
    out = tmp_path / "limited.wav"
    gain = loudness.apply_ceiling(src, out, CEILING, measured=before)
    assert gain < 0.0
    after = loudness.measure(out)
    assert after["true_peak_dbfs"] <= CEILING
    assert after["true_peak_bound_dbfs"] <= CEILING + 0.01
    assert after["true_peak_dbfs"] == pytest.approx(CEILING, abs=0.06)  # gain is not wildly conservative
    # plain float gain, written as float32, nothing else changed
    x, _ = sf.read(str(src), dtype="float32")
    y, _ = sf.read(str(out), dtype="float32")
    assert sf.info(str(out)).subtype == "FLOAT"
    np.testing.assert_allclose(y, x * np.float32(10 ** (gain / 20)), rtol=1e-6, atol=1e-7)


def test_apply_ceiling_measures_itself_and_works_in_place(decoded: dict[str, Path], tmp_path: Path) -> None:
    wav = tmp_path / "inplace.wav"
    shutil.copyfile(decoded["loud.opus"], wav)
    gain = loudness.apply_ceiling(wav, wav, CEILING)
    assert gain < 0.0
    assert loudness.measure(wav)["true_peak_dbfs"] <= CEILING
    assert not list(tmp_path.glob(".*.tmp"))


def test_apply_ceiling_leaves_quiet_audio_alone(decoded: dict[str, Path], tmp_path: Path) -> None:
    src = decoded["normal.wav"]  # -6 dBFS peak
    out = tmp_path / "same.wav"
    assert loudness.apply_ceiling(src, out, CEILING) == 0.0
    x, _ = sf.read(str(src), dtype="float32")
    y, _ = sf.read(str(out), dtype="float32")
    assert np.array_equal(x, y)


# ----------------------------------------------------------------------------------------- flags


def _flags(decoded: dict[str, Path], audio: dict[str, Path], name: str, **kw: object) -> dict:
    return flags.compute_flags(decoded[name], decode.probe(audio[name]), Config(), **kw)


def test_flags_normal_stereo(decoded: dict[str, Path], audio: dict[str, Path]) -> None:
    f = _flags(decoded, audio, "normal.wav")
    assert f["quasi_mono"] is False and f["mono_source"] is False
    assert f["side_mid_db"] > -10
    assert f["lossy_cutoff"] is False and f["lowpass_hz_est"] is None
    assert f["nonstandard_sr"] is False and f["drc"] is False


def test_flags_mono_source(decoded: dict[str, Path], audio: dict[str, Path]) -> None:
    f = _flags(decoded, audio, "mono.wav")
    assert f["mono_source"] is True
    assert f["quasi_mono"] is True
    assert f["side_mid_db"] == flags.SIDE_MID_FLOOR_DB


def test_flags_quasi_mono(decoded: dict[str, Path], audio: dict[str, Path]) -> None:
    f = _flags(decoded, audio, "quasi.wav")
    assert f["mono_source"] is False
    assert f["quasi_mono"] is True
    assert f["side_mid_db"] == pytest.approx(-40.0, abs=1.0)


def test_flags_quasi_mono_threshold_comes_from_config(decoded: dict[str, Path], audio: dict[str, Path]) -> None:
    cfg = Config.model_validate({"ingest": {"quasi_mono_side_mid_db": -50.0}})
    f = flags.compute_flags(decoded["quasi.wav"], decode.probe(audio["quasi.wav"]), cfg)
    assert f["quasi_mono"] is False


@pytest.mark.parametrize(("name", "lossy", "edge"), [("normal.mp3", True, (16000, 17500)), ("normal.m4a", True, (16500, 18500)), ("normal.opus", False, (19500, 21500))])
def test_flags_lossy_cutoff(name: str, lossy: bool, edge: tuple[int, int], decoded: dict[str, Path], audio: dict[str, Path]) -> None:
    f = _flags(decoded, audio, name)
    assert f["lossy_cutoff"] is lossy
    assert f["lowpass_hz_est"] is not None and edge[0] <= f["lowpass_hz_est"] <= edge[1]


def test_flags_nonstandard_rate_is_not_a_lossy_cutoff(decoded: dict[str, Path], audio: dict[str, Path]) -> None:
    f = _flags(decoded, audio, "sr32k.wav")
    assert f["nonstandard_sr"] is True and f["source_sample_rate"] == 32000
    # the 16 kHz edge is the 32 kHz rate's own band limit, not a codec lowpass
    assert f["lossy_cutoff"] is False


def test_flags_drc_from_format_id(decoded: dict[str, Path], audio: dict[str, Path]) -> None:
    assert _flags(decoded, audio, "normal.opus", format_id="251-drc")["drc"] is True
    assert _flags(decoded, audio, "normal.opus", format_id="251")["drc"] is False


def test_estimate_lowpass_on_constructed_spectra() -> None:
    freqs = np.linspace(0, 22050, 4097)
    flat = np.full_like(freqs, 1e-6)
    assert flags.estimate_lowpass(freqs, flat)[0] is None
    walled = np.where(freqs < 16000, 1e-6, 1e-14)
    edge, drop = flags.estimate_lowpass(freqs, walled)
    assert edge == pytest.approx(16000, abs=200) and drop >= 70
    # a steep but natural roll-off (-6 dB per kHz above 5 kHz) is not a brick wall
    rolloff = 1e-6 * 10 ** (-0.6 * np.maximum(freqs - 5000, 0) / 1000)
    assert flags.estimate_lowpass(freqs, rolloff)[0] is None


# ---------------------------------------------------------------------------------------- ingest


def test_ingest_file_builds_complete_input(audio: dict[str, Path], store: JobStore, cfg: Config) -> None:
    src = audio["normal.flac"]
    r = ingest(str(src), store=store, cfg=cfg)
    assert isinstance(r, IngestResult) and r.cache_hit is False
    assert r.song_key.startswith("f-") and len(r.song_key) == 18
    inp = store.input_dir(r.song_key)
    assert r.job_dir == store.job_dir(r.song_key) and inp.parent == r.job_dir

    # original bytes kept, never re-encoded
    assert (inp / "source.flac").read_bytes() == src.read_bytes()
    info = sf.info(str(inp / "mix_44k_f32.wav"))
    assert (info.samplerate, info.channels, info.subtype) == (44100, 2, "FLOAT")

    meta = atomic.read_json(inp / "meta.json")
    assert meta == r.meta
    assert meta["song_key"] == r.song_key and meta["bandscribe_version"]
    assert meta["source"]["kind"] == "file" and meta["source"]["filename"] == "normal.flac"
    assert meta["source"]["sha256"] == atomic.sha256_file(src)
    assert meta["source"]["probe"]["codec_name"] == "flac"
    assert meta["youtube"] is None
    assert meta["decode"]["sr"] == 44100 and r.song_key == "f-" + meta["decode"]["pcm_sha256"][:16]
    assert meta["duration_s"] == pytest.approx(synth_audio.SECONDS, abs=0.01)
    assert meta["loudness"]["gain_db"] == 0.0  # -6 dBFS master needs no gain
    assert set(meta["flags"]) >= {"quasi_mono", "side_mid_db", "lowpass_hz_est", "lossy_cutoff", "nonstandard_sr", "drc", "mono_source"}
    assert meta["files"]["mix_44k_f32.wav"]["sha256"] == atomic.sha256_file(inp / "mix_44k_f32.wav")

    assert meta["source"]["content_key"] == content_key(src)
    assert store.index_get(f"file:{meta['source']['content_key']}") == r.song_key
    assert _temp_dirs(store.root) == []


def test_ingest_limits_loud_master(audio: dict[str, Path], store: JobStore, cfg: Config) -> None:
    r = ingest(str(audio["loud.m4a"]), store=store, cfg=cfg)
    loud = r.meta["loudness"]
    assert loud["gain_db"] < 0.0 and loud["ceiling_dbfs"] == CEILING
    assert loud["output_true_peak_dbfs"] <= CEILING
    mix = store.input_dir(r.song_key) / "mix_44k_f32.wav"
    assert loudness.measure(mix)["true_peak_dbfs"] <= CEILING


def test_copy_under_other_name_is_a_fast_cache_hit(audio: dict[str, Path], store: JobStore, cfg: Config, tmp_path: Path) -> None:
    first = ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    copy = tmp_path / "다른 이름의 복사본 (1).wav"  # user files may have Korean names; only our own paths are ASCII
    shutil.copyfile(audio["normal.wav"], copy)
    t0 = time.perf_counter()
    again = ingest(str(copy), store=store, cfg=cfg)
    elapsed = time.perf_counter() - t0
    assert again.cache_hit is True and again.song_key == first.song_key
    assert elapsed < 1.0, f"cache hit took {elapsed:.3f} s"
    assert again.meta == first.meta


def test_changed_settings_are_not_served_from_cache(audio: dict[str, Path], store: JobStore, tmp_path: Path) -> None:
    """Regression: a cache hit used to ignore --set ingest.true_peak_ceiling_dbfs and still say "hit"."""
    default = Config()
    stricter = Config.model_validate({"ingest": {"true_peak_ceiling_dbfs": -3.0}})
    src = audio["loud.m4a"]
    a = ingest(str(src), store=store, cfg=default)
    assert a.meta["params"] == ingest_params(default) and a.meta["loudness"]["ceiling_dbfs"] == -1.0

    b = ingest(str(src), store=store, cfg=stricter)
    assert b.cache_hit is False and b.song_key == a.song_key  # same audio, same job: rebuilt in place
    assert b.meta["loudness"]["ceiling_dbfs"] == -3.0 and b.meta["params"] == ingest_params(stricter)
    assert b.meta["loudness"]["gain_db"] < a.meta["loudness"]["gain_db"]
    mix = store.input_dir(b.song_key) / "mix_44k_f32.wav"
    assert loudness.measure(mix)["true_peak_dbfs"] <= -3.0
    assert atomic.read_json(store.input_dir(b.song_key) / "meta.json") == b.meta

    # the same settings again (also via a copy under another name) are a real cache hit
    copy = tmp_path / "복사본.m4a"
    shutil.copyfile(src, copy)
    for path in (src, copy):
        again = ingest(str(path), store=store, cfg=stricter)
        assert again.cache_hit is True and again.meta == b.meta
    # back to the defaults: rebuilt again, and the job keeps its first source's name
    c = ingest(str(copy), store=store, cfg=default)
    assert c.cache_hit is False and c.meta["loudness"]["ceiling_dbfs"] == -1.0
    assert c.meta["source"]["filename"] == "loud.m4a"
    assert _temp_dirs(store.root) == [] and len(_complete_inputs(store.root)) == 1


def test_changed_flag_threshold_rebuilds(audio: dict[str, Path], store: JobStore) -> None:
    a = ingest(str(audio["quasi.wav"]), store=store, cfg=Config())
    assert a.meta["flags"]["quasi_mono"] is True
    b = ingest(str(audio["quasi.wav"]), store=store, cfg=Config.model_validate({"ingest": {"quasi_mono_side_mid_db": -50.0}}))
    assert b.cache_hit is False and b.meta["flags"]["quasi_mono"] is False


def test_same_audio_other_container_with_other_settings_is_rebuilt(audio: dict[str, Path], store: JobStore) -> None:
    """The PCM-hash hit (second cache path) honours the settings too."""
    a = ingest(str(audio["normal.wav"]), store=store, cfg=Config())
    b = ingest(str(audio["normal.flac"]), store=store, cfg=Config.model_validate({"ingest": {"lowpass_detect_hz": 15000}}))
    assert b.cache_hit is False and b.song_key == a.song_key
    assert b.meta["params"]["lowpass_detect_hz"] == 15000 and b.meta["source"]["stored_as"] == "source.flac"


def test_input_from_before_params_were_recorded_is_rebuilt(audio: dict[str, Path], store: JobStore, cfg: Config) -> None:
    r = ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    meta_path = store.input_dir(r.song_key) / "meta.json"
    old = atomic.read_json(meta_path)
    del old["params"]
    atomic.write_json(meta_path, old)
    again = ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    assert again.cache_hit is False and again.meta["params"] == ingest_params(cfg)
    assert ingest(str(audio["normal.wav"]), store=store, cfg=cfg).cache_hit is True


def test_content_key_covers_every_byte(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bandscribe.ingest import filekey

    monkeypatch.setattr(filekey, "CHUNK", 1 << 16)  # many chunks, so the thread pool path runs
    data = bytearray(os.urandom(5 * (1 << 16) + 123))
    a = tmp_path / "a.bin"
    a.write_bytes(bytes(data))
    k = content_key(a)
    assert len(k) == 64 and content_key(a) == k
    data[len(data) // 2] ^= 1  # a one-bit edit in the middle (a same-length re-bounce) changes the key
    b = tmp_path / "b.bin"
    b.write_bytes(bytes(data))
    assert content_key(b) != k
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert content_key(empty) != k


def _write_big_wav(path: Path, mib: int) -> None:
    """A 48 kHz / 24-bit stereo PCM WAV of about ``mib`` MiB (noise; only its bytes matter here)."""
    import struct

    frame = 6
    data_len = (mib << 20) // frame * frame
    fmt = struct.pack("<IHHIIHH", 16, 1, 2, 48000, 48000 * frame, frame, 24)
    header = b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVEfmt " + fmt + b"data" + struct.pack("<I", data_len)
    with open(path, "wb") as f:
        f.write(header)
        left = data_len
        while left:
            n = min(left, 16 << 20)
            f.write(os.urandom(n))
            left -= n


@pytest.mark.slow
def test_cli_cache_hit_on_a_large_wav_is_under_1s(tmp_path: Path) -> None:
    """DESIGN 10 / M0_SPEC 12.2 through the real launcher: a renamed copy of a 10-minute 48 kHz/24-bit
    bounce (~165 MiB) must be a cache hit in under 1 s, CLI start-up included. The job is seeded
    (index + meta) so the test times exactly the fast path, not a 20 s first ingest."""
    root = tmp_path / "root"
    big = tmp_path / "bounce.wav"
    _write_big_wav(big, 165)
    copy = tmp_path / "합주 bounce (복사본).wav"
    shutil.copyfile(big, copy)

    store = JobStore(root / "data" / "jobs")
    song_key = "f-00000000000000aa"
    store.index_put(f"file:{content_key(big)}", song_key)
    atomic.write_json(
        store.input_dir(song_key) / "meta.json",
        {"song_key": song_key, "params": ingest_params(Config()), "source": {"kind": "file", "filename": big.name}},
    )

    bandscribe_cmd = Path(__file__).resolve().parents[1] / "bandscribe.cmd"
    env = dict(os.environ, BANDSCRIBE_ROOT=str(root))
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = subprocess.run([str(bandscribe_cmd), "ingest", str(copy)], env=env, capture_output=True,
                             encoding="utf-8", errors="replace", timeout=60)
        times.append(time.perf_counter() - t0)
        assert out.returncode == 0, out.stderr
        assert f"song_key: {song_key}" in out.stdout and "캐시 적중: 예" in out.stdout
    assert min(times) < 1.0, f"cache hit took {[round(t, 2) for t in times]} s"


def test_cli_import_path_stays_light() -> None:
    # The in-process timing above cannot see import cost. scipy.signal (pulled in by flags' welch and by
    # pyloudnorm) takes ~0.9 s to import, which made a real cache-hit `bandscribe ingest` take 1.6 s; both are
    # imported lazily so only a fresh ingest pays for them.
    code = (
        "import sys, bandscribe.cli, bandscribe.ingest, bandscribe.jobs.store;"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('scipy', 'pyloudnorm')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=paths.child_env(), check=True,
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_same_audio_other_container_is_a_cache_hit(audio: dict[str, Path], store: JobStore, cfg: Config) -> None:
    a = ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    b = ingest(str(audio["normal.flac"]), store=store, cfg=cfg)
    assert b.cache_hit is True and b.song_key == a.song_key
    assert b.meta["source"]["filename"] == "normal.wav"  # the job keeps the first source
    assert store.index_get(f"file:{content_key(audio['normal.flac'])}") == a.song_key
    # a lossy encode of the same music is different audio -> different job
    c = ingest(str(audio["normal.mp3"]), store=store, cfg=cfg)
    assert c.cache_hit is False and c.song_key != a.song_key
    assert len(_complete_inputs(store.root)) == 2
    assert _temp_dirs(store.root) == []


def test_index_pointing_at_missing_input_rebuilds(audio: dict[str, Path], store: JobStore, cfg: Config) -> None:
    r = ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    shutil.rmtree(store.input_dir(r.song_key))
    again = ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    assert again.cache_hit is False and again.song_key == r.song_key
    assert (store.input_dir(r.song_key) / "meta.json").is_file()


def test_input_dir_without_meta_is_replaced(audio: dict[str, Path], store: JobStore, cfg: Config) -> None:
    r = ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    inp = store.input_dir(r.song_key)
    (inp / "meta.json").unlink()  # damaged: not complete any more
    again = ingest(str(audio["normal.flac"]), store=store, cfg=cfg)
    assert again.cache_hit is False and again.song_key == r.song_key
    assert (inp / "meta.json").is_file() and (inp / "source.flac").is_file()
    assert not (inp / "source.wav").exists()
    assert _temp_dirs(store.root) == []


def test_failure_mid_ingest_leaves_no_complete_input(
    audio: dict[str, Path], store: JobStore, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = loudness.apply_ceiling

    def boom(wav_in: Path, wav_out: Path, ceiling: float, **kw: object) -> float:
        real(wav_in, wav_out, ceiling, **kw)  # the mix is half-way into the scratch dir ...
        raise RuntimeError("simulated crash")  # ... when things go wrong

    monkeypatch.setattr(loudness, "apply_ceiling", boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    assert _complete_inputs(store.root) == []
    assert list(store.root.glob("*/input")) == []
    assert _temp_dirs(store.root) == []
    assert store.index_all() == {}

    monkeypatch.setattr(loudness, "apply_ceiling", real)
    r = ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    assert r.cache_hit is False and len(_complete_inputs(store.root)) == 1


_KILL_CHILD = r"""
import sys, time
from pathlib import Path
from bandscribe.config import Config
from bandscribe.ingest import ingest, loudness
from bandscribe.jobs.store import JobStore

src, root, marker = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
real = loudness.apply_ceiling

def stall(*a, **kw):
    gain = real(*a, **kw)
    marker.write_text("mix written", encoding="utf-8")
    time.sleep(300)
    return gain

loudness.apply_ceiling = stall
ingest(src, store=JobStore(root), cfg=Config())
"""


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="uses taskkill")
def test_killed_ingest_leaves_no_complete_input(audio: dict[str, Path], tmp_path: Path, cfg: Config) -> None:
    root = tmp_path / "jobs"
    marker = tmp_path / "marker.txt"
    env = paths.child_env({"PYTHONPATH": str(paths.ROOT)})
    proc = subprocess.Popen(
        [sys.executable, "-c", _KILL_CHILD, str(audio["normal.wav"]), str(root), str(marker)],
        env=env, cwd=str(tmp_path), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 90
        while not marker.exists():
            assert proc.poll() is None, proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            assert time.monotonic() < deadline, "child never reached the stalled step"
            time.sleep(0.1)
    finally:
        # Hard kill of the whole tree: the venv python.exe is a launcher with the real interpreter as its child.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
        proc.wait(timeout=30)
        if proc.stderr:
            proc.stderr.close()

    assert _complete_inputs(root) == []
    assert list(root.glob("*/input")) == []
    leftovers = _temp_dirs(root)
    assert leftovers and all(p.name.startswith(".tmp-ingest-") for p in leftovers)

    store = JobStore(root)
    deadline = time.monotonic() + 10
    while store.gc_temp() < len(leftovers) and _temp_dirs(root):  # handles close a moment after the kill
        assert time.monotonic() < deadline, f"gc_temp could not remove {_temp_dirs(root)}"
        time.sleep(0.2)
    assert _temp_dirs(root) == []
    r = ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    assert r.cache_hit is False and len(_complete_inputs(root)) == 1


def test_missing_file_is_a_clear_error(store: JobStore, cfg: Config, tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="파일을 찾을 수 없습니다"):
        ingest(str(tmp_path / "nope.flac"), store=store, cfg=cfg)


def test_non_audio_file_fails_without_leftovers(store: JobStore, cfg: Config, tmp_path: Path) -> None:
    bogus = tmp_path / "readme.txt"
    bogus.write_text("hello", encoding="utf-8")
    with pytest.raises(decode.DecodeError):
        ingest(str(bogus), store=store, cfg=cfg)
    assert _temp_dirs(store.root) == [] and _complete_inputs(store.root) == []


def test_youtube_input_is_on_by_default() -> None:
    """User request 2026-09-30: YouTube link input defaults to ON (M0_SPEC 3)."""
    defaults = tomllib.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    assert defaults["youtube"]["enabled"] is True
    assert Config().youtube.enabled is True
    # the extra-risk options stay off
    assert defaults["youtube"]["cookies"] is False and defaults["youtube"]["po_token"] is False


def test_temp_names_are_collectable_by_gc(audio: dict[str, Path], tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """The scratch dir uses the store's .tmp-<tag>-<pid>-<rand> naming, so gc_temp can find it."""
    store = JobStore(tmp_path / "jobs")
    seen: list[Path] = []
    real = decode.decode_to_f32

    def spy(src: Path, dst: Path, sr: int = 44100) -> None:
        seen.append(Path(dst).parent)
        real(src, dst, sr)

    monkeypatch.setattr(decode, "decode_to_f32", spy)
    ingest(str(audio["normal.wav"]), store=store, cfg=cfg)
    assert seen and seen[0].parent == store.root
    assert seen[0].name.startswith(".tmp-ingest-") and f"-{os.getpid()}-" in seen[0].name
