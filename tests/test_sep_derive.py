"""gtab.sep.derive (the ``stems`` stage): exact identities, mono views, stats, streaming, byte determinism."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from gtab.sep import derive

audio = pytest.importorskip("gtab.audio")  # P0's deterministic float WAV writer

STEMS = derive.STEMS


def _make(tmp: Path, n: int, sr: int = 44100, leftover_db: float = -30.0, seed: int = 0) -> tuple[Path, Path, dict]:
    rng = np.random.default_rng(seed)
    stems = {s: (rng.standard_normal((n, 2)) * (0.05 + 0.02 * i)).astype(np.float32) for i, s in enumerate(STEMS)}
    stems["piano"][: n // 3] = 0.0  # a partly silent stem
    total = np.zeros((n, 2), np.float32)
    for s in STEMS:
        total = total + stems[s]
    resid = rng.standard_normal((n, 2)).astype(np.float32)
    resid *= np.float32(np.sqrt(np.mean(total.astype(np.float64) ** 2) / np.mean(resid.astype(np.float64) ** 2))
                        * 10 ** (leftover_db / 20))
    mix = (total + resid).astype(np.float32)
    mix[5, 0] = 1.25  # > 0 dBFS sample must survive
    stems_dir = tmp / "sep" / "stems"
    stems_dir.mkdir(parents=True)
    for s in STEMS:
        sf.write(str(stems_dir / f"{s}.wav"), stems[s], sr, subtype="FLOAT")
    mix_wav = tmp / "mix.wav"
    sf.write(str(mix_wav), mix, sr, subtype="FLOAT")
    return mix_wav, stems_dir, {"mix": mix, **stems}


def _read(p: Path) -> np.ndarray:
    return sf.read(str(p), dtype="float32", always_2d=True)[0]


def _data_outputs(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def test_identities_views_and_streaming(tmp_path: Path) -> None:
    block = 4096
    n = 3 * block + 123  # longer than two blocks, ragged tail
    mix_wav, stems_dir, x = _make(tmp_path, n)
    out = tmp_path / "out"
    derive.derive(mix_wav, stems_dir, out, block_frames=block)
    m = x["mix"]
    np.testing.assert_array_equal(_read(out / "nonvox.wav"), m - x["vocals"] - x["drums"] - x["bass"])
    left = m - x["bass"] - x["drums"] - x["other"] - x["vocals"] - x["guitar"] - x["piano"]
    np.testing.assert_array_equal(_read(out / "leftover.wav"), left)
    np.testing.assert_array_equal(_read(out / "practice" / "mix_minus_guitar.wav"), m - x["guitar"])
    np.testing.assert_array_equal(_read(out / "practice" / "mix_minus_bass.wav"), m - x["bass"])
    assert _read(out / "nonvox.wav")[5, 0] > 1.0

    views = json.loads((out / "views.json").read_text(encoding="utf-8"))
    assert views["format"] == "gtab.views/1"
    expect = {"mix": m, "guitar": x["guitar"], "bass": x["bass"], "piano_other": x["piano"] + x["other"],
              "nonvox": m - x["vocals"] - x["drums"] - x["bass"], "guitar_other": x["guitar"] + x["other"]}
    assert set(views["views"]) == {f"{v}_mono" for v in expect}
    for v, st in expect.items():
        entry = views["views"][f"{v}_mono"]
        p = out / entry["path"]
        got = _read(p)
        assert got.shape == (n, 1) and sf.info(str(p)).channels == 1
        np.testing.assert_array_equal(got[:, 0], audio.to_mono(st))
        assert entry["pcm_sha256"] == audio.pcm_sha256(p), v
        assert entry["mono"] == "(L+R)/2" and entry["source"]


def test_stats_energy_and_flag(tmp_path: Path) -> None:
    n = 44100 * 2
    mix_wav, stems_dir, x = _make(tmp_path, n, leftover_db=-15.0)
    stats = derive.derive(mix_wav, stems_dir, tmp_path / "out", leftover_warn_db=-20.0)
    doc = json.loads((tmp_path / "out" / "stem_stats.json").read_text(encoding="utf-8"))
    assert doc == stats and doc["format"] == "gtab.stem_stats/1"
    mix_db = 10 * np.log10(np.mean(x["mix"].astype(np.float64) ** 2))
    assert abs(doc["mix_rms_db"] - mix_db) < 1e-3
    g_db = 10 * np.log10(np.mean(x["guitar"].astype(np.float64) ** 2))
    assert abs(doc["stems"]["guitar"]["rel_db"] - (g_db - mix_db)) < 1e-3
    assert abs(doc["stems"]["guitar"]["peak"] - float(np.max(np.abs(x["guitar"])))) < 1e-5
    assert abs(doc["leftover"]["rms_db_rel_mix"] + 15.0) < 0.2 and doc["leftover"]["flag"] is True
    assert "기준" in derive.leftover_warning(doc)
    assert doc["nonvox"]["rel_db"] < 0
    with np.load(tmp_path / "out" / "energy.npz") as e:
        assert set(e.files) == set(derive.ENERGY_STREAMS) | {"fps"}
        assert float(e["fps"][0]) == 10.0
        mix_e = e["mix"]
        assert mix_e.dtype == np.float32 and len(mix_e) == -(-n // 4410)
        first = 10 * np.log10(np.mean(x["mix"][:4410].astype(np.float64) ** 2) + 1e-20)
        assert abs(float(mix_e[0]) - first) < 1e-3
        assert float(e["piano"][0]) < -150  # silent start of the piano stem

    mix_wav2, stems_dir2, _ = _make(tmp_path / "quiet", n, leftover_db=-40.0)
    doc2 = derive.derive(mix_wav2, stems_dir2, tmp_path / "out2", leftover_warn_db=-20.0)
    assert doc2["leftover"]["flag"] is False and derive.leftover_warning(doc2) is None


def test_worst_windows(tmp_path: Path) -> None:
    sr = 8000
    n = sr * 35  # windows of 10 s: 0, 10, 20, 30 (partial)
    mix_wav, stems_dir, x = _make(tmp_path, n, sr=sr, leftover_db=-40.0)
    # make window 20-30 s leak: raise the mix there only (leftover = mix - stems grows)
    mix = x["mix"].copy()
    mix[20 * sr:30 * sr] += x["guitar"][20 * sr:30 * sr] * 0.5
    sf.write(str(mix_wav), mix, sr, subtype="FLOAT")
    doc = derive.derive(mix_wav, stems_dir, tmp_path / "out")
    worst = doc["leftover"]["worst_windows"]
    assert worst[0]["start_s"] == 20.0 and len(worst) == 4
    assert worst[0]["rel_db"] > worst[1]["rel_db"]


def test_byte_deterministic(tmp_path: Path) -> None:
    mix_wav, stems_dir, _ = _make(tmp_path, 44100 * 3 + 17)
    a, b = tmp_path / "a", tmp_path / "b"
    derive.derive(mix_wav, stems_dir, a, block_frames=1 << 15)
    import time

    time.sleep(1.1)  # a timestamp anywhere (WAV PEAK chunk, zip member dates) would now differ
    derive.derive(mix_wav, stems_dir, b, block_frames=1 << 15)
    da, db = _data_outputs(a), _data_outputs(b)
    assert da == db and len(da) == 13  # 4 stereo + 6 mono WAVs, 2 JSON, energy.npz
    with zipfile.ZipFile(io.BytesIO((a / "energy.npz").read_bytes())) as z:
        assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in z.infolist())


def test_mismatched_lengths_rejected(tmp_path: Path) -> None:
    mix_wav, stems_dir, x = _make(tmp_path, 5000)
    sf.write(str(stems_dir / "piano.wav"), x["piano"][:4000], 44100, subtype="FLOAT")
    with pytest.raises(ValueError, match="piano"):
        derive.derive(mix_wav, stems_dir, tmp_path / "out")


def test_stems_stage_run(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from gtab.sep import stage

    input_dir = tmp_path / "job" / "input"
    input_dir.mkdir(parents=True)
    mix_wav, stems_dir, _ = _make(tmp_path, 44100, leftover_db=-10.0)
    (input_dir / "mix_44k_f32.wav").write_bytes(mix_wav.read_bytes())
    events: list[tuple[str, dict]] = []
    ctx = SimpleNamespace(song_key="f-x", stage="stems", key="k" * 64, out_dir=tmp_path / "st",
                          dep_dirs={"sep": stems_dir.parent}, config={}, params={"leftover_warn_db": -20.0},
                          store=SimpleNamespace(input_dir=lambda k: input_dir), progress=lambda e, i: events.append((e, i)))
    stage.run_stems(ctx)
    assert (tmp_path / "st" / "views" / "guitar_mono.wav").is_file()
    assert events and events[0][0] == "warning" and "leftover" in events[0][1]["message"]
