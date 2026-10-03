"""Vendored model + our demix (CPU accumulators) vs upstream MSST ``demix`` at the pinned commit (gpu, slow).

Needs the upstream files at ``data/scratch/msst_ref/`` (at least ``utils/model_utils.py``; skipped if absent) and
the SW checkpoint. Both loops run in the ``sep_msst`` worker's test-only ``parity_upstream`` mode, started through
``gtab.gpu.run_worker`` (GPU lock + Job Object, like every gtab GPU job), on a 20 s clip at the top rung.
Pass: SNR > 60 dB per stem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gtab import paths

GTAB_ROOT = Path(__file__).resolve().parents[1]
REF = GTAB_ROOT / "data" / "scratch" / "msst_ref"
MODEL_DIR = GTAB_ROOT / "data" / "models" / "bs_roformer_sw"

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_demix_matches_upstream(tmp_path: Path) -> None:
    if not (REF / "utils" / "model_utils.py").is_file():
        pytest.skip(f"pristine MSST checkout not found at {REF}")
    if not (MODEL_DIR / "BS-Rofo-SW-Fixed.ckpt").is_file():
        pytest.skip("SW checkpoint not on disk")
    from gtab.audio import write_f32
    from gtab.bench import synth_clip
    from gtab.gpu import run_worker

    wav = tmp_path / "clip.wav"
    write_f32(wav, synth_clip(20.0), 44100)
    r = run_worker("sep_msst", {"mode": "parity_upstream", "ref_dir": str(REF), "wav": str(wav),
                                "ckpt": str(MODEL_DIR / "BS-Rofo-SW-Fixed.ckpt"),
                                "config_yaml": str(MODEL_DIR / "BS-Rofo-SW-Fixed.yaml"), "chunk_size": 588800},
                   tmp_path / "w", python=paths.GPU_PYTHON, timeout_s=1800, lock_timeout_s=3600)
    assert r.ok, r.stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    snr = r.output["snr_db"]
    print("SNR vs upstream demix (dB):", snr)
    assert set(snr) == {"bass", "drums", "other", "vocals", "guitar", "piano"}
    assert all(v > 60.0 for v in snr.values()), snr
