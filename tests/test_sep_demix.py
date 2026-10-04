"""The vendored demix loop (CPU accumulators) reconstructs its input exactly with an identity "model" (gpuenv).

Runs the ``sep_msst`` worker in ``selftest_identity`` mode in the gpu venv (CPU torch is enough): 30 s of random
stereo noise, chunk 262144 and 588800 (overlap 2), every stem must equal the input within 1e-5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bandscribe import paths
from bandscribe.gpu import run_worker

pytestmark = pytest.mark.gpuenv


def test_identity_reconstruction(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((44100 * 30, 2)) * 0.3).astype(np.float32)
    x[:, 0] += 1.5 * np.sin(np.arange(len(x)) / 44100 * 2 * np.pi * 3)  # > 1.0 peaks and a low tone too
    wav = tmp_path / "noise.wav"
    sf.write(str(wav), x, 44100, subtype="FLOAT")
    r = run_worker("sep_msst", {"mode": "selftest_identity", "wav": str(wav), "chunk_sizes": [262144, 588800],
                                "num_overlap": 2, "num_stems": 6}, tmp_path / "w", python=paths.GPU_PYTHON,
                   use_lock=False, timeout_s=600)
    assert r.ok, r.stderr_path.read_text(encoding="utf-8")[-3000:]
    res = r.output["results"]
    assert set(res) == {"262144", "588800"}
    for chunk, v in res.items():
        assert v["shape"] == [6, 2, len(x)], chunk
        assert v["max_abs_err"] < 1e-5, (chunk, v)


def test_identity_short_input(tmp_path: Path) -> None:
    """Shorter than 2 borders (no reflect padding) and shorter than one chunk (padded last chunk)."""
    x = (np.random.default_rng(1).standard_normal((100000, 2)) * 0.2).astype(np.float32)
    wav = tmp_path / "short.wav"
    sf.write(str(wav), x, 44100, subtype="FLOAT")
    r = run_worker("sep_msst", {"mode": "selftest_identity", "wav": str(wav), "chunk_sizes": [262144]},
                   tmp_path / "w", python=paths.GPU_PYTHON, use_lock=False, timeout_s=300)
    assert r.ok, r.stderr_path.read_text(encoding="utf-8")[-3000:]
    assert r.output["results"]["262144"]["max_abs_err"] < 1e-5
