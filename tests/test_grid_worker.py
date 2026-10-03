"""beats_beatthis worker (M1_M2_SPEC 7.5, 9.3; model beat_this.final0).

- ``gpu`` + ``slow``: the real CUDA rung through ``gtab.vram.run_gpu_stage`` (GPU lock, VRAM pre-check).
- ``gpuenv`` + ``slow``: the same synthetic song on the CPU rung (``final0@cpu``) — exercises the whole worker
  (local checkpoint, ``weights_only=True`` loads, in-memory audio, postprocessing, output files) while the GPU is
  busy; CPU torch is enough and no GPU lock is taken.
- ``gpuenv``: refusals (hub name, sha mismatch) happen before any model load.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from gtab import atomic

pytestmark = [pytest.mark.model("beat_this.final0")]


@pytest.fixture(scope="module")
def ckpt() -> tuple[Path, str]:
    from gtab import models

    p = models.local_path("beat_this.final0")
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _song(path: Path, bpm: float = 120.0, dur: float = 30.0, t0: float = 0.5, sr: int = 44100):
    """Rock pattern: kick on 1 and 3, snare on 2 and 4, eighth hats, a bass root per bar, crash on bar starts."""
    rng = np.random.default_rng(7)
    n = int(dur * sr)
    y = np.zeros(n)
    beat = 60.0 / bpm
    beats = np.arange(t0, dur - 0.5, beat)
    roots = [41.2, 55.0, 36.7, 49.0]  # E1 A1 D1 G1

    def put(x, t):
        i = int(round(t * sr))
        e = min(n, i + len(x))
        y[i:e] += x[:e - i]

    for k, t in enumerate(beats):
        pos = k % 4
        tt = np.arange(int(0.25 * sr)) / sr
        if pos in (0, 2):
            put(0.9 * np.sin(2 * np.pi * (50 + 90 * np.exp(-tt / 0.03)) * tt) * np.exp(-tt / 0.12), t)
        else:
            put(rng.normal(0, 0.35, tt.size) * np.exp(-tt / 0.05), t)
        for h in (0.0, 0.5):
            L = int(0.04 * sr)
            put(rng.normal(0, 0.06, L) * np.exp(-np.arange(L) / (0.006 * sr)), t + h * beat)
        if pos == 0:
            bar = k // 4
            f = roots[bar % 4]
            tb = np.arange(int(4 * beat * sr)) / sr
            bass = sum(np.sin(2 * np.pi * f * h_ * tb) / h_ for h_ in (1, 2, 3)) * 0.25 * np.exp(-tb / 1.5)
            put(bass, t)
            L = int(0.8 * sr)
            put(rng.normal(0, 0.12, L) * np.exp(-np.arange(L) / (0.3 * sr)), t)
    stereo = np.stack([y, y], 1).astype(np.float32)
    sf.write(str(path), stereo, sr, subtype="FLOAT")
    return beats, beats[::4]


def _request(wav: Path, out_dir: Path, ckpt: tuple[Path, str], **extra) -> dict:
    return {"format": "gtab.worker/1", "job": None, "out_dir": str(out_dir), "device": "auto", "determinism": True,
            "wav": str(wav), "checkpoint": "final0", "checkpoint_path": str(ckpt[0]), "checkpoint_sha256": ckpt[1],
            "dbn": False, "float16": False, "out_beats": "beats.json", "out_activations": "activations.npz",
            **extra}


def _check_outputs(out: dict, out_dir: Path, ckpt: tuple[Path, str], beats, downs) -> None:
    assert out["status"] == "ok"
    assert out["checkpoint_loads"] and all(x["weights_only"] is True for x in out["checkpoint_loads"])
    assert out["checkpoint_loads"][0]["path"] == str(ckpt[0])
    assert out["checkpoint_sha256"] == ckpt[1] and out["backend"]["model_sha256"] == ckpt[1]
    doc = atomic.read_json(out_dir / "beats.json")
    from gtab.schema.grid import BeatsRaw

    BeatsRaw.model_validate(doc)
    assert doc["format"] == "gtab.beats/1" and doc["tracker"] == "beat_this/final0" and doc["fps"] == 50.0
    assert out["n_beats"] == len(doc["beats_s"]) and out["n_downbeats"] == len(doc["downbeats_s"])
    got = np.asarray(doc["beats_s"])
    err = np.array([np.min(np.abs(got - b)) for b in beats[1:-1]])
    assert np.all(err <= 0.020 + 1e-6), err.max()  # 50 fps output: 20 ms quantisation
    got_d = np.asarray(doc["downbeats_s"])
    derr = np.array([np.min(np.abs(got_d - d)) for d in downs[1:-1]])
    assert np.all(derr <= 0.020 + 1e-6), (derr, got_d[:8], downs[:8])
    with np.load(out_dir / "activations.npz") as z:
        assert z["beat"].dtype == np.float32 and z["downbeat"].dtype == np.float32 and float(z["fps"]) == 50.0
        assert abs(z["beat"].size - 30 * 50) <= 2
    assert set(out["files"]) == {"beats.json", "activations.npz"}


@pytest.mark.gpu
@pytest.mark.slow
def test_worker_tracks_beats_and_downbeats(tmp_path, ckpt):
    from gtab import vram

    wav = tmp_path / "mix.wav"
    beats, downs = _song(wav)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = vram.run_gpu_stage("beats_beatthis", ["final0"], _request(wav, out_dir, ckpt), None, out_dir / "_worker")
    assert res.ok, (res.error, res.stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:])
    out = res.output
    _check_outputs(out, out_dir, ckpt, beats, downs)
    assert out["backend"]["device_used"] == "cuda" and out["vram"]["rung_label"] == "final0"
    gr = vram.write_gpu_run(out_dir, "beats_beatthis", ["final0"], out, res.waited_s)
    assert gr["rung"] == {"index": 0, "label": "final0", "device": "cuda"} and gr["degraded"] is False
    # the result carries every field the spec requires of a GPU worker
    r = res.result
    assert r["cuda"]["max_reserved_mb"] > 0
    for k in ("nvml_used_peak_mb", "nvml_used_start_mb", "pdh_self_shared_start_mb", "pdh_self_shared_peak_mb",
              "marks", "pdh_self_shared_growth_mb", "pdh_self_shared_delta_mb", "pdh_adapter_shared_delta_mb",
              "shared_growth"):
        assert k in r["sysmon"], k
    assert "model_loaded" in r["sysmon"]["marks"]
    print("beats worker (cuda):", {"max_reserved_mb": r["cuda"]["max_reserved_mb"], "ctx_mb": out["vram"]["ctx_mb"],
                                   "timing": out["timing"]})


@pytest.mark.gpuenv
@pytest.mark.slow
def test_worker_cpu_rung(tmp_path, ckpt):
    from gtab import gpu, vram

    wav = tmp_path / "mix.wav"
    beats, downs = _song(wav)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    req = _request(wav, out_dir, ckpt, device="cpu",
                   vram={"policy": "cpu", "cpu_allowed": True, "precheck": True, "worker_headroom_mb": 256,
                         "need_table": {}, "shared_growth_fail_mb": 64, "rtf_ref": {}, "rtf_fail_ratio": 0.5,
                         "cap_mb": None, "force_rung": None})
    res = gpu.run_worker("beats_beatthis", req, out_dir / "_worker", timeout_s=900, use_lock=False)
    assert res.ok, (res.error, res.stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:])
    out = res.output
    _check_outputs(out, out_dir, ckpt, beats, downs)
    assert out["backend"]["device_used"] == "cpu" and out["vram"]["rung_label"] == "final0@cpu"
    assert [a["status"] for a in out["vram"]["attempts"]] == ["ok"]
    gr = vram.write_gpu_run(out_dir, "beats_beatthis", ["final0"], out, 0.0)
    assert gr["rung"]["device"] == "cpu" and gr["degraded"] is True  # the runner warns about such a cache
    print("beats worker (cpu):", out["timing"])


@pytest.mark.gpuenv
def test_worker_refuses_hub_names_and_bad_sha(tmp_path, ckpt):
    from gtab import gpu

    for bad, why in (({"checkpoint_path": "final0", "checkpoint_sha256": "0" * 64}, "never a hub name"),
                     ({"checkpoint_path": str(ckpt[0]), "checkpoint_sha256": "0" * 64}, "sha256 mismatch"),
                     ({"checkpoint_path": str(ckpt[0]), "checkpoint_sha256": ckpt[1], "dbn": True}, "madmom")):
        req = {"out_dir": str(tmp_path), "device": "cpu", "wav": str(tmp_path / "x.wav"), "dbn": False, **bad}
        res = gpu.run_worker("beats_beatthis", req, tmp_path / "_w", timeout_s=300, use_lock=False)
        assert not res.ok
        assert why in (res.error or ""), res.error
