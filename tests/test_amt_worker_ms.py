"""MuScriptor worker on the real GPU (gpu env + CUDA + medium weights in the HF cache). Skipped otherwise.

The CPU-only checks (probe, bf16 refusal) are in test_amt_worker_probe.py (marker ``gpuenv``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from gtab import atomic, gpu, paths
from gtab.amt import instruments as I
from gtab.amt import stage as S

W = S.muscriptor_weights("medium", S.AMT_DEFAULTS["amt.muscriptor_revision"])
pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(W is None or not paths.GPU_PYTHON.is_file(),
                                 reason="gpu env or MuScriptor-medium weights missing")]
SR = 44100


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("ms")
    t = np.arange(int(SR * 10.0)) / SR
    x = np.zeros_like(t)
    rng = np.random.default_rng(0)
    for k in range(40):
        f = 110 * 2 ** (rng.integers(0, 24) / 12)
        t0 = k * 0.25
        idx = (t >= t0) & (t < t0 + 0.5)
        env = np.exp(-(t[idx] - t0) * 6)
        x[idx] += sum(np.sin(2 * np.pi * f * h * (t[idx] - t0)) / h for h in range(1, 12)) * env * 0.2
    p = d / "view.wav"
    sf.write(str(p), x.astype(np.float32)[:, None], SR, subtype="FLOAT")
    return p


def _req(clip: Path, out: Path, *, loader="upstream", need_medium=2400.0, **extra) -> dict:
    return {
        "format": "gtab.worker/1", "job": None, "out_dir": str(out.parent), "device": "auto",
        "model": {"size": "medium", "weights_path": str(W), "sha256": None, "revision": "x", "loader": loader,
                  "repo": "MuScriptor/muscriptor-medium",
                  "fallback": [{"size": "small", "repo": "MuScriptor/muscriptor-small", "weights_path": None}]},
        "decode": {"prelude_forcing": True, "beam_size": 1, "cfg_coef": 1.0, "use_sampling": False, "batch_size": 1},
        "dtype": "upstream",
        "views": [{"id": "guitar_mono", "wav": str(clip), "instruments": list(reversed(I.GUITAR_CLASSES)),
                   "out": str(out), "view_sha256": None}],
        "vram": {"policy": "wait", "cpu_allowed": False, "precheck": True, "worker_headroom_mb": 256,
                 "need_table": {"medium": {"reserved_peak_mb": need_medium, "ctx_mb": 500},
                                "medium-lean": {"reserved_peak_mb": need_medium, "ctx_mb": 500},
                                "small": {"reserved_peak_mb": 1400, "ctx_mb": 500}},
                 "shared_growth_fail_mb": 64, "rtf_ref": {}, "rtf_fail_ratio": 0.5, "cap_mb": None, "force_rung": None},
        "determinism": True, **extra}


def _run(req: dict, work: Path, *, use_lock: bool = True):
    return gpu.run_worker("amt_muscriptor", req, work, timeout_s=1800, lock_timeout_s=1800, use_lock=use_lock)


def test_transcribes_clip_with_tensor_input_and_full_result(tmp_path, clip):
    out_json = tmp_path / "raw" / "guitar_mono__muscriptor.json"
    res = _run(_req(clip, out_json), tmp_path / "_worker")
    assert res.ok, res.error
    o = res.output
    assert o["status"] == "ok" and o["api"]["input_kind"] == "tensor"
    assert o["views"][0]["n_notes"] >= 1
    assert o["views"][0]["instruments"] == list(I.GUITAR_CLASSES)  # re-sorted by MT3 id
    for k in ("name", "version", "model", "model_sha256", "dtype_policy", "device_used"):
        assert k in o["backend"]
    for k in ("device", "total_mb", "start_free_mb", "ctx_mb", "start_nvml_used_mb", "rung_index", "rung_label",
              "cap_mb", "attempts"):
        assert k in o["vram"]
    a = o["vram"]["attempts"][0]
    for k in ("rung_index", "rung_label", "device", "status", "max_allocated_mb", "max_reserved_mb",
              "nvml_used_peak_mb", "pdh_self_shared_growth_mb", "pdh_adapter_shared_delta_mb", "rtf", "slow", "error"):
        assert k in a
    assert a["status"] == "ok" and o["vram"]["rung_label"] == "medium"
    for k in ("load_s", "process_s", "audio_s", "rtf"):
        assert o["timing"][k] is not None
    assert res.result["cuda"]["max_reserved_mb"] > 0
    sm = res.result["sysmon"]
    assert "model_loaded" in sm["marks"] and "shared_growth" in sm
    doc = atomic.read_json(out_json)
    assert doc["time_offset_applied_s"] == 0.0 and doc["backend"]["instruments"] == list(I.GUITAR_CLASSES)
    assert {n["instrument"] for n in doc["notes"]} <= set(I.GUITAR_CLASSES)
    S.validate_noteset(doc)


def test_small_fallback_absent_is_unavailable_without_network(tmp_path, clip, monkeypatch):
    if S.muscriptor_weights("small") is not None:
        pytest.skip("MuScriptor-small is cached on this PC")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):  # any network attempt would fail loudly
        monkeypatch.setenv(k, "http://127.0.0.1:9")
    req = _req(clip, tmp_path / "o.json", need_medium=1e6)  # medium cannot fit -> skipped, then small
    res = _run(req, tmp_path / "_worker")
    assert res.ok, res.error
    o = res.output
    assert o["status"] == "insufficient_vram"
    assert [(a["rung_label"], a["status"]) for a in o["vram"]["attempts"]] == [("medium", "skipped_budget"),
                                                                              ("small", "unavailable")]
    assert not (tmp_path / "o.json").exists()


def test_lean_loader_parity(tmp_path, clip):
    """Gate for switching amt.muscriptor_loader to lean: same notes, lower reserved peak."""
    up = _run(_req(clip, tmp_path / "up.json"), tmp_path / "up_worker")
    lean = _run(_req(clip, tmp_path / "lean.json", loader="lean"), tmp_path / "lean_worker")
    assert up.ok and lean.ok, (up.error, lean.error)
    assert lean.output["loader"] == "lean" and lean.output["vram"]["rung_label"] == "medium-lean"
    assert atomic.read_json(tmp_path / "up.json")["notes"] == atomic.read_json(tmp_path / "lean.json")["notes"]
    assert lean.result["cuda"]["max_reserved_mb"] < up.result["cuda"]["max_reserved_mb"]
