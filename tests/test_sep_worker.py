"""Real SW separation through ``bandscribe.sep.run.separate`` (gpu, slow): 20 s clip -> 6 stems, every M1_M2_SPEC 7.1
field present, gpu_run.json written. Prints the measured VRAM / speed for the M2 report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bandscribe import paths

MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "bs_roformer_sw"

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_separate_20s_clip(tmp_path: Path) -> None:
    audio = pytest.importorskip("bandscribe.audio")
    if not (MODEL_DIR / "BS-Rofo-SW-Fixed.ckpt").is_file():
        pytest.skip("SW checkpoint not on disk")
    from bandscribe.bench import synth_clip
    from bandscribe.sep import run as seprun

    clip = synth_clip(20.0)
    mix = tmp_path / "mix.wav"
    audio.write_f32(mix, clip, 44100)
    cfg = {"gpu": {"wait_max_s": 600, "wait_poll_s": 10}}  # never wait an hour inside a test
    res = seprun.separate(mix, tmp_path / "out", cfg)

    assert set(res.stems) == set(seprun.STEM_ORDER)
    total = np.zeros_like(clip, dtype=np.float64)
    for name, p in res.stems.items():
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        assert sr == 44100 and x.shape == clip.shape, name
        total += x
    left = clip.astype(np.float64) - total
    left_db = 10 * np.log10(np.mean(left ** 2) / np.mean(clip.astype(np.float64) ** 2))
    assert left_db < -10.0  # the six stems explain the mix

    result = json.loads((tmp_path / "out" / "_worker" / "result.json").read_text(encoding="utf-8"))
    out = result["output"]
    assert out["status"] == "ok"
    assert set(out["backend"]) >= {"name", "version", "model", "model_sha256", "dtype_policy", "device_used"}
    assert out["backend"]["model_sha256"] == seprun.SEP_DEFAULTS["sep.ckpt_sha256"]
    v = out["vram"]
    assert set(v) >= {"device", "total_mb", "start_free_mb", "ctx_mb", "start_nvml_used_mb", "rung_index",
                      "rung_label", "cap_mb", "attempts"}
    a = v["attempts"][-1]
    assert set(a) >= {"rung_index", "rung_label", "device", "status", "max_allocated_mb", "max_reserved_mb",
                      "nvml_used_peak_mb", "pdh_self_shared_growth_mb", "pdh_adapter_shared_delta_mb", "rtf", "slow",
                      "error"}
    assert a["status"] == "ok"
    assert set(out["timing"]) == {"load_s", "process_s", "audio_s", "rtf"}
    assert set(out["files"]) == {f"stems/{s}.wav" for s in seprun.STEM_ORDER}
    assert out["stem_order"] == list(seprun.STEM_ORDER) and out["zero_dc"] is True
    assert result["cuda"]["max_allocated_mb"] > 0 and result["cuda"]["max_reserved_mb"] > 0
    sm = result["sysmon"]
    for k in ("nvml_used_peak_mb", "nvml_used_start_mb", "pdh_self_shared_start_mb", "pdh_self_shared_peak_mb",
              "marks", "pdh_self_shared_growth_mb", "pdh_self_shared_delta_mb", "pdh_adapter_shared_delta_mb",
              "shared_growth"):
        assert k in sm, k
    assert "model_loaded" in sm["marks"]
    gpu_run = json.loads((tmp_path / "out" / "gpu_run.json").read_text(encoding="utf-8"))
    assert gpu_run["format"] == "bandscribe.gpu_run/1" and gpu_run["rung"]["label"] == v["rung_label"]
    sep_json = json.loads((tmp_path / "out" / "sep.json").read_text(encoding="utf-8"))
    assert sep_json["chunk_size"] == int(v["rung_label"].split("=")[1].split("@")[0])
    print(json.dumps({"rung": v["rung_label"], "ctx_mb": v["ctx_mb"], "start_free_mb": v["start_free_mb"],
                      "max_reserved_mb": a["max_reserved_mb"], "max_allocated_mb": a["max_allocated_mb"],
                      "nvml_used_peak_mb": a["nvml_used_peak_mb"], "growth_mb": sm["pdh_self_shared_growth_mb"],
                      "timing": out["timing"], "leftover_db": round(float(left_db), 2)}))
