"""gtab.sep.run / gtab.sep.stage with a fake ``run_gpu_stage``: params = key, request contents, overrides,
input-loudness gain math, sep.json / gpu_run.json, the STAGES contract and plan-time missing models."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from gtab import paths, vram
from gtab.sep import run as seprun
from gtab.sep import stage as sepstage

SEP_KEYS = {"ckpt_sha256", "config_sha256", "msst_commit", "chunk_ladder", "num_overlap", "batch_size", "dtype",
            "input_lufs"}


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data = tmp_path / "data"
    # never read the real VRAM table or write the real sha cache / models lock
    for name, value in (("DATA", data), ("BENCH", data / "bench"), ("VRAM_TABLE", data / "bench" / "vram_table.json"),
                        ("MODELS", data / "models"), ("MODELS_LOCK", data / "models" / "models.lock.json")):
        monkeypatch.setattr(paths, name, value)
    mdir = tmp_path / "models"
    mdir.mkdir()
    (mdir / "sw.ckpt").write_bytes(b"weights")
    (mdir / "sw.yaml").write_text("model: {}\n", encoding="utf-8")
    return {"sep": {"model_dir": str(mdir), "ckpt": "sw.ckpt", "config": "sw.yaml", "ckpt_sha256": "c" * 64,
                    "chunk_ladder": [588800, 352256, 262144], "num_overlap": 2, "batch_size": 1, "dtype": "upstream",
                    "input_lufs": "off", "leftover_warn_db": -20.0},
            "gpu": {"insufficient_vram_policy": "wait", "wait_poll_s": 0, "ctx_mb_default": 500}}


def _mix(path: Path, seconds: float = 1.0) -> Path:
    n = int(44100 * seconds)
    x = (np.random.default_rng(0).standard_normal((n, 2)) * 0.1).astype(np.float32)
    sf.write(str(path), x, 44100, subtype="FLOAT")
    return path


class FakeStage:
    """Stands in for gtab.vram.run_gpu_stage: records the request and writes six stems like the worker."""

    def __init__(self, rung_label: str = "chunk=588800", status: str = "ok") -> None:
        self.calls: list[dict[str, Any]] = []
        self.rung_label = rung_label
        self.status = status

    def __call__(self, backend: str, labels: list[str], request: dict, cfg: Any, work_dir: Path, **kw: Any) -> Any:
        self.calls.append({"backend": backend, "labels": labels, "request": request, "work_dir": work_dir, **kw})
        out = Path(request["out_dir"])
        stems = {}
        for name in seprun.STEM_ORDER:
            rel = f"stems/{name}.wav"
            (out / "stems").mkdir(parents=True, exist_ok=True)
            sf.write(str(out / rel), np.zeros((100, 2), np.float32), 44100, subtype="FLOAT")
            stems[name] = rel
        output = {"status": self.status, "stems": stems, "stem_order": list(seprun.STEM_ORDER), "chunk_size": 588800,
                  "config_sha256": "y" * 64, "zero_dc": True, "input_gain_db": request["input_gain_db"],
                  "num_overlap": 2, "batch_size": 1, "msst_commit": seprun.SEP_DEFAULTS["sep.msst_commit"],
                  "backend": {"name": "sep_msst", "model_sha256": "c" * 64, "dtype_policy": "fp32+amp",
                              "device_used": "cuda"},
                  "vram": {"rung_index": labels.index(self.rung_label), "rung_label": self.rung_label, "attempts": []},
                  "timing": {"load_s": 5.0, "process_s": 2.0, "audio_s": 12.0, "rtf": 6.0}, "warnings": []}
        return SimpleNamespace(ok=True, output=output, waited_s=3.5, stderr_path=work_dir / "stderr.log")


def test_params_are_exactly_the_key_fields(cfg: dict) -> None:
    p = seprun.sep_params(cfg)
    assert set(p) == SEP_KEYS
    assert p["chunk_ladder"] == [588800, 352256, 262144] and p["input_lufs"] == "off" and p["dtype"] == "upstream"
    assert len(p["config_sha256"]) == 64
    # non-semantic knobs never reach the key
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["sep"]["leftover_warn_db"] = -30.0
    cfg2["gpu"]["vram_margin_gb"] = 2.0
    assert seprun.sep_params(cfg2) == p
    assert sepstage.STAGES["sep"]["params"](cfg) == p


@pytest.mark.parametrize("bad", [{"dtype": "bf16"}, {"input_lufs": -40}, {"input_lufs": "loud"}])
def test_invalid_overrides_rejected(cfg: dict, bad: dict) -> None:
    with pytest.raises(ValueError):
        seprun.sep_params(cfg, **bad)


def test_separate_builds_request_and_writes_provenance(cfg: dict, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeStage(rung_label="chunk=352256")
    monkeypatch.setattr(vram, "run_gpu_stage", fake)
    mix = _mix(tmp_path / "mix.wav")
    res = seprun.separate(mix, tmp_path / "out", cfg, force_rung="chunk=588800", dtype="fp16")
    call = fake.calls[0]
    req = call["request"]
    assert call["backend"] == "sep_msst" and call["labels"] == ["chunk=588800", "chunk=352256", "chunk=262144"]
    assert call["work_dir"] == tmp_path / "out" / "_worker"
    assert req["dtype"] == "fp16" and req["input_gain_db"] == 0.0 and req["mode"] == "separate"
    assert req["vram"]["force_rung"] == "chunk=588800" and req["vram"]["cpu_allowed"] is False
    assert req["model"]["ckpt"].endswith("sw.ckpt") and req["model"]["ckpt_sha256"] == "c" * 64
    assert req["chunk_ladder"] == [588800, 352256, 262144] and req["outputs"] == {"stems_dir": "stems"}
    assert Path(req["out_dir"]).is_absolute() and Path(req["mix_wav"]).is_absolute()
    assert set(res.stems) == set(seprun.STEM_ORDER) and all(p.is_file() for p in res.stems.values())
    assert res.rung == {"index": 1, "label": "chunk=352256", "device": "cuda"}
    gpu_run = json.loads((tmp_path / "out" / "gpu_run.json").read_text(encoding="utf-8"))
    assert gpu_run["degraded"] is True and gpu_run["waited_s"] == 3.5
    sep_json = json.loads((tmp_path / "out" / "sep.json").read_text(encoding="utf-8"))
    assert sep_json["format"] == "gtab.sep/1" and sep_json["zero_dc"] is True
    assert sep_json["stem_order"] == list(seprun.STEM_ORDER) and sep_json["rung"] == {"index": 1, "label": "chunk=352256"}
    assert sep_json["rtf"] == 6.0 and sep_json["input_lufs"] == "off"


def test_input_lufs_gain_math(cfg: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gtab.ingest import loudness

    fake = FakeStage()
    monkeypatch.setattr(vram, "run_gpu_stage", fake)
    monkeypatch.setattr(loudness, "measure", lambda wav: {"integrated_lufs": -9.374})
    mix = _mix(tmp_path / "mix.wav")
    res = seprun.separate(mix, tmp_path / "o1", cfg, input_lufs=-14.0)
    assert fake.calls[0]["request"]["input_gain_db"] == -4.63  # -14 - (-9.374), 0.01 dB steps
    assert res.sep_json["input_lufs"] == -14.0 and res.sep_json["input_gain_db"] == -4.63
    cfg["sep"]["input_lufs"] = -16.0
    assert seprun.sep_params(cfg)["input_lufs"] == -16.0
    seprun.separate(mix, tmp_path / "o2", cfg)
    assert fake.calls[1]["request"]["input_gain_db"] == -6.63
    monkeypatch.setattr(loudness, "measure", lambda wav: {"integrated_lufs": None})  # silence
    assert seprun.input_gain_db(mix, -14.0) == 0.0
    assert seprun.input_gain_db(mix, "off") == 0.0


def test_unexpected_worker_status(cfg: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vram, "run_gpu_stage", FakeStage(status="insufficient_vram"))
    with pytest.raises(RuntimeError):
        seprun.separate(_mix(tmp_path / "mix.wav"), tmp_path / "o", cfg)


def test_models_and_missing_model(cfg: dict, tmp_path: Path) -> None:
    m = sepstage.STAGES["sep"]["models"](cfg)
    assert set(m) == {"bs_roformer_sw.ckpt", "bs_roformer_sw.yaml"} and all(len(v) == 64 for v in m.values())
    (Path(cfg["sep"]["model_dir"]) / "sw.ckpt").unlink()
    with pytest.raises(Exception) as ei:
        sepstage.STAGES["sep"]["models"](cfg)
    assert type(ei.value).__name__ in ("ModelMissing", "FileNotFoundError")
    assert "분리 모델 파일이 없습니다" in (getattr(ei.value, "hint_ko", None) or str(ei.value))


def test_stage_contract() -> None:
    st = sepstage.STAGES
    assert set(st) == {"sep", "stems"}
    for name, spec in st.items():
        assert set(spec) >= {"run", "code_version", "params", "device", "models"}, name
        assert callable(spec["run"]) and callable(spec["params"]) and callable(spec["models"])
    assert st["sep"]["device"] == "gpu" and st["stems"]["device"] == "cpu"
    assert st["stems"]["params"]({"sep": {"leftover_warn_db": -25}}) == {"leftover_warn_db": -25.0}
    assert st["stems"]["models"]({}) == {}


def test_run_sep_stage_uses_job_mix_and_params(cfg: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeStage()
    monkeypatch.setattr(vram, "run_gpu_stage", fake)
    input_dir = tmp_path / "job" / "input"
    input_dir.mkdir(parents=True)
    _mix(input_dir / "mix_44k_f32.wav")
    events: list[tuple[str, dict]] = []
    ctx = SimpleNamespace(song_key="f-abc", stage="sep", key="k" * 64, out_dir=tmp_path / "stage", dep_dirs={},
                          config=cfg, store=SimpleNamespace(input_dir=lambda k: input_dir),
                          params=seprun.sep_params(cfg), progress=lambda e, i: events.append((e, i)))
    sepstage.run_sep(ctx)
    req = fake.calls[0]["request"]
    assert req["job"] == {"song_key": "f-abc", "stage": "sep", "stage_key": "k" * 64}
    assert Path(req["mix_wav"]) == (input_dir / "mix_44k_f32.wav").resolve()
    assert (tmp_path / "stage" / "sep.json").is_file() and (tmp_path / "stage" / "gpu_run.json").is_file()
    fake.calls[0]["notify"]("VRAM 부족: 테스트")
    assert events == [("vram_wait", {"stage": "sep", "message": "VRAM 부족: 테스트"})]
