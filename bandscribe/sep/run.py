"""The one entry point for BS-RoFormer SW separation (M1_M2_SPEC 9.1 item 6), used by the ``sep`` stage and by
code outside jobs (AMT's E1 / E24-sep, EVAL scenes, SEP's stereo-preservation check).

Both paths build the same worker request (``build_request``) and go through ``bandscribe.vram.run_gpu_stage``
(VRAM pre-check, wait policy, bounded retries) and ``bandscribe.vram.write_gpu_run`` (rung provenance), then write
``sep.json`` (M1_M2_SPEC 6.4) next to the stems.

Stage params (= the cache key, M1_M2_SPEC 3.2 row 4) are exactly: ``ckpt_sha256``, ``config_sha256``,
``msst_commit``, ``chunk_ladder``, ``num_overlap``, ``batch_size``, ``dtype``, ``input_lufs``. The model directory,
``leftover_warn_db`` and the VRAM knobs do not change the stems and stay out of the key.

``input_lufs`` (E1): ``"off"`` or a target loudness. The gain ``input_lufs - integrated LUFS of the mix`` (rounded
to 0.01 dB) is computed here, core-side, from the mix itself (so it is covered by the input hash); the worker
multiplies the mix by it and every stem by its inverse.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths, vram
from bandscribe.models import ModelMissing, sha256_cached

log = logging.getLogger(__name__)

BACKEND = "sep_msst"
SEP_FORMAT = "bandscribe.sep/1"
STEM_ORDER = ("bass", "drums", "other", "vocals", "guitar", "piano")  # SW yaml training.instruments
MODEL_CKPT = "bs_roformer_sw.ckpt"
MODEL_YAML = "bs_roformer_sw.yaml"
DTYPES = ("upstream", "fp16")

# Mirrors of the [sep] defaults (M1_M2_SPEC 1.3) for callers without a full Config (experiments and tests pass
# partial dicts); a real Config is read strictly by vram.cfg_get. tests/test_vram.py keeps them equal to Config().
SEP_DEFAULTS: dict[str, Any] = {
    "sep.backend": "sw_msst",
    "sep.model_dir": "data/models/bs_roformer_sw",
    "sep.ckpt": "BS-Rofo-SW-Fixed.ckpt",
    "sep.config": "BS-Rofo-SW-Fixed.yaml",
    "sep.ckpt_sha256": "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e",
    "sep.msst_commit": "84b1eac0887756b4f1a9d7a1ff49105939749ed2",
    "sep.chunk_ladder": [588800, 352256, 262144],
    "sep.num_overlap": 2,
    "sep.batch_size": 1,
    "sep.dtype": "upstream",
    "sep.input_lufs": "off",
    "sep.leftover_warn_db": -20.0,
}


def sep_value(cfg: Any, key: str) -> Any:
    return vram.cfg_get(cfg, key, SEP_DEFAULTS[key])


def rung_labels(ladder: list[int]) -> list[str]:
    """GPU rung labels in ladder order (the worker adds ``chunk=<smallest>@cpu`` only if CPU is allowed)."""
    return [f"chunk={int(c)}" for c in ladder]


def model_files(cfg: Any) -> tuple[Path, Path]:
    """(checkpoint, yaml) from ``sep.model_dir`` (relative to BANDSCRIBE_ROOT) + ``sep.ckpt`` / ``sep.config``."""
    d = Path(str(sep_value(cfg, "sep.model_dir")))
    if not d.is_absolute():
        d = paths.ROOT / d
    return d / str(sep_value(cfg, "sep.ckpt")), d / str(sep_value(cfg, "sep.config"))


def _model_missing(name: str, path: Path) -> ModelMissing:
    # The SW files have no download source (the uploader's account was deleted): the hint says where the kept
    # copy goes instead of naming `bandscribe models fetch`.
    return ModelMissing(name, f"분리 모델 파일이 없습니다: {path}. BS-RoFormer SW 체크포인트와 yaml 을 "
                              f"{path.parent} 에 두세요(원 계정이 삭제되어 받을 곳이 없습니다: 보관해 둔 사본을 쓰세요).")


def sep_models(cfg: Any) -> dict[str, str]:
    """Model name -> sha256 of the files the worker will load (plan time; raises ModelMissing if absent).

    The files come from ``sep.model_dir`` (the config names them; ``models.lock.json`` records the same files
    under these names for provenance). The memoised hash keeps a 700 MB checkpoint from being re-hashed on
    every ``bandscribe run``.
    """
    ckpt, yml = model_files(cfg)
    missing = [(n, p) for n, p in ((MODEL_CKPT, ckpt), (MODEL_YAML, yml)) if not p.is_file()]
    if missing:
        raise _model_missing(*missing[0])
    return {MODEL_CKPT: sha256_cached(ckpt), MODEL_YAML: sha256_cached(yml)}


def _check_dtype(dtype: str) -> str:
    if dtype not in DTYPES:
        raise ValueError(f"sep.dtype 는 {'|'.join(DTYPES)} 중 하나여야 합니다 (입력값: {dtype!r}); bf16 은 지원하지 않습니다")
    return dtype


def _check_input_lufs(v: Any) -> float | str:
    if v is None or (isinstance(v, str) and v.strip().lower() == "off"):
        return "off"
    try:
        f = float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"sep.input_lufs 는 \"off\" 또는 -30~-6 사이의 숫자여야 합니다 (입력값: {v!r})") from e
    if not (-30.0 <= f <= -6.0) or not math.isfinite(f):
        raise ValueError(f"sep.input_lufs 는 \"off\" 또는 -30~-6 사이의 숫자여야 합니다 (입력값: {v!r})")
    return f


def sep_params(cfg: Any, *, input_lufs: Any = None, dtype: str | None = None) -> dict[str, Any]:
    """The ``sep`` stage params = cache key (M1_M2_SPEC 3.2 row 4). Overrides are for experiments."""
    _ckpt, yml = model_files(cfg)
    config_sha = sha256_cached(yml) if yml.is_file() else None  # absent: sep_models raises at plan time
    return {
        "ckpt_sha256": str(sep_value(cfg, "sep.ckpt_sha256")),
        "config_sha256": config_sha,
        "msst_commit": str(sep_value(cfg, "sep.msst_commit")),
        "chunk_ladder": [int(c) for c in sep_value(cfg, "sep.chunk_ladder")],
        "num_overlap": int(sep_value(cfg, "sep.num_overlap")),
        "batch_size": int(sep_value(cfg, "sep.batch_size")),
        "dtype": _check_dtype(str(dtype if dtype is not None else sep_value(cfg, "sep.dtype"))),
        "input_lufs": _check_input_lufs(input_lufs if input_lufs is not None else sep_value(cfg, "sep.input_lufs")),
    }


def input_gain_db(mix_wav: Path, input_lufs: float | str) -> float:
    """Gain (dB, 0.01 dB steps) that brings the mix to ``input_lufs``; 0.0 when "off" or unmeasurable."""
    if input_lufs == "off":
        return 0.0
    from bandscribe.ingest import loudness

    lufs = loudness.measure(Path(mix_wav)).get("integrated_lufs")
    if lufs is None or not math.isfinite(float(lufs)):
        log.warning("integrated loudness of %s not measurable (silence?): input gain 0 dB", mix_wav)
        return 0.0
    return round(float(input_lufs) - float(lufs), 2)


def build_request(mix_wav: Path, out_dir: Path, params: Mapping[str, Any], cfg: Any, *, gain_db: float,
                  job: dict | None = None, force_rung: str | None = None, cap_mb: float | None = None,
                  precheck: bool = True, device: str = "auto") -> dict[str, Any]:
    """The sep_msst worker request (M1_M2_SPEC 7.1 envelope + 7.2)."""
    ckpt, yml = model_files(cfg)
    labels = rung_labels(list(params["chunk_ladder"]))
    if force_rung is not None and force_rung not in labels and not str(force_rung).endswith("@cpu"):
        raise ValueError(f"force_rung {force_rung!r} is not one of {labels}")
    return {
        "format": "bandscribe.worker/1",
        "job": job,
        "mode": "separate",
        "out_dir": str(Path(out_dir).resolve()),
        "device": device,
        "vram": vram.build_vram_request(BACKEND, labels, cfg, force_rung=force_rung, cap_mb=cap_mb,
                                        precheck=precheck),
        "determinism": True,
        "mix_wav": str(Path(mix_wav).resolve()),
        "model": {"ckpt": str(ckpt), "ckpt_sha256": params["ckpt_sha256"], "config_yaml": str(yml),
                  "arch": "bs_roformer"},
        "chunk_ladder": [int(c) for c in params["chunk_ladder"]],
        "num_overlap": int(params["num_overlap"]),
        "batch_size": int(params["batch_size"]),
        "dtype": params["dtype"],
        "input_gain_db": float(gain_db),
        "outputs": {"stems_dir": "stems"},
    }


@dataclass
class SepResult:
    stems: dict[str, Path]  # stem name -> stereo float32 WAV
    rung: dict[str, Any]  # {"index", "label", "device"} from gpu_run.json
    sep_json: dict[str, Any]
    out_dir: Path
    gpu_run: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def sep_json_doc(params: Mapping[str, Any], output: Mapping[str, Any], gpu_run: Mapping[str, Any]) -> dict[str, Any]:
    timing = output.get("timing") or {}
    be = output.get("backend") or {}
    return {
        "format": SEP_FORMAT,
        "backend": "sw_msst",
        "msst_commit": output.get("msst_commit", params.get("msst_commit")),
        "ckpt_sha256": be.get("model_sha256", params.get("ckpt_sha256")),
        "config_sha256": output.get("config_sha256", params.get("config_sha256")),
        "stem_order": list(output.get("stem_order") or STEM_ORDER),
        "chunk_size": output.get("chunk_size"),
        "num_overlap": output.get("num_overlap", params.get("num_overlap")),
        "batch_size": output.get("batch_size", params.get("batch_size")),
        "dtype": be.get("dtype_policy", "fp32+amp"),
        "zero_dc": output.get("zero_dc", True),
        "input_lufs": params.get("input_lufs"),
        "input_gain_db": output.get("input_gain_db", 0.0),
        "device_used": be.get("device_used"),
        "rung": {"index": gpu_run.get("rung", {}).get("index"), "label": gpu_run.get("rung", {}).get("label")},
        "audio_s": timing.get("audio_s"),
        "load_s": timing.get("load_s"),
        "process_s": timing.get("process_s"),
        "rtf": timing.get("rtf"),
        "vram": output.get("vram") or {},
        "warnings": list(output.get("warnings") or []),
    }


def run_separation(mix_wav: Path, out_dir: Path, cfg: Any, params: Mapping[str, Any], *, job: dict | None = None,
                   force_rung: str | None = None, notify: Callable[[str], None] | None = None) -> SepResult:
    """Run the worker into ``out_dir`` (stems/, _worker/), then write gpu_run.json and sep.json there."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gain = input_gain_db(Path(mix_wav), params["input_lufs"])
    request = build_request(mix_wav, out_dir, params, cfg, gain_db=gain, job=job, force_rung=force_rung)
    labels = rung_labels(list(params["chunk_ladder"]))
    res = vram.run_gpu_stage(BACKEND, labels, request, cfg, out_dir / "_worker", notify=notify)
    output = res.output
    if output.get("status") != "ok":
        raise RuntimeError(f"sep_msst 워커가 예상하지 못한 상태로 끝났습니다: {output.get('status')!r}")
    gpu_run = vram.write_gpu_run(out_dir, BACKEND, labels, output, getattr(res, "waited_s", 0.0))
    doc = sep_json_doc(params, output, gpu_run)
    atomic.write_json(out_dir / "sep.json", doc)
    stems: dict[str, Path] = {}
    for name, rel in (output.get("stems") or {}).items():
        p = out_dir / rel
        if not p.is_file():
            raise RuntimeError(f"sep_msst 워커가 스템을 쓰지 않았습니다: {p} (자세한 내용: {res.stderr_path})")
        stems[name] = p
    missing = [s for s in STEM_ORDER if s not in stems]
    if missing:
        raise RuntimeError(f"sep_msst 결과에 스템이 빠졌습니다: {missing}")
    if gpu_run.get("degraded"):
        log.warning("separation ran on a lower rung (%s)", gpu_run["rung"]["label"])
    return SepResult(stems=stems, rung=dict(gpu_run["rung"]), sep_json=doc, out_dir=out_dir, gpu_run=gpu_run,
                     warnings=list(doc["warnings"]))


def separate(mix_wav: Path, out_dir: Path, cfg: Any, *, force_rung: str | None = None,
             input_lufs: float | str | None = None, dtype: str | None = None,
             notify: Callable[[str], None] | None = None) -> SepResult:
    """Separate ``mix_wav`` (44.1 kHz float WAV) into ``out_dir``; overrides default to the config values.

    For experiments pass ``force_rung`` (e.g. ``"chunk=588800"``) so results never mix rungs (M1_M2_SPEC 10).
    """
    params = sep_params(cfg, input_lufs=input_lufs, dtype=dtype)
    if params["config_sha256"] is None:
        _ckpt, yml = model_files(cfg)
        raise _model_missing(MODEL_YAML, yml)
    return run_separation(Path(mix_wav), Path(out_dir), cfg, params, force_rung=force_rung, notify=notify)
