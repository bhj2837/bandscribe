"""sep_msst worker: BS-RoFormer SW 6-stem separation with the vendored MSST model (M1_M2_SPEC 7.2; DESIGN S3).

Request (+ the common envelope of M1_M2_SPEC 7.1)::

    {"mode": "separate" (default) | "selftest_identity",
     "mix_wav": "<abs>", "out_dir": "<abs>",
     "model": {"ckpt", "ckpt_sha256", "config_yaml", "arch": "bs_roformer"},
     "chunk_ladder": [588800, 352256, 262144], "num_overlap": 2, "batch_size": 1,
     "dtype": "upstream" | "fp16", "input_gain_db": 0.0, "outputs": {"stems_dir": "stems"},
     "device": "auto|cuda|cpu", "vram": {...}, "determinism": true}

Rungs ``chunk=<n>`` in ladder order; the CPU rung ``chunk=<smallest>@cpu`` exists only if ``vram.cpu_allowed``
(policy ``cpu`` and ``sep_msst`` in ``gpu.cpu_backends`` - not by default: ~23 min per 4-min song).

Steps: verify the checkpoint sha256 -> build ``BSRoformer`` from the yaml and load the weights strictly on the
CPU (``weights_only=True``) -> create the CUDA context and measure it (``ctx_mb``) -> ladder: model to the
rung's device, ``mark("model_loaded")`` + ``Watchdog.arm()``, our ``demix`` (CPU accumulators) with a watchdog
check after every chunk -> stems in yaml ``training.instruments`` order, each multiplied by the inverse input
gain, written as stereo float32 WAVs with ``bandscribe.audio.write_f32`` (no timestamped PEAK chunk).

dtype ``upstream`` = what MSST does (fp32 weights + fp16 autocast on CUDA, yaml ``training.use_amp: true``).
``fp16`` (E24-sep only) stores the ``nn.Linear`` weights in fp16 - the layers autocast runs in fp16 anyway, so
this halves their memory without changing which ops run in fp16 (norms, STFT and the complex masking stay
fp32). bfloat16 is refused (sm_75).

Modes: ``selftest_identity`` runs the vendored demix loop with an identity "model" on the CPU and reports the
reconstruction error (``tests/test_sep_demix.py``); ``parity_upstream`` (test-only, CUDA) compares our demix with
upstream MSST's on the SW model (``tests/test_sep_upstream_parity.py``).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from bandscribe.workers import gpu_common
from bandscribe.workers.base import MODEL_LOADED, WorkerContext, worker_main

log = logging.getLogger(__name__)

BACKEND = "sep_msst"
VERSION = "1"
SR = 44100
DTYPES = ("upstream", "fp16")


def rung_labels(ladder: list[int]) -> list[str]:
    return [f"chunk={int(c)}" for c in ladder]


def build_rungs(ladder: list[int]) -> list[dict[str, Any]]:
    rungs = [{"label": f"chunk={int(c)}", "device": "cuda", "params": {"chunk_size": int(c)}} for c in ladder]
    smallest = min(int(c) for c in ladder)
    rungs.append({"label": f"chunk={smallest}@cpu", "device": "cpu", "params": {"chunk_size": smallest}})
    return rungs


def _sha256(path: Path) -> str:
    # The (path, size, mtime) memo shared with the core: the core hashed the checkpoint at plan time already.
    from bandscribe import models

    return models.sha256_cached(path)


def _read_mix(path: Path) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, ch)
    return np.ascontiguousarray(data.T), int(sr)


def _write_stereo(path: Path, data_ct: Any, sr: int) -> int:
    """(channels, time) float32 -> deterministic float WAV; returns bytes written."""
    import numpy as np

    from bandscribe import audio

    path.parent.mkdir(parents=True, exist_ok=True)
    audio.write_f32(path, np.ascontiguousarray(data_ct.T, dtype=np.float32), sr)
    return path.stat().st_size


def _halve_linears(model: Any) -> int:
    import torch

    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.half()
            n += 1
    return n


def _separate(request: dict, ctx: WorkerContext) -> dict[str, Any]:
    gpu_common.refuse_bf16(request)
    # Before torch touches CUDA: a CPU-only request must not create a CUDA context (MSST's Attend reads the
    # device properties while the model is built, even on the CPU).
    gpu_common.hide_cuda_for_cpu_request(request)
    dtype = str(request.get("dtype", "upstream"))
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    gpu_common.apply_determinism(bool(request.get("determinism", True)))

    import numpy as np
    import torch

    from bandscribe.third_party.msst import MSST_COMMIT, loader
    from bandscribe.third_party.msst.demix import demix

    out: dict[str, Any] = ctx.partial
    warnings: list[str] = []
    out_dir = Path(request["out_dir"])
    stems_dir = (request.get("outputs") or {}).get("stems_dir", "stems")
    ladder = [int(c) for c in request.get("chunk_ladder", [588800, 352256, 262144])]
    num_overlap = int(request.get("num_overlap", 2))
    batch_size = int(request.get("batch_size", 1))
    gain_db = float(request.get("input_gain_db", 0.0) or 0.0)
    vram_req = request.get("vram") or {}
    model_req = request["model"]
    if model_req.get("arch", "bs_roformer") != "bs_roformer":
        raise ValueError(f"unsupported arch {model_req.get('arch')!r}")

    device = gpu_common.pick_device(request)
    vram: dict[str, Any] = {"device": str(device) if device.type == "cpu" else "cuda:0", "total_mb": None,
                            "start_free_mb": None, "ctx_mb": None, "start_nvml_used_mb": None, "rung_index": None,
                            "rung_label": None, "cap_mb": vram_req.get("cap_mb"), "attempts": []}
    out["vram"] = vram
    if device.type == "cuda":
        vram["ctx_mb"] = gpu_common.init_cuda_measure_ctx(ctx.sampler)
        free, total = gpu_common.mem_get_info_mb()
        vram["start_free_mb"], vram["total_mb"] = free, total
        snap = ctx.sampler.snapshot() if ctx.sampler is not None else {}
        vram["start_nvml_used_mb"] = snap.get("nvml_used_mb")
        gpu_common.apply_cap(request)

    # --- model (CPU first: the checkpoint is loaded with weights_only=True, strictly) ---
    t0 = time.perf_counter()
    ckpt = Path(model_req["ckpt"])
    cfg_path = Path(model_req["config_yaml"])
    ckpt_sha = _sha256(ckpt)
    want = str(model_req.get("ckpt_sha256") or "")
    if want and ckpt_sha != want:
        raise ValueError(f"checkpoint sha256 mismatch for {ckpt}: {ckpt_sha} != expected {want}")
    config_sha = _sha256(cfg_path)
    config = loader.load_yaml(cfg_path)
    stems = loader.stem_order(config)
    use_amp = bool((config.get("training") or {}).get("use_amp", True))
    model = loader.build_model(config)
    load_info = loader.load_weights(model, ckpt)
    model.eval()
    if dtype == "fp16":
        load_info["fp16_linears"] = _halve_linears(model)
    zero_dc = bool(getattr(model, "zero_dc", True))
    load_s = round(time.perf_counter() - t0, 3)
    out["load"] = load_info

    mix, sr = _read_mix(Path(request["mix_wav"]))
    if sr != SR:
        raise ValueError(f"mix must be {SR} Hz, got {sr}")
    audio_s = mix.shape[1] / sr
    gain = 10.0 ** (gain_db / 20.0)
    if gain_db != 0.0:
        mix *= np.float32(gain)

    def attempt(rung: dict, wd: gpu_common.Watchdog) -> tuple[Any, float]:
        dev = torch.device("cuda", 0) if rung["device"] == "cuda" else torch.device("cpu")
        try:
            if dtype == "fp16" and dev.type == "cpu":
                # fp16 weights rely on CUDA autocast (fp16 inputs); a CPU rung has no autocast -> plain fp32.
                model.float()
                warnings.append(f"{rung['label']}: CPU 칸은 fp16 가중치를 쓸 수 없어 fp32 로 실행했습니다")
            model.to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                if ctx.sampler is not None:
                    ctx.sampler.mark(MODEL_LOADED)
            wd.arm()
            t = time.perf_counter()
            est = demix(model, mix, chunk_size=int(rung["params"]["chunk_size"]), num_overlap=num_overlap,
                        num_stems=len(stems), device=dev, batch_size=batch_size, use_amp=use_amp,
                        on_chunk=lambda done, _total: wd.check(done), sample_rate=sr)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            return est, time.perf_counter() - t
        except BaseException:
            model.to("cpu")  # free the rung's VRAM before the ladder steps down
            raise

    result, attempts = gpu_common.run_ladder(build_rungs(ladder), attempt, sampler=ctx.sampler, req=request,
                                             audio_s_total=audio_s)
    vram["attempts"] = gpu_common.attempts_summary(attempts)
    for a in attempts:
        if a.slow:
            warnings.append(f"{a.rung_label}: 실시간 배수 {a.rtf}x 로 기준보다 느림(다른 GPU 작업과 경합했을 수 있음)")
    labels = rung_labels(ladder)
    backend = {"name": BACKEND, "version": VERSION, "model": "bs_roformer_sw", "model_sha256": ckpt_sha,
               "dtype_policy": "fp32+amp" if dtype == "upstream" else "fp16-linear+amp", "device_used": None}
    common = {"backend": backend, "vram": vram, "stem_order": stems, "config_sha256": config_sha,
              "zero_dc": zero_dc, "input_gain_db": gain_db, "num_overlap": num_overlap, "batch_size": batch_size,
              "msst_commit": MSST_COMMIT, "warnings": warnings}
    if result is None:
        log.warning("no rung fitted: answering insufficient_vram")
        return {"status": "insufficient_vram", **common,
                "timing": {"load_s": load_s, "process_s": None, "audio_s": round(audio_s, 3), "rtf": None},
                "files": {}}

    est, process_s = result
    ok = next(a for a in attempts if a.status == "ok")
    vram["rung_index"], vram["rung_label"] = ok.rung_index, ok.rung_label
    backend["device_used"] = "cpu" if ok.device == "cpu" else "cuda"
    chunk_size = build_rungs(ladder)[ok.rung_index]["params"]["chunk_size"]

    inv = np.float32(1.0 / gain)
    files: dict[str, dict[str, int]] = {}
    stem_paths: dict[str, str] = {}
    for k, name in enumerate(stems):
        data = est[k]
        if gain_db != 0.0:
            data = data * inv
        rel = f"{stems_dir}/{name}.wav"
        files[rel] = {"bytes": _write_stereo(out_dir / rel, data, sr)}
        stem_paths[name] = rel
    del est
    rtf = round(audio_s / process_s, 3) if process_s > 0 else None
    if not math.isfinite(rtf or 0.0):
        rtf = None
    return {"status": "ok", **common, "stems": stem_paths, "chunk_size": int(chunk_size),
            "timing": {"load_s": load_s, "process_s": round(process_s, 3), "audio_s": round(audio_s, 3), "rtf": rtf},
            "files": files, "rung_labels": labels}


def _selftest_identity(request: dict, ctx: WorkerContext) -> dict[str, Any]:
    """Vendored demix loop with an identity model on the CPU: max |output stem - input| per chunk size."""
    import numpy as np
    import torch

    from bandscribe.third_party.msst.demix import IdentityStems, demix

    mix, sr = _read_mix(Path(request["wav"]))
    n_stems = int(request.get("num_stems", 6))
    model = IdentityStems(n_stems)
    res: dict[str, Any] = {}
    for chunk in request.get("chunk_sizes", [262144, 588800]):
        t = time.perf_counter()
        est = demix(model, mix, chunk_size=int(chunk), num_overlap=int(request.get("num_overlap", 2)),
                    num_stems=n_stems, device=torch.device("cpu"), batch_size=int(request.get("batch_size", 1)),
                    use_amp=False)
        err = float(np.max(np.abs(est - mix[None]))) if est.size else 0.0
        res[str(int(chunk))] = {"max_abs_err": err, "shape": list(est.shape),
                                "seconds": round(time.perf_counter() - t, 3)}
    return {"status": "ok", "mode": "selftest_identity", "audio_s": round(mix.shape[1] / sr, 3), "results": res}


def _parity_upstream(request: dict, ctx: WorkerContext) -> dict[str, Any]:
    """Test-only (``tests/test_sep_upstream_parity.py``): our demix vs upstream MSST ``demix`` on CUDA.

    Request: ``{"mode": "parity_upstream", "ref_dir": <pristine checkout of msst_commit>, "wav", "ckpt",
    "config_yaml", "chunk_size"}``. Same model and weights, same clip, both loops in this process, so the
    comparison isolates the inference loop (CPU vs GPU accumulators, autocast API). Runs as a worker so the GPU
    is used only under the GPU lock and inside a Job Object, like every other bandscribe GPU job.
    Upstream ``utils/model_utils.py`` imports ``ml_collections`` (not in the gpu env) only for ``ConfigDict``
    type hints; a dict with attribute access stands in for it.
    """
    import sys
    import types

    import numpy as np
    import torch

    from bandscribe.third_party.msst import loader
    from bandscribe.third_party.msst.demix import demix as ours

    class _NS(dict):
        def __getattr__(self, k: str) -> Any:
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k) from None

    if "ml_collections" not in sys.modules:
        stub = types.ModuleType("ml_collections")
        stub.ConfigDict = _NS  # type: ignore[attr-defined]
        sys.modules["ml_collections"] = stub
    sys.path.insert(0, str(Path(request["ref_dir"])))
    from utils.model_utils import demix as upstream  # type: ignore[import-not-found]

    cfg = loader.load_yaml(request["config_yaml"])
    stems = loader.stem_order(cfg)
    model = loader.build_model(cfg)
    loader.load_weights(model, request["ckpt"])
    model.eval()
    mix, _sr = _read_mix(Path(request["wav"]))
    chunk = int(request.get("chunk_size", 588800))
    conf = _NS(audio=_NS(chunk_size=chunk), inference=_NS(batch_size=1, num_overlap=2),
               training=_NS(instruments=stems, use_amp=True))
    dev = torch.device("cuda", 0)
    model.to(dev)
    if ctx.sampler is not None:
        ctx.sampler.mark(MODEL_LOADED)
    up = upstream(conf, model, mix, dev, model_type="bs_roformer")
    mine = ours(model, mix, chunk_size=chunk, num_overlap=2, num_stems=len(stems), device=dev, use_amp=True)
    snr: dict[str, float] = {}
    for k, name in enumerate(stems):
        ref = np.asarray(up[name], dtype=np.float64)
        d = ref - mine[k].astype(np.float64)
        snr[name] = float(10 * np.log10(max(float((ref ** 2).sum()), 1e-30) / max(float((d ** 2).sum()), 1e-30)))
    del up, mine
    return {"status": "ok", "mode": "parity_upstream", "chunk_size": chunk, "snr_db": snr,
            "audio_s": round(mix.shape[1] / SR, 3)}


def handle(request: dict, ctx: WorkerContext) -> dict:
    mode = request.get("mode", "separate")
    if mode == "separate":
        return _separate(request, ctx)
    if mode == "selftest_identity":
        return _selftest_identity(request, ctx)
    if mode == "parity_upstream":
        return _parity_upstream(request, ctx)
    raise ValueError(f"unknown sep_msst mode: {mode!r}")


if __name__ == "__main__":
    worker_main(handle, name=BACKEND)
