"""beats_beatthis worker: Beat This! 1.1.0 beat/downbeat tracking (M1_M2_SPEC 7.5; DESIGN S2).

Request (+ the common envelope of M1_M2_SPEC 7.1)::

    {"wav": "<abs mix wav>", "out_dir": "<abs>",
     "checkpoint": "final0",                       # optional name for rung labels / tracker id (default: file stem)
     "checkpoint_path": "<abs local path>",        # from bandscribe.models.local_path("beat_this.final0")
     "checkpoint_sha256": "<hex>",
     "dbn": false, "float16": false,
     "out_beats": "beats.json", "out_activations": "activations.npz",
     "device": "auto|cuda|cpu", "vram": {...}, "determinism": true}

Rules (M1_M2_SPEC 0, 7.5):

- The worker **never** receives a hub name: a name would make ``beat_this.inference.load_checkpoint`` download
  inside the GPU lock through ``torch.hub.load_state_dict_from_url``, which loads with ``weights_only=False``.
  ``checkpoint_path`` must be an existing local file whose sha256 matches ``checkpoint_sha256``; beat_this
  then loads it with ``torch.load(..., weights_only=True)``. Verified 2026-09-30 on this PC: ``final0.ckpt``
  (sha256 8c328b45...) loads with ``weights_only=True`` as is - its ``hyper_parameters`` are plain ints,
  floats, strings, bools and dicts - so no ``torch.serialization.add_safe_globals`` entry is needed
  (``SAFE_GLOBALS`` stays empty; safe loading is never turned off).
- ``dbn`` is refused (needs madmom, not installed; ``grid.dbn = true`` is rejected at config load as well).
- Audio is read with soundfile (float32, (n, ch)) and passed **in memory** as ``(signal, sr)``:
  ``Audio2Frames`` averages the channels and resamples to 22050 Hz with soxr (torchaudio.load would need
  TorchCodec). One model load gives both outputs: ``Audio2Frames`` -> frame logits (50 fps) ->
  ``Postprocessor(type="minimal", fps=50)`` (its minimal postprocessing already snaps each downbeat to the
  nearest beat; times are quantised to 20 ms).

Rungs ``<checkpoint>`` (CUDA) and ``<checkpoint>@cpu`` (only if ``vram.cpu_allowed``: policy ``cpu`` and
``beats_beatthis`` in ``gpu.cpu_backends``, which is the default list). Outputs: ``beats.json`` (plain JSON
matching ``bandscribe.schema.grid.BeatsRaw``; workers never import bandscribe.schema) and ``activations.npz``
(``beat``/``downbeat`` logits float32, ``fps``).
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from bandscribe.workers import gpu_common
from bandscribe.workers.base import MODEL_LOADED, WorkerContext, worker_main

log = logging.getLogger(__name__)

BACKEND = "beats_beatthis"
VERSION = "1"
FPS = 50
# Types to allow for torch.load(weights_only=True) if a checkpoint's hyper-parameters needed them.
# Empty: final0 needs none (see module doc).
SAFE_GLOBALS: list[Any] = []


def rung_labels(checkpoint: str) -> list[str]:
    return [checkpoint]


def build_rungs(checkpoint: str) -> list[dict[str, Any]]:
    return [{"label": checkpoint, "device": "cuda", "params": {}},
            {"label": f"{checkpoint}@cpu", "device": "cpu", "params": {}}]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


@contextmanager
def _safe_torch_load(torch: Any, loads: list[dict[str, Any]]) -> Iterator[None]:
    """Refuse any ``torch.load`` without ``weights_only=True`` while the model is built (belt and braces: a
    beat_this update that loaded unsafely, or a hub fallback, fails loudly instead of unpickling code)."""
    orig = torch.load

    def guarded(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("weights_only") is not True:
            raise RuntimeError("torch.load without weights_only=True refused (Beat This! checkpoint)")
        loads.append({"weights_only": True, "path": str(args[0]) if args else None})
        return orig(*args, **kwargs)

    torch.load = guarded
    try:
        yield
    finally:
        torch.load = orig


def _dist_version(dist: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None


def _write_npz(path: Path, arrays: dict[str, Any]) -> int:
    """npz with a fixed member timestamp (np.savez stamps the current time into the zip headers)."""
    import numpy as np

    from bandscribe import atomic

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asanyarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 0
            info.external_attr = 0
            zf.writestr(info, member.getvalue())
    atomic.write_bytes(path, buf.getvalue())
    return path.stat().st_size


def _track(request: dict, ctx: WorkerContext) -> dict[str, Any]:
    gpu_common.refuse_bf16(request)
    # The one backend allowed a CPU rung (gpu.cpu_backends): a CPU-only request must not create a CUDA context.
    gpu_common.hide_cuda_for_cpu_request(request)
    if request.get("dbn"):
        raise ValueError("dbn=true needs madmom, which is not installed (grid.dbn is not supported)")
    if request.get("float16"):
        raise ValueError("float16=true is not supported: the grid stage runs Beat This! in fp32")
    gpu_common.apply_determinism(bool(request.get("determinism", True)))

    import numpy as np
    import soundfile as sf
    import torch

    out: dict[str, Any] = ctx.partial
    out_dir = Path(request["out_dir"])
    ckpt = Path(request["checkpoint_path"])
    checkpoint = str(request.get("checkpoint") or ckpt.stem)  # rung label / tracker id, e.g. "final0"
    if not ckpt.is_absolute() or not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint_path must be an existing local file (never a hub name): {ckpt}")
    ckpt_sha = _sha256(ckpt)
    want = str(request.get("checkpoint_sha256") or "")
    if not want:
        raise ValueError("checkpoint_sha256 is required")
    if ckpt_sha != want:
        raise ValueError(f"checkpoint sha256 mismatch for {ckpt}: {ckpt_sha} != expected {want}")
    if SAFE_GLOBALS:
        torch.serialization.add_safe_globals(SAFE_GLOBALS)

    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor

    data, sr = sf.read(str(request["wav"]), dtype="float32", always_2d=True)  # (n, ch)
    audio_s = data.shape[0] / float(sr)

    device = gpu_common.pick_device(request)
    vram_req = request.get("vram") or {}
    vram: dict[str, Any] = {"device": "cuda:0" if device.type == "cuda" else "cpu", "total_mb": None,
                            "start_free_mb": None, "ctx_mb": None, "start_nvml_used_mb": None, "rung_index": None,
                            "rung_label": None, "cap_mb": vram_req.get("cap_mb"), "attempts": []}
    out["vram"] = vram
    if device.type == "cuda":
        vram["ctx_mb"] = gpu_common.init_cuda_measure_ctx(ctx.sampler)
        free, total = gpu_common.mem_get_info_mb()
        vram["start_free_mb"], vram["total_mb"] = free, total
        snap = ctx.sampler.snapshot() if ctx.sampler is not None and hasattr(ctx.sampler, "snapshot") else {}
        vram["start_nvml_used_mb"] = snap.get("nvml_used_mb")
        gpu_common.apply_cap(request)

    timing: dict[str, float] = {}
    loads: list[dict[str, Any]] = []

    def attempt(rung: dict, wd: gpu_common.Watchdog) -> tuple[Any, Any, Any, Any]:
        dev = "cuda" if rung["device"] == "cuda" else "cpu"
        a2f = None
        try:
            t0 = time.perf_counter()
            with _safe_torch_load(torch, loads):
                a2f = Audio2Frames(checkpoint_path=str(ckpt), device=dev, float16=False)
            if dev == "cuda":
                torch.cuda.synchronize()
                if ctx.sampler is not None and hasattr(ctx.sampler, "mark"):
                    ctx.sampler.mark(MODEL_LOADED)
            timing["load_s"] = round(time.perf_counter() - t0, 3)
            wd.arm()
            t1 = time.perf_counter()
            beat_logits, down_logits = a2f(data, sr)
            if dev == "cuda":
                torch.cuda.synchronize()
            wd.check(audio_s)
            beats, downbeats = Postprocessor(type="minimal", fps=FPS)(beat_logits, down_logits)
            timing["process_s"] = round(time.perf_counter() - t1, 3)
            return (beat_logits.float().cpu().numpy(), down_logits.float().cpu().numpy(),
                    np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64))
        except BaseException:
            if a2f is not None:
                a2f.model.to("cpu")  # free the rung's VRAM before the ladder steps down
                del a2f
            raise

    with torch.inference_mode():
        result, attempts = gpu_common.run_ladder(build_rungs(checkpoint), attempt, sampler=ctx.sampler,
                                                 req=request, audio_s_total=audio_s)
    vram["attempts"] = gpu_common.attempts_summary(attempts)
    warnings = [f"{a.rung_label}: 실시간 배수 {a.rtf}x 로 기준보다 느림(다른 GPU 작업과 경합했을 수 있음)"
                for a in attempts if a.slow]
    backend = {"name": BACKEND, "version": _dist_version("beat-this"),
               "model": f"beat_this.{checkpoint}", "model_sha256": ckpt_sha, "dtype_policy": "fp32",
               "device_used": None}
    common = {"backend": backend, "vram": vram, "warnings": warnings, "checkpoint_path": str(ckpt),
              "checkpoint_sha256": ckpt_sha, "checkpoint_loads": loads, "rung_labels": rung_labels(checkpoint)}
    if result is None:
        log.warning("no rung fitted: answering insufficient_vram")
        return {"status": "insufficient_vram", **common,
                "timing": {"load_s": timing.get("load_s"), "process_s": None, "audio_s": round(audio_s, 3),
                           "rtf": None}, "files": {}}

    beat_logits, down_logits, beats, downbeats = result
    ok = next(a for a in attempts if a.status == "ok")
    vram["rung_index"], vram["rung_label"] = ok.rung_index, ok.rung_label
    backend["device_used"] = "cpu" if ok.device == "cpu" else "cuda"

    files: dict[str, dict[str, int]] = {}
    beats_rel = str(request.get("out_beats", "beats.json"))
    acts_rel = str(request.get("out_activations", "activations.npz"))
    doc = {"format": "bandscribe.beats/1", "tracker": f"beat_this/{checkpoint}", "checkpoint_sha256": ckpt_sha,
           "dbn": False, "fps": float(FPS), "beats_s": [round(float(t), 6) for t in beats],
           "downbeats_s": [round(float(t), 6) for t in downbeats], "duration_s": round(audio_s, 6)}
    from bandscribe import atomic

    atomic.write_json(out_dir / beats_rel, doc)
    files[beats_rel] = {"bytes": (out_dir / beats_rel).stat().st_size}
    files[acts_rel] = {"bytes": _write_npz(out_dir / acts_rel, {
        "beat": beat_logits.astype(np.float32), "downbeat": down_logits.astype(np.float32),
        "fps": np.asarray(float(FPS))})}
    process_s = timing.get("process_s") or 0.0
    rtf = round(audio_s / process_s, 3) if process_s > 0 else None
    if rtf is not None and not math.isfinite(rtf):
        rtf = None
    return {"status": "ok", **common, "n_beats": len(doc["beats_s"]), "n_downbeats": len(doc["downbeats_s"]),
            "timing": {"load_s": timing.get("load_s"), "process_s": process_s, "audio_s": round(audio_s, 3),
                       "rtf": rtf},
            "files": files}


def handle(request: dict, ctx: WorkerContext) -> dict:
    mode = request.get("mode", "track")
    if mode == "track":
        return _track(request, ctx)
    raise ValueError(f"unknown beats_beatthis mode: {mode!r}")


if __name__ == "__main__":
    worker_main(handle, name=BACKEND)
