"""f0_bass worker: CREPE (torchcrepe 0.0.24) and PESTO (2.0.1) F0 tracks of bass audio (DESIGN 5.3, M5).

Request (+ the common envelope of M1_M2_SPEC 7.1)::

    {"views": [{"id": "bass_mono", "wav": "<abs wav>", "out": "<abs .npz>"}, ...],
     "params": {"hop_s": 0.01,
                "crepe": {"model": "full", "fmin_hz": 32.0, "fmax_hz": 400.0, "batch_frames": 256,
                          "path_halfwidth_bins": 4},
                "pesto": {"model": "mir-1k_g7"}},
     "device": "auto|cuda|cpu", "vram": {...}, "determinism": true}

Per view it writes ``out`` (``crepe_hz``, ``crepe_conf``, ``pesto_hz``, ``pesto_conf``, ``hop_s``; frame ``i`` at
``i * hop_s``, ``floor(duration / hop_s) + 1`` frames). pyin runs in the core env (``bandscribe.bass.f0``).

- **Weights come from the installed packages only** (``torchcrepe/assets/<model>.pth``,
  ``pesto/weights/<model>.ckpt``), always loaded with ``torch.load(weights_only=True)``; their sha256 goes into
  the output. PESTO's own loader uses ``weights_only=False`` (its ``mir-1k_g7`` checkpoint pickles OmegaConf
  containers), so the model is built here from a checkpoint read with an allow-list of exactly the classes it
  holds (``PESTO_SAFE_GLOBALS``); anything else in a future checkpoint fails loudly.
- **CREPE:** activations in batches of ``batch_frames`` (resampled to 16 kHz with soxr), then one Viterbi over the
  whole view (torchcrepe's ``predict`` decodes every batch on its own and breaks the path at the batch edges) with
  torchcrepe's transition (a 12-bin triangle) on the salience normalised to sum 1 (torchcrepe's softmax of the
  sigmoid outputs is nearly flat and sticks to the lowest bin on bass), and the pitch as the activation-weighted mean of the bins within
  ``path_halfwidth_bins`` of the path (torchcrepe dithers instead: not reproducible). Bins outside
  ``fmin_hz .. fmax_hz`` are masked like torchcrepe does. ``fmin_hz`` below 31.71 Hz is refused: CREPE's lowest bin
  is 31.70 Hz and torchcrepe's bin index for a lower fmin is negative, which masks all but the top bins.
- **PESTO:** on the same 16 kHz audio after CREPE has left the GPU, in 30 s chunks with 1 s of context on each
  side (its CQT over a whole song at once took > 5 GB on the 6 GB card).

Rungs ``crepe-<model>`` (CUDA) and ``crepe-<model>@cpu`` (only with ``vram.cpu_allowed``; CREPE 'full' on a CPU is
~4x slower than real time).
"""

from __future__ import annotations

import collections
import hashlib
import logging
import math
import os
import time
import typing
from pathlib import Path
from typing import Any

from bandscribe.workers import gpu_common
from bandscribe.workers.base import MODEL_LOADED, WorkerContext, worker_main

log = logging.getLogger(__name__)

BACKEND = "f0_bass"
VERSION = "1"
CREPE_SR = 16000
CREPE_BINS = 360
CREPE_CENTS0 = 1997.3794084376191  # cents (re 10 Hz) of bin 0
CREPE_CENTS_PER_BIN = 20.0
MIN_FMIN_HZ = 31.71


def rung_labels(model: str) -> list[str]:
    return [f"crepe-{model}"]


def build_rungs(model: str) -> list[dict[str, Any]]:
    return [{"label": f"crepe-{model}", "device": "cuda", "params": {}},
            {"label": f"crepe-{model}@cpu", "device": "cpu", "params": {}}]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _dist_version(dist: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None


def pesto_safe_globals() -> list[Any]:
    """The classes ``mir-1k_g7.ckpt`` pickles besides tensors (checked 2026-10-08: OmegaConf containers and their
    metadata, typing.Any, defaultdict / OrderedDict, plain builtins)."""
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.dictconfig import DictConfig
    from omegaconf.listconfig import ListConfig
    from omegaconf.nodes import AnyNode

    return [ListConfig, DictConfig, ContainerMetadata, Metadata, AnyNode, typing.Any, collections.defaultdict,
            collections.OrderedDict, dict, list, int]


def _plain(x: Any) -> Any:
    """OmegaConf containers -> plain dict / list (PESTO's constructors take plain kwargs)."""
    if hasattr(x, "items"):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)) or type(x).__name__ == "ListConfig":
        return [_plain(v) for v in x]
    return x


def load_pesto(torch: Any, name: str, hop_ms: float, sr: int) -> tuple[Any, Path]:
    import pesto
    from pesto.data import Preprocessor
    from pesto.model import PESTO, Resnet1d

    path = Path(os.path.dirname(pesto.__file__)) / "weights" / f"{name}.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"PESTO checkpoint not in the installed package: {path}")
    with torch.serialization.safe_globals(pesto_safe_globals()):
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
    hp = _plain(ck["hparams"])
    hcqt = _plain(ck["hcqt_params"])
    pre = Preprocessor(hop_size=float(hop_ms), sampling_rate=int(sr), **hcqt)
    model = PESTO(Resnet1d(**hp["encoder"]), preprocessor=pre, crop_kwargs=hp["pitch_shift"],
                  reduction=hp["reduction"])
    res = model.load_state_dict(ck["state_dict"], strict=False)
    # the CQT kernels are rebuilt by the preprocessor and never stored: those two keys are the only allowed gap
    missing = [k for k in res.missing_keys if not k.startswith("preprocessor.hcqt_kernels.")]
    if missing or res.unexpected_keys:
        raise RuntimeError(f"PESTO checkpoint does not fit the model: missing {missing}, "
                           f"unexpected {list(res.unexpected_keys)}")
    model.eval()
    return model, path


def crepe_transition(np: Any) -> Any:
    xx, yy = np.meshgrid(range(CREPE_BINS), range(CREPE_BINS))
    t = np.maximum(12 - np.abs(xx - yy), 0).astype(np.float64)
    return t / t.sum(axis=1, keepdims=True)


def decode_crepe(np: Any, act: Any, fmin_hz: float, fmax_hz: float, halfwidth: int) -> tuple[Any, Any]:
    """(hz, conf) per frame from CREPE activations ``act`` (frames x 360, sigmoid outputs): one Viterbi over every
    frame, then the activation-weighted mean pitch within ``halfwidth`` bins of the path bin."""
    import librosa

    act = np.asarray(act, dtype=np.float64)
    lo = int(math.floor((1200.0 * math.log2(fmin_hz / 10.0) - CREPE_CENTS0) / CREPE_CENTS_PER_BIN))
    hi = int(math.ceil((1200.0 * math.log2(fmax_hz / 10.0) - CREPE_CENTS0) / CREPE_CENTS_PER_BIN))
    if lo < 0:
        raise ValueError(f"fmin {fmin_hz} Hz is below CREPE's lowest bin (31.70 Hz)")
    # observations = the salience normalised to sum 1 (original CREPE). torchcrepe's ``decode.viterbi`` takes the
    # softmax of the sigmoid outputs instead: values in [0, 1] give an almost flat distribution (max/min <= e), the
    # transition prior wins and on bass stems the path stays on the lowest bin through whole phrases (seisyun
    # complex 34-36 s: every frame at bin 0 with confidence 0 while the activations peak at 0.8-0.9 on the notes).
    masked = np.zeros_like(act)
    masked[:, lo:hi] = act[:, lo:hi]
    prob = masked / np.maximum(np.sum(masked, axis=1, keepdims=True), 1e-12)
    path = librosa.sequence.viterbi(prob.T, crepe_transition(np)).astype(np.int64)
    idx = np.arange(len(path))
    conf = act[idx, path]
    cents_bins = CREPE_CENTS0 + CREPE_CENTS_PER_BIN * np.arange(CREPE_BINS)
    w_lo = np.clip(path - halfwidth, 0, CREPE_BINS - 1)
    w_hi = np.clip(path + halfwidth + 1, 1, CREPE_BINS)
    cents = np.empty(len(path))
    for i in range(len(path)):  # 2e4 frames per 3 min: a plain loop is fast enough and obviously right
        w = act[i, w_lo[i]:w_hi[i]]
        s = float(np.sum(w))
        cents[i] = float(np.dot(w, cents_bins[w_lo[i]:w_hi[i]]) / s) if s > 0 else cents_bins[path[i]]
    hz = 10.0 * 2.0 ** (cents / 1200.0)
    return hz.astype(np.float32), conf.astype(np.float32)


PESTO_CHUNK_FRAMES = 3000  # 30 s per call
PESTO_CONTEXT_FRAMES = 100  # 1 s of audio on each side (the longest VQT kernel of mir-1k_g7 is ~0.13 s)


def run_pesto(np: Any, torch: Any, model: Any, x: Any, hop: int, n: int, dev: str) -> tuple[Any, Any]:
    """(hz, conf) of ``n`` frames of ``x`` (16 kHz) in chunks of ``PESTO_CHUNK_FRAMES`` with
    ``PESTO_CONTEXT_FRAMES`` of real audio around each (PESTO has no per-input normalisation: the log CQT and
    the encoder's layer norms work per frame, so chunks give the same frames as one call)."""
    hz = np.zeros(n, dtype=np.float32)
    conf = np.zeros(n, dtype=np.float32)
    for f0 in range(0, n, PESTO_CHUNK_FRAMES):
        f1 = min(n, f0 + PESTO_CHUNK_FRAMES)
        g0 = max(0, f0 - PESTO_CONTEXT_FRAMES)
        a, b = g0 * hop, min(len(x), (f1 + PESTO_CONTEXT_FRAMES) * hop + 1)
        if b <= a:
            break
        pred, c, _vol = model(torch.from_numpy(x[a:b]).to(dev), sr=CREPE_SR, convert_to_freq=True,
                              return_activations=False)
        pred = pred.float().cpu().numpy().reshape(-1)
        c = c.float().cpu().numpy().reshape(-1)
        k0, k1 = f0 - g0, min(f1 - g0, len(pred))
        if k1 > k0:
            hz[f0:f0 + (k1 - k0)] = pred[k0:k1]
            conf[f0:f0 + (k1 - k0)] = c[k0:k1]
    return hz, conf


def _frames(duration_s: float, hop_s: float) -> int:
    return int(math.floor(duration_s / hop_s)) + 1


def _fit(np: Any, a: Any, n: int) -> Any:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    return a[:n] if len(a) >= n else np.concatenate([a, np.zeros(n - len(a), dtype=np.float32)])


def _track(request: dict, ctx: WorkerContext) -> dict[str, Any]:
    gpu_common.refuse_bf16(request)
    gpu_common.hide_cuda_for_cpu_request(request)
    gpu_common.apply_determinism(bool(request.get("determinism", True)))

    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    import torchcrepe

    from bandscribe.bass import f0 as F0

    params = request.get("params") or {}
    hop_s = float(params.get("hop_s", 0.01))
    cp = {**F0.DEFAULT_PARAMS["crepe"], **dict(params.get("crepe") or {})}
    pp = {**F0.DEFAULT_PARAMS["pesto"], **dict(params.get("pesto") or {})}
    if float(cp["fmin_hz"]) < MIN_FMIN_HZ:
        raise ValueError(f"crepe fmin_hz {cp['fmin_hz']} < {MIN_FMIN_HZ}: below CREPE's lowest bin (31.70 Hz), "
                         "torchcrepe would mask every bin but the top ones")
    if abs(hop_s * CREPE_SR - round(hop_s * CREPE_SR)) > 1e-6:
        raise ValueError(f"hop_s {hop_s} is not a whole number of samples at 16 kHz")
    views = list(request.get("views") or [])
    if not views:
        raise ValueError("no views")
    out: dict[str, Any] = ctx.partial
    model_name = str(cp["model"])
    crepe_path = Path(os.path.dirname(torchcrepe.__file__)) / "assets" / f"{model_name}.pth"
    if not crepe_path.is_file():
        raise FileNotFoundError(f"torchcrepe weights not in the installed package: {crepe_path}")

    audio16: list[tuple[dict, Any, float]] = []  # (view, mono at 16 kHz, duration of the original)
    for v in views:
        data, sr = sf.read(str(v["wav"]), dtype="float32", always_2d=True)
        mono = data.mean(axis=1).astype(np.float64)
        x16 = mono if int(sr) == CREPE_SR else soxr.resample(mono, int(sr), CREPE_SR, quality="HQ")
        audio16.append((v, np.ascontiguousarray(x16, dtype=np.float32), mono.shape[0] / float(sr)))
    audio_s = float(sum(a[2] for a in audio16))

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
    shas: dict[str, str] = {"crepe": _sha256(crepe_path)}

    def attempt(rung: dict, wd: gpu_common.Watchdog) -> list[dict[str, Any]]:
        dev = "cuda" if rung["device"] == "cuda" else "cpu"
        pesto_model = None
        try:
            t0 = time.perf_counter()
            torchcrepe.load.model(dev, model_name)  # weights_only=True inside torchcrepe 0.0.24
            if dev == "cuda":
                torch.cuda.synchronize()
                if ctx.sampler is not None and hasattr(ctx.sampler, "mark"):
                    ctx.sampler.mark(MODEL_LOADED)
            timing["load_s"] = round(time.perf_counter() - t0, 3)
            wd.arm()
            results = []
            done = 0.0
            hop = int(round(hop_s * CREPE_SR))
            t1 = time.perf_counter()
            # pass 1: CREPE on every view
            for v, x16, dur in audio16:
                n = _frames(dur, hop_s)
                acts = []
                for frames in torchcrepe.preprocess(torch.from_numpy(x16)[None], CREPE_SR, hop, int(cp["batch_frames"]),
                                                    dev, pad=True):
                    acts.append(torchcrepe.infer(frames, model_name, dev).float().cpu().numpy())
                act = np.concatenate(acts, axis=0) if acts else np.zeros((0, CREPE_BINS), dtype=np.float32)
                c_hz, c_conf = decode_crepe(np, act, float(cp["fmin_hz"]), float(cp["fmax_hz"]),
                                            int(cp["path_halfwidth_bins"]))
                results.append({"view": v, "n": n, "audio_s": dur, "crepe_hz": _fit(np, c_hz, n),
                                "crepe_conf": _fit(np, c_conf, n)})
                done += dur / 2.0
                wd.check(done)
            torchcrepe.infer.model.to("cpu")
            del torchcrepe.infer.model
            gpu_common._free_cuda()
            # pass 2: PESTO on every view, in chunks (its CQT over a whole song needs several GB)
            pesto_model, ppath = load_pesto(torch, str(pp["model"]), hop_s * 1000.0, CREPE_SR)
            pesto_model = pesto_model.to(dev)
            shas["pesto"] = _sha256(ppath)
            for r, (_v, x16, dur) in zip(results, audio16):
                p_hz, p_conf = run_pesto(np, torch, pesto_model, x16, hop, r["n"], dev)
                r["pesto_hz"], r["pesto_conf"] = p_hz, p_conf
                if dev == "cuda":
                    torch.cuda.synchronize()
                done += dur / 2.0
                wd.check(done)
            timing["process_s"] = round(time.perf_counter() - t1, 3)
            return results
        except BaseException:
            if getattr(torchcrepe.infer, "model", None) is not None:
                torchcrepe.infer.model.to("cpu")
                del torchcrepe.infer.model
            if pesto_model is not None:
                pesto_model.to("cpu")
                del pesto_model
            raise

    with torch.inference_mode():
        result, attempts = gpu_common.run_ladder(build_rungs(model_name), attempt, sampler=ctx.sampler,
                                                 req=request, audio_s_total=audio_s)
    vram["attempts"] = gpu_common.attempts_summary(attempts)
    warnings = [f"{a.rung_label}: 실시간 배수 {a.rtf}x 로 기준보다 느림(다른 GPU 작업과 경합했을 수 있음)"
                for a in attempts if a.slow]
    backend = {"name": BACKEND, "version": VERSION, "torchcrepe": _dist_version("torchcrepe"),
               "pesto": _dist_version("pesto-pitch"), "crepe_model": model_name, "pesto_model": str(pp["model"]),
               "weights_sha256": shas, "device_used": None}
    common = {"backend": backend, "vram": vram, "warnings": warnings, "rung_labels": rung_labels(model_name)}
    if result is None:
        return {"status": "insufficient_vram", **common,
                "timing": {"load_s": timing.get("load_s"), "process_s": None, "audio_s": round(audio_s, 3),
                           "rtf": None}, "files": {}}
    ok = next(a for a in attempts if a.status == "ok")
    vram["rung_index"], vram["rung_label"] = ok.rung_index, ok.rung_label
    backend["device_used"] = "cpu" if ok.device == "cpu" else "cuda"
    files: dict[str, dict[str, Any]] = {}
    for r in result:
        tr = F0.F0Track(hz={"crepe": r["crepe_hz"], "pesto": r["pesto_hz"]},
                        conf={"crepe": r["crepe_conf"], "pesto": r["pesto_conf"]}, hop_s=hop_s)
        size = F0.save_npz(Path(r["view"]["out"]), tr)
        files[str(r["view"]["id"])] = {"path": str(r["view"]["out"]), "bytes": size, "frames": int(r["n"]),
                                       "audio_s": round(float(r["audio_s"]), 3)}
    process_s = timing.get("process_s") or 0.0
    rtf = round(audio_s / process_s, 3) if process_s > 0 else None
    return {"status": "ok", **common, "files": files,
            "timing": {"load_s": timing.get("load_s"), "process_s": process_s, "audio_s": round(audio_s, 3),
                       "rtf": rtf}}


def handle(request: dict, ctx: WorkerContext) -> dict:
    mode = request.get("mode", "track")
    if mode == "track":
        return _track(request, ctx)
    raise ValueError(f"unknown f0_bass mode: {mode!r}")


if __name__ == "__main__":
    worker_main(handle, name=BACKEND)
