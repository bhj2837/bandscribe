"""Self-test worker: CUDA sanity checks for `bandscribe doctor`, plus torch-free modes for runner tests.

Modes (request["mode"]):
  cuda (default)    torch/CUDA versions, device, fp16 matmul vs fp32, autocast fp16 conv2d, bf16 flags
  hold              sleep request["seconds"]
  spawn_grandchild  start `python -c "sleep 600"` with the same interpreter, record pids, sleep `seconds`
  echo              return the request unchanged
  fail              raise (tests the error path)
Only `cuda` imports torch.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.workers.base import WorkerContext, worker_main

log = logging.getLogger(__name__)

FP16_MATMUL_RTOL = 1e-2  # max |fp16 - fp32| relative to max |fp32|; fp16 has ~3 decimal digits
AUTOCAST_CONV_RTOL = 1e-2


class SelftestFailed(RuntimeError):
    pass


def _cuda(request: dict, ctx: WorkerContext) -> dict[str, Any]:
    import torch  # only this mode needs torch; the others must run in the torch-free core env

    out = ctx.partial
    out.update({
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
    })
    if not out["cuda_available"]:
        raise SelftestFailed("torch.cuda.is_available() is False")

    dev = torch.device("cuda", int(request.get("device", 0)))
    props = torch.cuda.get_device_properties(dev)
    out.update({
        "device_count": torch.cuda.device_count(),
        "device_name": props.name,
        "capability": list(torch.cuda.get_device_capability(dev)),
        "total_vram_mb": round(props.total_memory / 2**20, 1),
    })
    free, total = torch.cuda.mem_get_info(dev)
    out["mem_get_info_mb"] = {"free": round(free / 2**20, 1), "total": round(total / 2**20, 1)}

    # Turing has no native bf16: torch may report emulated support. We never use bf16 there (DESIGN 9.5).
    try:
        out["bf16_supported_native"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # older signature
        out["bf16_supported_native"] = bool(torch.cuda.is_bf16_supported())
    try:
        out["bf16_supported_emulated"] = bool(torch.cuda.is_bf16_supported(including_emulation=True))
    except TypeError:
        out["bf16_supported_emulated"] = None

    failures: list[str] = []
    gen = torch.Generator(device="cpu").manual_seed(1234)
    with torch.inference_mode():
        # fp16 matmul, compared with fp32 on the same inputs (Turing has no TF32, so fp32 is exact-ish).
        t = time.perf_counter()
        n = int(request.get("matmul_n", 1024))
        a = torch.randn(n, n, generator=gen).to(dev)
        b = torch.randn(n, n, generator=gen).to(dev)
        ref = a @ b
        c16 = a.half() @ b.half()
        torch.cuda.synchronize(dev)
        finite = bool(torch.isfinite(c16).all().item())
        rel = float(((c16.float() - ref).abs().max() / ref.abs().max()).item())
        out["fp16_matmul"] = {"n": n, "finite": finite, "max_rel_err": rel, "dtype": str(c16.dtype),
                              "ok": finite and rel <= FP16_MATMUL_RTOL and c16.dtype == torch.float16,
                              "ms": round((time.perf_counter() - t) * 1000, 1)}
        if not out["fp16_matmul"]["ok"]:
            failures.append(f"fp16 matmul: {out['fp16_matmul']}")

        # autocast conv2d: fp32 weights, fp16 compute - the MuScriptor/MSST pattern.
        t = time.perf_counter()
        conv = torch.nn.Conv2d(8, 16, 3, padding=1).to(dev)
        x = torch.randn(2, 8, 64, 64, generator=gen).to(dev)
        ref = conv(x)
        with torch.autocast("cuda", dtype=torch.float16):
            y = conv(x)
        torch.cuda.synchronize(dev)
        finite = bool(torch.isfinite(y).all().item())
        rel = float(((y.float() - ref).abs().max() / ref.abs().max()).item())
        out["autocast_conv"] = {"finite": finite, "max_rel_err": rel, "dtype": str(y.dtype),
                                "ok": finite and rel <= AUTOCAST_CONV_RTOL and y.dtype == torch.float16,
                                "ms": round((time.perf_counter() - t) * 1000, 1)}
        if not out["autocast_conv"]["ok"]:
            failures.append(f"autocast conv: {out['autocast_conv']}")
        del a, b, ref, c16, conv, x, y

    out["passed"] = not failures
    if failures:
        raise SelftestFailed("; ".join(failures))
    return out


def _hold(request: dict) -> dict[str, Any]:
    seconds = float(request.get("seconds", 1.0))
    time.sleep(seconds)
    return {"slept_s": seconds}


def _spawn_grandchild(request: dict) -> dict[str, Any]:
    pid_file = Path(request["pid_file"])
    # Same interpreter as ours (the venv path -> another launcher + base interpreter pair). The grandchild
    # prints its own pid so we also learn the pid of the process that actually sleeps.
    code = "import os, time; print(os.getpid(), flush=True); time.sleep(600)"
    gc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          encoding="utf-8", errors="replace")
    line = gc.stdout.readline().strip() if gc.stdout else ""
    info = {
        "worker_pid": os.getpid(),
        "worker_ppid": os.getppid(),
        "grandchild_pid": gc.pid,
        "grandchild_self_pid": int(line) if line.isdigit() else None,
    }
    atomic.write_json(pid_file, info)
    log.info("spawned grandchild %s", info)
    info.update(_hold(request))
    return info


def handle(request: dict, ctx: WorkerContext) -> dict:
    mode = request.get("mode", "cuda")
    if mode == "cuda":
        return _cuda(request, ctx)
    if mode == "hold":
        return _hold(request)
    if mode == "spawn_grandchild":
        return _spawn_grandchild(request)
    if mode == "echo":
        return request
    if mode == "fail":
        ctx.partial["reached"] = "fail"
        raise RuntimeError(str(request.get("message", "requested failure")))
    raise ValueError(f"unknown selftest mode: {mode!r}")


if __name__ == "__main__":
    worker_main(handle)
