"""vram_ballast worker: hold N MB of real VRAM so a measured worker runs short of memory (b3, M1_M2_SPEC 9.1).

``bandscribe bench gpu --ballast-mb N|auto`` starts this worker with ``winjob.spawn_in_job`` in **its own Job
Object, outside the GPU lock**, before the measured worker. Unlike ``--cap-mb`` (a PyTorch allocator limit,
which raises OOM before the driver could fall back), the ballast really occupies device memory, so with the
NVIDIA "CUDA - Sysmem Fallback Policy" on, the measured worker's allocations spill into shared memory instead
of failing - which is what b3 must detect.

Request: ``{"mode": "hold" (default) | "dry", "mb": N, "block_mb": 256, "max_s": 3600, "parent_pid": <pid>,
"touch_s": 0.2}``. ``touch_s > 0`` rewrites every block that often: WDDM pages out an *idle* process's VRAM when
another process needs it, so an idle ballast creates no pressure at all (measured on this PC, see bandscribe.bench).
Allocates N MB as ``torch.empty(..., dtype=uint8, device="cuda").fill_(1)`` blocks (touching the memory makes
the driver commit it), writes ``ready.json`` (``{"held_mb", "requested_mb", "short"}``) into its work dir, then
sleeps until a ``stop`` file appears there, its parent process dies, or ``max_s`` passes. Kill-on-close of its
job guarantees cleanup even if the bench crashes. ``dry`` allocates nothing (tests run it in the core env).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.workers.base import WorkerContext, worker_main

log = logging.getLogger(__name__)

READY_FILE = "ready.json"
STOP_FILE = "stop"
POLL_S = 0.1
_MB = 1024 * 1024


def _parent_alive(pid: int | None) -> bool:
    if not pid:
        return True
    from bandscribe.winjob import pid_alive

    return pid_alive(int(pid))


def handle(request: dict, ctx: WorkerContext) -> dict[str, Any]:
    mode = request.get("mode", "hold")
    want = max(0, int(request.get("mb", 0)))
    block_mb = max(1, int(request.get("block_mb", 256)))
    max_s = float(request.get("max_s", 3600))
    touch_s = float(request.get("touch_s", 0.0) or 0.0)
    parent = request.get("parent_pid")
    work = ctx.work_dir
    held = 0
    blocks: list[Any] = []
    error = None
    if mode == "hold" and want > 0:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        try:
            while held < want:
                n = min(block_mb, want - held)
                blocks.append(torch.empty(n * _MB, dtype=torch.uint8, device="cuda").fill_(1))
                held += n
            torch.cuda.synchronize()
        except torch.OutOfMemoryError as e:  # report what we got; the bench decides whether that is enough
            error = f"OOM after {held} MB: {str(e)[:200]}"
            log.warning(error)
    elif mode == "dry":
        held = want
    elif mode != "hold":
        raise ValueError(f"unknown vram_ballast mode: {mode!r}")

    atomic.write_json(Path(work) / READY_FILE, {"held_mb": held, "requested_mb": want, "short": held < want,
                                                "error": error})
    log.info("ballast holding %d MB (requested %d)", held, want)
    t0 = time.monotonic()
    reason = "max_s"
    touches = 0
    next_touch = t0
    k = 0  # next block to touch; one block per loop turn so a stop request is seen within one block's time
    while time.monotonic() - t0 < max_s:
        if (Path(work) / STOP_FILE).exists():
            reason = "stop_file"
            break
        if not _parent_alive(parent):
            reason = "parent_gone"
            break
        if touch_s > 0 and blocks and time.monotonic() >= next_touch:
            # Keep the blocks "hot" like a rendering game: WDDM pages out an idle process's allocations when
            # another process needs VRAM (measured 2026-09-30: an idle 4.5 GB ballast was simply evicted and
            # the measured worker never ran short), so an idle ballast does not create memory pressure.
            import torch

            blocks[k].fill_(1)
            torch.cuda.synchronize()
            k += 1
            if k == len(blocks):
                k = 0
                touches += 1
                next_touch = time.monotonic() + touch_s
            continue
        time.sleep(min(POLL_S, touch_s) if touch_s > 0 else POLL_S)
    held_s = round(time.monotonic() - t0, 3)
    blocks.clear()
    return {"held_mb": held, "requested_mb": want, "held_s": held_s, "released_because": reason, "error": error,
            "touch_s": touch_s, "touches": touches}


if __name__ == "__main__":
    worker_main(handle, name="vram_ballast")
