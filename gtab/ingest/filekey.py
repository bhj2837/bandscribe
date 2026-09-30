"""Whole-file content key for the ingest index (``file:<key>``), fast enough for the 1 s cache-hit goal.

A plain sha256 runs at about 400 MiB/s on this PC (CPU-bound, no SHA extensions used), so a 10-minute
48 kHz / 24-bit WAV bounce (165 MiB) alone took ~0.45 s on top of ~0.55 s of CLI start-up. Here the
file is cut into fixed chunks that are hashed with sha256 on a thread pool (hashlib releases the GIL),
and the key is the sha256 of the size plus the chunk digests. Every byte is still hashed; only the
construction differs from sha256(file), so it is exactly as collision-resistant. Sampling a few chunks
would be faster but unsafe: a re-bounce with a small same-length edit in the middle is a normal event
in a band's workflow, and it must not be served the old job.

Stdlib only.
"""

from __future__ import annotations

import hashlib
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

CHUNK = 8 << 20
# Bounded so the pool never holds more than about (WORKERS + 2) * CHUNK bytes in memory.
WORKERS = max(1, min(8, (os.cpu_count() or 2)))
_DOMAIN = b"gtab-content-key-v1\0"


def _digest(block: bytes) -> bytes:
    return hashlib.sha256(block).digest()


def _read_chunk(f: object, n: int) -> bytes:
    # Chunk boundaries are part of the key, so a short read must never end a chunk early.
    parts: list[bytes] = []
    need = n
    while need:
        b = f.read(need)  # type: ignore[attr-defined]
        if not b:
            break
        parts.append(b)
        need -= len(b)
    return b"".join(parts) if len(parts) != 1 else parts[0]


def content_key(path: Path) -> str:
    """Hex key of the file's bytes: sha256(domain || size || sha256(chunk_0) || sha256(chunk_1) ...)."""
    path = Path(path)
    top = hashlib.sha256(_DOMAIN)
    size = 0
    with open(path, "rb", buffering=0) as f:
        # Sequential reads run at several GB/s from the page cache; the hashing is what is spread out.
        with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="filekey") as pool:
            pending: deque[Future[bytes]] = deque()
            digests: list[bytes] = []
            while block := _read_chunk(f, CHUNK):
                size += len(block)
                pending.append(pool.submit(_digest, block))
                if len(pending) > WORKERS + 1:
                    digests.append(pending.popleft().result())
            digests.extend(p.result() for p in pending)
    top.update(size.to_bytes(8, "little"))
    for d in digests:
        top.update(d)
    return top.hexdigest()
