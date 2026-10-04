"""Transcription cache shared by S3.5 and S6 (M1_M2_SPEC 3.2 notes, DESIGN 9.3).

An entry is one raw NoteSet (backend times, **before** latency correction) for one (backend, model, view audio,
instrument mask, decode params, code version, rung). ``amt_ms1``, ``amt_gtr`` and ``amt_bp`` look views up
here, send only the missing ones to their worker, store the fresh NoteSets and then write their stage outputs
(latency applied) from the cache. The same guitar-stem transcription thus serves S3.5 and S6, and a rerun after a
cache wipe of a stage dir costs nothing.

Key rules:
- ``instruments`` is sorted by MT3 id before hashing (MuScriptor's conditioning depends on the order, so the
  core always sends the sorted list too): ``["b", "a"]`` and ``["a", "b"]`` are the same entry.
- ``rung_label`` (``medium``, ``medium-lean``, ``small``, ``medium@cpu`` ...) is part of the key. Lookups use
  the configured top rung, so a transcription made on a degraded rung (a game was holding VRAM) never answers
  for the top rung; it is stored under its own label (provenance) and simply re-made next time.

Layout: ``<root>/<key[:2]>/<key>.json`` with root ``jobs/<song_key>/cache/amt`` inside jobs and
``data/eval/reftx_cache`` outside (reftx, experiments). Writes are atomic; a corrupt entry is a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths
from bandscribe.amt.instruments import sort_mt3

log = logging.getLogger(__name__)

# Frozen: hashed into every key below. It keeps the project's old name on purpose (docs/RENAME.md): renaming it
# would orphan every cached transcription (data/jobs/*/cache/amt, data/eval/reftx_cache) for nothing.
KEY_FORMAT = "gtab.amt_cache_key/1"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def amt_key(backend: str, backend_version: str | None, model_sha256: str | None, view_pcm_sha256: str,
            instruments: list[str] | None, params: dict[str, Any], code_version: str, rung_label: str) -> str:
    """sha256 of the canonical JSON of everything that determines a raw transcription."""
    if not view_pcm_sha256:
        raise ValueError("view_pcm_sha256 is required for a transcription cache key")
    doc = {
        "format": KEY_FORMAT,
        "backend": str(backend),
        "backend_version": backend_version,
        "model_sha256": model_sha256,
        "view_pcm_sha256": str(view_pcm_sha256),
        "instruments": None if instruments is None else sort_mt3(instruments),
        "params": params,
        "code_version": str(code_version),
        "rung_label": str(rung_label),
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def job_cache_root(store: Any, song_key: str) -> Path:
    return Path(store.job_dir(song_key)) / "cache" / "amt"


def eval_cache_root() -> Path:
    return Path(paths.EVAL) / "reftx_cache"


class AmtCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self.path(key)
        if not p.is_file():
            return None
        try:
            doc = atomic.read_json(p)
        except (OSError, ValueError) as e:
            log.warning("ignoring unreadable transcription cache entry %s: %s", p, e)
            return None
        if not isinstance(doc, dict) or doc.get("format") != "bandscribe.notes/1":
            log.warning("ignoring transcription cache entry %s with unexpected content", p)
            return None
        return doc

    def put(self, key: str, doc: dict[str, Any]) -> Path:
        p = self.path(key)
        atomic.write_json(p, doc)
        return p
