"""On-disk format ids (``"bandscribe.<name>/<version>"``) and the legacy ``"gtab."`` prefix.

The project was called gtab until 2026-10-04 (docs/RENAME.md). Every file written before that says
``"format": "gtab.<name>/1"``. Those files are never rewritten (stage manifests record their sha256, and the
cached results are hours of GPU work), so readers normalise the id on load instead:
``normalize("gtab.notes/1") == "bandscribe.notes/1"``. New files always use the new prefix.

Where it is applied:
- ``atomic.read_json``: the top-level ``"format"`` of every JSON document read through it;
- the pydantic file models (``schema._common._Model``, ``eval.refscore._Model``): a before-validator, so
  ``model_validate_json`` and dicts from elsewhere (YAML) are covered too;
- the TOML/YAML readers that compare a ``format`` field themselves (registry, model sources, lines.yaml,
  Tier A songs.yaml).

Not format ids, so never touched here: the frozen hash salts ``ingest.filekey._DOMAIN`` and
``amt.cache.KEY_FORMAT`` keep their gtab spelling because they are part of existing cache keys.

Stdlib only, Python-3.10-safe (the bp310 worker imports ``atomic``, which imports this).
"""

from __future__ import annotations

from typing import Any

PREFIX = "bandscribe."
LEGACY_PREFIXES: tuple[str, ...] = ("gtab.",)


def normalize(fmt: Any) -> Any:
    """``"gtab.x/1"`` -> ``"bandscribe.x/1"``; anything else (new ids, non-strings, foreign ids) unchanged."""
    if isinstance(fmt, str):
        for legacy in LEGACY_PREFIXES:
            if fmt.startswith(legacy):
                return PREFIX + fmt[len(legacy):]
    return fmt


def normalize_doc(doc: Any) -> Any:
    """A dict whose top-level ``format`` is a legacy id comes back as a shallow copy with the new id; anything else
    is returned as is (same object)."""
    if isinstance(doc, dict):
        fmt = doc.get("format")
        new = normalize(fmt)
        if new is not fmt:
            return {**doc, "format": new}
    return doc
