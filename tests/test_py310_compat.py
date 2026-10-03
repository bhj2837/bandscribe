"""Modules a bp310 worker can import must parse as Python 3.10 (M1_M2_SPEC 1.7, 7.4).

The Basic Pitch worker runs on Python 3.10 (onnxruntime wheels) and imports gtab from the source tree via
PYTHONPATH, so 3.11+ syntax in any module on its import path (``except*``, PEP 695 generics, ...) would break
it only at run time. This test parses the fixed list below plus every ``gtab/**/*.py`` that declares
``# gtab: py310`` in its first 5 lines. Parse only: the ``bp310``-marked worker test covers real imports.
(``feature_version`` checks grammar, not library APIs such as ``datetime.UTC`` - see the API scan below.)
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GTAB = ROOT / "gtab"
FIXED = (
    "gtab/__init__.py",
    "gtab/workers/__init__.py",
    "gtab/workers/base.py",
    "gtab/atomic.py",
    "gtab/sysmon.py",
    "gtab/paths.py",
    "gtab/audio.py",
    "gtab/models.py",
)
MARK = "# gtab: py310"
# Stdlib names that only exist from 3.11 on (grammar-valid, so ast.parse(feature_version=...) misses them).
PY311_APIS = ("datetime.UTC", "dt.UTC", "_dt.UTC", "tomllib", "typing.Self", "StrEnum", "ExceptionGroup", "TaskGroup")


def _marked() -> list[str]:
    out = []
    for p in sorted(GTAB.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        head = p.read_text(encoding="utf-8").splitlines()[:5]
        if any(MARK in line for line in head):
            out.append(p.relative_to(ROOT).as_posix())
    return out


FILES = sorted(set(FIXED) | set(_marked()))


def test_basicpitch_worker_is_marked() -> None:
    assert "gtab/workers/amt_basicpitch.py" in _marked()


@pytest.mark.parametrize("rel", FILES)
def test_parses_as_python_310(rel: str) -> None:
    src = (ROOT / rel).read_text(encoding="utf-8")
    ast.parse(src, filename=rel, feature_version=(3, 10))


@pytest.mark.parametrize("rel", FILES)
def test_no_311_only_stdlib_apis(rel: str) -> None:
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    used = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts = []
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                dotted = ".".join([cur.id, *reversed(parts)])
                used += [api for api in PY311_APIS if dotted == api or dotted.startswith(api + ".")]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            names = [a.name for a in node.names]
            used += [api for api in PY311_APIS if api in names or api == mod]
    assert not used, f"{rel} uses Python 3.11+ stdlib APIs: {sorted(set(used))}"
