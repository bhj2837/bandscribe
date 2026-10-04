"""The torch boundary (M1_M2_SPEC 0): only bandscribe/workers/* and bandscribe/third_party/msst/* may import torch & co.

Core modules run in the core env, which never has torch (a CPU torch sneaking in would also change numerics
silently). The scan is static (AST), so it also catches imports inside functions and literal
``importlib.import_module("torch")`` / ``__import__("torch")`` calls.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "bandscribe"
FORBIDDEN = ("torch", "torchaudio", "beat_this", "muscriptor", "basic_pitch", "onnxruntime")
ALLOWED_DIRS = (PKG / "workers", PKG / "third_party" / "msst")


def _core_files() -> list[Path]:
    out = []
    for p in sorted(PKG.rglob("*.py")):
        if any(p.is_relative_to(d) for d in ALLOWED_DIRS) or "__pycache__" in p.parts:
            continue
        out.append(p)
    return out


def _forbidden(mod: str | None) -> bool:
    return bool(mod) and mod.split(".")[0] in FORBIDDEN


def violations(src: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, a.name) for a in node.names if _forbidden(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and _forbidden(node.module):
                found.append((node.lineno, str(node.module)))
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
            arg = node.args[0].value
            if name in ("import_module", "__import__", "find_spec") and isinstance(arg, str) and _forbidden(arg):
                found.append((node.lineno, arg))
    return found


def test_scanner_catches_every_form() -> None:
    src = (
        "import torch\nimport numpy\nfrom torch.nn import functional\nfrom . import torch_like\n"
        "def f():\n    import beat_this.inference\n    importlib.import_module('muscriptor')\n"
        "    __import__('onnxruntime')\n    import torchvision_not_listed\n"
    )
    assert [m for _, m in violations(src)] == ["torch", "torch.nn", "beat_this.inference", "muscriptor", "onnxruntime"]


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.relative_to(PKG).as_posix())
def test_core_module_does_not_import_torch(path: Path) -> None:
    found = violations(path.read_text(encoding="utf-8"))
    assert not found, f"{path.relative_to(PKG.parent)} imports GPU-env packages: {found}"
