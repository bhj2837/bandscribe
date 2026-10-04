"""Vendored MSST: files match VENDOR.md, strict load, no librosa, no sdp_kernel FutureWarning on CPU."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

from bandscribe import paths

BANDSCRIBE_ROOT = Path(__file__).resolve().parents[1]
MSST = BANDSCRIBE_ROOT / "bandscribe" / "third_party" / "msst"


def _vendor_table() -> dict[str, tuple[str, str]]:
    rows = {}
    for line in (MSST / "VENDOR.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| `([^`]+)` \| `[^`]+` \| `([0-9a-f]{64})` \| `([0-9a-f]{64})` \|$", line)
        if m:
            rows[m.group(1)] = (m.group(2), m.group(3))
    return rows


def test_vendored_files_match_vendor_md() -> None:
    table = _vendor_table()
    assert set(table) == {"LICENSE", "models/bs_roformer/__init__.py", "models/bs_roformer/bs_roformer.py",
                          "models/bs_roformer/attend.py"}
    for rel, (_upstream, local) in table.items():
        assert hashlib.sha256((MSST / rel).read_bytes()).hexdigest() == local, rel
    assert table["LICENSE"][0] == table["LICENSE"][1]  # verbatim
    assert "MIT License" in (MSST / "LICENSE").read_text(encoding="utf-8")


def test_pinned_commit_is_consistent() -> None:
    import bandscribe.third_party.msst as msst

    text = (MSST / "VENDOR.md").read_text(encoding="utf-8")
    assert msst.MSST_COMMIT in text
    from bandscribe.sep import run as seprun

    assert seprun.SEP_DEFAULTS["sep.msst_commit"] == msst.MSST_COMMIT


def test_trimmed_init_and_relative_import() -> None:
    init = (MSST / "models" / "bs_roformer" / "__init__.py").read_text(encoding="utf-8")
    assert "from .bs_roformer import BSRoformer" in init
    assert "mel_band" not in init and "conformer" not in init.lower()  # those import librosa upstream
    src = (MSST / "models" / "bs_roformer" / "bs_roformer.py").read_text(encoding="utf-8")
    assert "from models." not in src
    for f in MSST.rglob("*.py"):
        body = f.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+(librosa|matplotlib)", body, re.M), f


def _gpu_env(code: str, cwd: Path) -> dict:
    # cwd = a pytest tmp dir: the snippets write scratch checkpoints relative to it (never into the repo root).
    env = paths.child_env({"PYTHONPATH": str(BANDSCRIBE_ROOT)})
    proc = subprocess.run([str(paths.GPU_PYTHON), "-W", "default", "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, env=env,
                          cwd=str(cwd))
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.gpuenv
def test_strict_load_tiny_model_no_librosa_no_warning(tmp_path: Path) -> None:
    out = _gpu_env("""
        import json, sys, warnings
        warnings.simplefilter("always")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import torch
            import bandscribe.third_party.msst
            from bandscribe.third_party.msst.models.bs_roformer import BSRoformer
            from bandscribe.third_party.msst import loader
            cfg = {"model": {"dim": 16, "depth": 1, "stereo": True, "num_stems": 2, "time_transformer_depth": 1,
                             "freq_transformer_depth": 1, "dim_head": 8, "heads": 2, "flash_attn": True,
                             "freqs_per_bands": tuple([2] * 24 + [4] * 12 + [12] * 8 + [24] * 8 + [48] * 8 + [128, 129]),
                             "mask_estimator_depth": 1}, "training": {"instruments": ["a", "b"]}}
            torch.manual_seed(0)
            m1 = loader.build_model(cfg)
            path = "tiny.ckpt"
            torch.save(m1.state_dict(), path)
            m2 = loader.build_model(cfg)
            info = loader.load_weights(m2, path)
            same = all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
            bad = dict(m1.state_dict()); bad.pop(next(iter(bad)))
            torch.save(bad, "bad.ckpt")
            try:
                loader.load_weights(loader.build_model(cfg), "bad.ckpt"); strict = False
            except RuntimeError:
                strict = True
            m2.eval()
            with torch.inference_mode():
                y = m2(torch.randn(1, 2, 4096) * 0.1)
        msgs = [str(w.message)[:120] for w in caught]
        print(json.dumps({"info": info, "same": same, "strict": strict, "shape": list(y.shape),
                          "librosa": "librosa" in sys.modules, "matplotlib": "matplotlib" in sys.modules,
                          "sdp_warnings": [m for m in msgs if "sdp_kernel" in m], "stems": loader.stem_order(cfg)}))
    """, tmp_path)
    assert out["info"] == {"missing": 0, "unexpected": 0, "tensors": out["info"]["tensors"]}
    assert out["same"] and out["strict"] and out["shape"] == [1, 2, 2, 4096]
    assert not out["librosa"] and not out["matplotlib"]
    assert out["sdp_warnings"] == []
    assert out["stems"] == ["a", "b"]


@pytest.mark.gpuenv
def test_sw_yaml_parses_with_safe_loader(tmp_path: Path) -> None:
    yml = paths.DATA / "models" / "bs_roformer_sw" / "BS-Rofo-SW-Fixed.yaml"
    if not yml.is_file():
        pytest.skip("SW yaml not on disk")
    out = _gpu_env(f"""
        import json
        from bandscribe.third_party.msst import loader
        cfg = loader.load_yaml(r"{yml}")
        print(json.dumps({{"stems": loader.stem_order(cfg), "fpb": type(cfg["model"]["freqs_per_bands"]).__name__,
                          "chunk": cfg["audio"]["chunk_size"], "hop": cfg["model"]["stft_hop_length"],
                          "amp": cfg["training"]["use_amp"]}}))
    """, tmp_path)
    assert out == {"stems": ["bass", "drums", "other", "vocals", "guitar", "piano"], "fpb": "tuple",
                   "chunk": 588800, "hop": 512, "amp": True}
