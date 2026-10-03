"""Build the SW BS-RoFormer from its yaml and load the checkpoint safely and strictly.

Replaces MSST's settings loader (omegaconf / ml_collections, not installed; M1_M2_SPEC 9.1 item 3):

- the yaml is parsed with a ``yaml.SafeLoader`` subclass that maps ``tag:yaml.org,2002:python/tuple`` to a
  tuple (the SW yaml uses ``!!python/tuple`` for ``freqs_per_bands``); never unsafe ``yaml.load``;
- ``BSRoformer(**config["model"])``;
- ``torch.load(ckpt, map_location="cpu", weights_only=True)`` + ``load_state_dict(strict=True)`` (verified:
  0 missing / 0 unexpected keys with rotary-embedding-torch 0.9.1, which stores ``rotary_embed.freqs``).

The caller verifies the checkpoint sha256 before loading (sep_msst worker).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _safe_loader() -> Any:
    import yaml

    class _TupleSafeLoader(yaml.SafeLoader):
        pass

    def _tuple(loader: Any, node: Any) -> tuple:
        return tuple(loader.construct_sequence(node, deep=True))

    _TupleSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple)
    return _TupleSafeLoader


def load_yaml(path: Path | str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.load(f, Loader=_safe_loader())  # noqa: S506 - SafeLoader subclass, only tuples added
    if not isinstance(data, dict) or "model" not in data:
        raise ValueError(f"not an MSST model config: {path}")
    return data


def stem_order(config: dict[str, Any]) -> list[str]:
    """The model's output stem order = ``training.instruments`` (SW: bass, drums, other, vocals, guitar, piano)."""
    names = list((config.get("training") or {}).get("instruments") or [])
    if not names:
        raise ValueError("config has no training.instruments")
    return [str(n) for n in names]


def build_model(config: dict[str, Any]) -> Any:
    from .models.bs_roformer import BSRoformer

    return BSRoformer(**dict(config["model"]))


def load_weights(model: Any, ckpt_path: Path | str) -> dict[str, int]:
    """Strict, safe load onto CPU. Returns {"missing": 0, "unexpected": 0, "tensors": n}."""
    import torch

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]  # some MSST checkpoints wrap the weights
    res = model.load_state_dict(state, strict=True)
    n = len(state)
    del state
    return {"missing": len(res.missing_keys), "unexpected": len(res.unexpected_keys), "tensors": n}
