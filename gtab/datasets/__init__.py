"""Tier B/C datasets: public API (M1_M2_SPEC §9.5 item 5). EVAL and AMT import only this module.

```python
list_datasets() -> list[dict]                 # registry + local state, one row per dataset
dataset_root(name) -> Path                    # data/datasets/<name>
is_available(name) -> bool                    # required files present (fetched/verified) or imported by hand
iter_tracks(name, split=None) -> Iterator[DatasetTrack]
load_track(name, track_id) -> DatasetTrack
track_ids(name, split=None) -> list[str]      # same ids/order as iter_tracks, without parsing annotations
```

``track_ids`` (addition to the spec's API): sampling a subset (e.g. EVAL's bp-smoke, 20 of GuitarSet's 360
tracks) is then ``[load_track(name, i) for i in chosen]`` instead of parsing every JAMS file (~30 ms each).

``DatasetTrack`` is P0's ``gtab.schema.dataset.DatasetTrack`` (§6.8): audio paths inside it are POSIX paths
relative to ``dataset_root(name)``; string numbers follow §6 (1 = highest-pitched string); every loader fills
``families_present``. Loaders read files in place (``extracted/``, ``local/``, ``raw/``); nothing here
downloads — fetching is ``gtab data fetch`` (``gtab.datasets.fetch``).
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gtab.datasets import registry, store


class DatasetUnavailable(RuntimeError):
    """The dataset is not on this PC (message tells how to get it, in Korean)."""


def dataset_root(name: str) -> Path:
    registry.get(name)  # unknown name -> RegistryError with a suggestion
    return store.dataset_dir(name)


def is_available(name: str) -> bool:
    registry.get(name)
    return store.is_available(name)


def list_datasets() -> list[dict[str, Any]]:
    rows = []
    for name, spec in registry.load_registry().items():
        st = store.status(name)
        rows.append({"name": name, "title": spec.title, "tier": spec.tier, "license": spec.license,
                     "access": spec.access, "approx_bytes": spec.approx_bytes, "local_bytes": store.total_bytes(name),
                     "status": st, "path": store.dataset_dir(name), "page": spec.page,
                     "contamination": spec.contamination})
    return rows


def _loader(name: str) -> Any:
    spec = registry.get(name)
    return importlib.import_module(f"gtab.datasets.loaders.{spec.loader}")


def _require(name: str) -> Path:
    spec = registry.get(name)
    if not store.is_available(name):
        how = (f"`gtab data fetch {name}`" if spec.access != "manual"
               else f"직접 받은 뒤 `gtab data import-local {name} <폴더>`")
        st = store.status(name)
        extra = " (일부만 받았습니다)" if st == "partial" else ""
        raise DatasetUnavailable(f"데이터셋 '{name}' 이(가) 이 PC 에 없습니다{extra}. {how} 로 준비하세요.")
    return store.dataset_dir(name)


def iter_tracks(name: str, split: str | None = None) -> Iterator[Any]:
    root = _require(name)
    yield from _loader(name).iter_tracks(root, split=split)


def load_track(name: str, track_id: str) -> Any:
    return _loader(name).load_track(_require(name), track_id)


def track_ids(name: str, split: str | None = None) -> list[str]:
    return list(_loader(name).track_ids(_require(name), split=split))


__all__ = ["DatasetUnavailable", "dataset_root", "is_available", "iter_tracks", "list_datasets", "load_track",
           "track_ids"]
