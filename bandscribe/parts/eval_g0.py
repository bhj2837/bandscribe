"""G0 (plan change 2 and 3, docs/MIDTERM_REVIEW.md): the multi-guitar set separation work is judged on.

The six sheet songs are the easy case for parts (two distorted guitars hard left and right). The owner's J-pop songs are not: three guitars,
an acoustic among electrics, parts at the same position. G0 rebuilds those cases from Cambridge-MT takes, whole songs, so the part count and
the assignment can be measured against each take's own reference transcription. **The truth is a transcription of the isolated take, so G0
judges who plays a note, never whether the note is right**, and it is a development set: picks made here are checked on the sheet songs.

Scenarios, on top of E27's ``P``, ``A1``, ``B2`` and E28's ``D1`` (``data/eval/parts_track/``):

- ``T3``: three different parts hard left, centre, hard right.
- ``X2p``: an acoustic and an electric, hard left and right. ``X2c``: the same two at the centre (nothing to tell them apart but their sound).
- ``N1a``: the acoustic alone at the centre: one part. The control for timbre evidence, since the transcriber calls one take acoustic here and
  clean there (E25: class evidence split three one-guitar remixes).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from bandscribe import atomic

SUBDIR = "g0"
# song -> scenario -> {take index: pan}, from the inventory (data/tmp/g0/inventory.py): the takes that play most on different Lines
PLANS: dict[str, dict[str, dict[int, float]]] = {
    "dunningkruger_japansong": {"T3": {0: -1.0, 4: 0.0, 1: 1.0}},
    "butterflyeffect_fairytale": {"T3": {0: -1.0, 2: 0.0, 1: 1.0}},
    "bencarrigan_welltalkaboutitalltonight": {"X2p": {0: -1.0, 2: 1.0}, "X2c": {0: 0.0, 2: 0.0}, "N1a": {0: 0.0}},
    "flags_54": {"X2c": {0: 0.0, 2: 0.0}, "N1a": {0: 0.0}},
}


def cache_dir() -> Path:
    from bandscribe.parts import eval_track as ET

    return ET.cache_dir() / SUBDIR


def prepare(song: str, cfg: Any, *, e6: Path, progress: Callable[[str], None] | None = None) -> dict | None:
    """Fill the caches of one G0 song (GPU: separation, beats, transcription; ``eval_track.prepare_song``)."""
    from bandscribe.parts import eval_track as ET
    from bandscribe.router import tierb as T

    plans = [(sc, T.Plan(sc, 0, dict(pans), {"lines": sorted(pans)})) for sc, pans in PLANS[song].items()]
    cache_dir().mkdir(parents=True, exist_ok=True)
    return ET.prepare_song(song, cfg, e6=e6, progress=progress, plans=plans, out_name=f"{SUBDIR}/{song}.json")


def load_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for f in sorted(cache_dir().glob("*.json")):
        items.extend(atomic.read_json(f).get("items") or [])
    for it in items:
        it["truth"] = [set(t) for t in it["truth"]]  # JSON keeps lists; the scoring compares sets
    return items
