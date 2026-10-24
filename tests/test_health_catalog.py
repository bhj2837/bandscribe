"""The failure catalogue of the mid-term review replayed on this PC's jobs (plan 1-4 exit criterion; skipped where the
jobs are not, since jobs are local only). What it pins: every guitar-removed recording is caught by the stem level,
lulu.'s bass burst window is found and placed, the SNN tuning override and lulu.'s unplaced bass are shown, and the 13
valid songs raise at most one input / burst / unplaced false alarm."""

from __future__ import annotations

from pathlib import Path

import pytest

from bandscribe import atomic, paths
from bandscribe.tab import health as H

VALID = ("f-bd0dc18775389993", "f-56f32f7fc6873405", "f-cb4a96577087aaf4", "f-05d84be0010d8784", "f-2884a8577f49da56",
         "f-ade1a8bccc6acf6e", "f-250b3e07683d2d1a", "f-07f21d0ec2cb1bb4", "f-abd70434e91c65ba", "f-12befe96300880e4",
         "f-5959ab5fa06d86ad", "f-93e11bb95a10b9fc", "f-cc5cb9e81a50d3a4")
# guitar-removed: five practice backings (no score was made) and three wav files the owner exported without guitar
REMOVED = ("f-1385bbc40dc6b6cc", "f-1706c58d58f97fb5", "f-67e841b07a226989", "f-7986ec86028b9e7f", "f-87b0b65435cf7b40",
           "f-d92d41b599085227", "f-e064a271b0834683", "f-8eafb8187d7fe235")
LULU, SNN = "f-12befe96300880e4", "f-05d84be0010d8784"
CHECKED = ("guitar_removed", "burst_guitar_all", "burst_bass", "unplaced_")


def _job(key: str) -> Path:
    j = Path(paths.JOBS) / key
    if not j.is_dir():
        pytest.skip(f"{key} is not on this PC")
    return j


def _items(key: str) -> dict[str, dict]:
    doc = H.song_health(_job(key))
    if not doc.get("score_key"):
        pytest.skip(f"{key} has no exported score")
    return {i["id"]: i for i in doc["items"]}


@pytest.mark.parametrize("key", REMOVED)
def test_every_guitar_removed_recording_is_caught(key):
    stats = sorted((_job(key) / "stages" / "stems").glob("*/stem_stats.json"))
    if not stats:
        pytest.skip(f"{key} has no stems stage")
    levels = [H.guitar_level(atomic.read_json(p)) for p in stats]
    assert all(v is not None and v < H.GUITAR_REMOVED_DB for v in levels), levels
    if (_job(key) / "export" / "tab" / "score.alphatex").is_file():
        assert _items(key)["guitar_removed"]["level"] == "warn"


def test_the_valid_songs_raise_at_most_one_false_alarm():
    alarms = []
    for key in VALID:
        for iid, it in _items(key).items():
            if it["level"] == "warn" and iid.startswith(CHECKED) and not (key == LULU and iid in ("burst_bass",
                                                                                                    "unplaced_bass")):
                alarms.append((key, iid))
    assert len(alarms) <= 1, alarms


def test_lulu_bass_burst_window_and_unplaced_notes():
    items = _items(LULU)
    spans = items["burst_bass"]["spans"]
    assert any(a <= 150 and b >= 260 for a, b in spans), spans          # the measured burst, 150-260 s
    assert all(a >= 140 and b <= 280 for a, b in spans), spans          # and nothing well before or after it
    assert items["unplaced_bass"]["level"] == "warn"


def test_snn_tuning_override_is_shown():
    it = _items(SNN)["tuning_guitar"]
    assert it["level"] == "info" and "E standard" in it["title"]
