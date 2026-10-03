"""Real download of the smallest registered dataset (IDMT-SMT-Bass-Single-Track, 20 MB) into a temp root.

Marked ``network`` and ``slow``; it only runs when slow tests are selected explicitly
(``pytest -m slow tests/test_datasets_network.py`` or GTAB_RUN_SLOW=1), so the default ``pytest -q`` never
downloads 20 MB on every run.
"""

from __future__ import annotations

import os

import pytest

from gtab.datasets import fetch, store

pytestmark = [pytest.mark.network, pytest.mark.slow]


@pytest.fixture(autouse=True)
def _only_when_selected(request):
    if "slow" not in (request.config.getoption("-m") or "") and not os.environ.get("GTAB_RUN_SLOW"):
        pytest.skip("slow network test: run with -m slow or GTAB_RUN_SLOW=1")


def test_fetch_idmt_and_verify(tmp_path):
    res = fetch.fetch("idmt_bass_st", root=tmp_path, yes=True, progress=False)
    assert res["downloaded"] == 1 and res["status"] == "fetched"
    rec = store.read_lock(tmp_path)["datasets"]["idmt_bass_st"]["files"]["raw/IDMT-SMT-BASS-SINGLE-TRACKS.zip"]
    assert rec["verified_against"] == "md5" and rec["bytes"] == 20499169
    assert len(list((tmp_path / "idmt_bass_st" / "extracted").rglob("*.xml"))) >= 17
    assert store.verify("idmt_bass_st", tmp_path)["ok"]
    assert store.status("idmt_bass_st", tmp_path) == "verified"
