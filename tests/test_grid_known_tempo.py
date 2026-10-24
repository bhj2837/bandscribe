"""The tempo octave on this PC's songs whose tempo is known (mid-term plan 1-6): the grid is recomputed on each job's
cached beats + onsets (CPU, no audio decode), without and with the job's ``[grid] tempo_bpm`` hint.

What it pins: ``run`` asks the tempo on every new song because the automatic level is wrong on half of them, so the
question must always offer the right level, and the answer (the hint) must land on it. A grid change that moves an
automatic decision shows up here as a changed ``AUTO_WRONG``. Skipped for a job not on this PC (jobs are local only).
"""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

import pytest

from bandscribe import atomic, paths

# job key -> the automatic level is a factor 2 off the known tempo (measured 2026-10-24, grid code_version 5):
# lulu. 176 vs 88, 狭心症 136 vs 68, GKA 97 vs 193, the guitar-removed 青と夏 94 vs 185.
AUTO_WRONG = {"f-12befe96300880e4": True, "f-5959ab5fa06d86ad": True, "f-56f32f7fc6873405": True,
              "f-8eafb8187d7fe235": True, "f-05d84be0010d8784": False, "f-250b3e07683d2d1a": False,
              "f-2884a8577f49da56": False, "f-ade1a8bccc6acf6e": False}
SAME_LEVEL = 0.06  # |log2(bpm / known)|: 4 % (the trackers' 187.5 for a 192 BPM score is the same level)


def _job_inputs(job: Path):
    from bandscribe.analysis import onsets
    from bandscribe.analysis.stage import load_activations
    from bandscribe.tab import health as H

    hp = job / "hints.toml"
    known = float((tomllib.loads(hp.read_bytes().decode("utf-8-sig")).get("grid") or {}).get("tempo_bpm") or 0) \
        if hp.is_file() else 0.0
    grid_dir = H.stage_dirs(job).get("grid")
    if not known or grid_dir is None:
        return None
    man = json.loads((grid_dir / "manifest.json").read_text(encoding="utf-8"))
    beats_dir = job / "stages" / "beats" / str(man["deps"]["beats"])[:12]
    if not (beats_dir / "beats.json").is_file() or not (grid_dir / "onsets.npz").is_file():
        return None
    return (known, atomic.read_json(beats_dir / "beats.json"), load_activations(beats_dir / "activations.npz"),
            onsets.load(grid_dir / "onsets.npz"), dict(man["params"]))


@pytest.mark.parametrize("key", sorted(AUTO_WRONG))
def test_known_tempo_is_offered_and_the_hint_lands_on_it(key):
    from bandscribe.analysis import grid as G
    from bandscribe.commands import run as R

    job = Path(paths.JOBS) / key
    got = _job_inputs(job) if job.is_dir() else None
    if got is None:
        pytest.skip(f"{key}: job, its export or its known tempo is not on this PC")
    known, beats, acts, ons, params = got
    auto = G.derive_grid(beats, acts, ons, dict(params, tempo_bpm=0.0))
    cur, cands = R.tempo_choices(auto)
    assert (abs(math.log2(cur / known)) > 0.5) is AUTO_WRONG[key], (cur, known)
    assert any(abs(math.log2(float(v["bpm"]) / known)) < SAME_LEVEL for v in cands.values()), cands
    # the answer the question would save for a wrong level (h or d) is the known level
    if AUTO_WRONG[key]:
        ans = R.parse_tempo_answer("h" if cur > known else "d", cur)
        assert abs(math.log2(ans / known)) < SAME_LEVEL
    hinted = G.derive_grid(beats, acts, ons, dict(params, tempo_bpm=known))
    to = hinted["hypotheses"]["tempo_octave"]
    assert to["by"] == "hint" and abs(math.log2(to["scores"][to["decision"]]["bpm"] / known)) < SAME_LEVEL
