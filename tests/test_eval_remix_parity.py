"""a10: the ported generators + ``window_features`` reproduce router_exp.py's features within ±0.02.

The original script runs in a subprocess with the core python: ``synth_exp.py`` rewraps sys.stdout at import,
which must not happen inside pytest. ``R.features`` is wrapped to record the raw floats of every row of
``router_exp.main()`` (the table it prints is rounded).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from gtab import paths

EXPERIMENTS = paths.GTAB_ROOT / "docs" / "research" / "experiments"

HARNESS = r'''
import contextlib, io, json, sys
sys.path.insert(0, sys.argv[1])
import router_exp as R
rows = []
orig = R.features
def rec(st):
    f = orig(st)
    rows.append({k: float(v) for k, v in f.items()})
    return f
R.features = rec
with contextlib.redirect_stdout(io.StringIO()):
    R.main()
sys.stdout.write(json.dumps(rows))
sys.stdout.flush()
'''


def test_router_features_parity():
    if not (EXPERIMENTS / "router_exp.py").exists():
        pytest.skip("research scripts not present")
    from gtab.winjob import run_captured

    r = run_captured([sys.executable, "-c", HARNESS, str(EXPERIMENTS)], timeout_s=600, env=paths.child_env())
    assert r.returncode == 0, r.stderr[-2000:]
    ref = json.loads(r.stdout)

    from gtab.analysis.stereo import window_features
    from gtab.eval.remix import SR, parity_scenes

    scenes = parity_scenes()
    assert len(scenes) == len(ref) == 14
    worst = 0.0
    for (name, st), fr in zip(scenes.items(), ref, strict=True):
        f = window_features(st, SR)
        assert set(f) == set(fr), name
        for k, v in fr.items():
            if math.isnan(v):
                assert math.isnan(f[k]), (name, k)
                continue
            worst = max(worst, abs(f[k] - v))
            assert f[k] == pytest.approx(v, abs=0.02), (name, k, f[k], v)
    print(f"max |Δ feature| = {worst:.3g}")
