"""background_mask == repet3's inline mask (M1_M2_SPEC 9.3 item 6; DESIGN evidence 0 -> 17 dB lead SIR).

Both sides run in subprocesses with the core python: the research scripts import ``synth_exp``, which rewraps
``sys.stdout`` at import time (that must not happen inside pytest).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs" / "research" / "experiments"

HARNESS = r'''
import json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import numpy as np
import repet_exp as R
from bandscribe.analysis.repetition import background_mask
S = R.S; SR = R.SR; SEG = S.N
res = []
for m in (0, 20, 50):
    reps = 3
    takes = [S.rhythm_take(np.random.default_rng(300 + i)) for i in range(reps)]
    sh = int(m / 1000 * SR)
    if sh:
        takes[2] = np.concatenate([np.zeros(sh), takes[2][:-sh]])
    rhythm = np.concatenate(takes)
    lead = R.lead_random(np.random.default_rng(9), len(rhythm) / SR)
    lead[:SEG] = 0
    mix = rhythm + lead
    X = R.stft(mix); V = np.abs(X)
    p = int(round(SEG / R.HOP)); T = p * reps
    Xc, Vc = X[:, :T], V[:, :T]
    Vs = np.stack([Vc[:, i * p:(i + 1) * p] for i in range(reps)], 0)
    Mb = np.concatenate(list(background_mask(Vs)), axis=1)
    fg = R.istft(np.pad(Xc * (1 - Mb), ((0, 0), (0, X.shape[1] - T))), len(mix))
    sl = slice(SEG, 3 * SEG)
    l = R.sdr_sir(fg[sl], lead[sl], rhythm[sl]); b = R.sdr_sir(mix[sl], lead[sl], rhythm[sl])
    res.append({"misalign_ms": m, "base_sir": float(b[1]), "sdr": float(l[0]), "sir": float(l[1])})
print("JSON" + json.dumps(res))
'''

LINE = re.compile(r"^median\s+misalign\s+(\d+) ms \| mix-as-lead SIR\s+(-?[\d.]+) -> fg lead SDR\s+(-?[\d.]+) "
                  r"SIR\s+(-?[\d.]+)")


def _run(args, cwd):
    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=600, env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


@pytest.mark.slow  # ~60 s: the original script runs six configurations
@pytest.mark.skipif(not (EXP / "repet3.py").is_file(), reason="research scripts not present")
def test_repet3_parity(tmp_path):
    harness = tmp_path / "harness.py"
    harness.write_text(HARNESS, encoding="utf-8")
    ours_out = _run([str(harness), str(EXP), str(ROOT)], cwd=tmp_path)
    ours = json.loads(ours_out.split("JSON", 1)[1])
    orig_out = _run([str(EXP / "repet3.py")], cwd=EXP)
    orig = {}
    for line in orig_out.splitlines():
        m = LINE.match(line.strip())
        if m:
            orig[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    assert set(orig) == {0, 20, 50}, orig_out
    for row in ours:
        base, sdr, sir = orig[row["misalign_ms"]]
        assert abs(row["sir"] - sir) <= 0.5, (row, orig[row["misalign_ms"]])
        assert abs(row["base_sir"] - base) <= 0.1
    # the evidence DESIGN cites: a large lead SIR gain at 0 ms misalignment
    assert ours[0]["sir"] - ours[0]["base_sir"] > 10
