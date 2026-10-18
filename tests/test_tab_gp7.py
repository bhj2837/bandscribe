"""M9: the Guitar Pro 7/8 file (``tab/score.gp``): one sync point per beat, the backing choice, the drift measure,
and (with Node + alphaTab) the written file with its embedded backing track read back."""

from __future__ import annotations

import numpy as np
import pytest

from bandscribe import atomic
from bandscribe.tab import alphatex as ATX
from bandscribe.tab import stage as T


def _score(bars):
    return ATX.WScore(title="t", bars=bars, tempo_qpm=[120])


def _bar(no, start):
    from bandscribe.tab.notation import WBar

    return WBar(bar=no, start_gtick=0, n_beats=4, beat_unit="quarter", numerator=4, denominator=4, start_s=start,
                feel=None)


def test_one_sync_point_per_beat_and_the_drift_measure():
    grid = {"beats": [{"t_s": 0.5 * k + (0.04 if k == 6 else 0.0), "bar": 1 + k // 4} for k in range(8)]}
    score = _score([_bar(1, 0.0), _bar(2, 2.0), _bar(3, 4.0)])
    pts = T.gp7_sync_points(score, grid)
    assert [(p["barIndex"], p["barPosition"]) for p in pts[:5]] == [(0, 0.0), (0, 0.25), (0, 0.5), (0, 0.75), (1, 0.0)]
    assert pts[6]["millisecondOffset"] == pytest.approx(3040.0)
    assert pts[-1] == {"barIndex": 2, "barOccurence": 0, "barPosition": 0.0, "millisecondOffset": 4000.0}  # no beats
    # a player that put beat 6 on the straight 120 BPM line: 40 ms off there, exact elsewhere
    check = {"sync": [[0, 0, 0.0], [1, 3840, 2000.0], [2, 7680, 4000.0]], "bars": [[0, 3840], [3840, 3840]]}
    d = T.gp7_drift(check, score, grid)
    assert d["beats"] == 8 and d["max"] == pytest.approx(40.0) and d["median"] == pytest.approx(0.0)


def test_backing_choice(tmp_path):
    job = tmp_path / "job"
    assert T.backing_audio(job, "original") is None and T.backing_audio(None, "original") is None
    (job / "input").mkdir(parents=True)
    (job / "input" / "mix_44k_f32.wav").write_bytes(b"x")
    assert T.backing_audio(job, "original").name == "mix_44k_f32.wav" and T.backing_audio(job, "none") is None
    prac = job / "stages" / "stems" / "k1" / "practice"
    prac.mkdir(parents=True)
    (prac / "mix_minus_guitar.wav").write_bytes(b"x")
    assert T.backing_audio(job, "no_guitar") == prac / "mix_minus_guitar.wav"


def test_the_written_gp_holds_the_backing_and_follows_the_grid(tmp_path):
    from bandscribe.eval import node as NJ

    if not NJ.alphatab_available():
        pytest.skip("Node + alphaTab 1.8.4 not installed")
    import soundfile as sf

    from test_parts_tracks import make_parts_job

    from bandscribe.tab import edits as E

    job, d = make_parts_job(tmp_path)
    st = E.apply(E.load_base(job), [])
    dur = float(st.grid["beats"][-1]["t_s"]) + 1.0
    (job / "input").mkdir(parents=True, exist_ok=True)
    sr = 44100
    t = np.arange(int(dur * sr)) / sr
    sf.write(str(job / "input" / "mix_44k_f32.wav"), np.stack([0.1 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
             .astype(np.float32), sr, subtype="FLOAT")
    out = tmp_path / "out"
    T.write_score(out, song_key="s", meta={}, title="t", quant=st.quant, tab=st.tab, grid=st.grid,
                  sections=st.sections, params=T.score_params(None), job_dir=job)
    gp = out / "tab" / "score.gp"
    doc = atomic.read_json(out / "tab" / "score_gp.json")
    assert gp.is_file() and gp.read_bytes()[:2] == b"PK"  # a zip, as GP7/8 files are
    assert doc["backing"] == "mix_44k_f32.wav" and doc["backing_bytes"] > 1000
    assert doc["sync_points"] == 4 * doc["master_bars"] and doc["drift_ms"]["max"] <= 5.0  # a point per beat
    assert list((job / "cache" / "gp7").glob("*.mp3"))  # the MP3 is cached for the edits' re-exports
