"""M9: the Guitar Pro 7/8 file (``tab/score.gp``): one sync point per beat, the backing choice, the drift measure,
and (with Node + alphaTab) the written file with its embedded backing track read back."""

from __future__ import annotations

import numpy as np
import pytest

from bandscribe import atomic
from bandscribe.tab import alphatex as ATX
from bandscribe.tab import stage as T
from pathlib import Path


def _score(bars):
    return ATX.WScore(title="t", bars=bars, tempo_qpm=[120])


def _bar(no, start):
    from bandscribe.tab.notation import WBar

    return WBar(bar=no, start_gtick=0, n_beats=4, beat_unit="quarter", numerator=4, denominator=4, start_s=start,
                feel=None)


def test_one_sync_point_per_beat_and_the_drift_measure():
    grid = {"beats": [{"t_s": 0.5 * k + (0.04 if k == 6 else 0.0), "bar": 1 + k // 4} for k in range(8)]}
    score = _score([_bar(1, 0.0), _bar(2, 2.0), _bar(3, 4.0)])
    for k, b in enumerate(score.bars):
        b.start_gtick = 4 * 48 * k
    pts = T.gp7_sync_points(score, grid)
    assert [(p["barIndex"], p["barPosition"]) for p in pts[:5]] == [(0, 0.0), (0, 0.25), (0, 0.5), (0, 0.75), (1, 0.0)]
    assert pts[6]["millisecondOffset"] == pytest.approx(3040.0)
    assert pts[-1] == {"barIndex": 2, "barOccurence": 0, "barPosition": 0.0, "millisecondOffset": 4000.0}  # no beats
    # a player that put beat 6 on the straight 120 BPM line: 40 ms off there, exact elsewhere
    check = {"sync": [[0, 0, 0.0], [1, 3840, 2000.0], [2, 7680, 4000.0]], "bars": [[0, 3840], [3840, 3840]]}
    # notes: one written on bar 2's beat 3 that sounded 40 ms late, one on bar 1's beat 2 heard 10 ms early
    quant = {"tracks": {"guitar_all": {"notes": [{"bar": 2, "gtick": 4 * 48 + 2 * 48, "onset_s": 3.04},
                                                 {"bar": 1, "gtick": 48, "onset_s": 0.49}]}}}
    d = T.gp7_drift(check, score, grid, quant)
    b = d["beats"]
    assert b["beats"] == 8 and b["max"] == pytest.approx(40.0) and b["median"] == pytest.approx(0.0)
    assert d["guitar_all"]["notes"] == 2 and d["guitar_all"]["max"] == pytest.approx(40.0) and d["bass"] is None


def test_backing_choice(tmp_path):
    job = tmp_path / "job"
    assert T.backing_audio(job, "original") is None and T.backing_audio(None, "original") is None
    (job / "input").mkdir(parents=True)
    (job / "input" / "mix_44k_f32.wav").write_bytes(b"x")
    assert T.backing_audio(job, "original").name == "mix_44k_f32.wav" and T.backing_audio(job, "none") is None
    stems = job / "stages" / "stems" / "k1"
    (stems / "practice").mkdir(parents=True)
    (stems / "practice" / "mix_minus_guitar.wav").write_bytes(b"x")
    # review 2026-10-18: only the score stage's own stems dependency, never a newest-by-mtime guess
    assert T.backing_audio(job, "no_guitar") is None
    assert T.backing_audio(job, "no_guitar", stems_dir=stems) == stems / "practice" / "mix_minus_guitar.wav"
    sd = job / "stages" / "score" / "abcdefabcdef"
    sd.mkdir(parents=True)
    atomic.write_json(sd / "manifest.json", {"deps": {"stems": "k1"}})
    assert T.stems_dir_of(job, "abcdefabcdef") == stems and T.stems_dir_of(job, None) is None


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
    assert doc["sync_points"] == 4 * doc["master_bars"] and doc["drift_ms"]["beats"]["max"] <= 5.0  # point per beat
    assert doc["drift_ms"]["guitar_all"]["notes"] > 0  # written notes against their onsets: quantisation only here
    assert list((job / "cache" / "gp7").glob("*.mp3"))  # the MP3 is cached for the edits' re-exports


def test_an_export_with_a_locked_file_leaves_the_old_export_whole(tmp_path, monkeypatch):
    """Review 2026-10-18: a file another program holds open (here the role view's GP5, the 4th file) must not leave
    the export half new - every old file moves aside before any new one goes in."""
    from test_parts_tracks import make_parts_job

    from bandscribe.tab import edits as E

    job, d = make_parts_job(tmp_path)
    out = job / "export"
    s = E.Session(job)
    try:
        before = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()
                  and p.name != "export.json"}
        real = atomic._replace
        locked = (out / "tab" / "views" / "role.gp5").resolve()

        def refuse(src, dst):
            if Path(src).resolve() == locked:
                raise PermissionError("held open")
            return real(src, dst)

        monkeypatch.setattr(atomic, "_replace", refuse)
        (out / "tab" / "views").mkdir(parents=True, exist_ok=True)
        locked.write_bytes(b"old view")
        before["tab/views/role.gp5"] = b"old view"
        with pytest.raises(PermissionError):
            E.export(s.base, s.state, "h")
        after = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()
                 and p.name != "export.json" and ".edits-" not in p.as_posix()}
        assert after == before
    finally:
        monkeypatch.undo()
        s.close()
