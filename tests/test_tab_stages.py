"""M3 tab stages run on their dep dirs (stage contract, M1_M2_SPEC 3.1)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bandscribe import atomic
from bandscribe.tab import stage as T

from test_tab_quantize import beat_times, grid_doc, performance


def _ctx(tmp_path, deps: dict, params: dict, name: str):
    out = tmp_path / f"out_{name}"
    out.mkdir()
    return SimpleNamespace(song_key="k", stage=name, key="0" * 64, out_dir=out, dep_dirs=deps, config=None,
                           store=None, params=params)


def test_quant_stage_reads_guitar_all_and_pass1_bass(tmp_path):
    times = beat_times(8 * 4 + 4, warp=True)
    notes, _ = performance(times, 8, seed=2)
    d = {k: tmp_path / k for k in ("notes", "amt_ms1", "grid", "sections")}
    for p in d.values():
        p.mkdir()
    atomic.write_json(d["notes"] / "guitar_all.json", {"notes": notes})
    atomic.write_json(d["amt_ms1"] / "raw" / "bass_mono__muscriptor.json", {"notes": notes[::3]})
    atomic.write_json(d["grid"] / "grid.json", grid_doc(times))
    atomic.write_json(d["sections"] / "sections.json", {"sections": [{"id": "S1", "start_bar": 1, "end_bar": 9}]})
    ctx = _ctx(tmp_path, d, T.quant_params(None), "quant")
    T.run_quant(ctx)
    doc = atomic.read_json(ctx.out_dir / "quant.json")
    assert doc["format"] == "bandscribe.quant/1"
    assert doc["tracks"]["guitar_all"]["stats"]["n"] == len(notes)
    assert doc["tracks"]["bass_raw"]["stats"]["n"] == len(notes[::3])
    assert [s["id"] for s in doc["sections"]] == ["S1"]


def test_a4_stage_reads_the_pitched_views(tmp_path):
    pytest.importorskip("librosa")
    import soundfile as sf

    from test_tab_a4 import tones

    stems = tmp_path / "stems"
    (stems / "views").mkdir(parents=True)
    sf.write(str(stems / "views" / "guitar_mono.wav"), tones(-30.0, seed=7, seconds=6.0), 22050)
    ctx = _ctx(tmp_path, {"stems": stems}, T.a4_params(None), "a4")
    T.run_a4(ctx)
    doc = atomic.read_json(ctx.out_dir / "a4.json")
    assert doc["cents"] == pytest.approx(-30.0, abs=5.0)
    assert list(doc["per_view"]) == ["guitar_mono"]
    assert doc["warning"]  # 30 cents flat >= 25: warned


def test_a4_stage_without_views_fails_in_korean(tmp_path):
    (tmp_path / "stems" / "views").mkdir(parents=True)
    ctx = _ctx(tmp_path, {"stems": tmp_path / "stems"}, T.a4_params(None), "a4")
    with pytest.raises(T.TabStageError, match="기준음"):
        T.run_a4(ctx)
