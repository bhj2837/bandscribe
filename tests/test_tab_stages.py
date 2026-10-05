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


def test_tab_stage_frets_the_bass_and_the_guitar_only_as_one_part(tmp_path):
    from bandscribe.tab import quantize as Q

    from test_tab_tuning import E_STD, bass_line, song

    times = beat_times(40 * 4 + 4, warp=False)
    tm_grid = grid_doc(times)

    def notes_of(events):
        return [{"onset_s": e.onset_s, "offset_s": off, "pitch": p} for e in events for p, off in zip(e.pitches, e.offsets_s)]

    q = Q.quantize({"guitar_all": notes_of(song(E_STD, n_bars=8)), "bass_raw": notes_of(bass_line((28, 33, 38, 43)))},
                   tm_grid)
    d = {"quant": tmp_path / "quant", "a4": tmp_path / "a4"}
    atomic.write_json(d["quant"] / "quant.json", q)
    atomic.write_json(d["a4"] / "a4.json", {"cents": 2.0, "a4_hz": 440.5, "confidence": 0.5})
    for parts in (0, 1):
        params = T.tab_params(None)
        params["guitar_parts"] = parts
        ctx = _ctx(tmp_path, d, params, f"tab{parts}")
        T.run_tab(ctx)
        doc = atomic.read_json(ctx.out_dir / "tab.json")
        assert doc["tuning"]["guitar"]["label"] == "E standard" and doc["tuning"]["bass"]["label"] == "E standard"
        bass = doc["tracks"]["bass_raw"]
        assert bass["fretted"] and bass["check"]["bad_events"] == [] and bass["check"]["dropped"] == 0
        assert all(n["string"] is not None for n in bass["notes"])
        assert doc["tracks"]["guitar_all"]["fretted"] is (parts == 1)
        assert (ctx.out_dir / "text" / "bass_raw.txt").is_file()
        assert (ctx.out_dir / "text" / "guitar_all.txt").is_file() is (parts == 1)


def test_out_of_range_notes_are_written_an_octave_inside():
    from bandscribe.tab import fretting as F

    bass = F.Instrument(F.BASS_STD)
    assert T._octave_into_range(22, bass) == 12      # Bb0 on a 4-string bass -> Bb1
    assert T._octave_into_range(100, bass) == -36    # far above fret 24 of the G string -> E4
    assert T._octave_into_range(40, bass) == 0
    q = [{"gtick": 0, "dur": 48, "bar": 1, "tick": 0, "pitch": 22, "onset_s": 0.0, "offset_s": 0.4},
         {"gtick": 48, "dur": 48, "bar": 1, "tick": 48, "pitch": 33, "onset_s": 0.5, "offset_s": 0.9}]
    tr = T._fret_track(q, {"tuning": list(F.BASS_STD), "capo": 0, "label": "E standard"}, dict(F.DEFAULT_WEIGHTS))
    assert tr["notes"][0]["octave_shift"] == 12 and tr["notes"][0]["pitch"] == 34
    assert tr["notes"][0]["string"] is not None and tr["check"]["octave_shifted"] == 1 and tr["check"]["dropped"] == 0


def test_text_tab_keeps_a_note_in_the_last_ticks_of_a_bar():
    from bandscribe.tab import ascii as A

    bars = [{"bar": 1, "n_beats": 4}]
    txt = A.render([{"bar": 1, "tick": 186, "string": 1, "fret": 7}, {"bar": 1, "tick": 0, "string": 6, "fret": 3}],
                   bars, [40, 45, 50, 55, 59, 64])
    lines = txt.splitlines()
    assert any(ln.startswith("e|") and "7" in ln for ln in lines) and any(ln.startswith("E|3") for ln in lines)


def test_instrument_choice_does_not_depend_on_hash_order():
    """Review 2026-10-05: equal class counts picked the instrument by set iteration order (PYTHONHASHSEED)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    tests_dir = str(Path(__file__).parent)
    code = "\n".join([
        f"import sys; sys.path.insert(0, {tests_dir!r})",
        "from test_tab_quantize import beat_times, grid_doc",
        "from bandscribe.tab import quantize as Q, stage as S",
        "times = beat_times(20, warp=False); grid = grid_doc(times)",
        "cls = ['distorted_electric_guitar', 'clean_electric_guitar']",
        "notes = [{'onset_s': times[i], 'offset_s': times[i] + 0.2, 'pitch': 60, 'instrument': cls[i % 2]}"
        " for i in range(8)]",
        "q = Q.quantize({'guitar_all': notes}, grid)",
        "score, problems = S.build_score('t', q, {'tracks': {}}, grid, [], {'tempo_change_ratio': 0.04})",
        "print(score.tracks[0].instrument)",
    ])
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(seed)}).stdout.strip() for seed in range(12)}
    assert outs == {"electricguitarclean"}


def test_tab_params_take_a_user_tuning_by_name_or_alias():
    from bandscribe.config import ConfigError

    assert T.tab_params(None)["guitar_tuning"] == "auto" and T.tab_params(None)["bass_tuning"] == "auto"
    assert T.tab_params({"tab.guitar_tuning": "standard"})["guitar_tuning"] == "E standard"  # one cache key per tuning
    assert T.tab_params({"tab.bass_tuning": "drop d"})["bass_tuning"] == "Drop D"
    with pytest.raises(ConfigError, match="기타 튜닝"):
        T.tab_params({"tab.guitar_tuning": "banana"})


def test_tab_stage_writes_a_user_tuning_and_shifts_what_it_cannot_play(tmp_path):
    from bandscribe.tab import quantize as Q

    from test_tab_tuning import riff

    drop = (38, 45, 50, 55, 59, 64)
    times = beat_times(20 * 4 + 4, warp=False)
    evs = riff(drop, [0, 0, 3, 5, 0, 0, 7, 5])
    notes = [{"onset_s": e.onset_s + 0.5, "offset_s": off + 0.5, "pitch": p} for e in evs for p, off in zip(e.pitches, e.offsets_s)]
    q = Q.quantize({"guitar_all": notes, "bass_raw": []}, grid_doc(times))
    d = {"quant": tmp_path / "quant", "a4": tmp_path / "a4"}
    atomic.write_json(d["quant"] / "quant.json", q)
    atomic.write_json(d["a4"] / "a4.json", {"cents": 0.0, "a4_hz": 440.0, "confidence": 0.9})
    params = T.tab_params({"tab.guitar_tuning": "E standard", "tab.guitar_parts": 1})
    ctx = _ctx(tmp_path, d, params, "tab_user")
    T.run_tab(ctx)
    doc = atomic.read_json(ctx.out_dir / "tab.json")
    g = doc["tuning"]["guitar"]
    assert (g["label"], g["user"], g["auto_label"]) == ("E standard", True, "Drop D")
    tr = doc["tracks"]["guitar_all"]
    assert tr["tuning"] == [40, 45, 50, 55, 59, 64] and tr["check"]["octave_shifted"] > 0
    assert tr["check"]["bad_events"] == []
