"""Basic Pitch minimum-length semantics and parameter handling (worker helpers + amt_bp stage params)."""
from __future__ import annotations

import pytest

from bandscribe.amt import stage as S
from bandscribe.workers import amt_basicpitch as W


@pytest.mark.parametrize("frames", [1, 3, 4, 5, 7, 11, 13, 30])
def test_min_note_len_is_frames_minus_one(frames):
    assert W.min_note_len_for(frames) == frames - 1


def test_upstream_default_is_refused_unless_allowed():
    with pytest.raises(W.ParamError, match="127.7"):
        W.min_note_len_for(12)
    assert W.min_note_len_for(12, allow_default_min_len=True) == 11
    # 127.7 ms -> round(127.7 / 1000 * 22050 / 256) = 11 frames passed upstream = keeps >= 12 frames
    assert round(127.7 / 1000 * 22050 / 256) == W.min_note_len_for(12, allow_default_min_len=True)


@pytest.mark.parametrize("bad", [0, 31, -1, 4.5, "x", True])
def test_min_note_frames_validation(bad):
    with pytest.raises(W.ParamError):
        W.min_note_len_for(bad)


def test_normalize_params_defaults_and_errors():
    p = W.normalize_params(None)
    assert p == {"onset_threshold": 0.5, "frame_threshold": 0.3, "min_note_frames": 4, "minimum_frequency": None,
                 "maximum_frequency": None, "melodia_trick": True, "multiple_pitch_bends": False}
    assert W.normalize_params({"minimum_frequency": 0.0, "maximum_frequency": 1200})["maximum_frequency"] == 1200.0
    with pytest.raises(W.ParamError):
        W.normalize_params({"onset_threshold": 1.5})
    with pytest.raises(W.ParamError):
        W.normalize_params({"min_note_ms": 46.4})
    with pytest.raises(W.ParamError):
        W.normalize_params({"min_note_frames": 12})
    assert W.normalize_params({"min_note_frames": 12}, allow_default_min_len=True)["min_note_frames"] == 12


def test_stage_params_come_from_config_frames():
    cfg = {"amt": {"bp_min_note_frames": 5, "bp_onset_threshold": 0.4, "bp_frame_threshold": 0.2, "bp_threads": 2,
                   "bp_min_freq_hz": 0.0, "bp_max_freq_hz": 0.0, "bp_melodia_trick": False}}
    p = S.bp_params(cfg)
    assert p["params"]["min_note_frames"] == 5 and p["params"]["onset_threshold"] == 0.4
    assert p["params"]["minimum_frequency"] is None and p["params"]["melodia_trick"] is False
    assert p["threads"] == 2 and p["views"] == ["guitar_mono", "bass_mono"]
    with pytest.raises(W.ParamError):
        S.bp_params({"amt": {"bp_min_note_frames": 12}})


def test_default_config_is_not_the_upstream_default():
    assert S.bp_params({})["params"]["min_note_frames"] == 7  # E23 (2026-10-03); DESIGN §9.7: never 127.7 ms (12)


def test_events_to_notes_bends_and_sorting():
    fps = 22050 / 256
    events = [(1.0, 1.5, 64, 0.8, [0, 0, 0]), (0.5, 0.9, 60, 0.61234567, [0, 1, 2, -3])]
    notes = W.events_to_notes(events)
    assert [n["pitch"] for n in notes] == [60, 64]
    assert "bend" not in notes[1]  # all-zero bins -> no bend
    b = notes[0]["bend"]
    assert b["cents"] == [0.0, 33.3333, 66.6667, -100.0]
    assert b["t_s"] == [0.5, round(0.5 + 1 / fps, 6), round(0.5 + 2 / fps, 6), round(0.5 + 3 / fps, 6)]
    assert notes[0]["amplitude"] == 0.612346


def test_sweep_out_paths():
    from pathlib import Path

    assert W.sweep_out_path(Path("a/raw/guitar_mono__basicpitch.json"), 3).name == "guitar_mono__basicpitch__s3.json"


def test_bp_models_absent_without_env(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "bp310_python", lambda: tmp_path / "missing" / "python.exe")
    assert S.bp_models({}) == {"basic_pitch": "absent"}
