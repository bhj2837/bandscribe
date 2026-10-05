"""Basic Pitch minimum-length semantics and parameter handling (worker helpers + amt_bp stage params)."""
from __future__ import annotations

import pytest

from bandscribe.amt import stage as S
from bandscribe.workers import amt_basicpitch as W


@pytest.mark.parametrize("frames", [1, 3, 4, 5, 7, 11])
def test_min_note_len_is_frames_minus_one(frames):
    assert W.min_note_len_for(frames) == frames - 1


@pytest.mark.parametrize("frames", [13, 20, 30])
def test_minimum_above_the_upstream_default_needs_the_flag_too(frames):
    """Review 2026-10-05: the guard used to refuse only exactly 12; 13+ deletes even more short notes."""
    with pytest.raises(W.ParamError, match="127.7"):
        W.min_note_len_for(frames)
    assert W.min_note_len_for(frames, allow_default_min_len=True) == frames - 1
    with pytest.raises(W.ParamError):
        W.normalize_params({"min_note_frames": frames})


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


def test_thread_count_is_in_every_basic_pitch_cache_key():
    """The amt_bp stage key had bp_threads but the transcription cache key did not (review 2026-10-05): the thread
    count changes the ONNX output in the last bit, so both keys (and the model-output file) carry it now."""
    from bandscribe.amt import cache as amt_cache

    p2, p4 = S.bp_params({"amt": {"bp_threads": 2}}), S.bp_params({"amt": {"bp_threads": 4}})
    assert S._bp_cache_params(p2) == {"params": p2["params"], "threads": 2}
    keys = {amt_cache.amt_key("basicpitch", "0.4.0", "m" * 64, "v" * 64, None, S._bp_cache_params(p), "1", S.BP_RUNG)
            for p in (p2, p4)}
    assert len(keys) == 2
    # model-output cache file: the pre-2026-10-05 name (all written with 4 threads) stays valid for 4 threads
    assert W.model_output_name("v" * 64, "abcdef0123456789", 4) == f"{'v' * 64}__abcdef012345.npz"
    assert W.model_output_name("v" * 64, "abcdef0123456789", 2) == f"{'v' * 64}__abcdef012345__t2.npz"


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
