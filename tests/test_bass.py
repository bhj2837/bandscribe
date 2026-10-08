"""M5 bass: F0 vote and contours (bandscribe.bass.f0), the S11 clean-up (bandscribe.bass.clean), the bass stage, and
the technique marks through fretting, spelling, alphaTex, GP5 and the text tab."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from bandscribe import atomic
from bandscribe.bass import clean as CL
from bandscribe.bass import f0 as F0

HOP = F0.HOP_S


def track_from(midi_per_frame: np.ndarray, *, trackers=("crepe", "pyin"), conf: float = 0.9, octave_off=()) -> F0.F0Track:
    """A track whose named trackers all report ``midi_per_frame`` (NaN = unvoiced); trackers in ``octave_off``
    report it an octave higher."""
    hz, cf = {}, {}
    for name in trackers:
        m = np.asarray(midi_per_frame, dtype=float) + (12.0 if name in octave_off else 0.0)
        h = np.where(np.isfinite(m), F0.midi_to_hz(np.nan_to_num(m, nan=60.0)), 0.0)
        hz[name] = h.astype(np.float32)
        cf[name] = np.where(np.isfinite(m), conf, 0.0).astype(np.float32)
    return F0.F0Track(hz=hz, conf=cf)


def note(on: float, off: float, pitch: int) -> dict:
    return {"onset_s": on, "offset_s": off, "pitch": pitch, "instrument": "electric_bass"}


def frames(seconds: float) -> int:
    return F0.n_frames(seconds)


# ------------------------------------------------------------------------------------------------- vote


@pytest.mark.parametrize("pitch, ests, action, new", [
    (40, {"crepe": 40.1, "pyin": 39.9}, "confirmed", 40),
    (52, {"crepe": 40.1, "pyin": 39.9, "pesto": 52.0}, "octave", 40),  # two say an octave lower
    (40, {"crepe": 41.2, "pyin": 40.9}, "semitone", 41),
    (40, {"crepe": 47.0, "pyin": 47.2}, "disagree", 40),  # a fifth: left alone (review)
    (40, {"crepe": 40.0, "pyin": 46.0}, "split", 40),
    (40, {"crepe": 40.0}, "no_f0", 40),
])
def test_vote_moves_only_octaves_and_one_semitone(pitch, ests, action, new):
    v = F0.vote(pitch, ests)
    assert v.action == action
    assert v.pitch == new


def test_note_estimates_use_the_evidence_window_and_confidence():
    m = np.full(frames(2.0), 40.0)
    m[100:104] = 52.0  # the attack transient right at the onset (outside onset + 30 ms .. + 120 ms)
    tr = track_from(m)
    ests = F0.note_estimates(tr, 1.0, 1.5)
    assert ests == {"crepe": pytest.approx(40.0, abs=0.01), "pyin": pytest.approx(40.0, abs=0.01)}
    tr.conf["pyin"][:] = 0.2  # below conf_min: not a vote
    assert set(F0.note_estimates(tr, 1.0, 1.5)) == {"crepe"}


def test_consensus_folds_single_tracker_octave_slips_to_the_reference():
    m = np.full(frames(1.0), 33.0)
    tr = track_from(m, trackers=("crepe", "pesto"), octave_off=("pesto",))
    assert np.all(np.isnan(F0.consensus(tr)))  # the two disagree by an octave: no consensus without a reference
    c = F0.consensus(tr, ref_midi=np.full(tr.n, 33.0))
    assert np.nanmax(np.abs(c - 33.0)) < 0.01


def test_running_median_bridges_single_dropouts_and_keeps_steps():
    x = np.array([40.0] * 6 + [np.nan] + [40.0] * 3 + [43.0] * 8)
    y = F0.running_median(x)
    assert y[6] == pytest.approx(40.0)
    assert y[12] == pytest.approx(43.0) and y[8] == pytest.approx(40.0)


def test_accuracy_counts_bands_and_octave_errors():
    ref = np.full(200, np.nan)
    ref[10:100] = 31.0  # G1, 49 Hz: the 30-100 Hz band
    ref[100:190] = 55.0  # G3, 196 Hz
    est = ref.copy()
    est[100:145] += 12.0  # half the high band an octave up
    tr = track_from(est, trackers=("crepe",))
    acc = F0.accuracy(tr, ref)["crepe"]
    assert acc["30-100"]["rpa"] == pytest.approx(1.0)
    assert acc["100-400"]["rpa"] == pytest.approx(0.5)
    assert acc["100-400"]["rca"] == pytest.approx(1.0)
    assert acc["100-400"]["octave"] == pytest.approx(0.5)


def test_npz_round_trip_is_byte_stable(tmp_path):
    tr = track_from(np.linspace(30, 50, 300))
    F0.save_npz(tmp_path / "a.npz", tr)
    F0.save_npz(tmp_path / "b.npz", F0.load_npz(tmp_path / "a.npz"))
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    back = F0.load_npz(tmp_path / "a.npz")
    assert np.allclose(back.hz["crepe"], tr.hz["crepe"]) and back.names == ["crepe", "pyin"]


def test_crepe_fmin_stays_above_the_lowest_bin():
    """DESIGN 5.3 / M5: torchcrepe's lowest bin is 31.70 Hz; a lower fmin gives a negative bin index (fmin 30 =
    every bin masked but the top ones). The default stays above it and the decoder refuses lower values."""
    from bandscribe.workers import f0_bass as W

    assert F0.DEFAULT_PARAMS["crepe"]["fmin_hz"] >= W.MIN_FMIN_HZ > F0.CREPE_LOWEST_BIN_HZ
    act = np.zeros((5, W.CREPE_BINS))
    with pytest.raises(ValueError):
        W.decode_crepe(np, act, 30.0, 400.0, 4)


def test_crepe_decoder_follows_peaked_activations_over_a_noise_floor():
    """The salience is normalised (original CREPE): a path through clear peaks, not the softmax of sigmoid outputs
    that sat on bin 0 on bass stems."""
    from bandscribe.workers import f0_bass as W

    rng = np.random.default_rng(0)
    act = rng.uniform(0.0, 0.05, size=(60, W.CREPE_BINS))
    true_bin = 69  # 69.3 Hz = C#2
    act[10:50, true_bin] = 0.9
    act[10:50, true_bin - 1] = act[10:50, true_bin + 1] = 0.5
    act[30:32, 2] = 0.8  # a kick thump on a bottom bin for two frames
    hz, conf = W.decode_crepe(np, act, 32.0, 400.0, 4)
    cents = 1200.0 * np.log2(hz[20:40] / 10.0)
    assert np.allclose(cents, W.CREPE_CENTS0 + 20.0 * true_bin, atol=5.0)
    assert np.all(conf[20:40] == pytest.approx(0.9))
    # torchcrepe's decoding (softmax of the sigmoid outputs) is nearly flat and does not follow the peak
    z = np.exp(act - act.max(axis=1, keepdims=True))
    import librosa

    flat = librosa.sequence.viterbi((z / z.sum(axis=1, keepdims=True)).T, W.crepe_transition(np))
    assert np.mean(flat[20:40] == true_bin) < 0.5


# ------------------------------------------------------------------------------------------------ clean-up


# seisyun complex 3.0-5.6 s (CREPE on the bass stem, 2026-10-08): a plucked F#2 sliding down to G1, slower in the
# middle; MuScriptor wrote it as the staircase in ``slide_case``
SC_SLIDE = ((3.03, 42.5), (3.10, 41.8), (3.15, 41.3), (3.25, 41.0), (3.30, 40.6), (3.40, 40.3), (3.50, 39.6),
            (3.55, 39.3), (3.70, 39.1), (3.75, 38.5), (3.85, 38.2), (3.95, 37.4), (4.00, 37.2), (4.15, 36.8),
            (4.25, 36.1), (4.40, 35.5), (4.50, 35.3), (4.60, 34.7), (4.70, 34.2), (4.85, 33.3), (5.00, 32.6),
            (5.10, 32.1), (5.30, 31.2), (5.55, 31.2))


def slide_case():
    """seisyun complex's intro: one slide, transcribed as seven notes."""
    n = frames(7.0)
    m = np.full(n, np.nan)
    t = np.arange(n) * HOP
    on = (t >= 3.03) & (t <= 5.55)
    m[on] = np.interp(t[on], [x for x, _ in SC_SLIDE], [y for _, y in SC_SLIDE])
    notes = [note(3.0, 3.48, 41), note(3.48, 3.72, 39), note(3.72, 3.96, 38), note(3.96, 4.2, 37),
             note(4.2, 4.44, 36), note(4.44, 5.11, 35), note(5.11, 6.0, 31)]
    return track_from(m), notes


def test_a_glide_written_as_a_staircase_becomes_one_slide():
    tr, notes = slide_case()
    out, rep = CL.clean(notes, tr, CL.DEFAULT_PARAMS)  # E13's adopted steps: merges on, anchor off
    assert [(n["pitch"], n.get("tech", {}).get("slide")) for n in out] == [(41, "legato"), (31, None)]
    assert out[0]["onset_s"] == pytest.approx(3.0) and out[0]["offset_s"] == pytest.approx(5.11)
    assert out[0]["merged"] == 5 and rep["slides"]["chains"] == 1


def test_the_anchor_misreads_the_ends_of_a_slide():
    """Why E13 left the anchor off: its 30-120 ms window catches a slide still moving - the pluck reads a
    semitone up and the landing note (G1, still arriving) a semitone up too."""
    tr, notes = slide_case()
    out, _ = CL.clean(notes, tr, CL.arm_params("anchor_merge"))
    assert [(n["pitch"], n.get("from_pitch")) for n in out] == [(42, 41), (32, 31)]


def test_a_plucked_chromatic_run_is_not_a_slide():
    n = frames(3.0)
    m = np.full(n, np.nan)
    notes = []
    for k, p in enumerate((40, 39, 38, 37, 36)):  # 160 ms notes, the pitch steps at each pluck
        a, b = 0.5 + 0.16 * k, 0.5 + 0.16 * (k + 1)
        m[int(a / HOP) + 1:int(b / HOP)] = p
        notes.append(note(a, b, p))
    out, rep = CL.clean(notes, track_from(m), CL.arm_params("anchor_merge"))
    assert [x["pitch"] for x in out] == [40, 39, 38, 37, 36]
    assert not any(x.get("tech") for x in out) and rep["slides"]["chains"] == 0


def test_anchor_moves_an_octave_error_and_drops_its_harmonic_twin():
    m = np.full(frames(2.0), 33.0)
    tr = track_from(m, trackers=("crepe", "pyin", "pesto"), octave_off=("pesto",))
    notes = [note(0.5, 1.0, 45), note(1.2, 1.6, 33), note(1.2, 1.6, 52)]  # 45: octave error; 52: 33 + 19 (harmonic)
    out, rep = CL.clean(notes, tr, CL.arm_params("anchor"))
    assert [(x["onset_s"], x["pitch"]) for x in out] == [(0.5, 33), (1.2, 33)]
    assert out[0]["from_pitch"] == 45 and out[0]["f0"] == "octave"
    assert rep["vote"]["octave"] == 1 and rep["dupes_dropped"] == 1
    raw, _ = CL.clean(notes, tr, CL.arm_params("raw"))
    assert [x["pitch"] for x in raw] == [45, 33, 52]  # the raw arm changes nothing


def test_fragments_and_decaying_tails_join_but_replucks_stay():
    m = np.full(frames(3.0), np.nan)
    m[50:150] = 40.0  # 0.5-1.5 s sounding; 1.5-1.8 s silent (the tail has no F0)
    m[200:260] = 40.0
    tr = track_from(m)
    notes = [note(0.5, 1.0, 40), note(1.0, 1.05, 40),  # a 50 ms fragment
             note(1.05, 1.5, 40), note(1.5, 1.8, 40),  # a tail without F0 (no attack: silence in the test audio)
             note(2.0, 2.6, 40)]  # after a gap: a new note
    attack = SimpleNamespace(rise_db=lambda t: 0.0)
    out, rep = CL.clean(notes, tr, CL.arm_params("anchor_merge"), attack=attack)
    assert [(x["onset_s"], x["offset_s"]) for x in out] == [(0.5, 1.05), (1.05, 1.8), (2.0, 2.6)]
    assert rep["fragments_merged"] == 2


def test_kick_rejection_drops_only_notes_on_a_kick_without_f0():
    m = np.full(frames(3.0), np.nan)
    m[100:150] = 36.0
    tr = track_from(m)
    notes = [note(1.0, 1.4, 36), note(2.0, 2.1, 28), note(2.5, 2.6, 28)]
    out, rep = CL.clean(notes, tr, CL.arm_params("anchor_merge_kick"), kicks=np.array([1.0, 2.01]))
    assert [x["onset_s"] for x in out] == [1.0, 2.5]  # 1.0 has F0, 2.5 is not on a kick
    assert rep["kick_dropped"] == 1


def test_gap_fill_adds_low_confidence_notes_only_where_nothing_sounds():
    m = np.full(frames(3.0), np.nan)
    m[50:100] = 40.0  # covered by a MuScriptor note
    m[150:200] = 43.0  # nothing transcribed here
    m[230:236] = 45.0  # too short (60 ms)
    tr = track_from(m)
    out, rep = CL.clean([note(0.5, 1.0, 40)], tr, CL.arm_params("two_voter"))
    added = [x for x in out if x["src"] == "f0"]
    assert [(x["pitch"], x["conf"]) for x in added] == [(43, 0.3)]
    assert added[0]["onset_s"] == pytest.approx(1.48) and rep["gap_filled"] == 1


def test_and_policy_keeps_only_f0_confirmed_notes():
    m = np.full(frames(2.0), np.nan)
    m[50:100] = 40.0
    tr = track_from(m)
    out, rep = CL.clean([note(0.5, 1.0, 40), note(1.2, 1.5, 45)], tr, CL.arm_params("and"))
    assert [x["pitch"] for x in out] == [40] and rep["and_dropped"] == 1


def test_vibrato_and_dead_note_suggestions():
    n = frames(3.0)
    t = np.arange(n) * HOP
    m = np.full(n, np.nan)
    on = (t >= 0.5) & (t < 1.5)
    m[on] = 40.0 + 0.3 * np.sin(2 * np.pi * 5.5 * t[on])  # 5.5 Hz, ±30 cents
    tr = track_from(m)
    attack = SimpleNamespace(rise_db=lambda x: 20.0 if abs(x - 2.0) < 0.01 else 0.0)
    out, rep = CL.clean([note(0.5, 1.5, 40), note(2.0, 2.1, 38)], tr, CL.DEFAULT_PARAMS, attack=attack)
    assert out[0]["tech"] == {"vibrato": True}
    assert out[1]["tech"] == {"dead": True}
    assert rep["techniques"] == {"vibrato": 1, "dead": 1}


def test_arm_params_cover_every_arm():
    assert set(CL.ARMS) == {"raw", "anchor", "anchor_merge", "anchor_merge_kick", "two_voter", "and"}
    for arm in CL.ARMS:
        assert set(CL.arm_params(arm)) == set(CL.DEFAULT_PARAMS)
    with pytest.raises(ValueError):
        CL.arm_params("nope")


# ------------------------------------------------------------------------------------- written techniques


def test_slide_partners_are_fretted_on_one_string():
    from bandscribe.tab import fretting as F

    inst = F.Instrument(F.BASS_STD)
    ev = F.group_events([{"gtick": 0, "onset_s": 0.0, "offset_s": 1.0, "pitch": 42},
                         {"gtick": 48, "onset_s": 1.0, "offset_s": 2.0, "pitch": 31}])
    free = [(a.string, a.fret) for a in F.assign(ev, inst)]
    assert free[0][0] != free[1][0]  # alone, F#2 goes to the D string
    for model in ("window", "v1"):
        assert [(a.string, a.fret) for a in F.assign(ev, inst, model=model, links={1})] == [(4, 14), (4, 3)]


def test_tied_pieces_keep_dead_first_and_slide_last():
    from bandscribe.tab import notation as N

    a = N.WNote(40, 4, 12, False, ("b:0",), ("x", "v", "sl"))
    b = N.WNote(40, 4, 12, True, ("b:0",), ("x", "v", "sl"))
    c = N.WNote(40, 4, 12, True, ("b:0",), ("x", "v", "sl"))
    N.place_effects([[N.WBeat(0, 48, 4, notes=[a]), N.WBeat(48, 48, 4, notes=[b])], [N.WBeat(0, 48, 4, notes=[c])]])
    assert (a.fx, b.fx, c.fx) == (("x", "v"), ("v",), ("v", "sl"))


def test_alphatex_gp5_and_text_tab_write_the_techniques(tmp_path):
    import guitarpro as gp

    from bandscribe.tab import alphatex as ATX
    from bandscribe.tab import ascii as ascii_tab
    from bandscribe.tab import gp5 as GP5
    from bandscribe.tab import notation as N
    from bandscribe.tab import stage as T

    bar = N.WBar(bar=1, start_gtick=0, n_beats=4, beat_unit="quarter", numerator=4, denominator=4, start_s=0.0,
                 feel=None)
    # the writers know every mark (vibrato and dead notes are not written by default: notation.WRITTEN_TECH)
    notes = [N.WNote(42, 4, 14, fx=T.note_fx({"tech": {"slide": "legato"}})), N.WNote(31, 4, 3, fx=("v",)),
             N.WNote(31, 4, 3, fx=("x",)), N.WNote(33, 4, 5, fx=T.note_fx({"src": "f0"}))]
    beats = [N.WBeat(48 * k, 48, 4, notes=[x]) for k, x in enumerate(notes)]
    score = ATX.WScore(title="t", bars=[bar], tempo_qpm=[120],
                       tracks=[ATX.WTrack("Bass", "Bass", "electricbassfinger", True, [28, 33, 38, 43], 0, [beats])])
    tex = ATX.write(score)
    assert "14.4{sl}.4" in tex and "3.4{v}.4" in tex and "3.4{x}.4" in tex and "5.4{g}.4" in tex
    GP5.write(score, tmp_path / "s.gp5")
    song = gp.parse(str(tmp_path / "s.gp5"))
    got = [n for be in song.tracks[0].measures[0].voices[0].beats for n in be.notes]
    assert got[0].effect.slides == [gp.SlideType.legatoSlideTo] and got[1].effect.vibrato
    assert got[2].type == gp.NoteType.dead and got[3].effect.ghostNote
    rows = [{"bar": 1, "tick": 0, "string": 4, "fret": 14, "tech": {"slide": "legato"}},
            {"bar": 1, "tick": 48, "string": 4, "fret": 3, "tech": {"vibrato": True}},
            {"bar": 1, "tick": 96, "string": 4, "fret": 3, "tech": {"dead": True}}]
    text = ascii_tab.render(rows, [{"bar": 1, "n_beats": 4}], [28, 33, 38, 43])
    e_row = next(line for line in text.splitlines() if line.startswith("E|"))
    assert "14\\" in e_row and "~" not in e_row and "x" not in e_row  # suggestions not written stay out


def test_note_fx_maps_every_technique():
    from bandscribe.tab import stage as T

    from bandscribe.tab import notation as N

    assert N.WRITTEN_TECH == ("slide",)  # E13: vibrato and dead-note marks only reach score.json
    assert T.note_fx({}) == ()
    assert T.note_fx({"src": "f0", "tech": {"dead": True, "vibrato": True, "slide": "out_down"}}) == ("g", "sod")
    assert T.note_fx({"tech": {"slide": "out_up"}}) == ("sou",)


def test_quantiser_carries_technique_fields():
    from bandscribe.tab import quantize as Q

    from test_tab_quantize import beat_times, grid_doc

    times = beat_times(12, warp=False)
    ns = [{"onset_s": times[2], "offset_s": times[3], "pitch": 40, "tech": {"slide": "legato"}, "src": "ms",
           "from_pitch": 52}, {"onset_s": times[4], "offset_s": times[5], "pitch": 33}]
    q = Q.quantize({"bass": ns}, grid_doc(times))
    rows = Q.notes_of(q, "bass")
    assert rows[0]["tech"] == {"slide": "legato"} and rows[0]["from_pitch"] == 52 and "tech" not in rows[1]


# ----------------------------------------------------------------------------------------------- the stage


def test_bass_stage_cleans_the_gated_bass(tmp_path):
    import soundfile as sf

    from bandscribe.bass import stage as BS

    from test_tab_quantize import beat_times, grid_doc

    tr, notes = slide_case()
    d = {k: tmp_path / k for k in ("notes", "bass_f0", "stems", "sep", "grid")}
    for p in d.values():
        p.mkdir()
    atomic.write_json(d["notes"] / "bass_raw.json", {"format": "bandscribe.notes/1", "notes": notes,
                                                     "audio_duration_s": 7.0})
    F0.save_npz(d["bass_f0"] / BS.F0_FILE, tr)
    (d["stems"] / "views").mkdir()
    sf.write(str(d["stems"] / "views" / "bass_mono.wav"), np.zeros(7 * 8000, dtype=np.float32), 8000)
    atomic.write_json(d["grid"] / "grid.json", grid_doc(beat_times(16, warp=False)))
    ctx = SimpleNamespace(song_key="k", stage="bass", key="0" * 64, out_dir=tmp_path / "out", dep_dirs=d,
                          config=None, store=None, params=BS.bass_params(None))
    BS.run_bass(ctx)
    doc = atomic.read_json(tmp_path / "out" / BS.BASS_FILE)
    assert doc["format"] == "bandscribe.bass/1"
    assert [n["pitch"] for n in doc["notes"]] == [41, 31]  # E13 defaults: merges on, the anchor (41 -> 42) off
    rep = atomic.read_json(tmp_path / "out" / BS.REPORT_FILE)
    assert rep["in"] == 7 and rep["out"] == 2
    assert (tmp_path / "out" / "midi" / "bass.mid").is_file()


def test_bass_params_follow_the_config():
    from bandscribe.bass import stage as BS
    from bandscribe.config import load_config

    p = BS.bass_params(load_config(overrides=["bass.kick=true", "bass.slides=false"]))
    assert p["clean"]["kick"] is True and p["clean"]["slides"] is False
    assert set(p["f0"]) == set(BS.VOTE_KEYS)


def test_quant_reads_the_bass_stage_when_it_is_a_dep(tmp_path):
    from bandscribe.tab import stage as T

    (tmp_path / "notes").mkdir()
    (tmp_path / "bass").mkdir()
    atomic.write_json(tmp_path / "notes" / "guitar_all.json", {"notes": []})
    atomic.write_json(tmp_path / "notes" / "bass_raw.json", {"notes": [note(1.0, 2.0, 40)]})
    atomic.write_json(tmp_path / "bass" / "bass.json", {"notes": [note(1.0, 2.0, 28)]})
    assert T.quant_tracks({"notes": tmp_path / "notes", "bass": tmp_path / "bass"})["bass"][0]["pitch"] == 28
    assert T.quant_tracks({"notes": tmp_path / "notes"})["bass"][0]["pitch"] == 40  # a run from before M5


def test_old_edit_logs_name_the_bass_track_bass_raw(tmp_path):
    from bandscribe.tab import edits as E

    (tmp_path / E.EDITS_FILE).write_text(
        '{"id": "e00001", "op": "pitch", "note": {"track": "bass_raw", "onset_s": 1.0, "pitch": 40}, "to": 41, "v": 1}\n'
        '{"id": "e00002", "op": "assign", "note": {"track": "guitar_all", "onset_s": 2.0, "pitch": 52}, "to": "bass_raw", "v": 1}\n'
        '{"id": "e00003", "op": "add", "track": "bass_raw", "gtick": 96, "dur": 12, "pitch": 40, "v": 1}\n',
        encoding="utf-8")
    log = E.read_log(tmp_path)
    assert [log[0]["note"]["track"], log[1]["to"], log[2]["track"]] == ["bass", "bass", "bass"]


def test_tempo_map_from_dataset_beats_has_a_pickup_bar():
    from bandscribe.bass import experiments as X

    beats = [0.5 * k for k in range(14)]
    tm = X.tempo_map(beats, [1.0, 3.0, 5.0])  # two beats before the first downbeat
    bars, ticks = tm.time_to_bar_tick(np.array([1.0, 3.25, 0.5]))
    assert list(np.atleast_1d(bars)) == [1, 2, 0]
    assert float(np.atleast_1d(ticks)[1]) == pytest.approx(24.0)
    gt = X.gt_ticks([{"onset_s": 3.0, "pitch": 40}], tm)
    assert gt.notes[0].bar == 2 and gt.notes[0].tick == pytest.approx(0.0)


def test_technique_counts_against_idmt_styles():
    from bandscribe.bass import experiments as X

    it = X.Item("idmt_bass_st", "idmt_bass_st:x", "x", {},
                gt=[note(0.0, 0.5, 40), note(0.5, 1.0, 45), note(1.0, 1.2, 40)],
                styles=[{"expression": "NO"}, {"expression": "SL"}, {"expression": "DN"}])
    est = [{**note(0.01, 0.5, 40), "tech": {"slide": "legato"}}, note(0.5, 1.0, 45),
           {**note(1.0, 1.2, 38), "tech": {"dead": True}}]
    c = X.technique_counts(it, est)
    assert c["slide"] == {"gt": 1, "hit": 1, "marked": 1, "marked_hit": 1}
    assert c["dead"] == {"gt": 1, "hit": 1, "marked": 1, "marked_hit": 1}
    assert c["vibrato"]["gt"] == 0


def test_step_rules_follow_the_registration():
    from bandscribe.bass import experiments as X

    rows = []
    for g in range(8):
        for arm, tp in (("raw", 80), ("anchor", 90), ("anchor_merge", 90), ("anchor_merge_kick", 90),
                        ("two_voter", 90), ("and", 70)):
            rows.append({"system": arm, "dataset": "idmt_bass_st", "item": f"i{g}", "group": f"i{g}",
                         "scenario": "idmt_bass_st", "metric": X.METRIC, "value": 0.0, "tp": tp + g % 3,
                         "fp": 10, "fn": 100 - tp})
    out = X.step_decisions(rows, 500, 1)
    assert out["anchor"]["decision"] == "adopt"
    assert out["merge"]["decision"] == "adopt"  # no change: non-inferior
    assert out["gap_fill"]["decision"] == "hold"  # no change: CI contains 0
    assert out["kick"]["decision"] == "hold" and out["kick"]["why"] == "no rows"  # no Tier B rows
    assert math.isfinite(out["anchor"]["delta"])
