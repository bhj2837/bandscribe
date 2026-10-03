"""Performance MIDI: tempo map from the grid, lead-in, pickup, 6/8, determinism (gtab.amt.midi)."""
from __future__ import annotations

import math

import mido
import pytest

from gtab.amt import midi as M


def grid(beat_times, beats_per_bar, *, pickup=0, unit="quarter", den=4):
    """grid.json-like dict: bar 0 = pickup (if any), bars 1.. full; beats numbered within bars."""
    beats, bars = [], []
    i = 0
    bar = 0 if pickup else 1
    while i < len(beat_times):
        n = pickup if (bar == 0) else beats_per_bar
        chunk = beat_times[i:i + n]
        for k, t in enumerate(chunk):
            beats.append({"t_s": t, "t_raw_s": t, "bar": bar, "beat": k + 1, "is_downbeat": k == 0 and bar > 0,
                          "shift_ms": 0.0, "activation": None})
        end = beat_times[i + n] if i + n < len(beat_times) else chunk[-1] + (chunk[-1] - chunk[-2] if len(chunk) > 1 else 0.5)
        num = len(chunk) * (3 if unit == "dotted_quarter" else 1) if bar == 0 else beats_per_bar * (3 if unit == "dotted_quarter" else 1)
        bars.append({"bar": bar, "start_s": chunk[0], "end_s": end, "n_beats": len(chunk), "numerator": num,
                     "denominator": den, "beat_unit": unit, "pickup": bar == 0})
        i += n
        bar += 1
    return {"format": "gtab.grid/1", "tpb": 48, "duration_s": beat_times[-1] + 1, "beats": beats, "bars": bars,
            "tempo": [], "meter": [], "beat_unit": unit, "pickup": {"present": bool(pickup), "beats": pickup},
            "hypotheses": {}, "confidence": {}, "params": {}}


def read(path):
    mid = mido.MidiFile(str(path))
    assert mid.ticks_per_beat == 480
    tempos, sigs = [], []
    t = 0
    for msg in mid.tracks[0]:
        t += msg.time
        if msg.type == "set_tempo":
            tempos.append((t, msg.tempo))
        elif msg.type == "time_signature":
            sigs.append((t, msg.numerator, msg.denominator))
    notes = []
    for tr in mid.tracks[1:]:
        t = 0
        on = {}
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                on[msg.note] = t
            elif msg.type in ("note_off", "note_on"):
                notes.append((on.pop(msg.note), t, msg.note))
    return mid, tempos, sigs, sorted(notes)


def tick_to_s(tempos, tick):
    s, last_t, last_tempo = 0.0, 0, 500000
    for t, tempo in tempos:
        if t > tick:
            break
        s += (t - last_t) * last_tempo / 480e6
        last_t, last_tempo = t, tempo
    return s + (tick - last_t) * last_tempo / 480e6


def bar_starts(sigs, n_bars, total_ticks=None):
    """Tick of every MIDI bar start given the time signature list."""
    starts = []
    tick = 0
    k = 0
    while len(starts) < n_bars:
        while k + 1 < len(sigs) and sigs[k + 1][0] <= tick:
            k += 1
        num, den = sigs[k][1], sigs[k][2]
        starts.append(tick)
        tick += num * 480 * 4 // den
    return starts


NOTES = [{"onset_s": 0.5, "offset_s": 0.9, "pitch": 64, "instrument": "clean_electric_guitar"},
         {"onset_s": 1.234, "offset_s": 1.6, "pitch": 67, "instrument": "clean_electric_guitar"},
         {"onset_s": 2.001, "offset_s": 2.5, "pitch": 40, "instrument": "distorted_electric_guitar"}]


def test_lead_in_puts_bar1_on_first_downbeat(tmp_path):
    beats = [0.37 + 0.5 * k for k in range(16)]
    g = grid(beats, 4)
    info = M.write_performance_midi(tmp_path / "a.mid", [("gtr", 27, NOTES)], g)
    assert info["midi_bar_offset"] == 1 and info["lead_in"]["quarters"] == 1
    _mid, tempos, sigs, notes = read(tmp_path / "a.mid")
    assert sigs[0] == (0, 1, 4) and sigs[1] == (480, 4, 4)
    starts = bar_starts(sigs, 3)
    assert starts[1] == 480
    assert tick_to_s(tempos, starts[1]) == pytest.approx(0.37, abs=0.5 / 480 + 1e-9)  # +-1 tick
    assert tick_to_s(tempos, starts[2]) == pytest.approx(0.37 + 2.0, abs=0.002)
    assert tempos[0] == (0, 370000) and tempos[1] == (480, 500000)
    for (on, off, pitch), n in zip(notes, sorted(NOTES, key=lambda n: n["onset_s"])):
        assert tick_to_s(tempos, on) == pytest.approx(n["onset_s"], abs=0.001)
        assert tick_to_s(tempos, off) == pytest.approx(n["offset_s"], abs=0.001)


def test_long_silence_lead_in_uses_k_quarters(tmp_path):
    beats = [3.2 + 0.5 * k for k in range(12)]
    info = M.write_performance_midi(tmp_path / "b.mid", [("gtr", 27, NOTES[:1] + [dict(NOTES[1], onset_s=4.0, offset_s=4.2)])],
                                    grid(beats, 4))
    k = math.ceil(3.2 / 0.5)
    assert info["lead_in"]["quarters"] == k == 7
    _mid, tempos, sigs, notes = read(tmp_path / "b.mid")
    assert sigs[0] == (0, 7, 4)
    assert tempos[0] == (0, round(3.2 / 7 * 1e6))
    assert tick_to_s(tempos, 7 * 480) == pytest.approx(3.2, abs=1e-6)
    assert tick_to_s(tempos, notes[1][0]) == pytest.approx(4.0, abs=0.001)


def test_pickup_bar_gets_its_own_signature(tmp_path):
    beats = [0.25 + 0.5 * k for k in range(14)]  # 2 pickup beats, then 4/4
    info = M.write_performance_midi(tmp_path / "c.mid", [("gtr", 27, NOTES)], grid(beats, 4, pickup=2))
    assert info["midi_bar_offset"] == 2  # lead-in + pickup
    _mid, tempos, sigs, _n = read(tmp_path / "c.mid")
    assert sigs[:3] == [(0, 1, 4), (480, 2, 4), (480 + 960, 4, 4)]
    assert tick_to_s(tempos, 480 + 960) == pytest.approx(0.25 + 1.0, abs=0.0011)  # bar 1 = first downbeat


def test_compound_6_8_720_ticks_per_dotted_quarter(tmp_path):
    beats = [0.6 * k for k in range(12)]  # dotted quarter = 0.6 s -> quarter 0.4 s
    info = M.write_performance_midi(tmp_path / "d.mid", [("gtr", 27, NOTES)],
                                    grid(beats, 2, unit="dotted_quarter", den=8))
    assert info["lead_in"] is None and info["midi_bar_offset"] == 0
    _mid, tempos, sigs, notes = read(tmp_path / "d.mid")
    assert sigs == [(0, 6, 8)]
    assert tempos == [(0, 400000)]
    assert tick_to_s(tempos, 720) == pytest.approx(0.6)
    for (on, _off, _p), n in zip(notes, sorted(NOTES, key=lambda n: n["onset_s"])):
        assert tick_to_s(tempos, on) == pytest.approx(n["onset_s"], abs=0.001)


def test_tempo_ramp_follows_beats(tmp_path):
    t, beats = 0.0, []
    for k in range(20):
        beats.append(t)
        t += 60.0 / (100 + 2 * k)
    M.write_performance_midi(tmp_path / "e.mid", [("gtr", 27, NOTES)], grid(beats, 4))
    _mid, tempos, _s, notes = read(tmp_path / "e.mid")
    assert len(tempos) == 19  # one per beat interval; the last beat repeats the previous tempo (dropped)
    for k, bt in enumerate(beats):
        assert tick_to_s(tempos, 480 * k) == pytest.approx(bt, abs=0.001)
    for (on, _off, _p), n in zip(notes, sorted(NOTES, key=lambda n: n["onset_s"])):
        assert tick_to_s(tempos, on) == pytest.approx(n["onset_s"], abs=0.001)


def test_no_grid_constant_120(tmp_path):
    info = M.write_performance_midi(tmp_path / "f.mid", [("gtr", 27, NOTES)], None)
    assert info["midi_bar_offset"] == 0
    _mid, tempos, sigs, notes = read(tmp_path / "f.mid")
    assert tempos == [(0, 500000)] and sigs == [(0, 4, 4)]
    assert notes[0][0] == 480  # 0.5 s at 120 BPM


def test_deterministic_bytes_tracks_and_velocity(tmp_path):
    g = grid([0.37 + 0.5 * k for k in range(16)], 4)
    tracks = M.class_tracks({"notes": NOTES}, ("acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"))
    assert [t[0] for t in tracks] == ["clean_electric_guitar", "distorted_electric_guitar"]
    assert [t[1] for t in tracks] == [27, 30]
    M.write_performance_midi(tmp_path / "x.mid", tracks, g)
    M.write_performance_midi(tmp_path / "y.mid", tracks, g)
    assert (tmp_path / "x.mid").read_bytes() == (tmp_path / "y.mid").read_bytes()
    mid = mido.MidiFile(str(tmp_path / "x.mid"))
    assert [tr.name for tr in mid.tracks] == ["tempo", "clean_electric_guitar", "distorted_electric_guitar"]
    vels = [m.velocity for tr in mid.tracks[1:] for m in tr if m.type == "note_on"]
    assert set(vels) == {90}
    chans = {m.channel for tr in mid.tracks[1:] for m in tr if hasattr(m, "channel")}
    assert 9 not in chans


def test_same_pitch_overlap_trimmed_and_amplitude_velocity(tmp_path):
    notes = [{"onset_s": 0.0, "offset_s": 1.0, "pitch": 60, "amplitude": 0.5},
             {"onset_s": 0.5, "offset_s": 1.5, "pitch": 60, "amplitude": 1.0}]
    M.write_performance_midi(tmp_path / "o.mid", [("bp", 27, notes)], None)
    _mid, _t, _s, got = read(tmp_path / "o.mid")
    assert got == [(0, 480, 60), (480, 1440, 60)]
    vels = [m.velocity for m in mido.MidiFile(str(tmp_path / "o.mid")).tracks[1] if m.type == "note_on"]
    assert vels == [64, 127]
