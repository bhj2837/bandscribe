"""Distractor suite (bandscribe.eval.distractors): window choice, conditions, batching, mastering, the whole run with
fake GPU backends, the report, and the CLI / registry wiring."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bandscribe.eval import distractors as D
from bandscribe.eval import remix
from bandscribe.eval import stem_taxonomy as T

SR = 44100


# ------------------------------------------------------------------------------------------- windows


def _power(n: int, active: dict[str, tuple[int, int]], level: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    out = {}
    for c, (a, z) in active.items():
        v = np.full(n, 1e-12)
        v[a:z] = (level or {}).get(c, 1e-2)
        out[c] = v
    return out


def test_select_windows_prefers_most_present_categories_then_adds_a_new_one():
    n = 2400  # 240 s at 10 Hz
    p = _power(n, {"guitar_electric": (0, n), "drums": (0, n), "synth_pad": (0, 1000), "strings": (500, 1000),
                   "organ": (1700, 2400)})
    ws = D.select_windows(p, window_s=75, step_s=5, max_windows=2)
    assert [w.wid for w in ws] == ["w1", "w2"]
    w1, w2 = ws
    assert set(w1.present) == {"synth_pad", "strings"} and 25 <= w1.start_s <= 50
    assert "organ" in w2.present and w2.start_s >= w1.end_s
    assert w1.guitar_share >= D.GUITAR_SHARE
    # deterministic
    assert [(w.start_s, w.present) for w in D.select_windows(p, max_windows=2)] == [(w.start_s, w.present) for w in ws]


def test_select_windows_needs_guitar_and_a_distractor():
    n = 1200
    assert D.select_windows(_power(n, {"guitar_electric": (0, n), "drums": (0, n)})) == []
    assert D.select_windows(_power(n, {"synth_pad": (0, n), "drums": (0, n)})) == []
    # a guitar 30 dB under the sum is not "active" (threshold -20 dB)
    quiet = _power(n, {"guitar_electric": (0, n), "synth_pad": (0, n)}, {"guitar_electric": 1e-5, "synth_pad": 1e-2})
    assert D.select_windows(quiet) == []


def test_conditions_for():
    conds = D.conditions_for(["guitar_electric", "drums", "bass", "vocals", "synth_pad", "sfx", "keys_piano"],
                             ["synth_pad", "keys_piano"])
    names = [c.name for c in conds]
    assert names == ["base", "+synth_pad", "+keys_piano", "full"]
    by = {c.name: c.categories for c in conds}
    assert set(by["base"]) == {"guitar_electric", "drums", "bass", "vocals"}
    assert set(by["+synth_pad"]) == set(by["base"]) | {"synth_pad"}
    assert "sfx" in by["full"] and "sfx" not in by["+synth_pad"]
    only_base = D.conditions_for(["guitar_electric", "drums"], [], guitar_alone=True)
    assert [c.name for c in only_base] == ["base", "guitar_alone"]


# ------------------------------------------------------------------------------------------ batching


def test_batch_layout_aligns_segments_to_the_chunk_grid_with_silence_around():
    chunk, step = 1000, 500
    lengths = [1700, 300, 2600, 900]
    batches = D.batch_layout(lengths, chunk=chunk, step=step, max_samples=6000)
    flat = [i for b in batches for i, _ in b]
    assert flat == [0, 1, 2, 3]
    for b in batches:
        prev_end = 0
        for i, off in b:
            assert off % step == 0
            assert off - prev_end >= chunk          # silence before
            prev_end = off + lengths[i]
        assert D.batch_total(lengths, b, chunk=chunk, step=step) - prev_end >= chunk  # and after the last
        assert D.batch_total(lengths, b, chunk=chunk, step=step) <= 6000 or len(b) == 1
    with pytest.raises(ValueError):
        D.batch_layout([10], chunk=1000, step=300, max_samples=10_000)


def test_master_gain_reproduces_master_on_the_full_mix():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.05, (SR * 3, 2))
    x[SR:SR + 2000] *= 30  # a peak for the limiter
    g = remix.master_gain(x, SR, lufs=-8.0)
    y = np.clip(x * g[:, None], -10 ** (-1 / 20), 10 ** (-1 / 20)).astype(np.float32)
    assert np.max(np.abs(y - remix.master(x, SR, lufs=-8.0))) < 1e-5
    assert np.all(remix.master_gain(np.zeros((SR, 2)), SR) == 1.0)


# ------------------------------------------------------------------------------- full run with fakes


def _write(path: Path, x: np.ndarray) -> None:
    from bandscribe.audio import write_f32

    path.parent.mkdir(parents=True, exist_ok=True)
    write_f32(path, np.asarray(x, dtype=np.float32).reshape(len(x), -1), SR)


def _tone(f: float, t: np.ndarray, amp: float) -> np.ndarray:
    return amp * np.sin(2 * np.pi * f * t)


def make_song(root: Path, dur: float = 40.0) -> D.SongInfo:
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(1)
    g = np.zeros(n)
    for k in range(int(dur * 2)):  # a guitar note every 0.5 s
        a = int(k * 0.5 * SR)
        z = min(n, a + int(0.4 * SR))
        g[a:z] += _tone(110 * 2 ** ((k % 5) / 12), t[: z - a], 0.2) * np.exp(-np.arange(z - a) / SR * 4)
    stems = {
        "01_Kick.wav": rng.normal(0, 0.02, n) * (np.sin(2 * np.pi * 2 * t) > 0.9),
        "02_Bass.wav": _tone(55, t, 0.1),
        "03_ElecGtr1.wav": g,
        "04_LeadVox.wav": _tone(330, t, 0.05),
        "05_SynthPad.wav": _tone(440, t, 0.08) * (t < 25),
        "06_Rhodes.wav": _tone(523.25, t, 0.06) * (t > 15),
    }
    files = []
    for name, x in stems.items():
        p = root / "song" / name
        _write(p, x)
        files.append(D.StemFile(p, name, T.categorize(name)))
    return D.SongInfo("cambridge_mt", "testsong_one", "testsong", "Test - One", files)


class FakeGPU:
    def __init__(self) -> None:
        self.sep_calls: list[Path] = []
        self.ms_calls: list[list[dict]] = []

    def separate(self, mix_wav: Path, out_dir: Path, cfg, *, force_rung: str):
        import soundfile as sf

        from bandscribe.audio import write_f32

        self.sep_calls.append(Path(mix_wav))
        x, sr = sf.read(str(mix_wav), dtype="float32", always_2d=True)
        stems = {}
        for name, k in (("bass", 0.1), ("drums", 0.1), ("other", 0.1), ("vocals", 0.1), ("guitar", 0.8), ("piano", 0.1)):
            p = Path(out_dir) / "stems" / f"{name}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            write_f32(p, x * k, sr)
            stems[name] = p
        return SimpleNamespace(stems=stems, sep_json={"audio_s": len(x) / sr, "process_s": 1.0},
                               gpu_run={"waited_s": 0.0}, rung={"label": force_rung})

    def ms(self, views, params, cfg, *, work_dir, force_rung, cache=None):
        self.ms_calls.append([dict(v) for v in views])
        out = {}
        for v in views:
            vid = v["id"]
            base = [{"onset_s": round(0.5 * k + 0.003, 3), "offset_s": round(0.5 * k + 0.4, 3), "pitch": 45 + k % 5,
                     "velocity": 80, "instrument": "clean_electric_guitar"} for k in range(30)]
            if vid.startswith("r"):
                notes = base
            elif vid.startswith("p"):
                seg = (v.get("segments") or [[0.0, 10.0]])[0]
                notes = [{"onset_s": seg[0] + 0.5 * k, "offset_s": seg[0] + 0.5 * k + 0.3, "pitch": 72, "velocity": 70,
                          "instrument": "electric_piano"} for k in range(12)]
            else:  # guitar views: miss one note, add a pad-pitch false note and an octave
                notes = base[1:] + [{"onset_s": 3.0, "offset_s": 3.5, "pitch": 69, "velocity": 70,
                                     "instrument": "clean_electric_guitar"},
                                    {"onset_s": 5.0, "offset_s": 5.3, "pitch": 57, "velocity": 70,
                                     "instrument": "distorted_electric_guitar"}]
                if vid.startswith("m"):
                    notes = notes + [{"onset_s": 6.0, "offset_s": 6.3, "pitch": 72, "velocity": 70,
                                      "instrument": "electric_piano"}]
            out[vid] = {"notes": notes}
        return out, {"worker_output": {"vram": {"rung_label": force_rung}}, "rung_label": force_rung, "waited_s": 0.0}


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    song = make_song(tmp_path / "data")
    gpu = FakeGPU()
    monkeypatch.setattr(D, "load_songs", lambda dataset="cambridge_mt", only=None: [song])
    monkeypatch.setattr(D, "_separate", gpu.separate)
    monkeypatch.setattr(D, "_ms", gpu.ms)
    monkeypatch.setattr(D, "_ms_cached", lambda views, params, rung: set())
    return SimpleNamespace(song=song, gpu=gpu, cache=tmp_path / "cache", tmp=tmp_path)


def test_run_end_to_end_with_fakes(fake_env):
    from bandscribe.eval import runs

    run_dir = fake_env.tmp / "run"
    run_dir.mkdir()
    args = {"cache_root": fake_env.cache, "window_s": 15.0, "gpu_budget_min": 1000}
    s = D.run(run_dir, {}, args)
    assert s["n_windows"] >= 1
    w = s["windows"][0]
    assert set(w["present"]) <= {"synth_pad", "keys_epiano"} and w["present"]
    conds = s["conditions"]["production"]
    assert "base" in conds and "full" in conds and D.NULL_COND in conds
    # decoder null: the fake transcribes the noisy view like the clean one -> no change
    assert s["null"]["windows"] and s["null"]["median_abs"]["d_fp_per_min"] == 0.0
    assert any(i.startswith("n") for call in fake_env.gpu.ms_calls for i in (v["id"] for v in call))
    # one separation worker for every condition (they fit one batch), cached on the second run
    assert len(fake_env.gpu.sep_calls) == 1
    # the reference and every condition's guitar view in call 1, presence windows too
    ids = [v["id"] for v in fake_env.gpu.ms_calls[0]]
    assert any(i.startswith("r") for i in ids) and any(i.startswith("g") for i in ids)
    # fake scores: base has 30 ref notes, 29 hits, 2 FPs, 1 FN per window
    base = conds["base"]
    assert base["tp"] == 29 * s["n_windows"] and base["fn"] == 1 * s["n_windows"]
    for f in ("metrics.csv", "summary.json", "report.md", "report.html", "windows.json"):
        assert (run_dir / f).is_file(), f
    rows = runs.read_metrics_csv(run_dir / "metrics.csv")
    assert {r["metric"] for r in rows} >= {"note_f1_onset50", "fp_per_min", "fn_per_min", "delta_fp_per_min"}
    assert any(r["metric"].startswith("fp_cause:") for r in rows)
    assert any(r["metric"].startswith("leak_db:") for r in rows)
    fpc = {r["metric"] for r in rows if r["metric"].startswith("fp_cause:")}
    assert "fp_cause:guitar_octave" in fpc  # pitch 57 over the reference's 45
    md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "먼저 고칠 것" in md and "범주별 효과" in md
    # separated views are the segments of the batch: guitar_mono = 0.8 x mono(condition audio)
    import soundfile as sf

    sep_dirs = [d for d in (fake_env.cache / "sep").iterdir() if (d / "done.json").is_file()]
    assert sep_dirs
    g, _ = sf.read(str(sep_dirs[0] / "guitar_mono.wav"), dtype="float32")
    assert len(g) == int(round(15.0 * SR))
    # second run: everything comes from the caches (no separation call)
    run2 = fake_env.tmp / "run2"
    run2.mkdir()
    D.run(run2, {}, args)
    assert len(fake_env.gpu.sep_calls) == 1
    a = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((run2 / "summary.json").read_text(encoding="utf-8"))
    assert a["deltas"] == b["deltas"]


def test_dry_run_and_budget(fake_env):
    run_dir = fake_env.tmp / "dry"
    run_dir.mkdir()
    s = D.run(run_dir, {}, {"cache_root": fake_env.cache, "window_s": 15.0, "dry_run": True})
    assert s["dry_run"] and s["windows"] and not fake_env.gpu.sep_calls
    with pytest.raises(D.DistractorError, match="예산"):
        D.run(run_dir, {}, {"cache_root": fake_env.cache, "window_s": 15.0, "gpu_budget_min": 0.01})


def test_null_view_is_seeded_and_inaudible():
    x = np.zeros(SR, dtype=np.float32)
    a, b = D.null_view(x, "s:w1"), D.null_view(x, "s:w1")
    assert np.array_equal(a, b) and not np.array_equal(a, D.null_view(x, "s:w2"))
    rms_db = 20 * np.log10(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    assert rms_db == pytest.approx(D.NULL_RMS_DBFS, abs=0.5)


def test_leak_summary_means_over_windows():
    leaks = {("s1:w1", "base"): {"drums": {"leak": 0.01, "share": 0.1, "bins": 5}},
             ("s2:w1", "base"): {"drums": {"leak": 0.001, "share": 0.2, "bins": 5}},
             ("s1:w1", "+synth_pad"): {"synth_pad": {"leak": 0.1, "share": 0.3, "bins": 5}},
             ("s2:w1", "+synth_pad"): {"synth_pad": {"leak": float("nan"), "share": float("nan"), "bins": 0}},
             ("s1:w1", "full"): {"synth_pad": {"leak": 1.0, "share": 0.5, "bins": 5}}}
    out = D._leak_summary(leaks, [])
    assert out["drums"]["base_leak_db"] == pytest.approx(-25.0)
    assert out["synth_pad"]["own_leak_db"] == pytest.approx(-10.0)  # NaN windows are skipped
    assert out["synth_pad"]["full_leak_db"] == pytest.approx(0.0) and out["synth_pad"]["n"]["own_leak_db"] == 2


def test_fix_first_ranks_by_weighted_errors():
    summ = {"n_windows": 4, "deltas": {"production": {
        "+synth_pad": {"n_items": 4, "n_groups": 4, "fp_per_min": {"delta": 3.0, "ci": [1, 5]},
                       "fn_per_min": {"delta": 1.0, "ci": [0, 2]}},
        "+organ": {"n_items": 1, "n_groups": 1, "fp_per_min": {"delta": 8.0, "ci": [None, None]},
                   "fn_per_min": {"delta": 0.0, "ci": [None, None]}},
        "full": {"n_items": 4, "n_groups": 4, "fp_per_min": {"delta": 9.0, "ci": [1, 5]},
                 "fn_per_min": {"delta": 1.0, "ci": [0, 2]}}}},
        "conditions": {"production": {"full": {"x": 1}}},
        "fp_cause_matrix": {"full": {"guitar_octave": 2.5, "synth_pad": 3.0}}, "fn_cause_matrix": {"full": {}}}
    r = D.fix_first(summ)
    assert [x["source"] for x in r] == ["synth_pad", "guitar_octave", "organ"]
    assert r[0]["weighted"] == pytest.approx(4.0) and r[2]["weighted"] == pytest.approx(2.0)


def test_report_renders_minimal_summary(tmp_path):
    from bandscribe.eval import distractor_report as R

    p_md, p_html = R.write(tmp_path, {"n_windows": 0, "n_songs": 0, "rung": "x", "gpu": {}, "arms": ["production"]})
    assert "distractors" in p_md.read_text(encoding="utf-8")
    assert p_html.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_registry_entry_and_cli_routing():
    from bandscribe.eval import experiments, suites

    e = experiments.load_registry()["distractors"]
    assert e.spec.get("suite") == "distractors" and e.spec.get("rule") == "descriptive"
    assert "distractors" in suites.SUITES and "backing" in suites.SUITES
    if e.decided:
        pytest.skip("distractors 는 이미 결정됨")
    with pytest.raises(experiments.ExperimentStateError, match="--suite distractors"):
        experiments.run_experiment("distractors", {}, dry_run=True)
