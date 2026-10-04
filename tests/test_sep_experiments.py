"""bandscribe.sep.checks.stereo_preservation and the ``stereo-preservation`` experiment with a fake separator (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from bandscribe.sep import checks, experiments

SR = 44100


def _stereo(n: int = SR * 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    left = np.sin(2 * np.pi * 330 * t) * 0.3 + rng.standard_normal(n) * 0.05
    right = 0.4 * left + rng.standard_normal(n) * 0.05  # partly correlated, left-heavy
    return np.stack([left, right])


def test_identical_stems_preserve_everything() -> None:
    x = _stereo()
    sp = checks.stereo_preservation(x, x.T.copy(), SR)  # (2, n) vs (n, 2) accepted
    assert sp["delta_coh"] == pytest.approx(0.0, abs=1e-9) and sp["delta_ild_db"] == pytest.approx(0.0, abs=1e-9)
    assert abs(sp["energy_ratio_db"]) < 1e-6
    assert all(v is None or abs(v) < 1e-9 for v in sp["window_feature_deltas"].values())


def test_collapsed_image_is_detected() -> None:
    x = _stereo()
    mono = np.stack([x.mean(axis=0)] * 2)  # a separator that flattened the stereo image
    sp = checks.stereo_preservation(x, mono, SR)
    assert sp["delta_ild_db"] > 3.0 and sp["delta_coh"] > 0.02
    swapped = x[::-1]
    assert checks.stereo_preservation(x, swapped, SR)["delta_ild_db"] > 3.0


def test_length_mismatch_and_silence() -> None:
    x = _stereo()
    sp = checks.stereo_preservation(x, x[:, :-1000], SR)
    assert sp["samples"] == x.shape[1] - 1000
    with pytest.raises(ValueError):
        checks.stereo_preservation(np.zeros((2, 5000)), x[:, :5000], SR)


class FakeSeparate:
    """Returns the scene's true guitar stem as SW's guitar stem (perfect separation) with a recorded rung."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, mix_wav: Path, out_dir: Path, cfg: Any, *, force_rung: str | None = None) -> Any:
        from bandscribe.eval import remix

        self.calls.append({"mix": mix_wav, "out": out_dir, "force_rung": force_rung})
        item = json.loads((Path(mix_wav).parent / "truth.json").read_text(encoding="utf-8"))
        scene = remix.make_scene(item["scenario"], item["seed"], dur_s=item["dur_s"], full_band=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        g = out_dir / "guitar.wav"
        sf.write(str(g), scene.guitar_stereo().T.astype(np.float32), SR, subtype="FLOAT")
        return SimpleNamespace(stems={"guitar": g}, rung={"index": 0, "label": force_rung, "device": "cuda"})


def test_synthetic_rows_with_fake_separator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("bandscribe.audio")
    pytest.importorskip("bandscribe.schema.gt")  # remix builds GT lines with P0's schema
    remix = pytest.importorskip("bandscribe.eval.remix")
    runs = pytest.importorskip("bandscribe.eval.runs")
    fake = FakeSeparate()
    real_make = remix.make_scene

    def make_and_note(scenario: str, seed: int = 0, **kw: Any) -> Any:
        scene = real_make(scenario, seed, **kw)
        d = tmp_path / "scenes" / runs.slug(scene.item_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "truth.json").write_text(json.dumps({"scenario": scenario, "seed": seed, "dur_s": kw["dur_s"]}),
                                      encoding="utf-8")
        return scene

    monkeypatch.setattr(remix, "make_scene", make_and_note)
    rows = experiments.synthetic_rows(tmp_path, {}, scenarios=["A", "P70"], seeds=[0], dur_s=3.0,
                                      force_rung="chunk=588800", separate=fake)
    assert [c["force_rung"] for c in fake.calls] == ["chunk=588800"] * 2
    by = {(r["scenario"], r["metric"]): r for r in rows}
    assert by[("A", "stereo_delta_coh")]["value"] == pytest.approx(0.0, abs=1e-6)
    assert by[("P70", "stereo_delta_ild_db")]["value"] == pytest.approx(0.0, abs=1e-6)
    assert all(r["rung"] == "chunk=588800" and r["dataset"] == "synthetic" for r in rows)
    for r in rows:
        runs.normalize_row(r)  # only metrics.csv columns
    assert (tmp_path / "scenes" / "synth-a_s0_band" / "stereo_preservation.json").is_file()


def test_archive_rows_for_a_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bandscribe import paths

    monkeypatch.setattr(paths, "JOBS", tmp_path / "jobs")
    job = tmp_path / "jobs" / "f-1234"
    (job / "input").mkdir(parents=True)
    sep_dir = job / "stages" / "sep" / "abcdef123456"
    (sep_dir / "stems").mkdir(parents=True)
    x = _stereo(SR).T.astype(np.float32)
    sf.write(str(job / "input" / "mix_44k_f32.wav"), x, SR, subtype="FLOAT")
    sf.write(str(sep_dir / "stems" / "guitar.wav"), x * 0.5, SR, subtype="FLOAT")
    sf.write(str(sep_dir / "stems" / "piano.wav"), x * 1e-4, SR, subtype="FLOAT")  # -80 dB: sparse
    (sep_dir / "gpu_run.json").write_text(json.dumps({"format": "bandscribe.gpu_run/1", "rung": {
        "index": 1, "label": "chunk=352256", "device": "cuda"}}), encoding="utf-8")
    (sep_dir / "manifest.json").write_text(json.dumps({"status": "complete", "created_utc": "2026-09-30"}),
                                           encoding="utf-8")
    assert experiments._latest_sep_stages() == [("f-1234", sep_dir)]
    rows = experiments.archive_rows([("f-1234", sep_dir)], tmp_path)
    by = {r["line"]: r for r in rows}
    assert by["guitar"]["value"] < -100 and by["guitar"]["note"].startswith("ok")
    assert by["piano"]["value"] is None and by["piano"]["note"].startswith("skipped_sparse")
    assert all(r["rung"] == "chunk=352256" for r in rows)


def test_registry_and_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bandscribe import paths

    monkeypatch.setattr(paths, "JOBS", tmp_path / "none")
    assert set(experiments.EXPERIMENTS) == {"stereo-preservation"}
    rows = experiments.EXPERIMENTS["stereo-preservation"](tmp_path / "run", {}, {"dry_run": True})
    assert rows == []
    plan = json.loads((tmp_path / "run" / "plan.json").read_text(encoding="utf-8"))
    assert plan["force_rung"] == "chunk=588800" and plan["scenarios"] == list(experiments.DEFAULT_SCENARIOS)
