"""Auxiliary backing suite (bandscribe.eval.backing): note matching, pair statistics, the run on fake jobs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandscribe.eval import backing as B


def n(on: float, p: int, cls: str = "clean_electric_guitar") -> dict:
    return {"onset_s": on, "offset_s": on + 0.2, "pitch": p, "instrument": cls}


def test_hits_same_pitch_within_50ms():
    a = [n(1.0, 60), n(2.0, 62), n(3.0, 64)]
    b = [n(1.04, 60), n(2.0, 63), n(3.2, 64)]
    assert B.hits(a, b) == [True, False, False]


def test_pair_stats_shares_and_classes():
    orig = [n(1.0, 60), n(2.0, 62, "distorted_electric_guitar"), n(3.0, 64), n(4.0, 65)]
    back = [n(1.01, 60)]
    back_any = back + [n(2.02, 62)]
    st = B.pair_stats(orig, back, back_any)
    assert st["orig_n"] == 4 and st["on_backing"] == 1 and st["share"] == 0.25
    assert st["on_backing_any"] == 2 and st["share_any"] == 0.5
    assert st["by_class"]["distorted_electric_guitar"] == {"n": 1, "on_backing_any": 1}


def test_load_pairs_errors(tmp_path):
    with pytest.raises(B.BackingError, match="반주 쌍"):
        B.load_pairs(tmp_path / "missing.toml")
    p = tmp_path / "p.toml"
    p.write_text('[[pair]]\nlabel = "곡"\noriginal = "a"\nbacking = "b"\n', encoding="utf-8")
    with pytest.raises(B.BackingError, match="ASCII"):
        B.load_pairs(p)
    p.write_text('[[pair]]\nlabel = "s1"\noriginal = "a.wav"\nbacking = "b.wav"\n', encoding="utf-8")
    assert B.load_pairs(p) == [{"label": "s1", "original": "a.wav", "backing": "b.wav"}]


def _job(jobs: Path, key: str, notes: list[dict], name: str = "aaaaaaaaaaaa") -> None:
    d = jobs / key / "stages" / "notes" / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    (d / "guitar_all.json").write_text(json.dumps({"notes": notes}), encoding="utf-8")


def test_run_on_fake_jobs(tmp_path, monkeypatch):
    from bandscribe import paths
    from bandscribe.ingest.filekey import content_key

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    orig, back = tmp_path / "o.bin", tmp_path / "b.bin"
    orig.write_bytes(b"original audio bytes")
    back.write_bytes(b"backing audio bytes")
    (jobs / "index.json").write_text(json.dumps({f"file:{content_key(orig)}": "f-orig", f"file:{content_key(back)}": "f-back"}),
                                     encoding="utf-8")
    _job(jobs, "f-orig", [n(1.0, 60), n(2.0, 62), n(3.0, 64), n(4.0, 65, "electric_piano")])
    _job(jobs, "f-back", [n(1.02, 60)])
    _job(jobs, "f-back", [n(3.0, 64)], name="bbbbbbbbbbbb")  # an older / other-mask result of the backing
    monkeypatch.setattr(paths, "JOBS", jobs)
    pairs = tmp_path / "pairs.toml"
    pairs.write_text(f'[[pair]]\nlabel = "s1"\noriginal = "{orig.as_posix()}"\nbacking = "{back.as_posix()}"\n'
                     f'[[pair]]\nlabel = "s2"\noriginal = "{(tmp_path / "none.wav").as_posix()}"\n'
                     f'backing = "{back.as_posix()}"\n', encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    s = B.run(run_dir, None, {"pairs": str(pairs)})
    p1, p2 = s["pairs"]
    assert p1["orig_n"] == 3  # non-guitar classes are not guitar notes
    assert p1["on_backing_any"] == 2 and p1["share_any"] == pytest.approx(2 / 3)
    assert p1["on_backing"] == 1  # either backing result alone matches one note
    assert "error" in p2
    assert (run_dir / "metrics.csv").is_file() and "반주 대조" in (run_dir / "report.md").read_text(encoding="utf-8")
