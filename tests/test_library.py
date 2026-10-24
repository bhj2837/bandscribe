"""``bandscribe library`` (mid-term plan 1-7): scored songs under their titles, guitar-removed audio left out."""

from __future__ import annotations

from bandscribe import library as L
from bandscribe import paths
from bandscribe.tab import health as H
from test_tab_edits import make_job


def test_folder_names_are_windows_safe():
    assert L.folder_name("lulu.", "f-1") == "lulu" and L.folder_name('a/b:c?"', "f-1") == "a_b_c__"
    assert L.folder_name(" . ", "f-1") == "f-1" and L.folder_name("狭心症", "f-1") == "狭心症"


def test_library_copies_the_files_once_and_leaves_out_guitar_removed_audio(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    a = make_job(jobs).rename(jobs / "f-aaaaaaaaaaaaaaaa")
    b = make_job(jobs).rename(jobs / "f-bbbbbbbbbbbbbbbb")
    for j in (a, b):
        (j / "export" / "tab" / "score.gp").write_bytes(b"PK-gp7")
        (j / "export" / "tab" / "score.gp5").write_bytes(b"gp5")
    monkeypatch.setattr(paths, "JOBS", jobs)
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    real = H.load_health

    def health(job_dir):
        doc = real(job_dir)
        if job_dir.name == b.name:
            doc = {**doc, "items": doc["items"] + [{"id": "guitar_removed", "level": "warn", "title": "x"}]}
        return doc

    monkeypatch.setattr(H, "load_health", health)
    res = L.build()
    assert [s["key"] for s in res["songs"]] == [a.name] and res["skipped"][0]["why"] == "기타가 빠진 음원"
    folder = tmp_path / "data" / "library" / L.folder_name(res["songs"][0]["title"], a.name)
    name = folder.name
    assert (folder / f"{name}.gp").read_bytes() == b"PK-gp7" and (folder / f"{name}.gp5").is_file()
    text = (folder / "README.md").read_text(encoding="utf-8")
    assert text.startswith(f"# {res['songs'][0]['title']}") and "**템포**" in text and "**튜닝**" in text
    index = (tmp_path / "data" / "library" / "README.md").read_text(encoding="utf-8")
    assert "1곡" in index and "기타가 빠진 음원" in index
    assert L.build()["songs"][0]["changed"] == 0                         # nothing changed, nothing copied
