"""A job's hints.toml writer (mid-term plan 1-6): one key at a time, the user's comments and other keys kept."""

from __future__ import annotations

import tomllib

import pytest

from bandscribe.jobs import hints as HI


def test_set_hint_keeps_comments_and_other_keys(tmp_path):
    p = HI.path(tmp_path)
    p.write_text('# my notes\n[tab]\nguitar_tuning = "Eb standard"  # band score\n\n[grid]\n# counted by ear\n'
                 'tempo_bpm = 120.0\n', encoding="utf-8")
    HI.set_hint(tmp_path, "grid", "tempo_bpm", 88.0)          # replaced in place
    HI.set_hint(tmp_path, "tab", "guitar_parts", 2)           # added at the end of its section, before the blank line
    HI.set_hint(tmp_path, "hints", "title", 'lulu. "live"', comment="song title")  # a new section
    text = p.read_text(encoding="utf-8")
    assert "# my notes" in text and "# band score" in text and "# counted by ear" in text and "# song title" in text
    assert text.index("guitar_parts = 2") < text.index("[grid]")
    d = tomllib.loads(text)
    assert d == {"tab": {"guitar_tuning": "Eb standard", "guitar_parts": 2}, "grid": {"tempo_bpm": 88.0},
                 "hints": {"title": 'lulu. "live"'}}
    assert HI.read_hints(tmp_path) == d


def test_set_hint_on_a_new_file_and_a_bom_file(tmp_path):
    HI.set_hint(tmp_path, "hints", "tempo_confirmed", True)
    assert HI.read_hints(tmp_path) == {"hints": {"tempo_confirmed": True}}
    (tmp_path / "b").mkdir()
    HI.path(tmp_path / "b").write_bytes("﻿[grid]\ntempo_bpm = 68\n".encode("utf-8"))  # Notepad's UTF-8 with BOM
    assert HI.read_hints(tmp_path / "b") == {"grid": {"tempo_bpm": 68}}
    HI.set_hint(tmp_path / "b", "grid", "tempo_bpm", 136.0)
    assert HI.path(tmp_path / "b").read_bytes().startswith(b"[grid]")
    assert HI.read_hints(tmp_path / "b") == {"grid": {"tempo_bpm": 136.0}}


def test_a_broken_file_is_not_overwritten(tmp_path):
    HI.path(tmp_path).write_text("[grid\ntempo_bpm = \n", encoding="utf-8")
    assert HI.read_hints(tmp_path) == {}
    with pytest.raises(tomllib.TOMLDecodeError):
        HI.set_hint(tmp_path, "grid", "tempo_bpm", 88.0)
    assert HI.path(tmp_path).read_text(encoding="utf-8") == "[grid\ntempo_bpm = \n"


def test_a_new_key_does_not_take_the_next_sections_comment(tmp_path):
    p = HI.path(tmp_path)
    p.write_text('[hints]\ntitle = "lulu."\n\n# 사용자 지정: lulu. 는 88 BPM\n[grid]\ntempo_bpm = 88\n', encoding="utf-8")
    HI.set_hint(tmp_path, "hints", "tempo_confirmed", True)
    assert p.read_text(encoding="utf-8") == ('[hints]\ntitle = "lulu."\ntempo_confirmed = true\n\n'
                                             '# 사용자 지정: lulu. 는 88 BPM\n[grid]\ntempo_bpm = 88\n')
