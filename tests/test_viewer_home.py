"""One viewer server for every song (mid-term plan 1-7): the home page, /s/<job>/ routes, one edit session per job."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from bandscribe.tab import viewer
from test_tab_edits import make_job


def _two_jobs(tmp_path):
    a = make_job(tmp_path)
    a = a.rename(tmp_path / "f-aaaaaaaaaaaaaaaa")
    b = make_job(tmp_path).rename(tmp_path / "f-bbbbbbbbbbbbbbbb")
    for j in (a, b):
        (j / "input" / "source.wav").write_bytes(b"RIFF")
    (tmp_path / "f-noscore").mkdir()
    at = tmp_path.parent / (tmp_path.name + "_at")
    at.mkdir()
    (at / "alphaTab.min.js").write_text("//", encoding="utf-8")
    return a, b, at


def _get(url, **headers):
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
        return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):
        return None


def test_home_lists_the_songs_and_each_song_lives_under_its_prefix(tmp_path):
    a, b, at = _two_jobs(tmp_path)
    httpd, t = viewer.serve(None, alphatab=at, jobs_root=tmp_path)
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, home = _get(base + "/")
        assert code == 200 and "곡 목록" in home and "/s/f-aaaaaaaaaaaaaaaa/" in home and "/s/f-bbbbbbbbbbbbbbbb/" in home
        assert "f-noscore" not in home and _get(base + "/home")[1] == home
        songs = json.loads(_get(base + "/api/songs")[1])["songs"]
        assert [s["key"] for s in songs] == ["f-aaaaaaaaaaaaaaaa", "f-bbbbbbbbbbbbbbbb"]
        assert songs[0]["tempo"] != "" and "edits" in songs[0]
        code, page = _get(base + "/s/f-aaaaaaaaaaaaaaaa/")
        assert code == 200 and 'const TOKEN = "' in page and 'href="/home"' in page
        snap = json.loads(_get(base + "/s/f-aaaaaaaaaaaaaaaa/api/score")[1])
        assert snap["readonly"] == "" and snap["map"]["notes"]
        assert _get(base + "/s/f-aaaaaaaaaaaaaaaa/api/health")[0] == 200
        assert _get(base + "/alphatab/alphaTab.min.js")[0] == 200
        opener = urllib.request.build_opener(_NoRedirect)
        with pytest.raises(urllib.error.HTTPError) as e:
            opener.open(base + "/s/f-aaaaaaaaaaaaaaaa")
        assert e.value.code == 301 and e.value.headers["Location"] == "/s/f-aaaaaaaaaaaaaaaa/"
        for bad in ("/api/score", "/s/f-zzzz/api/score", "/s/f-noscore/", "/s/..%2F..%2Fx/api/score", "/s/a/../b"):
            assert _get(base + bad)[0] == 404, bad
        assert _get(base + "/home", Host="evil.example")[0] == 403
    finally:
        httpd.shutdown()
        t.join(timeout=5)
        for s in httpd.sessions():
            s.close()


def test_an_edit_lands_in_its_own_job_and_the_root_shares_the_session(tmp_path):
    a, b, at = _two_jobs(tmp_path)
    httpd, t = viewer.serve(a, alphatab=at)                   # started for a: a at the root, every song under /s/
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        page = _get(base + "/")[1]
        token = page.split('const TOKEN = "', 1)[1].split('"', 1)[0]
        assert _get(base + "/s/f-aaaaaaaaaaaaaaaa/")[0] == 200 and _get(base + "/s/f-bbbbbbbbbbbbbbbb/")[0] == 200
        assert len(httpd.sessions()) == 2                        # a once (root and /s/ share it), b once
        snap = json.loads(_get(base + "/s/f-bbbbbbbbbbbbbbbb/api/score")[1])
        uid = next(u for u, n in snap["map"]["notes"].items() if n["track"] == "bass")
        body = json.dumps({"cmd": "delete", "uid": uid, "version": snap["version"]}).encode()
        req = urllib.request.Request(base + "/s/f-bbbbbbbbbbbbbbbb/api/cmd", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "X-Bandscribe-Token": token})
        res = json.loads(urllib.request.urlopen(req).read())
        assert res["applied"] and res["edits"] == 1
        assert (b / "edits.jsonl").is_file() and not (a / "edits.jsonl").is_file()
        req = urllib.request.Request(base + "/s/f-noscore/api/cmd", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "X-Bandscribe-Token": token})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 404
    finally:
        httpd.shutdown()
        t.join(timeout=5)
        for s in httpd.sessions():
            s.close()


def test_home_page_puts_guitar_removed_audio_apart_and_marks_an_unconfirmed_tempo():
    rows = [{"key": "f-1", "title": "lulu.", "url": "/s/f-1/", "tempo": "176", "tempo_state": "확인 안 됨",
             "tuning": "기타 E standard", "warnings": [], "guitar_removed": False, "edits": 3, "duration": "4:31",
             "updated": "10-24 13:05"},
            {"key": "f-2", "title": "<b>x</b>", "url": "/s/f-2/", "tempo": "150", "tempo_state": "정함", "tuning": "",
             "warnings": ["이 음원에는 기타 소리가 거의 없습니다"], "guitar_removed": True, "edits": 0}]
    page = viewer.home_page(rows)
    assert "1곡 (기타가 빠진 음원 1개 따로)." in page and "기타가 빠진 음원 (참고용)" in page
    assert '<span class="warn">확인 안 됨</span>' in page and "&lt;b&gt;x&lt;/b&gt;" in page and "⚠ 1" in page
    assert page.index("lulu.") < page.index("기타가 빠진 음원 (참고용)") < page.index("&lt;b&gt;x")
    assert "아직 악보가 있는 곡이 없습니다" in viewer.home_page([])


def test_view_finds_a_song_by_key_prefix_or_title(tmp_path):
    from bandscribe import atomic
    from bandscribe.commands.view import find_jobs

    a, b, _ = _two_jobs(tmp_path)
    for j, title in ((a, "lulu."), (b, "狭心症")):
        doc = atomic.read_json(j / "export" / "score.json")
        atomic.write_json(j / "export" / "score.json", {**doc, "song": {**(doc.get("song") or {}), "title": title}})
    assert find_jobs("f-aaaa", tmp_path) == [a]
    assert find_jobs("f-", tmp_path) == [a, b, tmp_path / "f-noscore"]   # a key prefix lists every job it fits
    assert find_jobs("狭心症", tmp_path) == [b] and find_jobs("LULU", tmp_path) == [a] and find_jobs("lu lu.", tmp_path) == [a]
    assert find_jobs("nothing", tmp_path) == []
