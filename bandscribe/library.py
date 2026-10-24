"""The song library (mid-term plan 1-7): ``data/library/<title>/`` holds each scored song's Guitar Pro files under
its title, its chord chart and a README (tempo, tunings, parts, the song health), so the scores can be found and
opened without knowing job keys. Guitar-removed recordings (song health ``guitar_removed``) stay out.

Copies only what changed (size and modification time); never deletes anything in the library.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths

_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def library_dir() -> Path:
    return Path(paths.DATA) / "library"


def folder_name(title: str, key: str) -> str:
    """A Windows-safe folder / file name for ``title`` (forbidden characters to '_', no trailing dot or space)."""
    name = _BAD.sub("_", title).strip().rstrip(". ")
    return name or key


def _copy(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    if dst.is_file():
        a, b = src.stat(), dst.stat()
        if a.st_size == b.st_size and int(a.st_mtime) <= int(b.st_mtime):
            return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def readme(title: str, key: str, card: list[tuple[str, str]], health: dict[str, Any]) -> str:
    lines = [f"# {title}", "", f"bandscribe 작업 `{key}` 에서 만든 악보 ({time.strftime('%Y-%m-%d')}).", ""]
    lines += [f"- **{label}** {text.replace('(위 목록, 뷰어 위쪽에도 나옵니다)', '(아래 점검; 뷰어 위쪽에도 나온다)')}"
              for label, text in card if label != "파일"]
    items = [i for i in health.get("items") or [] if i.get("id") != "run"]
    if items:
        lines += ["", "## 점검", ""]
        for it in items:
            tag = "경고" if it.get("level") == "warn" else "알림"
            spans = ", ".join(f"{int(a // 60)}:{int(a % 60):02d}–{int(b // 60)}:{int(b % 60):02d}"
                              for a, b in (it.get("spans") or [])[:6])
            lines.append(f"- {tag}: {it.get('title', '')}" + (f" ({spans})" if spans else ""))
    lines += ["", "## 파일", "",
              f"- `{folder_name(title, key)}.gp` — Guitar Pro 7 (원곡 백킹 트랙 포함). 제목이 그대로 들어간다.",
              f"- `{folder_name(title, key)}.gp5` — Guitar Pro 5 (TuxGuitar 등). 한국어 코드 페이지라 제목이 다를 수 있다.",
              f"- `{folder_name(title, key)} 코드.txt` — 코드표(있을 때).",
              "", "이 폴더는 `bandscribe library` 가 다시 만든다. 손으로 고친 내용은 덮어쓰일 수 있다.", ""]
    return "\n".join(lines)


def build(*, progress=None) -> dict[str, Any]:
    """Put every valid scored song into the library; returns {songs: [...], skipped: [...]}."""
    from bandscribe.commands.view import find_jobs
    from bandscribe.tab import health as HL
    from bandscribe.tab import summary as SM
    from bandscribe.tab import viewer

    root = Path(paths.JOBS)
    lib = library_dir()
    songs, skipped, used = [], [], {}
    for key in viewer.song_keys(root):
        job = root / key
        try:
            title = viewer.job_files(job)["title"]
            health = HL.load_health(job)
        except (viewer.ViewerError, OSError, ValueError) as e:
            skipped.append({"key": key, "why": str(e)})
            continue
        if any(i.get("id") == "guitar_removed" for i in health.get("items") or []):
            skipped.append({"key": key, "title": title, "why": "기타가 빠진 음원"})
            continue
        name = folder_name(title, key)
        if name.casefold() in used:
            name = f"{name} ({key[:8]})"
        used[name.casefold()] = key
        dest = lib / name
        tab = job / "export" / "tab"
        changed = sum([_copy(tab / "score.gp", dest / f"{name}.gp"), _copy(tab / "score.gp5", dest / f"{name}.gp5"),
                       _copy(job / "export" / "chords" / "chords.txt", dest / f"{name} 코드.txt")])
        arg = title if find_jobs(title, root) == [job] else key
        text = readme(title, key, SM.card(job, health, view_arg=arg), health)
        if not (dest / "README.md").is_file() or (dest / "README.md").read_text(encoding="utf-8") != text:
            atomic.write_text(dest / "README.md", text)
            changed += 1
        songs.append({"key": key, "title": title, "folder": str(dest), "changed": changed})
        if progress:
            progress(f"{title}: {'갱신' if changed else '그대로'}")
    index = ["# bandscribe 곡 서고", "", f"{len(songs)}곡 ({time.strftime('%Y-%m-%d %H:%M')}). 곡마다 폴더에 .gp · .gp5 · 코드표 · "
             "README 가 있다. 뷰어로 보려면 `bandscribe view`.", ""]
    index += [f"- [{s['title']}](<{Path(s['folder']).name}/README.md>)" for s in sorted(songs, key=lambda s: s["title"].casefold())]
    if skipped:
        index += ["", "넣지 않은 작업:", ""] + [f"- {s.get('title') or s['key']} — {s['why']}" for s in skipped]
    atomic.write_text(lib / "README.md", "\n".join(index) + "\n")
    return {"dir": str(lib), "songs": songs, "skipped": skipped}
