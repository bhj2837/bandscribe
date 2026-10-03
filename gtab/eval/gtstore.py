"""Tier A ground-truth song folders: ``data/eval/tierA/<song_id>/`` (spec §6.7).

Files: ``gt.yaml`` (human-edited spec), ``ref.<ext>`` (byte copy of the reference tab), ``refscore.json`` (neutral
import, unexpanded), ``gt_notes.json`` (expanded), ``tempo_map.json``, ``render.wav``, ``taps.json``,
``align_report.json``, ``align_check.wav``.

``gt.yaml`` is written once as a skeleton with Korean comments and then owned by the user (never overwritten).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gtab import atomic, paths

log = logging.getLogger(__name__)

SONG_ID_RE = re.compile(r"^[a-z0-9_-]+$")

# GM programs (0-based) -> family (FAMILIES keys of gtab.schema.instrumentation)
_PROGRAM_FAMILIES = [
    (range(0, 8), "keys"), (range(8, 16), "keys"), (range(16, 24), "keys"), (range(24, 32), "guitar"),
    (range(32, 40), "bass"), (range(40, 48), "strings"), (range(48, 52), "strings"), (range(52, 55), "vocals"),
    (range(55, 56), "brass"), (range(56, 64), "brass"), (range(64, 80), "winds"), (range(80, 104), "synth"),
]


class GtError(RuntimeError):
    """User-facing (Korean) ground-truth problem."""


def check_song_id(song_id: str) -> str:
    if not SONG_ID_RE.fullmatch(song_id or ""):
        raise GtError(f"곡 ID '{song_id}' 는 영문 소문자·숫자·_·- 만 쓸 수 있습니다 (예: my_song-01)")
    return song_id


def tier_a_root() -> Path:
    return paths.EVAL / "tierA"


def song_dir(song_id: str, root: Path | None = None) -> Path:
    return Path(root or tier_a_root()) / check_song_id(song_id)


def guess_kind(track: Any) -> str:
    """guitar | bass | other from percussion flag, GM program, string count/tuning and the name."""
    if track.percussion:
        return "other"
    name = (track.name or "").lower()
    if "bass" in name or "베이스" in name:
        return "bass"
    if any(k in name for k in ("guitar", "gtr", "기타")):
        return "guitar"
    prog = track.program
    if prog is not None and 24 <= prog <= 31:
        return "guitar"
    if prog is not None and 32 <= prog <= 39:
        return "bass"
    if track.n_strings in (4, 5) and track.tuning and min(track.tuning) < 40:
        return "bass"
    if track.n_strings >= 6 and prog is None:
        return "guitar"
    return "other"


def guess_families(tracks: list[Any]) -> list[str]:
    fams: set[str] = set()
    for t in tracks:
        if t.percussion:
            fams.add("drums")
            continue
        k = guess_kind(t)
        if k in ("guitar", "bass"):
            fams.add(k)
            continue
        if t.program is not None:
            for rng, fam in _PROGRAM_FAMILIES:
                if t.program in rng:
                    fams.add(fam)
    order = ["guitar", "bass", "keys", "synth", "strings", "brass", "winds", "vocals", "drums"]
    return [f for f in order if f in fams]


def _q(v: Any) -> str:
    import json

    return json.dumps(v, ensure_ascii=False)


def skeleton_yaml(song_id: str, rs: Any, *, song_key: str | None, ref_file: str, expanded_bars: int,
                  encoding: str | None, pickup: bool) -> str:
    """gt.yaml skeleton: tracks from the file, one line per guitar/bass track, Korean guidance comments."""
    guitars = [t for t in rs.tracks if guess_kind(t) == "guitar"]
    basses = [t for t in rs.tracks if guess_kind(t) == "bass"]
    line_of: dict[int, str] = {}
    for i, t in enumerate(guitars, 1):
        line_of[t.index] = f"L{i}"
    for i, t in enumerate(basses, 1):
        line_of[t.index] = f"B{i}"
    fams = guess_families(rs.tracks)
    out = [
        "# gtab 정답(Tier A) 명세. 처음 한 번만 자동으로 만들고, 이후에는 사람이 고친다(gtab 이 덮어쓰지 않음).",
        "# 채울 것: sections(정답 구간), families_present(실제 편성), lines 의 relation(더블·유니즌·트윈), 구간별 roles.",
        "format: gtab.gt/1",
        f"song_id: {_q(song_id)}",
        f"title: {_q(rs.title or None)}",
        f"artist: {_q(rs.artist or None)}",
        "audio:",
        f"  song_key: {_q(song_key)}   # gtab ingest 가 준 작업 키 (오디오 파일이 바뀌면 다시 gt import)",
        '  note: ""                    # 예: "CD 리핑", "구입 음원"',
        "version_flags:",
        "  version: album              # album | mv | live",
        "  trimmed: false",
        '  note: ""',
        "reference:",
        f"  file: {_q(ref_file)}",
        f"  format: {_q(rs.source.get('format'))}",
        f"  expanded_bars: {expanded_bars}   # 반복·다른 엔딩을 펼친 마디 수 (정렬 보고서가 대조한다)",
        "  bar_order: null             # D.S./코다가 있으면 연주 순서대로 적은 원본 마디 번호 목록 (예: [1, 2, 3, 2, 4])",
        f"  encoding: {_q(encoding)}   # GP3-5 문자 인코딩 (자동: cp949 -> cp1252). 트랙 이름이 깨지면 바꾼다",
        f"  pickup: {'true' if pickup else 'false'}               # 첫 마디가 못갖춘마디면 true (마디 0)",
        "# 실제로 연주되는 악기 계열 (b13 편성 정답): guitar, bass, keys, synth, strings, brass, winds, vocals, drums",
        f"families_present: {_q(fams)}",
        "tracks:                       # 참조 파일의 트랙. kind: guitar | bass | other, line: 이 트랙이 속한 Line",
    ]
    for t in rs.tracks:
        kind = guess_kind(t)
        out += [f"  - ref_track: {t.index}",
                f"    name: {_q(t.name)}",
                f"    kind: {kind}",
                f"    line: {_q(line_of.get(t.index))}",
                f"    tuning: {_q(t.tuning if kind != 'other' else None)}   # 1번 줄(가장 높은 줄)부터",
                f"    capo: {t.capo}",
                "    player: null"]
    out.append("lines:                        # Line = 한 기타리스트가 매 순간 쳐야 하는 한 줄기 (DESIGN 4.1)")
    for t in guitars + basses:
        lid = line_of[t.index]
        out += [f"  - id: {lid}",
                f"    name: {_q(t.name)}",
                f"    kind: {'bass' if lid.startswith('B') else 'guitar'}",
                f"    takes: [{t.index}]          # 더블 테이크면 같은 Line 에 트랙 번호를 모두 적는다 (R1)",
                "    relation: {double: false, unison_with: [], twin_with: null, overdub: false}"]
    out += [
        "players: []",
        "# 정답 구간. 마디 범위는 반열림: end_bar 는 포함하지 않는다.",
        "#   예) 17~24마디면 start_bar: 17, end_bar: 25",
        "# scenario: A | A_rhythm_only | B | F | ADT | quad | P70 | P_hard | twin | twin_double | centre_unison … (DESIGN 4.5)",
        "# roles: Line 별 lead | rhythm | both | silent;  split: dev | test (약 1/3은 test 로 떼어 둔다)",
        "sections: []",
        "#  - id: verse1",
        "#    start_bar: 9",
        "#    end_bar: 17",
        "#    scenario: A",
        "#    roles: {L1: rhythm, L2: lead}",
        "#    split: dev",
        "alignment:",
        "  method: null",
        "  approved: false             # align_check.wav 를 듣고 gtab gt approve 로 승인한다",
        "  approved_utc: null",
        "  minutes_spent: null",
    ]
    return "\n".join(out) + "\n"


def load_spec(path: Path) -> Any:
    import yaml

    from gtab.schema.gt import GtSpec

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as e:
        raise GtError(f"gt.yaml 문법 오류: {path} ({e})") from e
    try:
        return GtSpec.model_validate(data)
    except Exception as e:  # pydantic ValidationError -> Korean summary
        raise GtError(f"gt.yaml 값이 올바르지 않습니다: {path}\n{e}") from e


def save_spec_fields(path: Path, updates: dict[str, Any]) -> None:
    """Update top-level blocks of gt.yaml (used by approve / align). Comments in other blocks are kept."""
    import yaml

    text = Path(path).read_text(encoding="utf-8-sig")
    for key, value in updates.items():
        block = yaml.safe_dump({key: value}, allow_unicode=True, sort_keys=False, default_flow_style=False).rstrip("\n")
        pat = re.compile(rf"^{re.escape(key)}:.*?(?=^\S|\Z)", re.MULTILINE | re.DOTALL)
        if pat.search(text):
            text = pat.sub(lambda _m: block + "\n", text, count=1)
        else:
            text = text.rstrip("\n") + "\n" + block + "\n"
    atomic.write_text(Path(path), text)


@dataclass
class GtSong:
    dir: Path
    spec: Any
    gt_notes: Any | None
    tempo_map: Any | None
    refscore: Any | None


def track_lines(spec: Any) -> dict[int, str]:
    out: dict[int, str] = {}
    for t in spec.tracks:
        if t.line and t.kind in ("guitar", "bass"):
            out[t.ref_track] = t.line
    for ln in spec.lines:
        for take in ln.takes:
            out.setdefault(int(take), ln.id)
    return out


def load_tempo_map(song_dir_: Path) -> Any | None:
    from gtab.eval.refscore import tempo_map_from_dict

    p = Path(song_dir_) / "tempo_map.json"
    return tempo_map_from_dict(atomic.read_json(p)) if p.is_file() else None


def load_song(song_dir_: Path) -> GtSong:
    from gtab.eval.refscore import RefScore
    from gtab.schema.gt import GtNotes

    d = Path(song_dir_)
    spec = load_spec(d / "gt.yaml")
    gtn = GtNotes.model_validate_json((d / "gt_notes.json").read_bytes()) if (d / "gt_notes.json").is_file() else None
    rs = RefScore.load(d / "refscore.json") if (d / "refscore.json").is_file() else None
    return GtSong(d, spec, gtn, load_tempo_map(d), rs)


def rebuild_gt_notes(song: GtSong) -> Any:
    """Expand refscore.json with gt.yaml's line map / bar_order / pickup; fill times from tempo_map.json if any."""
    from gtab.eval.refscore import expand, fill_times

    if song.refscore is None:
        raise GtError(f"{song.dir.name}: refscore.json 이 없습니다 (gtab gt import 먼저)")
    ref = song.spec.reference or {}
    gtn = expand(song.refscore, song_id=song.spec.song_id, track_lines=track_lines(song.spec),
                 bar_order=ref.get("bar_order"), pickup=ref.get("pickup"))
    if song.tempo_map is not None:
        gtn = fill_times(gtn, song.tempo_map)
    atomic.write_text(song.dir / "gt_notes.json", gtn.model_dump_json(indent=1) + "\n")
    song.gt_notes = gtn
    return gtn
