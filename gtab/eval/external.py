"""External system outputs as predictions (DESIGN 8.5 external baselines; ``gtab eval import-external``).

Stored in ``data/eval/external/<slug(system)>/<slug(item)>.json`` (a ``Prediction``). MIDI files are read per
track with their own tempo map (seconds); Guitar Pro files go through RefScore (bar, tick) and are timed with the
Tier A song's ``tempo_map.json`` when the item is a Tier A song (else the file's nominal tempo).
``--map "1=L1,2=L2"`` maps source track numbers (1-based) to line ids; unmapped tracks become ``T<n>``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gtab import atomic, paths
from gtab.eval.runs import slug

log = logging.getLogger(__name__)

SYSTEMS = ("songsterr_ai", "klangio", "mvsep101", "other")
MIDI_EXTS = {".mid", ".midi"}
GP_EXTS = {".gp3", ".gp4", ".gp5", ".gp", ".gpx"}


class ExternalError(RuntimeError):
    """User-facing (Korean)."""


def external_path(system: str, item: str) -> Path:
    return paths.EVAL / "external" / slug(system) / f"{slug(item)}.json"


def parse_map(text: str | None) -> dict[int, str]:
    out: dict[int, str] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, sep, v = part.partition("=")
        if not sep or not k.strip().isdigit() or not v.strip():
            raise ExternalError(f"--map 형식이 잘못되었습니다: {part!r} (예: \"1=L1,2=L2\")")
        out[int(k)] = v.strip()
    return out


def midi_notes(path: Path) -> dict[int, list[tuple[float, float, int]]]:
    """Track number (1-based, as stored) -> [(onset_s, offset_s, pitch)], drums (channel 10) left out."""
    import mido

    mid = mido.MidiFile(str(path))
    tpq = mid.ticks_per_beat
    tempo_events: list[tuple[int, int]] = []
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                tempo_events.append((t, msg.tempo))
    tempo_events.sort()

    def to_s(tick: int) -> float:
        sec, last_t, tempo = 0.0, 0, 500000
        for et, tp in tempo_events:
            if et >= tick:
                break
            sec += (et - last_t) * tempo / 1e6 / tpq
            last_t, tempo = et, tp
        return sec + (tick - last_t) * tempo / 1e6 / tpq

    out: dict[int, list[tuple[float, float, int]]] = {}
    for ti, tr in enumerate(mid.tracks, 1):
        t = 0
        open_: dict[tuple[int, int], list[int]] = {}
        notes = []
        for msg in tr:
            t += msg.time
            if msg.type not in ("note_on", "note_off") or msg.channel == 9:
                continue
            key = (msg.channel, msg.note)
            if msg.type == "note_on" and msg.velocity > 0:
                open_.setdefault(key, []).append(t)
            elif open_.get(key):
                start = open_[key].pop(0)
                notes.append((to_s(start), to_s(t), int(msg.note)))
        for (_ch, note), starts in open_.items():
            notes += [(to_s(s), to_s(t), int(note)) for s in starts]
        if notes:
            out[ti] = sorted(notes)
    return out


def import_external(item: str, file: Path, system: str, *, mapping: dict[int, str] | None = None,
                    tempo_map: Any = None) -> Path:
    """Convert an external result to a Prediction and store it; returns the stored path."""
    from gtab.schema.evalio import PredNote, Prediction

    if system not in SYSTEMS:
        raise ExternalError(f"--system 은 {', '.join(SYSTEMS)} 중 하나여야 합니다 (받은 값: {system})")
    file = Path(file)
    ext = file.suffix.lower()
    mapping = dict(mapping or {})
    notes: list[Any] = []
    meta: dict[str, Any] = {"source_file": file.name, "source_sha256": atomic.sha256_file(file), "map": mapping}
    if ext in MIDI_EXTS:
        for track, ns in midi_notes(file).items():
            line = mapping.get(track, f"T{track}")
            notes += [PredNote(onset_s=round(a, 6), offset_s=round(max(a, b), 6), pitch=p, line=line) for a, b, p in ns]
    elif ext in GP_EXTS:
        from gtab.eval.gp_import import import_reference
        from gtab.eval.refscore import expand, nominal_tempo_map

        rs = import_reference(file)
        lines = {t.index: mapping.get(t.index, f"T{t.index}") for t in rs.tracks if not t.percussion}
        gtn = expand(rs, song_id=slug(item), track_lines=lines)
        tm = tempo_map or nominal_tempo_map(gtn)
        meta["timing"] = "gt_tempo_map" if tempo_map is not None else "nominal"
        import numpy as np

        if gtn.notes:
            on = np.asarray(tm.bar_tick_to_time(np.array([n.bar for n in gtn.notes]), np.array([n.tick for n in gtn.notes])))
            off = np.asarray(tm.bar_tick_to_time(np.array([n.bar for n in gtn.notes]),
                                                 np.array([n.tick + n.dur_ticks for n in gtn.notes])))
            notes = [PredNote(onset_s=round(max(0.0, float(a)), 6), offset_s=round(max(float(a), float(b), 0.0), 6),
                              pitch=n.pitch, line=n.line, shared=n.shared, bar=n.bar, tick=n.tick)
                     for n, a, b in zip(gtn.notes, on, off)]
    else:
        raise ExternalError(f"지원하지 않는 파일 형식입니다: {file.name} (.mid, .gp3/.gp4/.gp5/.gp/.gpx)")
    notes.sort(key=lambda n: (n.onset_s, n.pitch, n.line))
    lines_meta = [{"id": ln, "name": ln} for ln in sorted({n.line for n in notes})]
    pred = Prediction(format="gtab.prediction/1", system=f"external:{system}", item=item, lines=lines_meta,
                      notes=notes, meta=meta)
    out = external_path(system, item)
    atomic.write_text(out, pred.model_dump_json(indent=1) + "\n")
    return out


def load_external(system: str, item: str) -> Any | None:
    """Stored prediction for ``item`` (falls back to the song id before ':' for Tier A section items)."""
    from gtab.schema.evalio import Prediction

    for key in (item, item.split(":", 1)[0]):
        p = external_path(system, key)
        if p.is_file():
            pred = Prediction.model_validate_json(p.read_bytes())
            return pred.model_copy(update={"item": item}) if pred.item != item else pred
    return None
