"""Evaluation suites and systems (``bandscribe eval run``, spec §9.4 item 8, DESIGN §8.7).

Suites:
- ``quick``   — synthetic scenes (12 scenarios × 2 seeds, GT only) scored against ``oracle-noisy``; < 60 s; the
                pre-merge gate of DESIGN 9.7. Also the determinism check (metrics.csv / summary.json byte-identical).
- ``synth``   — every scenario × 5 seeds (+ lead −6 dB and reverb variants).
- ``bp-smoke``— 20 GuitarSet tracks drawn with ``eval.seed``, Basic Pitch via AMT's ``amt_basicpitch`` worker (bp310),
                onset-only F1 at 50 ms. GuitarSet is in Basic Pitch's training data: 판정용 아님 (flagged).
- ``tierA``   — songs under ``data/eval/tierA/`` (gt.yaml sections, GT tempo map); ``dev`` split unless ``--set``.
- ``tierB``   — Tier B dataset tracks (line GT from the dataset or AMT's reference transcription).
- ``components`` — runs every registered experiment that is still open (``bandscribe eval exp`` for each).
- ``distractors`` — which kinds of sound (synth pad/lead, keys, strings, FX ...) add false or missed guitar notes:
                Cambridge-MT excerpts remixed with and without each sound category, production guitar path,
                per-note cause attribution (``bandscribe.eval.distractors``; pre-registered as ``distractors``).
- ``backing`` — auxiliary: share of an original song's guitar notes that also appear on its guitar-removed backing
                (``bandscribe.eval.backing``; pairs in ``data/eval/backing_pairs.toml``, local only).

Systems: ``oracle`` (GT as prediction, sanity 1.0), ``oracle-noisy`` (seeded deletions / octave errors / jitter /
label swaps / unassigned / insertions), ``job:<stage>`` (NoteSets of a job's stage dir; rung from ``gpu_run.json``),
``external:<name>`` (``data/eval/external/<slug(name)>/<slug(item)>.json``), ``b0``, ``no_split``, ``majority``.
"""

from __future__ import annotations

import logging
import math
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic, paths
from bandscribe.eval import metrics, stats
from bandscribe.eval.metrics import UNASSIGNED, Event

log = logging.getLogger(__name__)

SUITES = ("quick", "bp-smoke", "synth", "tierA", "tierB", "components", "distractors", "backing")
DEFAULT_SYSTEM = {"quick": "oracle-noisy", "synth": "oracle-noisy", "bp-smoke": "basicpitch", "tierA": "job:notes",
                  "tierB": "oracle", "components": "experiments", "distractors": "production", "backing": "job:notes"}
BP_SMOKE_NOTE = "학습 데이터 포함(Basic Pitch 학습에 GuitarSet 사용), 판정용 아님"
Progress = Callable[[str, dict], None]


class SuiteError(RuntimeError):
    """User-facing (Korean) suite problem."""


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg.get(key, default) if cfg is not None else default
    except Exception:  # noqa: BLE001 - a partial config object must not break evaluation
        return default


def item_seed(base: int, item: str) -> int:
    """Per-item seed that is stable across processes (``hash()`` of a str is salted per process)."""
    return (int(base) * 1_000_003 + zlib.crc32(item.encode("utf-8"))) % (2**63)


# --------------------------------------------------------------------------------------------- items


@dataclass
class EvalItem:
    item: str                          # id as written inside JSON (may contain ':')
    dataset: str
    group: str                         # bootstrap block (song / performer / scene seed)
    scenario: str
    gt: Any                            # GtNotes | DatasetTrack
    lines: list[str]                   # GT lines to evaluate (guitar lines)
    tempo_map: Any = None
    sections: list[Any] | None = None  # half-open bar ranges or GtSection objects
    unison_windows: list[tuple[float, float]] | None = None
    song_key: str | None = None
    meta: dict = field(default_factory=dict)


def synth_items(seeds: Sequence[int], *, variants: bool = False) -> list[EvalItem]:
    from bandscribe.eval import remix

    out = []
    for scenario in remix.SCENARIOS:
        for s in seeds:
            kws: list[dict] = [{}]
            if variants:
                kws += [{"lead_gain_db": -6.0}, {"reverb": True}]
            for kw in kws:
                sc = remix.make_scene(scenario, s, render=False, **kw)
                guitar_lines = [ln.id for ln in sc.lines if ln.kind == "guitar"]
                uw = [(0.0, sc.meta["dur_s"])] if any(n.shared for n in sc.gt.notes) else None
                out.append(EvalItem(item=sc.item_id, dataset="synth", group=f"{scenario}:s{s}", scenario=scenario,
                                    gt=sc.gt, lines=guitar_lines, tempo_map=remix.scene_tempo_map(sc),
                                    unison_windows=uw, meta=dict(sc.meta)))
    return out


def tier_a_items(root: Path | None = None, *, split: str = "dev") -> list[EvalItem]:
    from bandscribe.eval import gtstore

    base = Path(root or paths.EVAL / "tierA")
    out = []
    for song_dir in sorted(p for p in base.glob("*") if (p / "gt.yaml").is_file()):
        try:
            song = gtstore.load_song(song_dir)
        except gtstore.GtError as e:  # a broken gt.yaml must be fixed, not silently left out of the score
            raise SuiteError(f"{song_dir.name}: {e}") from e
        if song.gt_notes is None:
            log.warning("%s: gt_notes.json 없음 (bandscribe gt import 먼저)", song_dir.name)
            continue
        if not song.spec.alignment.get("approved"):
            log.warning("%s: 정렬이 승인되지 않았습니다 (bandscribe gt approve)", song_dir.name)
        secs = [s for s in song.spec.sections if split == "all" or s.split == split]
        if not secs:
            continue
        tm = song.tempo_map
        if tm is None and any(n.onset_s is None for n in song.gt_notes.notes):
            # without the GT tempo map there are no GT times: scoring would silently compare against nothing
            log.warning("%s: 정답 템포맵(tempo_map.json)이 없어 건너뜁니다 (bandscribe gt align)", song_dir.name)
            continue
        gl = [ln.id for ln in song.spec.lines if ln.kind == "guitar"]
        uw = None
        if tm is not None:
            uw = [(float(tm.bar_tick_to_time([s.start_bar], [0])[0]), float(tm.bar_tick_to_time([s.end_bar], [0])[0]))
                  for s in secs if "unison" in s.scenario.lower()] or None
        for s in secs:
            out.append(EvalItem(item=f"{song.spec.song_id}:{s.id}", dataset="tierA", group=song.spec.song_id,
                                scenario=s.scenario, gt=song.gt_notes, lines=gl, tempo_map=tm, sections=[s],
                                unison_windows=uw, song_key=(song.spec.audio or {}).get("song_key"),
                                meta={"song_dir": str(song_dir), "section": s.id, "split": s.split}))
    return out


def dataset_items(name: str, *, split: str | None = None, limit: int | None = None, seed: int = 0,
                  reftx_root: Path | None = None) -> list[EvalItem]:
    """Items of a Tier B/C dataset. Tracks without note GT (Tier B multitracks) use AMT's reference
    transcription of the true solo tracks (``bandscribe eval reftx <dataset>``) as line GT; tracks with neither are
    skipped with a warning."""
    from bandscribe import datasets

    try:
        # Sample ids first and parse only the chosen tracks (GuitarSet: ~24 ms for the ids vs 11-30 s to load all
        # 360 annotations). Same order (sorted ids) and same RNG draw as sampling loaded tracks.
        ids = sorted(datasets.track_ids(name, split=split))
        if limit is not None and len(ids) > limit:
            rng = np.random.Generator(np.random.PCG64(seed))
            idx = sorted(rng.choice(len(ids), size=limit, replace=False).tolist())
            ids = [ids[i] for i in idx]
        tracks = [datasets.load_track(name, tid) for tid in ids]
    except datasets.DatasetUnavailable as e:
        raise SuiteError(str(e)) from e
    out = []
    for tr in tracks:
        meta: dict[str, Any] = {"track": tr}
        if not tr.notes:
            ref = load_reftx(tr, root=reftx_root)
            if ref is None:
                log.warning("%s:%s: 정답 음표도 참조 전사도 없어 건너뜁니다 (bandscribe eval reftx %s)", tr.dataset,
                            tr.track_id, tr.dataset)
                continue
            tr = tr.model_copy(update={"notes": ref})
            meta = {"track": tr, "reference": "reftx (MuScriptor 참조 전사: 입력·설정 비교용, 절대 성능 주장 금지)"}
        gl = [ln.id for ln in tr.lines if ln.kind == "guitar" and ln.id in tr.notes] or sorted(tr.notes)
        out.append(EvalItem(item=f"{tr.dataset}:{tr.track_id}", dataset=tr.dataset, group=tr.group or tr.track_id,
                            scenario=str(tr.meta.get("scenario", "")), gt=tr, lines=gl, meta=meta))
    return out


def reftx_dir(track: Any, root: Path | None = None) -> Path:
    """The item folder ``bandscribe eval reftx`` hands to ``bandscribe.amt.reftx.reference_transcribe`` (which writes
    ``<slug(line)>__reftx.json`` straight into it): ``data/eval/tierB/<slug(dataset)>/<slug(track)>/``."""
    from bandscribe.eval.runs import slug

    return Path(root or paths.EVAL / "tierB") / slug(track.dataset) / slug(track.track_id)


def load_reftx(track: Any, *, root: Path | None = None) -> dict[str, Any] | None:
    """{line id: NoteSet} from the stored reference transcriptions (None if there are none)."""
    from bandscribe.eval.runs import slug
    from bandscribe.schema.notes import NoteSet

    d = reftx_dir(track, root)
    out: dict[str, Any] = {}
    for ln in track.lines:
        f = d / f"{slug(ln.id)}__reftx.json"
        if f.is_file():
            try:
                out[ln.id] = NoteSet.model_validate_json(f.read_bytes())
            except ValueError as e:  # pydantic ValidationError is a ValueError
                log.warning("%s: 참조 전사 파일을 읽지 못했습니다 (%s)", f, e)
    return out or None


# ------------------------------------------------------------------------------------------- systems


def _pred(system: str, item: str, notes: list[Any], meta: dict | None = None) -> Any:
    from bandscribe.schema.evalio import Prediction

    lines = sorted({n.line for n in notes})
    return Prediction(format="bandscribe.prediction/1", system=system, item=item,
                      lines=[{"id": ln, "name": ln} for ln in lines], notes=notes, meta=dict(meta or {}))


def _gt_note_rows(it: EvalItem) -> list[tuple[float, float, int, str, bool]]:
    """(onset, offset, pitch, line, shared) of the item's GT notes, restricted to its lines and sections."""
    gt = it.gt
    keep = set(it.lines)
    rows = []
    if hasattr(gt, "string_fret") and isinstance(getattr(gt, "notes", None), dict):
        for ln, ns in gt.notes.items():
            if ln in keep:
                rows += [(float(n.onset_s), float(n.offset_s), int(n.pitch), ln, False) for n in ns.notes]
        return rows
    notes = [n for n in gt.notes if n.line in keep]
    if it.sections:
        rng = metrics.section_ranges(it.sections)
        notes = [n for n in notes if any(a <= n.bar < b for a, b in rng)]
    if notes and any(n.onset_s is None for n in notes):
        if it.tempo_map is None:
            return []
        timed = metrics._times_from_map(notes, it.tempo_map)
        return [(t.onset_s, t.offset_s, n.pitch, n.line, bool(n.shared)) for n, t in zip(notes, timed)]
    return [(float(n.onset_s), float(n.offset_s), int(n.pitch), n.line, bool(n.shared)) for n in notes]


class System:
    name = "system"
    rung = ""
    warnings: Sequence[str] = ()  # Korean, user-facing (systems that warn keep their own list)

    def predict(self, it: EvalItem) -> Any | None:
        raise NotImplementedError

    def rung_for(self, it: EvalItem) -> str:
        return self.rung

    def note_for(self, it: EvalItem) -> str:
        """Provenance for the metrics ``note`` column of ``it`` (job systems: the stage key used)."""
        return ""

    def stage_keys(self) -> dict[str, dict[str, str]]:
        """{item: {stage: key12}} of the job stage dirs this system read (run.json provenance)."""
        return {}


class Oracle(System):
    name = "oracle"

    def predict(self, it: EvalItem) -> Any:
        from bandscribe.schema.evalio import PredNote

        notes = [PredNote(onset_s=a, offset_s=max(a, b), pitch=p, line=ln, posterior=1.0, shared=sh)
                 for a, b, p, ln, sh in _gt_note_rows(it)]
        return _pred(self.name, it.item, notes)


class OracleNoisy(System):
    """Seeded corruptions of the GT, for harness checks (every rate is a knob; defaults are moderate)."""

    name = "oracle-noisy"

    def __init__(self, seed: int, *, p_delete: float = 0.10, p_octave: float = 0.05, jitter_s: float = 0.015,
                 p_swap: float = 0.10, p_unassigned: float = 0.05, p_insert: float = 0.05) -> None:
        self.seed = int(seed)
        self.p = dict(delete=p_delete, octave=p_octave, swap=p_swap, unassigned=p_unassigned, insert=p_insert)
        self.jitter_s = jitter_s

    def predict(self, it: EvalItem) -> Any:
        from bandscribe.schema.evalio import PredNote

        rng = np.random.Generator(np.random.PCG64(item_seed(self.seed, it.item)))
        rows = sorted(_gt_note_rows(it), key=lambda r: (r[0], r[2], r[3]))
        lines = sorted(set(it.lines))
        notes = []
        for a, b, p, ln, sh in rows:
            u = rng.random(6)
            if u[0] < self.p["delete"]:
                continue
            pitch = p + (12 if u[1] < self.p["octave"] and p <= 115 else 0)
            on = max(0.0, a + float(np.clip(rng.normal(0, self.jitter_s), -0.045, 0.045)))
            line = ln
            if u[2] < self.p["swap"] and len(lines) > 1:
                line = lines[(lines.index(ln) + 1 + int(u[3] * (len(lines) - 1))) % len(lines)]
            if u[4] < self.p["unassigned"]:
                line = UNASSIGNED
            post = float(0.3 + 0.7 * u[5]) if line == ln else float(0.2 + 0.5 * u[5])
            notes.append(PredNote(onset_s=round(on, 6), offset_s=round(max(on, b), 6), pitch=int(pitch), line=line,
                                  posterior=round(post, 4), shared=sh))
        n_ins = int(round(self.p["insert"] * len(rows)))
        span = max((r[1] for r in rows), default=1.0)
        for _ in range(n_ins):
            on = float(rng.uniform(0, span))
            notes.append(PredNote(onset_s=round(on, 6), offset_s=round(on + 0.2, 6), pitch=int(rng.integers(40, 88)),
                                  line=lines[int(rng.integers(0, len(lines)))] if lines else UNASSIGNED,
                                  posterior=round(float(rng.uniform(0.1, 0.6)), 4)))
        notes.sort(key=lambda n: (n.onset_s, n.pitch, n.line))
        return _pred(self.name, it.item, notes, {"seed": self.seed, **self.p, "jitter_s": self.jitter_s})


class Majority(System):
    """Oracle content, every note in the largest GT line: the floor of macro assignment accuracy."""

    name = "majority"

    def predict(self, it: EvalItem) -> Any:
        from bandscribe.eval.baselines import majority

        pred = Oracle().predict(it)
        return majority(pred, metrics.events_from_prediction(pred))


class External(System):
    def __init__(self, name: str) -> None:
        self.name = f"external:{name}"
        self.ext = name

    def predict(self, it: EvalItem) -> Any | None:
        from bandscribe.eval.external import load_external

        return load_external(self.ext, it.item)


B0_FILE = "raw/mix_mono__muscriptor.json"  # amt_ms1's full-mix transcription: only the eval profile makes it


def with_profile(cfg: Any, profile: str) -> Any:
    """A copy of ``cfg`` (Config or plain dict) with ``run.profile`` replaced; anything else is returned as is."""
    from bandscribe.config import Config

    if isinstance(cfg, Config):
        return cfg.model_copy(update={"run": cfg.run.model_copy(update={"profile": profile})})
    if isinstance(cfg, dict):
        return {**cfg, "run": {**(cfg.get("run") or {}), "profile": profile}}
    return cfg


class JobStage(System):
    """NoteSets of a job's stage: ``job:notes`` (guitar_all.json), ``job:amt_gtr`` (raw guitar view), …

    The stage dir is the one the **current settings** would make (the planned key, ``runner.plan``). When that
    dir does not exist the item is skipped with a Korean warning: the newest dir made with other settings is a
    different system, and scoring it under this name silently mixes results (review 2026-10-05). Opt in with
    ``allow_stale=True`` (``bandscribe eval run ... --arg allow_stale=true``): the newest finished dir is used
    and its note says ``(stale)``. Only when the job cannot be planned at all (no config, a foreign layout) is
    the newest dir used without that flag. Every item's metrics note records ``stage=<key12>``; run.json
    lists all stage keys read (``stage_keys``).

    ``b0`` (baseline B0) needs the full-mix transcription, which only the ``eval`` profile runs (2026-10-04: the
    default ``quality`` profile skips it for speed). So B0 plans its stage keys with ``run.profile = "eval"``;
    without that dir B0 is skipped with a Korean hint to run ``bandscribe run <곡> --profile eval``.
    """

    def __init__(self, stage: str, *, mode: str = "lines", cfg: Any = None, allow_stale: bool = False) -> None:
        self.stage = stage
        self.mode = mode
        self.name = f"job:{stage}" if mode == "lines" else mode
        self.cfg = with_profile(cfg, "eval") if mode == "b0" else cfg
        self.allow_stale = bool(allow_stale)
        self._need = B0_FILE if mode == "b0" else None
        self._warned: set[str] = set()
        self._rung: dict[str, str] = {}
        self._planned: dict[str, dict[str, str]] = {}
        self._dirs: dict[str, dict[str, str]] = {}  # song_key -> {stage: key12 [+ " (stale)"]}
        self._item_song: dict[str, str] = {}
        self._note: dict[str, str] = {}
        self._skipped_stale: set[tuple[str, str]] = set()
        self.warnings: list[str] = []

    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)
            log.warning("%s", msg)

    def _planned_keys(self, song_key: str) -> dict[str, str]:
        """{stage: key12} the current config would use for this job (``bandscribe.pipeline.runner.plan``), or {} when
        it cannot be planned (no config, missing model, foreign job layout)."""
        if self.cfg is None or not hasattr(self.cfg, "get"):
            return {}
        if song_key not in self._planned:
            keys: dict[str, str] = {}
            try:
                from bandscribe.jobs.store import JobStore
                from bandscribe.pipeline import runner

                for it in runner.plan(song_key, self.cfg, store=JobStore(paths.JOBS), missing_ok=True):
                    if it.key12:
                        keys[it.stage] = it.key12
            except Exception as e:  # planning is a preference, never a reason to fail an evaluation
                log.info("could not plan %s for job:<stage> lookup: %s", song_key, e)
            self._planned[song_key] = keys
        return self._planned[song_key]

    def _stage_dir(self, song_key: str, stage: str, need: str | None = None, *, quiet: bool = False) -> Path | None:
        """The stage dir made with the current settings (planned key; with ``need``, a file relative to the stage
        dir, it must contain it). Missing: None with a Korean warning (``quiet``: none, for provenance lookups),
        or the newest finished dir under ``allow_stale``. Unplannable job: the newest finished dir."""
        d = paths.JOBS / song_key / "stages" / stage
        key12 = self._planned_keys(song_key).get(stage)

        def ok(p: Path) -> bool:
            return (p / "manifest.json").is_file() and (need is None or (p / need).is_file())

        def used(p: Path, stale: bool = False) -> Path:
            self._dirs.setdefault(song_key, {})[stage] = p.name + (" (stale)" if stale else "")
            return p

        if key12 and ok(d / key12):
            return used(d / key12)
        cands = sorted((p for p in d.glob("*") if ok(p)), key=lambda p: p.stat().st_mtime) if d.is_dir() else []
        if not cands:
            return None
        if key12 is None:  # the job could not be planned (no config, a foreign layout): newest is all there is
            return used(cands[-1])
        if not self.allow_stale:
            if quiet:
                return None
            self._skipped_stale.add((song_key, stage))
            prof = " --profile eval" if self.mode == "b0" else ""
            self._warn(f"{song_key}: 현재 설정으로 만든 {stage} 단계 결과({key12})가 없어 이 곡을 평가에서 뺍니다(다른 설정의 "
                       f"결과 {len(cands)}개는 쓰지 않음). 'bandscribe run {song_key} --until {stage}{prof}' 로 만든 뒤 다시 "
                       "평가하거나, 예전 결과로 평가하려면 --arg allow_stale=true 를 주세요.")
            return None
        if not quiet:
            self._warn(f"{song_key}: 현재 설정의 {stage} 단계({key12})가 없어 가장 최근 결과({cands[-1].name})로 "
                       "평가합니다(allow_stale: 다른 설정의 결과, 지표 note 에 stale 표시).")
        return used(cands[-1], stale=True)

    def _read_rung(self, song_key: str) -> str:
        labels = []
        for st in ("sep", "amt_ms1" if self.mode == "b0" else "amt_gtr"):
            sd = self._stage_dir(song_key, st, self._need if st == "amt_ms1" else None, quiet=True)
            gr = sd / "gpu_run.json" if sd else None
            if gr and gr.is_file():
                lab = (atomic.read_json(gr).get("rung") or {}).get("label")
                if lab:
                    labels.append(f"{st}={lab}")
        return "|".join(labels)

    def predict(self, it: EvalItem) -> Any | None:
        from bandscribe.eval.baselines import b0, no_split
        from bandscribe.schema.notes import NoteSet

        if not it.song_key:
            return None
        stage = "amt_ms1" if self.mode == "b0" else self.stage
        if self.mode == "b0":
            sd = self._stage_dir(it.song_key, "amt_ms1", B0_FILE)
            f = sd / B0_FILE if sd else None
            if not f or not f.is_file():
                if it.song_key not in self._warned and (it.song_key, "amt_ms1") not in self._skipped_stale:
                    self._warned.add(it.song_key)
                    self._warn(f"{it.song_key}: B0 는 eval 프로필의 믹스 전사가 필요합니다. 'bandscribe run {it.song_key} "
                               "--profile eval' 로 만든 뒤 다시 평가하세요(B0 생략).")
                return None
            pred = b0(NoteSet.model_validate_json(f.read_bytes()), item=it.item)
        else:
            sd = self._stage_dir(it.song_key, self.stage)
            f = self._guitar_file(sd) if sd is not None else None
            if f is None:
                return None
            ns = NoteSet.model_validate_json(f.read_bytes())
            pred = no_split(ns, item=it.item, system=self.name)
        self._item_song[it.item] = it.song_key
        self._note[it.item] = f"stage={self._dirs.get(it.song_key, {}).get(stage, sd.name if sd else '')}"
        self._rung[it.item] = self._read_rung(it.song_key)
        return pred

    def rung_for(self, it: EvalItem) -> str:
        return self._rung.get(it.item, "")

    def note_for(self, it: EvalItem) -> str:
        return self._note.get(it.item, "")

    def stage_keys(self) -> dict[str, dict[str, str]]:
        return {item: dict(self._dirs.get(song, {})) for item, song in sorted(self._item_song.items())}

    @staticmethod
    def _guitar_file(sd: Path) -> Path | None:
        """The stage's guitar transcription: ``guitar_all.json`` (notes), else the raw guitar view
        (``raw/guitar*__*.json``, amt_gtr), else the mix view (amt_ms1: the B0 input), else the first raw file.
        Never a bass / piano+other view by accident of sort order."""
        if (sd / "guitar_all.json").is_file():
            return sd / "guitar_all.json"
        raw = sorted((sd / "raw").glob("*.json")) if (sd / "raw").is_dir() else []
        for prefix in ("guitar", "mix_mono__"):
            hit = [f for f in raw if f.name.startswith(prefix)]
            if hit:
                return hit[0]
        return raw[0] if raw else None


class BasicPitchWorker(System):
    """Basic Pitch through AMT's ``amt_basicpitch`` worker (bp310 env, CPU, no GPU lock), all items in one call."""

    name = "basicpitch"

    def __init__(self, cfg: Any, work_root: Path) -> None:
        self.cfg = cfg
        self.work_root = Path(work_root)
        self.cache: dict[str, Any] = {}

    def prepare(self, items: Sequence[EvalItem], run_worker: Callable[..., Any] | None = None) -> None:
        from bandscribe import gpu
        from bandscribe.eval.runs import slug
        from bandscribe.schema.notes import NoteSet

        run_worker = run_worker or gpu.run_worker
        views = []
        out_dir = self.work_root / "basicpitch"
        out_dir.mkdir(parents=True, exist_ok=True)
        for it in items:
            wav = _item_audio(it)
            if wav is None:
                log.warning("%s: 오디오 파일을 찾지 못했습니다", it.item)
                continue
            views.append({"id": slug(it.item), "wav": str(wav), "out": str(out_dir / f"{slug(it.item)}__basicpitch.json"),
                          "view_sha256": None})
        if not views:
            raise SuiteError("bp-smoke: 평가할 GuitarSet 오디오가 없습니다 (bandscribe data fetch guitarset)")
        params = {"onset_threshold": float(_cfg(self.cfg, "amt.bp_onset_threshold", 0.5)),
                  "frame_threshold": float(_cfg(self.cfg, "amt.bp_frame_threshold", 0.3)),
                  "min_note_frames": int(_cfg(self.cfg, "amt.bp_min_note_frames", 4)),
                  "minimum_frequency": (float(_cfg(self.cfg, "amt.bp_min_freq_hz", 0.0)) or None),
                  "maximum_frequency": (float(_cfg(self.cfg, "amt.bp_max_freq_hz", 0.0)) or None),
                  "melodia_trick": bool(_cfg(self.cfg, "amt.bp_melodia_trick", True)), "multiple_pitch_bends": False}
        request = {"format": "bandscribe.worker/1", "job": None, "out_dir": str(out_dir), "device": "cpu",
                   "vram": {"policy": "error", "cpu_allowed": True, "precheck": False, "force_rung": None, "cap_mb": None},
                   "determinism": True, "mode": "transcribe", "threads": int(_cfg(self.cfg, "amt.bp_threads", 4)),
                   "model_output_cache": None, "views": views, "params": params, "allow_default_min_len": False}
        res = run_worker("amt_basicpitch", request, self.work_root / "_worker_bp",
                         timeout_s=float(_cfg(self.cfg, "gpu.worker_timeout_s", 7200)),
                         python=paths.BP310_PYTHON, use_lock=False)
        if not getattr(res, "ok", False):
            raise SuiteError(f"Basic Pitch 워커 실패: {getattr(res, 'error', None) or getattr(res, 'stderr_path', '')}")
        for v, it in zip(views, [i for i in items if _item_audio(i) is not None]):
            f = Path(v["out"])
            if f.is_file():
                ns = NoteSet.model_validate_json(f.read_bytes())
                from bandscribe.eval.baselines import prediction_from_notes

                # one GT line -> name the prediction after it, so the identity mapping (label_swap_raw) is meaningful
                line = it.lines[0] if len(it.lines) == 1 else "guitar"
                self.cache[it.item] = prediction_from_notes(ns.notes, system=self.name, item=it.item, line=line,
                                                            meta={"backend": ns.backend.model_dump()})

    def predict(self, it: EvalItem) -> Any | None:
        return self.cache.get(it.item)


def _item_audio(it: EvalItem) -> Path | None:
    tr = it.meta.get("track")
    if tr is None:
        return None
    from bandscribe import datasets

    root = Path(datasets.dataset_root(tr.dataset))
    parts = sorted(tr.parts, key=lambda p: (0 if "mic" in p.id else 1, 0 if p.kind in ("guitar", "mix") else 1, p.id))
    for p in parts:
        if p.kind in ("guitar", "mix", "di"):
            return root / p.audio
    return root / tr.mix if tr.mix else None


def make_system(name: str, cfg: Any, *, work_root: Path | None = None, allow_stale: bool = False) -> System:
    """``allow_stale``: job systems may score a stage dir made with other settings (``JobStage``)."""
    seed = int(_cfg(cfg, "eval.seed", stats.DEFAULT_SEED))
    if name == "oracle":
        return Oracle()
    if name == "oracle-noisy":
        return OracleNoisy(seed)
    if name == "majority":
        return Majority()
    if name == "b0":
        return JobStage("amt_ms1", mode="b0", cfg=cfg, allow_stale=allow_stale)
    if name == "no_split":
        return JobStage("amt_gtr", mode="no_split", cfg=cfg, allow_stale=allow_stale)
    if name == "basicpitch":
        return BasicPitchWorker(cfg, work_root or paths.DATA / "tmp")
    if name.startswith("job:"):
        return JobStage(name.split(":", 1)[1], cfg=cfg, allow_stale=allow_stale)
    if name.startswith("external:"):
        return External(name.split(":", 1)[1])
    raise SuiteError(f"알 수 없는 시스템 '{name}' (oracle, oracle-noisy, majority, b0, no_split, job:<단계>, external:<이름>)")


# --------------------------------------------------------------------------------------- per-item metrics


def _merged_arrays(rows: Iterable[tuple[float, float, int]], tol: float = metrics.DEDUPE_TOL_S) -> metrics.NoteArrays:
    """Merge notes of equal pitch within ``tol`` (unison copies / doubled predictions count once)."""
    by_p: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for a, b, p in rows:
        by_p[int(p)].append((a, b))
    iv, pp = [], []
    for p in sorted(by_p):
        cur: list[float] | None = None
        for a, b in sorted(by_p[p]):
            if cur is not None and a - cur[0] <= tol:
                cur[1] = max(cur[1], b)
                continue
            if cur is not None:
                iv.append(cur)
                pp.append(p)
            cur = [a, b]
        if cur is not None:
            iv.append(cur)
            pp.append(p)
    if not iv:
        return np.zeros((0, 2)), np.zeros(0)
    order = np.lexsort((np.array(pp), np.array([x[0] for x in iv])))
    return np.array(iv)[order], np.array(pp, dtype=float)[order]


@dataclass
class ItemResult:
    rows: list[dict[str, Any]]
    confusion: dict[str, Any]
    coverage: list[tuple[float, float, float]]
    worst_bars: list[dict[str, Any]]
    majority_macro: float


def evaluate_item(it: EvalItem, pred: Any, *, suite: str, system: str, rung: str = "", note: str = "",
                  onset_tol: float = 0.05) -> ItemResult:
    base = {"suite": suite, "system": system, "dataset": it.dataset, "item": it.item, "group": it.group,
            "scenario": it.scenario, "section": it.meta.get("section", ""), "rung": rung, "note": note}
    windows = None
    if it.sections and it.tempo_map is not None:
        rng = metrics.section_ranges(it.sections)
        windows = [(float(it.tempo_map.bar_tick_to_time([a], [0])[0]), float(it.tempo_map.bar_tick_to_time([b], [0])[0]))
                   for a, b in rng]
    gt_rows = _gt_note_rows(it)
    pred_notes = [n for n in pred.notes if windows is None or any(s <= n.onset_s < e for s, e in windows)]
    ref = _merged_arrays((a, b, p) for a, b, p, _l, _s in gt_rows)
    est = _merged_arrays((n.onset_s, n.offset_s, n.pitch) for n in pred_notes)
    rows: list[dict[str, Any]] = []

    def add(metric: str, r: metrics.PRF | None = None, value: float | None = None, **kw: Any) -> None:
        row = {**base, "metric": metric, **kw}
        if r is not None:
            row.update(value=r.f1, tp=r.tp, fp=r.fp, fn=r.fn, n=r.tp + r.fn)
        else:
            row["value"] = value
        rows.append(row)

    for name, r in metrics.note_prf_variants(ref, est).items():
        add(f"note_f1_{name}", r)
    add("chroma_f1_onset50", metrics.note_prf(ref, est, chroma=True, onset_tol=onset_tol))
    add("octave_error_rate", value=metrics.octave_error_rate(ref, est, onset_tol=onset_tol))
    gt_notes = it.gt if hasattr(it.gt, "tpb") else None
    if gt_notes is not None and it.tempo_map is not None:
        add("tick_f1", metrics.tick_prf(gt_notes, est, it.tempo_map, sections=it.sections, lines=it.lines))
    ge = metrics._dedupe([(a, p, ln, None, sh, None) for a, _b, p, ln, sh in gt_rows], metrics.DEDUPE_TOL_S)
    pe = metrics._dedupe([(n.onset_s, n.pitch, n.line or UNASSIGNED, n.posterior, bool(n.shared), n.bar)
                          for n in pred_notes], metrics.DEDUPE_TOL_S)
    res = metrics.assignment(ge, pe, onset_tol=onset_tol)
    add("assign_macro", value=res.macro, n=res.n_matched)
    add("assign_weighted", value=res.weighted, n=res.n_matched)
    add("unassigned_ratio", value=res.unassigned_ratio, n=res.n_est)
    add("label_swap_raw", value=res.label_swap_raw, n=res.n_matched)
    for ln, r in metrics.lines_f1(ge, pe, res.mapping, onset_tol=onset_tol).items():
        add("line_f1_onset50", r, line=ln)
    if any(len(e.lines) > 1 or e.shared for e in ge):
        add("shared_f1", metrics.shared_prf(ge, pe, it.unison_windows, onset_tol=onset_tol))
    from bandscribe.eval.baselines import largest_line

    maj = float("nan")
    if ge:
        target = largest_line(ge)
        mp = [Event(e.onset_s, e.pitch, frozenset({target}) if e.lines - {UNASSIGNED} else e.lines, e.posterior)
              for e in pe]
        maj = metrics.assignment(ge, mp, onset_tol=onset_tol).macro
        add("assign_macro_majority", value=maj, n=res.n_matched)
    cov = metrics.coverage_accuracy(ge, pe, onset_tol=onset_tol)
    worst = _worst_bars(it, ge, pe, res)
    return ItemResult(rows, res.confusion | {"mapping": res.mapping}, cov, worst, maj)


def _worst_bars(it: EvalItem, ge: Sequence[Event], pe: Sequence[Event], res: metrics.AssignResult) -> list[dict[str, Any]]:
    """Per bar: missed, extra and wrongly assigned events (bars from the GT tempo map; 2 s bins without one)."""
    def bar_of(t: float) -> int:
        if it.tempo_map is not None:
            return int(np.asarray(it.tempo_map.time_to_bar_tick(t)[0]).reshape(-1)[0])
        return int(t // 2.0) + 1

    matched_g = {g for g, _ in res.pairs}
    matched_e = {e for _, e in res.pairs}
    acc: dict[int, dict[str, int]] = defaultdict(lambda: {"missed": 0, "extra": 0, "wrong_line": 0})
    for i, e in enumerate(ge):
        if i not in matched_g:
            acc[bar_of(e.onset_s)]["missed"] += 1
    for j, e in enumerate(pe):
        if j not in matched_e:
            acc[bar_of(e.onset_s)]["extra"] += 1
    for gi, ei in res.pairs:
        if not any(res.mapping.get(p) in ge[gi].lines for p in pe[ei].lines if p != UNASSIGNED):
            acc[bar_of(ge[gi].onset_s)]["wrong_line"] += 1
    out = [{"item": it.item, "song_key": it.song_key or "", "bar": b, **v, "total": sum(v.values())}
           for b, v in acc.items()]
    return sorted(out, key=lambda r: (-r["total"], r["bar"]))


# -------------------------------------------------------------------------------------------- running


@dataclass
class SuiteRun:
    run_dir: Path
    summary: dict[str, Any]


def items_for_suite(suite: str, cfg: Any, args: dict | None = None) -> list[EvalItem]:
    args = dict(args or {})
    seed = int(_cfg(cfg, "eval.seed", stats.DEFAULT_SEED))
    if suite == "quick":
        return synth_items((0, 1))
    if suite == "synth":
        return synth_items((0, 1, 2, 3, 4), variants=True)
    if suite == "bp-smoke":
        return dataset_items("guitarset", limit=int(args.get("limit", 20)), seed=seed)
    if suite == "tierA":
        return tier_a_items(split=str(args.get("split", "dev")))
    if suite == "tierB":
        from bandscribe import datasets

        out: list[EvalItem] = []
        names = args.get("datasets") or ["cambridge_mt", "medleydb"]
        for name in names:
            if datasets.is_available(name):
                out += dataset_items(name)
        if not out:
            raise SuiteError("Tier B 항목이 없습니다. Cambridge-MT 곡을 `bandscribe data import-local cambridge_mt <폴더> --song <이름>` "
                             "으로 등록하고 lines.yaml 을 채운 뒤 `bandscribe eval reftx cambridge_mt` 로 참조 전사를 만드세요.")
        return out
    raise SuiteError(f"알 수 없는 평가 세트 '{suite}' ({', '.join(SUITES)})")


BASELINE_METRICS = ("note_f1_onset50", "tick_f1", "assign_macro")


def join_notes(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


def _job_baselines(items: Sequence[EvalItem], system: str, cfg: Any, *, suite: str, note: str,
                   onset_tol: float, allow_stale: bool = False,
                   used: list[System] | None = None) -> list[dict[str, Any]]:
    """DESIGN 8.5 "기준선 (모든 리포트에 포함)": for items that come from a job (Tier A songs), score B0 (guitar
    classes of the mix transcription) and no-split (the guitar-stem transcription as one line) next to the
    system. Their rows go to metrics.csv with their own ``system``; jobs without those stages are skipped.
    ``used`` collects the baseline systems (their warnings and stage keys go to summary.json / run.json)."""
    if system in ("b0", "no_split") or not any(it.song_key for it in items):
        return []
    out: list[dict[str, Any]] = []
    for bname in ("b0", "no_split"):
        bsys = make_system(bname, cfg, allow_stale=allow_stale)
        if used is not None:
            used.append(bsys)
        for it in items:
            pred = bsys.predict(it)
            if pred is not None:
                out += evaluate_item(it, pred, suite=suite, system=bsys.name, rung=bsys.rung_for(it),
                                     note=join_notes(note, bsys.note_for(it)), onset_tol=onset_tol).rows
    return out


def _summary(rows: Sequence[dict[str, Any]], *, n: int, seed: int,
             mde: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from bandscribe.eval.experiments import aggregate_rows

    aggs = aggregate_rows(rows, rows[0]["suite"] if rows else "", n=n, seed=seed)
    out: dict[str, Any] = {"aggregates": {}, "mde_checks": {}, "baselines": {}}
    for a in aggs:
        key = a["metric"] + (f"|{a['line']}" if a.get("line") else "") + (f"|{a['rung']}" if a.get("rung") else "")
        out["aggregates"][key] = {"value": a["value"], "ci": [a["ci_low"], a["ci_high"]], "n_items": a["n"]}
        if a["metric"] in ("note_f1_onset50", "tick_f1", "assign_macro") and np.isfinite(a["ci_low"]):
            half = (a["ci_high"] - a["ci_low"]) * 100 / 2
            out["mde_checks"][key] = {"ci_half_width_points": half, "mde_points": mde, "resolves_mde": half <= mde}
    if "assign_macro_majority" in out["aggregates"]:
        out["baselines"]["majority_macro"] = out["aggregates"].pop("assign_macro_majority")
    return out, aggs


def run_suite(suite: str, system: str | None, cfg: Any, *, out: Path | None = None, args: dict | None = None,
              progress: Progress | None = None, now: Any = None, gitsha: str | None = None,
              run_worker: Callable[..., Any] | None = None) -> SuiteRun:
    from bandscribe.eval import report, runs

    if suite == "components":
        return _run_components(cfg, out=out, args=args, now=now, gitsha=gitsha)
    if suite in ("distractors", "backing"):
        return _run_special(suite, cfg, out=out, args=args, progress=progress, now=now, gitsha=gitsha)
    system = system or DEFAULT_SYSTEM[suite]
    note = BP_SMOKE_NOTE if suite == "bp-smoke" else ""
    items = items_for_suite(suite, cfg, args)
    if not items:
        hint = (" (Tier A 곡은 bandscribe gt import → gt.yaml 의 sections 채우기 → bandscribe gt align 순서로 준비합니다)"
                if suite == "tierA" else "")
        raise SuiteError(f"'{suite}' 에 평가할 항목이 없습니다{hint}")
    run_dir = runs.new_run_dir(system, root=out, now=now, gitsha=gitsha)
    started = runs.utc_now()
    allow_stale = _truthy((args or {}).get("allow_stale"))
    sysobj = make_system(system, cfg, work_root=run_dir, allow_stale=allow_stale)
    if isinstance(sysobj, BasicPitchWorker):
        sysobj.prepare(items, run_worker=run_worker)
    n = int(_cfg(cfg, "eval.bootstrap_iterations", stats.DEFAULT_ITERATIONS))
    seed = int(_cfg(cfg, "eval.seed", stats.DEFAULT_SEED))
    mde = float(_cfg(cfg, "eval.mde_points", 1.0))
    onset_tol = float(_cfg(cfg, "eval.onset_tol_s", 0.05))
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    skipped = []
    for k, it in enumerate(items):
        if progress:
            progress("item", {"i": k + 1, "n": len(items), "item": it.item})
        pred = sysobj.predict(it)
        if pred is None:
            skipped.append(it.item)
            continue
        r = evaluate_item(it, pred, suite=suite, system=sysobj.name, rung=sysobj.rung_for(it),
                          note=join_notes(note, sysobj.note_for(it)), onset_tol=onset_tol)
        rows += r.rows
        worst += r.worst_bars
        details.append({"item": it.item, "scenario": it.scenario, "confusion": r.confusion, "coverage": r.coverage})
    if not rows:
        hint = "".join(f"\n- {w}" for w in sysobj.warnings)
        raise SuiteError(f"시스템 '{system}' 의 예측이 하나도 없습니다 (건너뜀 {len(skipped)}개){hint}")
    summ, aggs = _summary(rows, n=n, seed=seed, mde=mde)
    summary = {"suite": suite, "system": sysobj.name, "n_items": len({r["item"] for r in rows}),
               "skipped": sorted(skipped), "note": note, **summ}
    base_systems: list[System] = []
    base_rows = _job_baselines(items, sysobj.name, cfg, suite=suite, note=note, onset_tol=onset_tol,
                               allow_stale=allow_stale, used=base_systems)
    warnings = [w for s in [sysobj, *base_systems] for w in s.warnings]
    if warnings:
        summary["warnings"] = list(dict.fromkeys(warnings))
        if progress:
            for w in summary["warnings"]:
                progress("message", {"message": f"주의: {w}"})
    stage_keys = {s.name: s.stage_keys() for s in [sysobj, *base_systems] if s.stage_keys()}
    if base_rows:
        from bandscribe.eval.experiments import aggregate_rows

        base_aggs = aggregate_rows(base_rows, suite, n=n, seed=seed)
        for a in base_aggs:
            if a["metric"] in BASELINE_METRICS and not a.get("line"):
                summary["baselines"][f"{a['system']}:{a['metric']}"] = {
                    "value": a["value"], "ci": [a["ci_low"], a["ci_high"]], "n_items": a["n"], "rung": a["rung"]}
        base_rows += base_aggs
    runs.write_metrics_csv(run_dir / "metrics.csv", rows + aggs + base_rows)
    runs.write_summary(run_dir / "summary.json", summary)
    _write_worst_bars(run_dir / "worst_bars.csv", worst)
    runs.write_run_json(run_dir / "run.json", {
        "suite": suite, "system": sysobj.name, "args": args or {}, "git_sha": run_dir.name.split("_")[1],
        "started_utc": started, "finished_utc": runs.utc_now(),
        "config": cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else None,
        **({"stage_keys": stage_keys, "allow_stale": allow_stale} if stage_keys or allow_stale else {}),
        **runs.provenance()})
    report.write_suite_report(run_dir, summary, rows + aggs, details, worst)
    return SuiteRun(run_dir, summary)


def _truthy(v: Any) -> bool:
    """``--arg allow_stale=true`` arrives as JSON (bool) or as a string."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _write_worst_bars(path: Path, worst: Sequence[dict[str, Any]]) -> None:
    import csv
    import io

    cols = ["item", "song_key", "bar", "missed", "extra", "wrong_line", "total"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in sorted(worst, key=lambda r: (-r["total"], r["item"], r["bar"])):
        w.writerow({c: r[c] for c in cols})
    atomic.write_bytes(path, buf.getvalue().encode("utf-8"))


def _run_special(suite: str, cfg: Any, *, out: Path | None, args: dict | None, progress: Progress | None,
                 now: Any, gitsha: str | None) -> SuiteRun:
    """Suites with their own runner and report (``distractors``, ``backing``). A registry entry with the suite's
    id (docs/decisions.md) goes from 사전 등록 to 실행 중 and gets a result line, like ``bandscribe eval exp``."""
    import datetime as dt

    from bandscribe.eval import experiments, runs

    args = dict(args or {})
    run_dir = runs.new_run_dir(suite, root=out, now=now, gitsha=gitsha)
    started = runs.utc_now()
    say = (lambda msg: progress("message", {"message": msg})) if progress else None
    try:
        if suite == "distractors":
            from bandscribe.eval import distractors

            summary = distractors.run(run_dir, cfg, args, progress=say)
        else:
            from bandscribe.eval import backing

            summary = backing.run(run_dir, cfg, args, progress=say)
    except Exception as e:
        try:
            atomic.write_json(run_dir / "failed.json", {"suite": suite, "error_type": type(e).__name__, "error": str(e)})
        except OSError:
            pass
        if type(e).__name__ in ("DistractorError", "BackingError"):
            raise SuiteError(str(e)) from e
        raise
    runs.write_run_json(run_dir / "run.json", {
        "suite": suite, "system": DEFAULT_SYSTEM[suite], "args": {k: v for k, v in args.items() if k != "cache_root"},
        "git_sha": run_dir.name.split("_")[1], "started_utc": started, "finished_utc": runs.utc_now(),
        "config": cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else None, **runs.provenance()})
    if not summary.get("dry_run") and not args.get("no_registry"):
        try:
            reg_path = experiments.decisions_path()
            entry = experiments.load_registry(reg_path).get(suite)
            if entry is not None and not entry.decided:
                if entry.status == experiments.STATE_PRE:
                    experiments.set_status(suite, experiments.STATE_RUNNING, reg_path)
                experiments.append_result(suite, f"{(now or dt.datetime.now()).strftime('%Y-%m-%d')} `{run_dir.name}`: "
                                                 "계산된 판정 = 서술(판정 없음)", reg_path)
        except OSError as e:
            log.warning("decisions.md 를 갱신하지 못했습니다: %s", e)
    return SuiteRun(run_dir, summary)


def _run_components(cfg: Any, *, out: Path | None, args: dict | None, now: Any, gitsha: str | None) -> SuiteRun:
    from bandscribe.eval import experiments, runs

    reg = experiments.load_registry()
    open_ids = [e for e, v in sorted(reg.items()) if not v.decided]
    run_dir = runs.new_run_dir("components", root=out, now=now, gitsha=gitsha)
    started = runs.utc_now()
    results: dict[str, Any] = {}
    for eid in open_ids:
        if reg[eid].spec.get("suite"):  # a suite registered for its pre-registration (e.g. distractors)
            try:
                r = _run_special(reg[eid].spec["suite"], cfg, out=out, args=args, progress=None, now=now, gitsha=gitsha)
                results[eid] = {"run_dir": r.run_dir.name, "decision": "서술(판정 없음)"}
            except SuiteError as e:
                results[eid] = {"error": str(e)}
            continue
        try:
            r = experiments.run_experiment(eid, cfg, args=args, runs_root=out, now=now, gitsha=gitsha)
            results[eid] = {"run_dir": r.run_dir.name, "decision": r.summary.get("decision_ko")}
        except (experiments.ExperimentStateError, experiments.ExperimentRunError) as e:
            results[eid] = {"error": str(e)}
    summary = {"suite": "components", "experiments": results}
    runs.write_summary(run_dir / "summary.json", summary)
    runs.write_metrics_csv(run_dir / "metrics.csv", [])
    runs.write_run_json(run_dir / "run.json", {
        "suite": "components", "system": "components", "args": args or {}, "git_sha": run_dir.name.split("_")[1],
        "started_utc": started, "finished_utc": runs.utc_now(), **runs.provenance()})
    return SuiteRun(run_dir, summary)


def coverage_points(details: Sequence[dict[str, Any]]) -> list[tuple[float, float, float]]:
    """Mean coverage/accuracy per τ over items (for the report curve)."""
    by: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for d in details:
        for tau, cov, acc in d["coverage"]:
            if not (math.isnan(cov) or math.isnan(acc)):
                by[tau].append((cov, acc))
    return [(t, float(np.mean([c for c, _ in v])), float(np.mean([a for _, a in v]))) for t, v in sorted(by.items())]
