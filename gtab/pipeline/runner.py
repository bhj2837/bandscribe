"""``gtab run``: plan, run and publish the stage graph for one job (M1_M2_SPEC 4; DESIGN 9.3).

- :func:`plan` evaluates every selected stage's ``models(cfg)`` **first** and reports all missing models at
  once (:class:`ModelsMissing`, Korean message naming ``gtab models fetch ...``) before anything runs, so
  nothing ever downloads during a run and stage keys cannot change halfway (M1_M2_SPEC 0, A19). Keys are
  computed exactly like M0's ``run_dag`` (``JobStore.stage_key`` over name, code version, params, dep keys +
  input hashes, model shas).
- :func:`run` then hands the stages to M0's ``run_dag`` (cache, ``force`` cascade, atomic stage dirs) with each
  stage's ``run`` wrapped to report progress and to expose the callback as ``ctx.progress`` (the SEP/AMT/GRID
  stages send ``vram_wait`` notices through it). Cached GPU stages whose ``gpu_run.json`` says ``degraded``
  (made on a lower ladder rung) produce a warning suggesting ``--force``; the rung is provenance, not key
  (M1_M2_SPEC 3.1, A16). A run that includes ``notes`` is published into ``export/``.
- :func:`data_outputs` is the one implementation of "data outputs" for determinism checks: every file in a
  stage dir except ``manifest.json`` and ``_worker/**`` (M1_M2_SPEC 0).

Progress events: ``("stage_start"|"stage_done"|"stage_cached"|"stage_degraded_cache"|"vram_wait"|
"model_missing", info)``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gtab import atomic
from gtab.jobs.dag import Stage, run_dag, topo_order
from gtab.jobs.store import MANIFEST, JobStore, canonical_json
from gtab.pipeline import graph

log = logging.getLogger(__name__)

Progress = Callable[[str, dict], None]
WORKER_DIR = "_worker"


@dataclass
class RunPlanItem:
    stage: str
    key12: str
    cached: bool
    device: str
    key: str = ""
    forced: bool = False
    missing_models: list[str] = field(default_factory=list)
    stage_dir: Path | None = None


class JobInputMissing(RuntimeError):
    pass


class ModelsMissing(RuntimeError):
    """One or more stage models are not on disk. ``errors`` = the individual ``gtab.models.ModelMissing``."""

    def __init__(self, errors: list[tuple[str, BaseException]]) -> None:
        self.errors = errors
        self.names = [getattr(e, "name", None) or str(e) for _, e in errors]
        lines = [f"필요한 모델이 없습니다 ({len(errors)}개). 실행 전에 받아 두세요:"]
        for stage, e in errors:
            hint = getattr(e, "hint_ko", None) or str(e)
            lines.append(f"  - {stage}: {hint}")
        super().__init__("\n".join(lines))


def _is_model_missing(e: BaseException) -> bool:
    from gtab.models import ModelMissing

    return isinstance(e, ModelMissing)


def data_outputs(stage_dir: Path) -> dict[str, str]:
    """rel path -> sha256 of every data output (everything but ``manifest.json`` and ``_worker/**``)."""
    root = Path(stage_dir)
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == MANIFEST or rel == WORKER_DIR or rel.startswith(WORKER_DIR + "/"):
            continue
        out[rel] = atomic.sha256_file(p)
    return out


def input_hashes(meta: dict[str, Any]) -> dict[str, str]:
    """The job-input part of every stage key (M1_M2_SPEC 3.1)."""
    try:
        pcm = meta["decode"]["pcm_sha256"]
    except (KeyError, TypeError) as e:
        raise JobInputMissing("작업 입력 정보(meta.json)에 decode.pcm_sha256 이 없습니다. 'gtab ingest' 로 다시 "
                              "처리하세요.") from e
    return {"input.mix": pcm,
            "input.ingest": atomic.sha256_bytes(canonical_json(meta.get("params") or {}).encode("utf-8"))}


def load_meta(store: JobStore, song_key: str) -> dict[str, Any]:
    path = store.input_dir(song_key) / "meta.json"
    if not path.is_file():
        raise JobInputMissing(f"작업 '{song_key}' 의 입력이 없습니다({path}). 'gtab ingest' 부터 실행하세요.")
    return atomic.read_json(path)


def resolve_job(source: str, store: JobStore, cfg: Any) -> Any:
    """``IngestResult`` for a file / URL (ingested, cache-aware) or an existing job key (prefix ok)."""
    from gtab import ingest as ingest_mod

    text = str(source).strip()
    if ingest_mod.is_url(text) or Path(text).is_file():
        return ingest_mod.ingest(text, store=store, cfg=cfg)
    if not text or any(ch in text for ch in "/\\:") or text.startswith("."):
        raise FileNotFoundError(f"파일이나 작업을 찾을 수 없습니다: {source}")
    jobs = store.list_jobs()
    matches = [j for j in jobs if j == text] or [j for j in jobs if j.startswith(text)]
    if not matches:
        raise FileNotFoundError(f"파일이나 작업을 찾을 수 없습니다: {source} (작업 목록: gtab status)")
    if len(matches) > 1:
        raise ValueError(f"'{text}'(으)로 시작하는 작업이 여러 개입니다: {', '.join(matches)}")
    song_key = matches[0]
    meta = load_meta(store, song_key)
    return ingest_mod.IngestResult(song_key, store.job_dir(song_key), meta, True)


def _emit(progress: Progress | None, event: str, info: dict[str, Any]) -> None:
    if progress is None:
        return
    try:
        progress(event, info)
    except Exception as e:  # display problems never fail a run
        log.debug("progress callback failed: %s", e)


def _prepare(song_key: str, cfg: Any, until: str, force: Iterable[str], store: JobStore, *, missing_ok: bool,
             progress: Progress | None = None) -> tuple[list[Stage], dict[str, str], list[RunPlanItem]]:
    graph.check_name(until, "--until 단계")
    force = set(force)
    for f in force:
        graph.check_name(f, "--force/--from 단계")
    meta = load_meta(store, song_key)
    inputs = input_hashes(meta)
    stages = graph.build_stages(cfg, until)
    impls = graph.implementations(s.name for s in stages)
    errors: list[tuple[str, BaseException]] = []
    missing: dict[str, list[str]] = {}
    for st in stages:
        fn = impls[st.name].get("models")
        if fn is None:
            continue
        try:
            st.models = dict(fn(cfg))
        except Exception as e:
            if not _is_model_missing(e):
                raise
            errors.append((st.name, e))
            missing.setdefault(st.name, []).append(getattr(e, "name", None) or str(e))
            _emit(progress, "model_missing", {"stage": st.name, "message": getattr(e, "hint_ko", None) or str(e)})
    if errors and not missing_ok:
        raise ModelsMissing(errors)

    by_name = {s.name: s for s in stages}
    forced = graph.descendants(force) if force else set()
    keys: dict[str, str | None] = {}
    items: list[RunPlanItem] = []
    for name in topo_order(stages):
        st = by_name[name]
        if name in missing or any(keys.get(d) is None for d in st.deps):
            keys[name] = None
            items.append(RunPlanItem(name, "", False, st.device, missing_models=missing.get(name, [])))
            continue
        dep_keys = {d: keys[d] for d in st.deps}
        key = JobStore.stage_key(name, st.code_version, st.params, dep_keys | inputs, st.models)
        keys[name] = key
        complete = store.is_complete(song_key, name, key)
        items.append(RunPlanItem(name, key[:12], complete and name not in forced, st.device, key=key,
                                 forced=name in forced, stage_dir=store.stage_dir(song_key, name, key)))
    return stages, inputs, items


def plan(song_key: str, cfg: Any, *, until: str = graph.DEFAULT_UNTIL, force: Iterable[str] = (),
         store: JobStore | None = None, missing_ok: bool = False) -> list[RunPlanItem]:
    """Stage table (DAG order) without running anything. Raises ModelsMissing unless ``missing_ok``."""
    store = store or JobStore()
    return _prepare(song_key, cfg, until, force, store, missing_ok=missing_ok)[2]


def degraded_message(stage: str, gpu_run: dict[str, Any], song: str) -> str:
    label = (gpu_run.get("rung") or {}).get("label") or "?"
    return (f"{stage}: 낮은 칸({label})으로 만든 캐시입니다. 여유가 있을 때 "
            f"'gtab run {song} --force {stage}' 로 다시 만드세요.")


def _wrap(st: Stage, progress: Progress | None) -> None:
    inner = st.run

    def run(ctx: Any) -> None:
        if progress is not None:
            try:
                ctx.progress = progress  # the stages' notifiers look here (vram_wait)
            except AttributeError:
                pass
        _emit(progress, "stage_start", {"stage": st.name, "key12": ctx.key[:12], "device": st.device})
        t0 = time.monotonic()
        inner(ctx)
        _emit(progress, "stage_done", {"stage": st.name, "key12": ctx.key[:12], "device": st.device,
                                       "duration_s": round(time.monotonic() - t0, 3)})

    st.run = run


def run(song_key: str, cfg: Any, *, until: str = graph.DEFAULT_UNTIL, force: Iterable[str] = (),
        progress: Progress | None = None, store: JobStore | None = None, publish: bool = True) -> dict[str, Path]:
    """Run (or reuse) every stage ``until`` needs; publish ``export/`` when ``notes`` is included.

    Returns {stage: final stage dir}. Raises ModelsMissing before running anything if a model is absent.
    """
    from gtab import vram

    store = store or JobStore()
    force = set(force)
    stages, inputs, items = _prepare(song_key, cfg, until, force, store, missing_ok=False, progress=progress)
    for it in items:
        if not it.cached:
            continue
        _emit(progress, "stage_cached", {"stage": it.stage, "key12": it.key12, "device": it.device})
        if it.device == "gpu" and it.stage_dir is not None:
            gr = vram.read_gpu_run(it.stage_dir)
            if gr and gr.get("degraded"):
                msg = degraded_message(it.stage, gr, song_key)
                log.warning(msg)
                _emit(progress, "stage_degraded_cache", {"stage": it.stage, "key12": it.key12,
                                                         "rung": gr.get("rung"), "message": msg})
    for st in stages:
        _wrap(st, progress)
    results = run_dag(stages, song_key, store, cfg, until=until, force=force, input_hashes=inputs)
    if publish and "notes" in results:
        from gtab.pipeline import publish as publish_mod

        publish_mod.publish(store.job_dir(song_key), results)
    return results


def stage_summary(results: dict[str, Path], items: list[RunPlanItem] | None = None) -> list[dict[str, Any]]:
    """Per stage: key12, cached/ran, wall time (from the manifest), device and GPU rung (gpu_run.json)."""
    from gtab import vram

    cached = {it.stage: it.cached for it in items or []}
    rows = []
    for stage, d in results.items():
        d = Path(d)
        row: dict[str, Any] = {"stage": stage, "key12": d.name, "cached": cached.get(stage), "duration_s": None,
                               "device": None, "rung": None, "degraded": False}
        try:
            m = atomic.read_json(d / MANIFEST)
            row["duration_s"] = m.get("duration_s")
            row["device"] = m.get("device")
        except (OSError, ValueError):
            pass
        gr = vram.read_gpu_run(d) if row["device"] == "gpu" else None
        if gr:
            row["rung"] = (gr.get("rung") or {}).get("label")
            row["degraded"] = bool(gr.get("degraded"))
        rows.append(row)
    return rows


def run_warnings(results: dict[str, Path], *, exclude_stages: Iterable[str] = ()) -> list[str]:
    """Korean warnings worth showing after a run, also for stages that came from the cache: grid repairs
    (``grid.json`` hypotheses), leftover energy (``stems``), a skipped Basic Pitch pass (``amt_bp``
    ``skipped.json``) and instrumentation warnings (``instr``). ``exclude_stages``: stages whose warnings were
    already shown live during this run (``warning`` progress events)."""
    skip = set(exclude_stages)
    out: list[str] = []

    def doc(stage: str, name: str) -> dict[str, Any] | None:
        d = results.get(stage)
        if stage in skip or d is None or not (Path(d) / name).is_file():
            return None
        try:
            v = atomic.read_json(Path(d) / name)
        except (OSError, ValueError):
            return None
        return v if isinstance(v, dict) else None

    grid = doc("grid", "grid.json")
    if grid is not None:
        hyp = grid.get("hypotheses") or {}
        out += [f"박자 격자: {w}" for w in hyp.get("warnings") or [] if isinstance(w, str)]
    st = doc("stems", "stem_stats.json")
    if st is not None:
        lo = st.get("leftover") or {}
        if lo.get("flag"):
            worst = ", ".join(f"{w.get('start_s', 0):.0f}s({w.get('rel_db', 0):.1f} dB)"
                              for w in (lo.get("worst_windows") or [])[:5])
            out.append(f"leftover(분리 잔여) 에너지가 믹스 대비 {lo.get('rms_db_rel_mix', 0):.1f} dB 로 큽니다. "
                       f"구간: {worst or '-'}")
    bp = doc("amt_bp", "skipped.json")
    if bp is not None:
        out.append(f"Basic Pitch 보조 전사를 건너뛰었습니다({bp.get('reason') or '-'}). "
                   "D:\\gtab\\envs\\bp310 에서 `D:\\gtab\\tools\\uvw.cmd sync` 로 환경을 만들 수 있습니다.")
    ins = doc("instr", "instrumentation.json")
    if ins is not None:
        out += [f"편성: {w}" for w in ins.get("warnings") or [] if isinstance(w, str)]
    return out
