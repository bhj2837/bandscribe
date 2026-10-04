"""SEP's registered experiment: ``stereo-preservation`` (M1_M2_SPEC 10; M2 b7 + b8). Descriptive, no decision.

``EXPERIMENTS[id](out_dir: Path, cfg, args: dict) -> list[dict]`` (metrics.csv rows, ``rung`` filled) is called by
``bandscribe eval exp`` (EVAL), which writes the run directory and the decisions.md entry.

1. **Synthetic** (b8): ``bandscribe.eval.remix.make_scene(scenario, seed, full_band=True)`` scenes go through SW at the
   top rung (``force_rung``; never a ``bandscribe run`` cache), and the true guitar-only stem (sum of the scene's
   guitar takes) is compared with SW's guitar stem by ``bandscribe.sep.checks.stereo_preservation``: energy-weighted
   mean |Δcoherence|, |ΔILD| per bin and the router's window-feature deltas.
2. **Real stems** (b7): for the ``sep`` stage of the given jobs (``args["jobs"]``; default: the newest finished
   one), every stem is archived to 24-bit FLAC and restored; the error is reported relative to the mix RMS
   (``skipped_sparse`` below -60 dB rel. mix).

args: ``scenarios`` (default ``A, P70, B, twin``), ``seeds`` (default ``[0]``), ``dur_s`` (default 8),
``jobs`` (song keys), ``force_rung`` (default: top rung of ``sep.chunk_ladder``), ``dry_run``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic
from bandscribe.sep import archive, checks
from bandscribe.sep import run as seprun

log = logging.getLogger(__name__)

EXPERIMENT_ID = "stereo-preservation"
SUITE = "components"
SYSTEM = "sep:sw_msst"
DEFAULT_SCENARIOS = ("A", "P70", "B", "twin")
WINDOW_FEATURES = ("SM", "r0", "xcoff", "lag", "coh", "cen", "hard", "fcoh", "ildcoh")


def _row(**kw: Any) -> dict[str, Any]:
    base = {"suite": SUITE, "system": SYSTEM, "line": "guitar", "section": "", "note": ""}
    base.update(kw)
    return base


def _latest_sep_stages(limit: int = 1) -> list[tuple[str, Path]]:
    """(song_key, sep stage dir) of the newest complete ``sep`` stages in the job store."""
    from bandscribe import paths

    found: list[tuple[str, str, Path]] = []
    root = Path(paths.JOBS)
    if not root.is_dir():
        return []
    for manifest in root.glob("*/stages/sep/*/manifest.json"):
        try:
            m = atomic.read_json(manifest)
        except (OSError, ValueError):
            continue
        if m.get("status") == "complete":
            found.append((str(m.get("created_utc", "")), manifest.parents[3].name, manifest.parent))
    found.sort(reverse=True)
    return [(k, d) for _t, k, d in found[:limit]]


def _job_sep_dir(song_key: str) -> Path | None:
    from bandscribe import paths

    cands = []
    for manifest in (Path(paths.JOBS) / song_key / "stages" / "sep").glob("*/manifest.json"):
        try:
            m = atomic.read_json(manifest)
        except (OSError, ValueError):
            continue
        if m.get("status") == "complete":
            cands.append((str(m.get("created_utc", "")), manifest.parent))
    return max(cands)[1] if cands else None


def synthetic_rows(out_dir: Path, cfg: Any, *, scenarios: list[str], seeds: list[int], dur_s: float,
                   force_rung: str, separate: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    import soundfile as sf

    from bandscribe.audio import write_f32
    from bandscribe.eval import remix
    from bandscribe.eval.runs import slug

    separate = separate or seprun.separate
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            scene = remix.make_scene(scenario, seed, dur_s=dur_s, full_band=True)
            sdir = Path(out_dir) / "scenes" / slug(scene.item_id)
            sdir.mkdir(parents=True, exist_ok=True)
            mix_wav = sdir / "mix.wav"
            write_f32(mix_wav, np.ascontiguousarray(scene.mix.T, dtype=np.float32), scene.sr)
            res = separate(mix_wav, sdir / "sep", cfg, force_rung=force_rung)
            est, sr = sf.read(str(res.stems["guitar"]), dtype="float32", always_2d=True)
            sp = checks.stereo_preservation(scene.guitar_stereo(), est, sr)
            atomic.write_json(sdir / "stereo_preservation.json", {"item": scene.item_id, "rung": res.rung, **sp})
            rung = str(res.rung.get("label") or "")
            common = {"dataset": "synthetic", "item": scene.item_id, "group": scenario, "scenario": scenario,
                      "rung": rung, "n": sp["samples"]}
            rows.append(_row(metric="stereo_delta_coh", value=sp["delta_coh"], **common))
            rows.append(_row(metric="stereo_delta_ild_db", value=sp["delta_ild_db"], **common))
            rows.append(_row(metric="stereo_energy_ratio_db", value=sp["energy_ratio_db"], **common))
            for k in WINDOW_FEATURES:
                v = sp["window_feature_deltas"].get(k)
                rows.append(_row(metric=f"stereo_wf_delta_{k}", value=v, **common,
                                 note="" if v is not None else "undefined (NaN in one of the stems)"))
    return rows


def archive_rows(jobs: list[tuple[str, Path]], work_dir: Path) -> list[dict[str, Any]]:
    from bandscribe import paths, vram

    rows: list[dict[str, Any]] = []
    for song_key, sep_dir in jobs:
        mix = Path(paths.JOBS) / song_key / "input" / "mix_44k_f32.wav"
        if not mix.is_file():
            log.warning("job %s has no mix; skipped", song_key)
            continue
        mix_db = archive.rms_db(mix)
        gpu_run = vram.read_gpu_run(sep_dir) or {}
        rung = str((gpu_run.get("rung") or {}).get("label") or "")
        for wav in sorted((sep_dir / "stems").glob("*.wav")):
            chk = archive.roundtrip_check(wav, mix_rms_db=mix_db, work_dir=work_dir)
            rows.append(_row(dataset="job", item=song_key, group=song_key, scenario="real", line=wav.stem,
                             rung=rung, metric="archive_error_db_rel_mix", value=chk["error_db"],
                             note=f"{chk['status']} (stem {chk['rel_rms_db']:+.1f} dB rel. mix)"))
    return rows


def run_stereo_preservation(out_dir: Path, cfg: Any, args: dict | None = None) -> list[dict[str, Any]]:
    args = dict(args or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [str(s) for s in args.get("scenarios") or DEFAULT_SCENARIOS]
    seeds = [int(s) for s in args.get("seeds") or [0]]
    dur_s = float(args.get("dur_s", 8.0))
    labels = seprun.rung_labels(list(seprun.sep_value(cfg, "sep.chunk_ladder")))
    force = str(args.get("force_rung") or labels[0])
    jobs_arg = args.get("jobs")
    if jobs_arg:
        jobs = [(k, d) for k in jobs_arg if (d := _job_sep_dir(str(k))) is not None]
    else:
        jobs = _latest_sep_stages(1)
    plan = {"experiment": EXPERIMENT_ID, "scenarios": scenarios, "seeds": seeds, "dur_s": dur_s,
            "force_rung": force, "jobs": [k for k, _ in jobs]}
    atomic.write_json(out_dir / "plan.json", plan)
    if args.get("dry_run"):
        log.info("dry run: %s", plan)
        return []
    rows = synthetic_rows(out_dir, cfg, scenarios=scenarios, seeds=seeds, dur_s=dur_s, force_rung=force)
    if jobs:
        rows += archive_rows(jobs, out_dir)
    else:
        log.warning("no finished sep stage found: real-stem archive check (b7) skipped")
    return rows


EXPERIMENTS: dict[str, Callable[[Path, Any, dict], list[dict]]] = {EXPERIMENT_ID: run_stereo_preservation}
