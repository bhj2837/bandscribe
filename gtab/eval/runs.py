"""Evaluation run directories, the ASCII ``slug()`` for every generated path, and deterministic metric files.

Layout (DESIGN §8.7, spec §6.10)::

    data/runs/<YYYYMMDD-HHMMSS>_<gitsha7>_<slug(system)>/
        run.json       suite, system, config, model/dataset shas, git sha, timestamps (the ONLY file with them)
        metrics.csv    long format, sorted, "\\n" line ends, fixed float formatting -> byte-identical reruns
        summary.json   aggregates + CIs + baselines + MDE checks (deterministic)
        report.html    human report
        worst_bars.csv

Why ``slug``: ids such as ``job:amt_gtr`` or ``guitarset:00_BN1-129-Eb_comp`` keep their spelling inside JSON,
but ``a:b`` is illegal in NTFS names (it even creates an alternate data stream) and non-ASCII paths break
several Windows tools (M0 rule). So every generated file/folder name goes through ``slug``.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gtab import atomic, paths

log = logging.getLogger(__name__)

METRIC_COLUMNS: tuple[str, ...] = (
    "suite", "system", "dataset", "item", "group", "scenario", "section", "line", "rung",
    "metric", "value", "tp", "fp", "fn", "n", "ci_low", "ci_high", "note",
)
# Sort order of metrics.csv (spec §6.10); the remaining columns break ties so the order is total.
SORT_COLUMNS: tuple[str, ...] = ("suite", "dataset", "item", "scenario", "section", "line", "metric", "rung")

_SLUG_KEEP = re.compile(r"[a-z0-9_-]")


def slug(s: str) -> str:
    """ASCII file-name form of an id: lower-case, ``[a-z0-9_-]`` kept, ``:`` -> ``-``, anything else -> ``_``.

    Runs of ``_`` / ``-`` collapse to one character and ``_`` is trimmed from both ends, so an id made only
    of other characters (e.g. Korean ``기타``) is empty and raises ValueError: user-facing ids are ASCII.
    ``job:amt_gtr`` -> ``job-amt_gtr``; ``guitarset:00_BN1`` -> ``guitarset-00_bn1``.
    """
    if not isinstance(s, str):
        raise TypeError(f"slug() needs a str, got {type(s).__name__}")
    out: list[str] = []
    for ch in s.lower():
        if _SLUG_KEEP.fullmatch(ch):
            out.append(ch)
        elif ch == ":":
            out.append("-")
        else:
            out.append("_")
    text = re.sub(r"_+", "_", re.sub(r"-+", "-", "".join(out))).strip("_")
    if not text or text == "-":
        raise ValueError(f"id {s!r} 에서 ASCII 파일 이름을 만들 수 없습니다 (영문·숫자·_·- 로 된 이름을 쓰세요)")
    return text


def git_sha7(root: Path | None = None) -> str:
    """Short git sha of the working tree (``0000000`` when git or the repo is unavailable)."""
    try:
        from gtab.winjob import run_captured

        r = run_captured(["git", "rev-parse", "--short=7", "HEAD"], timeout_s=30,
                         cwd=str(root or paths.GTAB_ROOT), env=paths.child_env())
    except OSError:
        return "0000000"
    sha = r.stdout.strip()
    return sha if r.returncode == 0 and re.fullmatch(r"[0-9a-f]{7,40}", sha) else "0000000"


def new_run_dir(system: str, *, root: Path | None = None, now: dt.datetime | None = None,
                gitsha: str | None = None) -> Path:
    """Create ``<root>/<YYYYMMDD-HHMMSS>_<gitsha7>_<slug(system)>`` (``-2``, ``-3`` … on a same-second clash)."""
    base = Path(root) if root is not None else paths.RUNS
    stamp = (now or dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    sha = (gitsha or git_sha7())[:7]
    name = f"{stamp}_{sha}_{slug(system)}"
    base.mkdir(parents=True, exist_ok=True)
    for i in range(1, 1000):
        cand = base / (name if i == 1 else f"{name}-{i}")
        try:
            cand.mkdir()
            return cand
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot create a run dir under {base}")


# ------------------------------------------------------------------------------------------ formatting


def fmt_value(v: Any) -> str:
    """Stable text for one csv cell: floats with at most 6 decimals, no ``-0``, ``None``/NaN -> empty."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float) or hasattr(v, "__float__") and not isinstance(v, str):
        f = float(v)
        if not math.isfinite(f):
            return ""
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
        text = f"{round(f, 6):.6f}".rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text
    return str(v)


def normalize_row(row: Mapping[str, Any]) -> dict[str, str]:
    unknown = set(row) - set(METRIC_COLUMNS)
    if unknown:
        raise ValueError(f"unknown metrics.csv columns: {sorted(unknown)}")
    if not row.get("metric"):
        raise ValueError("a metrics row needs 'metric'")
    return {c: fmt_value(row.get(c)) for c in METRIC_COLUMNS}


def _sort_key(r: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(r[c] for c in SORT_COLUMNS) + tuple(r[c] for c in METRIC_COLUMNS if c not in SORT_COLUMNS)


def metrics_csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    norm = sorted((normalize_row(r) for r in rows), key=_sort_key)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(METRIC_COLUMNS), lineterminator="\n")
    w.writeheader()
    w.writerows(norm)
    return buf.getvalue()


def write_metrics_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic.write_bytes(Path(path), metrics_csv_text(rows).encode("utf-8"))


def read_metrics_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def round_floats(obj: Any, nd: int = 6) -> Any:
    """Recursively round floats (JSON outputs must be byte-stable; NaN/inf become None)."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        r = round(obj, nd)
        return 0.0 if r == 0 else r
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        return round_floats(obj.item(), nd)
    if isinstance(obj, Mapping):
        return {str(k): round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, nd) for v in obj]
    return obj


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    atomic.write_json(Path(path), round_floats(summary))


def write_run_json(path: Path, info: Mapping[str, Any]) -> None:
    """run.json is the one file allowed to hold timestamps and machine details."""
    atomic.write_json(Path(path), dict(info))


def _file_sha(p: Path) -> str | None:
    return atomic.sha256_file(p) if Path(p).is_file() else None


def provenance() -> dict[str, Any]:
    """Model and dataset provenance for run.json (spec §6.10: "model shas, dataset lock shas").

    ``models`` = name -> sha256 as recorded in models.lock.json (recorded values, not recomputed: hashing
    every checkpoint on each run would cost minutes; ``gtab models verify`` checks the files). Never raises:
    a broken registry is reported in the dict instead of failing the evaluation.
    """
    out: dict[str, Any] = {"models_lock_sha256": _file_sha(paths.MODELS_LOCK),
                           "datasets_lock_sha256": _file_sha(paths.DATASETS / "datasets.lock.json")}
    try:
        from gtab import models

        out["models"] = {name: e.sha256 for name, e in sorted(models.entries().items())}
    except Exception as e:  # noqa: BLE001 - provenance must not break a run
        out["models"] = None
        out["models_error"] = str(e)
    return out


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
