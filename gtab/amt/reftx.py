"""Reference transcription of Tier B true solo tracks ("Tier B 참 단독 트랙의 MuScriptor 참조 전사", DESIGN 8.2, M2).

For every guitar / bass line of a multitrack, the line's true solo takes are summed and down-mixed to mono and
transcribed by MuScriptor (guitar lines: the three guitar classes; bass lines: bass classes). Comparing a
separated-stem transcription with this reference isolates what separation and routing cost from what the
transcriber itself gets wrong (the reference is MuScriptor-derived, so it is valid for comparing inputs and
settings, never for absolute accuracy claims).

The rung is pinned (``force_rung`` = the configured top rung, e.g. ``medium``) and is part of the cache key
(``data/eval/reftx_cache``), so a reference made on a degraded rung never stands in for the top rung.
Generated names are ASCII (``gtab.eval.runs.slug``). ``out_dir`` is the item's own folder (``gtab eval reftx`` passes
``data/eval/tierB/<dataset>/<track>``, experiments their per-item work dir). Outputs: ``<slug(line)>_mono.wav`` and
``<slug(line)>__reftx.json`` (raw NoteSet; latency is not applied to references), plus ``_worker/`` provenance.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gtab import atomic
from gtab.amt import cache as amt_cache
from gtab.amt import instruments as I
from gtab.audio import write_f32

log = logging.getLogger(__name__)

SR = 44100


def _slug(s: str) -> str:
    from gtab.eval.runs import slug

    return slug(s)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def write_mono_f32(path: Path, x: np.ndarray, sr: int) -> str:
    """Write a mono float32 WAV (deterministic writer); returns its PCM sha256.

    The hash is taken over the same little-endian float32 sample bytes ``gtab.audio.pcm_sha256`` reads back, so
    it equals the view hash of the written file without a second pass over it.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype="<f4").reshape(-1, 1))
    write_f32(path, x, int(sr))
    return hashlib.sha256(x.tobytes()).hexdigest()


def line_audio(track: Any, line_id: str, root: Path) -> tuple[np.ndarray, int] | None:
    """Sum of the line's true solo parts, mono (L+R)/2, float32; None if the line has no audio part.

    A ``di`` part is left out when the line also has an amped/miked part, exactly like the Tier B remix
    (``gtab.eval.remix.mix_parts``): the reference must transcribe the same audio the mix contains, and a clean
    DI under the amp track would double every note's attack.
    """
    import soundfile as sf

    parts = [p for p in (_get(track, "parts") or []) if _get(p, "line") == line_id
             and _get(p, "kind") in ("guitar", "bass", "di")]
    if any(_get(p, "kind") != "di" for p in parts):
        parts = [p for p in parts if _get(p, "kind") != "di"]
    if not parts:
        return None
    acc: np.ndarray | None = None
    sr0 = None
    for p in parts:
        x, sr = sf.read(str(Path(root) / _get(p, "audio")), dtype="float32", always_2d=True)
        mono = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        if sr0 is None:
            sr0, acc = sr, mono.astype(np.float32)
        else:
            if sr != sr0:
                raise ValueError(f"parts of line {line_id} have different sample rates ({sr} vs {sr0})")
            n = max(len(acc), len(mono))
            acc = np.pad(acc, (0, n - len(acc))) + np.pad(mono, (0, n - len(mono)))
    return acc.astype(np.float32), int(sr0)


def line_mask(kind: str) -> list[str]:
    return I.mask_bass() if kind == "bass" else I.mask_guitar_only()


def reference_transcribe(track: Any, out_dir: Path, cfg: Any, *, force_rung: str | None = None,
                         cache_root: Path | None = None, root: Path | None = None,
                         transcribe: Any = None) -> dict[str, dict[str, Any]]:
    """{line id: raw NoteSet dict} for every guitar/bass line of a Tier B ``DatasetTrack``.

    ``transcribe`` (tests) replaces ``gtab.amt.stage.transcribe_views_ms``; ``root`` the dataset root.
    """
    from gtab.amt import stage as S

    if root is None:
        from gtab import datasets

        root = datasets.dataset_root(str(_get(track, "dataset")))
    params = S.ms_params(cfg)
    labels, _model = S.ms_ladder(params)
    rung = force_rung or labels[0]
    item = f"{_get(track, 'dataset')}:{_get(track, 'track_id')}"
    item_dir = Path(out_dir)
    views = []
    for line in _get(track, "lines") or []:
        kind = str(_get(line, "kind"))
        if kind not in ("guitar", "bass"):
            continue
        lid = str(_get(line, "id"))
        audio = line_audio(track, lid, Path(root))
        if audio is None:
            log.info("reftx: %s line %s has no solo audio part; skipped", item, lid)
            continue
        x, sr = audio
        wav = item_dir / f"{_slug(lid)}_mono.wav"
        sha = write_mono_f32(wav, x, sr)
        views.append({"id": lid, "wav": wav, "pcm_sha256": sha, "instruments": line_mask(kind)})
    if not views:
        return {}
    cache = amt_cache.AmtCache(cache_root or amt_cache.eval_cache_root())
    fn = S.transcribe_views_ms if transcribe is None else transcribe
    raws, _info = fn(views, params, cfg, work_dir=item_dir / "_worker", cache=cache, job=None, force_rung=rung)
    out: dict[str, dict[str, Any]] = {}
    for v in views:
        doc = raws[v["id"]]
        atomic.write_json(item_dir / f"{_slug(v['id'])}__reftx.json", doc)
        out[v["id"]] = doc
    return out
