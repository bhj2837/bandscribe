"""MuScriptor 0.3.0 transcription worker (gpu env) - M1_M2_SPEC 7.3, DESIGN S3.5 / S6.

Started by ``gtab.vram.run_gpu_stage`` as ``<gpu venv>\\Scripts\\python.exe -m gtab.workers.amt_muscriptor
request.json result.json``. One model load per worker; all views of the pass are transcribed with it.

Installed-API findings (muscriptor 0.3.0, checked against the wheel and on this PC 2026-09-30; ``probe`` mode
re-reports them from the installed package without loading weights):

- ``TranscriptionModel.load_model(weights_path=None, device=None, dtype=None)``. ``weights_path`` may be a size
  keyword (resolved to ``hf://MuScriptor/muscriptor-<size>`` -> **download**), a URL, or a local path (used as
  is). We only ever pass the local ``model.safetensors`` path, and set ``HF_HUB_OFFLINE=1`` before importing
  muscriptor, so nothing downloads under the GPU lock. The architecture is read from ``config.json`` next to the
  weights (present in the HF snapshot).
- **Device is fixed at build time.** ``_build_model(device)`` enables fp16 ``TorchAutocast`` only when the
  device is CUDA; weights stay fp32, conditioners (mel spectrogram / class embeddings) stay fp32. The model keeps
  its build device and moves audio, masks and prompts there, so CPU-build-then-``.to("cuda")`` fails with device
  mismatches (and would run plain fp32). Hence: GPU rungs build on CUDA, the CPU rung builds on CPU.
- The upstream loader calls ``safetensors.load_file(path, device="cuda")``: model + a transient second copy of
  the weights on the GPU (measured: load 10.3 s, peak 2343 MB allocated / 2354 MB reserved; transcribing 30 s:
  max reserved 2360 MB, 3.1x realtime). ``loader: "lean"`` builds on CUDA but loads the state dict on the CPU
  (private helpers ``_build_model`` / ``_resolve_config`` / ``_remap_single_codebook_keys``, strict load);
  pinned to 0.3.0 and gated by the parity test before it may become the default.
- ``transcribe(audio: str|Path|tuple[Tensor, int], use_sampling=False, temperature=1.0, cfg_coef=1.0,
  instruments=None, batch_size=None, no_eos_is_ok=True, beam_size=1, prelude_forcing=True)`` yields
  ``NoteStartEvent(pitch, start_time, index, instrument)``, ``NoteEndEvent(end_time, start_event)`` and
  ``ProgressEvent(completed, total)`` (the latter is defined in ``muscriptor.events``, not exported by the
  package root). Audio is split into 5 s chunks at 16 kHz.
- **Audio goes in as a tensor** ``[C, T]`` float32 (channels are mean-downmixed, resampled to 16 kHz inside):
  a ``.wav`` path would be read with the stdlib ``wave`` module, which cannot read float32 WAV ("unknown format:
  3"). We read our float32 WAVs with soundfile.
- ``instruments`` is a **hard mask** (every program token outside the listed MT3_FULL_PLUS groups is forbidden)
  **and** the conditioning string ``" ".join(str(group_id) ...)`` in the given order - so the list order changes
  the conditioning. The core sorts by MT3 id (``gtab.amt.instruments.sort_mt3``); we re-sort defensively.
- MT3_FULL_PLUS group table: 35 entries, ids 0-33 + drums 36 (= ``gtab.schema.instrumentation.MT3_GROUPS``).
- Times may carry a constant lag of up to ~25 ms (model bias, upstream docstring); raw NoteSets keep backend
  times (``time_offset_applied_s: 0``) and the core subtracts ``amt.latency_s``.
- **Segments** (S3.5 presence windows, 2026-10-04): a view may carry ``segments: [[start_s, end_s], ...]``. Each
  segment is transcribed by its own ``transcribe`` call (fresh decoder state: no prelude carried over from
  another part of the song), its note times are shifted back by the segment start, and the NoteSet records
  ``segments`` and the full view's ``audio_duration_s``. Progress, watchdog and realtime factor count only the
  transcribed audio.
- **bfloat16 is refused** (RTX 2060 = sm_75, no native bf16). ``dtype: "fp16"`` (E24-amt only) = fp16 weights
  except the conditioners, i.e. ``load_model(..., dtype=float16)`` which re-floats ``condition_provider``.

Ladder (M1_M2_SPEC 8.2): ``medium`` (or ``medium-lean``) -> ``small`` (only if its weights are already in the HF
cache: ``huggingface_hub.try_to_load_from_cache``, never a download; otherwise the rung is ``unavailable``)
-> ``medium@cpu`` only if ``vram.cpu_allowed``. After the model is resident: ``sampler.mark("model_loaded")`` and
``Watchdog.arm()``; ``ProgressEvent``s drive ``Watchdog.check``.

This module is importable in the core env (no torch at module level): ``assemble_notes`` and friends are pure
and unit-tested there. It never imports ``gtab.schema`` (M1_M2_SPEC 0); it writes plain ``gtab.notes/1`` JSON.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gtab import atomic
from gtab.workers import gpu_common
from gtab.workers.base import WorkerContext, worker_main

log = logging.getLogger(__name__)

BACKEND = "muscriptor"
PINNED_VERSION = "0.3.0"
MODEL_LOADED = "model_loaded"
DTYPES = ("upstream", "fp16")
LOADERS = ("upstream", "lean")
MINIMUM_NOTE_DURATION_SEC = 0.01  # = muscriptor.tokenizer.notes.MINIMUM_NOTE_DURATION_SEC (validate_notes rule)
SMALL_REPO = "MuScriptor/muscriptor-small"

# MT3_FULL_PLUS ids (duplicated from gtab.schema.instrumentation.MT3_GROUPS on purpose: workers never import
# gtab.schema). Used only to re-sort the requested mask; `probe` compares it with the installed table.
MT3_IDS: dict[str, int] = {
    "acoustic_piano": 0, "electric_piano": 1, "chromatic_percussion": 2, "organ": 3, "acoustic_guitar": 4,
    "clean_electric_guitar": 5, "distorted_electric_guitar": 6, "acoustic_bass": 7, "electric_bass": 8,
    "violin": 9, "viola": 10, "cello": 11, "contrabass": 12, "orchestral_harp": 13, "timpani": 14,
    "string_ensemble": 15, "synth_strings": 16, "voice": 17, "orchestra_hit": 18, "trumpet": 19, "trombone": 20,
    "tuba": 21, "french_horn": 22, "brass_section": 23, "soprano_and_alto_sax": 24, "tenor_sax": 25,
    "baritone_sax": 26, "oboe": 27, "english_horn": 28, "bassoon": 29, "clarinet": 30, "flutes": 31,
    "synth_lead": 32, "synth_pad": 33, "drums": 36}


# ------------------------------------------------------------------------------------ pure helpers


def sort_instruments(instruments: Iterable[str] | None) -> list[str] | None:
    """MT3-id order, duplicates removed; None stays None (= every class). Unknown names are an error."""
    if instruments is None:
        return None
    names = list(dict.fromkeys(str(n) for n in instruments))
    unknown = [n for n in names if n not in MT3_IDS]
    if unknown:
        raise ValueError(f"unknown MT3 instrument group(s): {unknown}")
    return sorted(names, key=lambda n: MT3_IDS[n])


def _r6(x: float) -> float:
    return round(float(x), 6)


def _kind(ev: Any) -> str:
    """Event kind by duck typing (tests feed plain objects; the worker feeds muscriptor dataclasses)."""
    if hasattr(ev, "completed") and hasattr(ev, "total"):
        return "progress"
    if hasattr(ev, "start_event") and hasattr(ev, "end_time"):
        return "end"
    if hasattr(ev, "start_time") and hasattr(ev, "pitch") and hasattr(ev, "index"):
        return "start"
    return "other"


def note_sort_key(n: dict[str, Any]) -> tuple:
    return (n["onset_s"], n["pitch"], n["offset_s"], n.get("instrument") or "")


def assemble_notes(events: Iterable[Any], audio_s: float, *, on_progress: Any = None) -> list[dict[str, Any]]:
    """NoteStart/NoteEnd/Progress events -> sorted NoteEvent dicts (raw backend times).

    Pairs ends with starts by ``index``. A start without an end (not expected from 0.3.0, which closes
    everything at end of stream, but cheap to guard) ends at ``audio_s``. Non-drum notes shorter than
    10 ms are stretched to 10 ms, as muscriptor's own ``validate_notes(fix=True)`` does for its MIDI output.
    ``on_progress(completed, total)`` is called for every ProgressEvent (drives the watchdog).
    """
    open_notes: dict[int, Any] = {}
    done: list[tuple[Any, float]] = []
    for ev in events:
        kind = _kind(ev)
        if kind == "progress":
            if on_progress is not None:
                on_progress(int(ev.completed), int(ev.total))
        elif kind == "start":
            open_notes[int(ev.index)] = ev
        elif kind == "end":
            start = ev.start_event
            open_notes.pop(int(start.index), None)
            done.append((start, float(ev.end_time)))
    for start in open_notes.values():
        done.append((start, max(float(audio_s), float(start.start_time))))
    notes: list[dict[str, Any]] = []
    for start, end in done:
        onset = max(0.0, float(start.start_time))
        instrument = str(start.instrument) if start.instrument is not None else None
        offset = max(end, onset)
        if instrument != "drums" and offset - onset < MINIMUM_NOTE_DURATION_SEC:
            offset = onset + MINIMUM_NOTE_DURATION_SEC
        notes.append({"onset_s": _r6(onset), "offset_s": _r6(offset), "pitch": int(start.pitch),
                      "instrument": instrument})
    notes.sort(key=note_sort_key)
    return notes


def segment_bounds(segments: Iterable[Iterable[float]] | None, sr: int, n_samples: int) -> list[tuple[int, int]]:
    """Sample ranges ``[(a, z), ...]`` of ``segments`` (seconds) in a view of ``n_samples``; None = whole view.

    Segments are clipped to the view; empty ones are dropped. They must be sorted and disjoint (the core sends
    them that way; checked here because the NoteSet schema requires it)."""
    if segments is None:
        return [(0, int(n_samples))]
    out: list[tuple[int, int]] = []
    prev = 0
    for seg in segments:
        a_s, z_s = (float(x) for x in seg)
        a = max(0, min(int(n_samples), int(round(a_s * sr))))
        z = max(0, min(int(n_samples), int(round(z_s * sr))))
        if a < prev:
            raise ValueError(f"segments must be sorted and disjoint: {list(segments)}")
        if z > a:
            out.append((a, z))
            prev = z
    return out


def offset_notes(notes: Iterable[dict[str, Any]], offset_s: float) -> list[dict[str, Any]]:
    """Notes of a segment moved to view time (``+ offset_s``), rounded like every raw time."""
    off = float(offset_s)
    out = []
    for n in notes:
        m = dict(n)
        m["onset_s"] = _r6(float(n["onset_s"]) + off)
        m["offset_s"] = _r6(max(float(n["offset_s"]) + off, m["onset_s"]))
        out.append(m)
    return out


def noteset_doc(*, view_id: str, view_sha256: str | None, backend: dict[str, Any], instruments: list[str] | None,
                audio_s: float, notes: list[dict[str, Any]],
                segments: list[tuple[float, float]] | None = None) -> dict[str, Any]:
    """A ``gtab.notes/1`` document (schema §6.1): raw times, fixed key order, no timestamps.

    ``segments`` (seconds, view time) is written only for a segmented view, so whole-view documents keep their
    M2 bytes."""
    be = dict(backend)
    be["instruments"] = None if instruments is None else list(instruments)
    doc = {"format": "gtab.notes/1", "view": view_id, "view_sha256": view_sha256, "backend": be,
           "time_offset_applied_s": 0.0, "audio_duration_s": _r6(audio_s)}
    if segments is not None:
        doc["segments"] = [[_r6(a), _r6(z)] for a, z in segments]
    doc["notes"] = notes
    return doc


def build_rungs(model_req: dict[str, Any], loader: str) -> list[dict[str, Any]]:
    """GPU rungs (largest first) + the CPU rung; run_ladder drops the CPU rung unless cpu_allowed."""
    medium_label = "medium-lean" if loader == "lean" else "medium"
    size = str(model_req.get("size", "medium"))
    rungs = [{"label": medium_label if size == "medium" else size, "device": "cuda",
              "params": {"size": size, "weights_path": model_req.get("weights_path"), "loader": loader,
                         "repo": model_req.get("repo")}}]
    for fb in model_req.get("fallback") or []:
        fsize = str(fb.get("size"))
        rungs.append({"label": fsize, "device": "cuda",
                      "params": {"size": fsize, "weights_path": fb.get("weights_path"), "loader": "upstream",
                                 "repo": fb.get("repo") or f"MuScriptor/muscriptor-{fsize}"}})
    rungs.append({"label": f"{size}@cpu", "device": "cpu",
                  "params": {"size": size, "weights_path": model_req.get("weights_path"), "loader": "upstream",
                             "repo": model_req.get("repo")}})
    return rungs


def rung_labels(model_req: dict[str, Any], loader: str) -> list[str]:
    """GPU rung labels in ladder order (the core passes the same list to run_gpu_stage / write_gpu_run)."""
    return [r["label"] for r in build_rungs(model_req, loader) if r["device"] != "cpu"]


# ----------------------------------------------------------------------------------------- worker


def _muscriptor_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("muscriptor")
    except Exception:
        return None


def _resolve_weights(params: dict[str, Any]) -> str | None:
    """Local weights path for a rung, or None if not available offline (never downloads)."""
    p = params.get("weights_path")
    if p:
        return str(p) if Path(p).is_file() else None
    repo = params.get("repo")
    if not repo:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(repo, "model.safetensors")
    except Exception as e:  # offline / odd cache: unavailable, never fatal
        log.info("HF cache lookup for %s failed: %s", repo, e)
        return None
    return found if isinstance(found, str) and Path(found).is_file() else None


def _sha256(path: Path) -> str:
    """sha256 via gtab.models' (path, size, mtime) memo (1.2 GB file; the core side already filled it)."""
    from gtab import models

    return models.sha256_cached(Path(path))


def _load(rung: dict[str, Any], weights: str, dtype: str) -> tuple[Any, str]:
    """Build a TranscriptionModel on the rung's device. Returns (model, loader actually used)."""
    import torch
    from muscriptor.transcription_model import TranscriptionModel

    dev = torch.device("cuda", 0) if rung["device"] == "cuda" else torch.device("cpu")
    tdtype = torch.float16 if dtype == "fp16" else None  # None = upstream per-device default (fp32 on CUDA/CPU)
    loader = rung["params"].get("loader", "upstream")
    if loader == "lean" and dev.type == "cuda":
        import safetensors.torch
        from muscriptor.tokenizer.mt3 import MT3Tokenizer
        from muscriptor.transcription_model import _build_model, _remap_single_codebook_keys, _resolve_config

        p = Path(weights)
        model = _build_model(dev, _resolve_config(p, p))
        model.eval()
        sd = safetensors.torch.load_file(str(p), device="cpu")
        sd = _remap_single_codebook_keys(sd)
        model.load_state_dict(sd, strict=True)
        del sd
        # Not everything is built on `dev` (the token embedding / output head are created on the CPU); upstream
        # load_model also ends with model.to(device). Moving a few CPU modules costs no second weight copy.
        model.to(dev)
        if tdtype is not None:
            model.to(tdtype)
            model.condition_provider.float()  # conditioners keep fp32 numerics (as load_model does)
        tm = TranscriptionModel(model, MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001), dev)
        return tm, "lean"
    return TranscriptionModel.load_model(Path(weights), device=dev, dtype=tdtype), "upstream"


def _read_view(wav: Path) -> tuple[Any, int, float]:
    import numpy as np
    import soundfile as sf
    import torch

    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)  # (n, ch)
    t = torch.from_numpy(np.ascontiguousarray(x.T))  # [C, T] float32, as TranscriptionModel expects
    return t, int(sr), x.shape[0] / float(sr)


def _probe() -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from muscriptor import events as mevents
    from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
    from muscriptor.transcription_model import TranscriptionModel

    groups = dict(MT3_FULL_PLUS_GROUP_NAMES)
    return {
        "status": "ok", "mode": "probe",
        "muscriptor_version": _muscriptor_version(),
        "transcribe_signature": str(inspect.signature(TranscriptionModel.transcribe)),
        "load_model_signature": str(inspect.signature(TranscriptionModel.load_model)),
        "progress_event_module": getattr(mevents, "ProgressEvent").__module__,
        "groups": groups,
        "n_groups": len(groups),
        "groups_match_gtab": groups == MT3_IDS,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }


def _transcribe(request: dict[str, Any], ctx: WorkerContext) -> dict[str, Any]:
    # Before muscriptor/huggingface_hub are imported: the hub reads this at import time.
    os.environ["HF_HUB_OFFLINE"] = "1"
    gpu_common.refuse_bf16(request)
    # Before torch touches CUDA: a CPU-only request must not create a CUDA context on a GPU judged too full.
    gpu_common.hide_cuda_for_cpu_request(request)
    dtype = str(request.get("dtype", "upstream"))
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    model_req = dict(request["model"])
    loader = str(model_req.get("loader", "upstream"))
    if loader not in LOADERS:
        raise ValueError(f"model.loader must be one of {LOADERS}, got {loader!r}")
    decode = dict(request.get("decode") or {})
    views = list(request.get("views") or [])
    if not views:
        raise ValueError("request has no views")
    gpu_common.apply_determinism(bool(request.get("determinism", True)))

    import torch

    out: dict[str, Any] = ctx.partial
    warn: list[str] = []
    version = _muscriptor_version()
    if loader == "lean" and version != PINNED_VERSION:
        warn.append(f"lean 로더는 muscriptor {PINNED_VERSION} 전용입니다(설치: {version}); upstream 로더를 씁니다.")
        loader = "upstream"
    from muscriptor.transcription_model import TranscriptionModel

    api = {"muscriptor_version": version, "transcribe_signature": str(inspect.signature(TranscriptionModel.transcribe)),
           "input_kind": "tensor"}
    out["api"] = api
    vram_req = request.get("vram") or {}
    device = gpu_common.pick_device(request)
    vram: dict[str, Any] = {"device": "cpu" if device.type == "cpu" else "cuda:0", "total_mb": None,
                            "start_free_mb": None, "ctx_mb": None, "start_nvml_used_mb": None, "rung_index": None,
                            "rung_label": None, "cap_mb": vram_req.get("cap_mb"), "attempts": []}
    out["vram"] = vram
    if device.type == "cuda":
        vram["ctx_mb"] = gpu_common.init_cuda_measure_ctx(ctx.sampler)
        free, total = gpu_common.mem_get_info_mb()
        vram["start_free_mb"], vram["total_mb"] = free, total
        snap = ctx.sampler.snapshot() if ctx.sampler is not None and hasattr(ctx.sampler, "snapshot") else {}
        vram["start_nvml_used_mb"] = snap.get("nvml_used_mb")
        gpu_common.apply_cap(request)

    # Views: masks re-sorted, audio read once (CPU tensors; the model moves them to its device). A segmented
    # view keeps one slice per segment; "audio_s" is the transcribed audio, "view_s" the whole view.
    prepared: list[dict[str, Any]] = []
    for v in views:
        tensor, sr, view_s = _read_view(Path(v["wav"]))
        segs = v.get("segments")
        bounds = segment_bounds(segs, sr, int(tensor.shape[-1]))
        pieces = [{"offset_s": a / float(sr), "audio_s": (z - a) / float(sr),
                   "tensor": tensor if segs is None else tensor[:, a:z].contiguous()} for a, z in bounds]
        prepared.append({"id": str(v["id"]), "out": Path(v["out"]), "view_sha256": v.get("view_sha256"),
                         "instruments": sort_instruments(v.get("instruments")), "pieces": pieces, "sr": sr,
                         "view_s": view_s, "audio_s": sum(pc["audio_s"] for pc in pieces),
                         "segments": None if segs is None else [(a / float(sr), z / float(sr))
                                                                for a, z in bounds]})
    audio_total = sum(p["audio_s"] for p in prepared)

    medium_weights = model_req.get("weights_path")
    if not medium_weights or not Path(medium_weights).is_file():
        raise FileNotFoundError(f"MuScriptor weights not found: {medium_weights} "
                                "(gtab computes this path from HF_HOME + amt.muscriptor_revision)")
    rungs = build_rungs(model_req, loader)
    labels = [r["label"] for r in rungs if r["device"] != "cpu"]
    # Rungs whose weights are not on disk are 'unavailable' (small not in the HF cache): recorded, never tried.
    available: list[dict[str, Any]] = []
    unavailable: dict[str, gpu_common.Attempt] = {}
    for i, r in enumerate(rungs):
        w = _resolve_weights(r["params"])
        if w is None:
            unavailable[r["label"]] = gpu_common.Attempt(
                rung_index=i, rung_label=r["label"], device=r["device"], status="unavailable",
                error=f"weights for {r['label']} are not on disk (HF cache: {r['params'].get('repo')}); not downloaded")
            continue
        r = dict(r)
        r["params"] = dict(r["params"], weights_path=w)
        available.append(r)
    force = (vram_req.get("force_rung") or None)
    if force and force in unavailable:
        vram["attempts"] = [unavailable[force].to_dict()]
        raise FileNotFoundError(f"force_rung {force!r} is unavailable: {unavailable[force].error}")

    decode_kw = {"use_sampling": bool(decode.get("use_sampling", False)),
                 "cfg_coef": float(decode.get("cfg_coef", 1.0)),
                 "beam_size": int(decode.get("beam_size", 1)),
                 "prelude_forcing": bool(decode.get("prelude_forcing", True)),
                 "batch_size": int(decode.get("batch_size", 1)),
                 "no_eos_is_ok": True}

    def attempt(rung: dict[str, Any], wd: gpu_common.Watchdog) -> dict[str, Any]:
        state: dict[str, Any] = {"tm": None}
        try:
            t0 = time.perf_counter()
            state["tm"], used_loader = _load(rung, rung["params"]["weights_path"], dtype)
            if rung["device"] == "cuda":
                torch.cuda.synchronize()
                if ctx.sampler is not None and hasattr(ctx.sampler, "mark"):
                    ctx.sampler.mark(MODEL_LOADED)
            load_s = time.perf_counter() - t0
            wd.arm()
            results: list[dict[str, Any]] = []
            done_s = 0.0
            for p in prepared:
                t = time.perf_counter()
                notes: list[dict[str, Any]] = []
                for pc in p["pieces"]:
                    base = done_s

                    def progress(completed: int, total: int, _base: float = base,
                                 _dur: float = pc["audio_s"]) -> None:
                        if total > 0 and completed > 0:
                            wd.check(_base + _dur * completed / total)

                    with warnings.catch_warnings(record=True) as caught, torch.inference_mode():
                        warnings.simplefilter("always", RuntimeWarning)
                        events = state["tm"].transcribe((pc["tensor"], p["sr"]), instruments=p["instruments"],
                                                        **decode_kw)
                        piece = assemble_notes(events, pc["audio_s"], on_progress=progress)
                    if rung["device"] == "cuda":
                        torch.cuda.synchronize()
                    for w in caught:
                        msg = str(w.message)
                        if "EOS" in msg:
                            warn.append(f"{p['id']}: {msg}")
                    notes += offset_notes(piece, pc["offset_s"]) if p["segments"] is not None else piece
                    done_s += pc["audio_s"]
                notes.sort(key=note_sort_key)
                results.append({"notes": notes, "process_s": time.perf_counter() - t})
            return {"results": results, "load_s": load_s, "loader": used_loader}
        except BaseException:
            state["tm"] = None  # drop the model before the ladder frees the CUDA cache
            raise

    result, attempts = gpu_common.run_ladder(available, attempt, sampler=ctx.sampler, req=request,
                                             audio_s_total=audio_total)
    # run_ladder numbers the rungs it was given; report indices of the full configured ladder instead.
    label_index = {r["label"]: i for i, r in enumerate(rungs)}
    for a in attempts:
        a.rung_index = label_index.get(a.rung_label, a.rung_index)
    # Put 'unavailable' rungs back where the ladder reached them (a rung below the last one tried is not listed).
    reached = max((a.rung_index for a in attempts), default=-1)
    if result is None:
        reached = len(rungs)
    merged = list(attempts) + [a for a in unavailable.values()
                               if a.rung_index <= reached and (a.device != "cpu" or bool(vram_req.get("cpu_allowed")))
                               and not force]
    merged.sort(key=lambda a: a.rung_index)
    vram["attempts"] = gpu_common.attempts_summary(merged)
    for a in merged:
        if a.slow:
            warn.append(f"{a.rung_label}: 실시간 배수 {a.rtf}x 로 기준보다 느림(다른 GPU 작업과 경합했을 수 있음)")

    model_sha = model_req.get("sha256")
    backend = {"name": BACKEND, "version": version, "model": f"muscriptor-{model_req.get('size', 'medium')}",
               "model_sha256": model_sha, "dtype_policy": ("fp32+autocast-fp16" if dtype == "upstream"
                                                            else "fp16-weights+fp32-conditioners"),
               "device_used": None}
    common = {"backend": backend, "vram": vram, "api": api, "loader": loader, "rung_labels": labels,
              "warnings": warn}
    if result is None:
        log.warning("no MuScriptor rung fitted: answering insufficient_vram")
        return {"status": "insufficient_vram", **common,
                "timing": {"load_s": None, "process_s": None, "audio_s": round(audio_total, 3), "rtf": None},
                "files": {}, "views": [], "load_s": None}

    ok = next(a for a in attempts if a.status == "ok")
    ok_rung = rungs[label_index[ok.rung_label]]
    vram["rung_index"], vram["rung_label"] = ok.rung_index, ok.rung_label
    backend["device_used"] = "cpu" if ok.device == "cpu" else "cuda"
    if ok_rung["params"]["size"] != model_req.get("size", "medium"):
        backend["model"] = f"muscriptor-{ok_rung['params']['size']}"
        backend["model_sha256"] = _sha256(Path(next(r for r in available if r["label"] == ok.rung_label)
                                                ["params"]["weights_path"]))
    params_doc = {"rung_label": ok.rung_label, "loader": result["loader"], "dtype": dtype, **{
        k: decode_kw[k] for k in ("prelude_forcing", "beam_size", "cfg_coef", "use_sampling", "batch_size")}}
    nb_backend = {"name": BACKEND, "version": version, "model": backend["model"],
                  "model_sha256": backend["model_sha256"], "params": params_doc,
                  "device": backend["device_used"], "dtype": backend["dtype_policy"]}
    out_dir = Path(request["out_dir"]) if request.get("out_dir") else None
    files: dict[str, dict[str, int]] = {}
    view_rows: list[dict[str, Any]] = []
    process_total = 0.0
    for p, r in zip(prepared, result["results"]):
        doc = noteset_doc(view_id=p["id"], view_sha256=p["view_sha256"], backend=nb_backend,
                          instruments=p["instruments"], audio_s=p["view_s"], notes=r["notes"],
                          segments=p["segments"])
        data = (json.dumps(doc, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        atomic.write_bytes(p["out"], data)
        key = str(p["out"])
        if out_dir is not None:
            try:
                key = p["out"].resolve().relative_to(out_dir.resolve()).as_posix()
            except ValueError:
                pass
        files[key] = {"bytes": len(data)}
        process_total += r["process_s"]
        view_rows.append({"id": p["id"], "out": str(p["out"]), "n_notes": len(r["notes"]),
                          "audio_s": round(p["audio_s"], 3), "process_s": round(r["process_s"], 3),
                          "instruments": p["instruments"]})
    rtf = round(audio_total / process_total, 3) if process_total > 0 else None
    return {"status": "ok", **common, "loader": result["loader"], "views": view_rows, "files": files,
            "load_s": round(result["load_s"], 3),
            "timing": {"load_s": round(result["load_s"], 3), "process_s": round(process_total, 3),
                       "audio_s": round(audio_total, 3), "rtf": rtf}}


def handle(request: dict, ctx: WorkerContext) -> dict:
    mode = str(request.get("mode", "transcribe"))
    if mode == "probe":
        return _probe()
    if mode != "transcribe":
        raise ValueError(f"unknown mode {mode!r} (transcribe | probe)")
    return _transcribe(request, ctx)


if __name__ == "__main__":
    worker_main(handle)
