# bandscribe: py310  (runs in the bp310 env on Python 3.10; tests/test_py310_compat.py parses it with feature_version 3.10)
"""Basic Pitch 0.4.0 worker (bp310 env, CPU/ONNX, no GPU lock) - M1_M2_SPEC 7.4, DESIGN S6.

Started by the core as ``<bp310 venv>\\Scripts\\python.exe -m bandscribe.workers.amt_basicpitch request.json result.json``
through ``bandscribe.gpu.run_worker(..., python=BP310_PYTHON, use_lock=False)``; bandscribe is not installed in bp310, it is
imported from the source tree via PYTHONPATH. This module therefore imports only the stdlib, numpy, soundfile,
basic_pitch, onnxruntime and ``bandscribe.workers.base`` / ``bandscribe.atomic`` (all Python-3.10-safe) - never bandscribe.schema.

Why Python 3.10: basic-pitch 0.4.0 on Windows pulls onnxruntime only for Python < 3.11 (TensorFlow < 2.15.1,
without cp312 wheels, otherwise). On 3.10 the lock resolves onnxruntime<=1.23.2, numpy<=2.2.6, scipy<=1.15.3.

Two-step path in both modes, so settings tuned by E23 reproduce exactly in the pipeline:

1. ``basic_pitch.inference.run_inference(wav, model)`` once per view. The model is loaded once per worker; the
   ONNX Runtime session is ours, built with ``SessionOptions(intra_op_num_threads=threads,
   inter_op_num_threads=1, ORT_SEQUENTIAL)``: pinned threads keep the float reductions (and so the NoteSets)
   independent of the machine's core count. The raw model output (``note``/``onset``/``contour`` arrays) is
   optionally cached as ``<cache>/<view_sha256>__<model sha12>.npz``.
2. ``note_creation.model_output_to_notes(output, onset_thresh, frame_thresh, infer_onsets=True,
   min_note_len=min_note_frames - 1, ...)``. Verified in basic_pitch 0.4.0 (note_creation.py): a note is kept
   only if its length in frames is **strictly greater** than ``min_note_len`` (``if i - note_start_idx <=
   min_note_len: continue``), and ``predict()`` converts ms with ``round(ms / 1000 * 22050 / 256)``. So the
   upstream default 127.7 ms = 11 frames keeps notes of >= 12 frames (~139 ms), which deletes 16th notes above
   ~117 BPM, fast runs and ghost notes (DESIGN S6). Our parameter ``min_note_frames`` is the shortest note
   *kept*; 12 (= the upstream default) is refused unless ``allow_default_min_len`` is set (E23's reference arm).
   ``constrain_frequency`` mutates its inputs in place, so every setting gets a fresh copy of the cached output.

Pitch bends: ``get_pitch_bends`` returns per-frame offsets in contour bins of 1/3 semitone; cents = 100/3 x bin,
``t_s`` = note onset + frame index / (22050 / 256). A note whose bins are all zero gets ``bend: null``.

Modes: ``transcribe`` (default) writes one NoteSet per view to ``view["out"]``; ``sweep`` (E23) runs step 1 once
per view and step 2 for every entry of ``settings``, writing ``<out stem>__s<k>.json``; ``probe`` reports
versions (and the interpreter version) without running the model.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from bandscribe import atomic
from bandscribe.workers.base import WorkerContext, worker_main

log = logging.getLogger(__name__)

BACKEND = "basicpitch"
SR_MODEL = 22050
FFT_HOP = 256
FRAMES_PER_SECOND = SR_MODEL / FFT_HOP  # 86.1328... (basic_pitch's ANNOTATIONS_FPS is the int 86; times use this)
UPSTREAM_DEFAULT_MIN_NOTE_FRAMES = 12  # round(127.7 ms / 1000 * 22050 / 256) = 11 -> keeps >= 12 frames
CENTS_PER_CONTOUR_BIN = 100.0 / 3.0

DEFAULT_PARAMS: Dict[str, Any] = {
    "onset_threshold": 0.5,
    "frame_threshold": 0.3,
    "min_note_frames": 4,
    "minimum_frequency": None,
    "maximum_frequency": None,
    "melodia_trick": True,
    "multiple_pitch_bends": False,
}


class ParamError(ValueError):
    pass


# ------------------------------------------------------------------------------------------ params


def min_note_len_for(min_note_frames: Any, *, allow_default_min_len: bool = False) -> int:
    """``min_note_len`` to pass to Basic Pitch for the shortest note we want to KEEP (in frames).

    Basic Pitch drops notes whose length is <= ``min_note_len``, so keeping notes of >= N frames means passing
    N - 1. N == 12 reproduces the upstream 127.7 ms default and is refused unless explicitly allowed.
    """
    if isinstance(min_note_frames, bool):
        raise ParamError(f"min_note_frames must be an int, got {min_note_frames!r}")
    if not isinstance(min_note_frames, int):
        try:
            as_float = float(min_note_frames)
        except (TypeError, ValueError):
            raise ParamError(f"min_note_frames must be an int, got {min_note_frames!r}") from None
        if not as_float.is_integer():
            raise ParamError(f"min_note_frames must be an int, got {min_note_frames!r}")
        min_note_frames = int(as_float)
    if min_note_frames < 1 or min_note_frames > 30:
        raise ParamError(f"min_note_frames must be in [1, 30], got {min_note_frames}")
    if min_note_frames == UPSTREAM_DEFAULT_MIN_NOTE_FRAMES and not allow_default_min_len:
        raise ParamError(
            "min_note_frames=12 is Basic Pitch's upstream 127.7 ms default, which deletes short notes "
            "(DESIGN S6); set allow_default_min_len only for E23's reference arm")
    return min_note_frames - 1


def _opt_hz(v: Any) -> Optional[float]:
    """0 / None -> None (= backend default); otherwise a positive frequency."""
    if v is None:
        return None
    f = float(v)
    if f <= 0.0:
        return None
    return f


def normalize_params(params: Optional[Dict[str, Any]], *, allow_default_min_len: bool = False) -> Dict[str, Any]:
    """Defaults filled in, types checked, key order fixed (the dict is written into every NoteSet)."""
    p = dict(DEFAULT_PARAMS)
    for k, v in (params or {}).items():
        if k not in DEFAULT_PARAMS:
            raise ParamError(f"unknown Basic Pitch parameter {k!r}")
        p[k] = v
    min_len = min_note_len_for(p["min_note_frames"], allow_default_min_len=allow_default_min_len)
    out = {
        "onset_threshold": float(p["onset_threshold"]),
        "frame_threshold": float(p["frame_threshold"]),
        "min_note_frames": min_len + 1,
        "minimum_frequency": _opt_hz(p["minimum_frequency"]),
        "maximum_frequency": _opt_hz(p["maximum_frequency"]),
        "melodia_trick": bool(p["melodia_trick"]),
        "multiple_pitch_bends": bool(p["multiple_pitch_bends"]),
    }
    for k in ("onset_threshold", "frame_threshold"):
        if not 0.0 < out[k] < 1.0:
            raise ParamError(f"{k} must be in (0, 1), got {out[k]}")
    return out


# ------------------------------------------------------------------------------------- note events


def _r6(x: float) -> float:
    return round(float(x), 6)


def events_to_notes(note_events: List[Any]) -> List[Dict[str, Any]]:
    """Basic Pitch note events ``(start_s, end_s, pitch, amplitude, bends|None)`` -> NoteEvent dicts, sorted."""
    notes: List[Dict[str, Any]] = []
    for ev in note_events:
        start, end, pitch, amplitude, bends = ev[0], ev[1], ev[2], ev[3], ev[4] if len(ev) > 4 else None
        onset = _r6(start)
        note: Dict[str, Any] = {
            "onset_s": onset,
            "offset_s": max(onset, _r6(end)),
            "pitch": int(pitch),
            "amplitude": _r6(amplitude),
        }
        if bends is not None and len(bends) and any(int(b) != 0 for b in bends):
            note["bend"] = {
                "t_s": [_r6(float(start) + k / FRAMES_PER_SECOND) for k in range(len(bends))],
                "cents": [round(int(b) * CENTS_PER_CONTOUR_BIN, 4) for b in bends],
            }
        notes.append(note)
    notes.sort(key=lambda n: (n["onset_s"], n["pitch"], n["offset_s"], ""))
    return notes


def noteset_doc(*, view_id: str, view_sha256: Optional[str], backend: Dict[str, Any], params: Dict[str, Any],
                audio_s: Optional[float], notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A ``bandscribe.notes/1`` document (schema §6.1) built with a fixed key order, no timestamps."""
    be = dict(backend)
    be["params"] = dict(params)
    be["instruments"] = None
    return {
        "format": "bandscribe.notes/1",
        "view": view_id,
        "view_sha256": view_sha256,
        "backend": be,
        "time_offset_applied_s": 0.0,
        "audio_duration_s": None if audio_s is None else _r6(audio_s),
        "notes": notes,
    }


def dumps_noteset(doc: Dict[str, Any]) -> str:
    # Same serialisation as bandscribe.atomic.write_json (indent=1, insertion order), so files are byte-stable.
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"


def sweep_out_path(out: Path, k: int) -> Path:
    return out.with_name(f"{out.stem}__s{k}{out.suffix}")


# ------------------------------------------------------------------------------------------- model


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _Runner:
    """Holds the one ONNX session of this worker and runs step 1 / step 2."""

    def __init__(self, threads: int) -> None:
        import onnxruntime as ort
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch import inference

        self.threads = int(threads)
        path = Path(ICASSP_2022_MODEL_PATH)
        if path.suffix != ".onnx":  # basic_pitch picks the ONNX path only when onnxruntime is the backend
            path = path.with_name("nmp.onnx")
        if not path.is_file():
            raise FileNotFoundError(f"bundled Basic Pitch ONNX model not found: {path}")
        self.model_path = path
        self.model_sha256 = _sha256(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # basic_pitch.inference.Model builds its own session with default (all-core) threading; we bypass its
        # constructor and hand it our pinned session instead.
        model = inference.Model.__new__(inference.Model)
        model.model_type = inference.Model.MODEL_TYPES.ONNX
        model.model = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.model = model
        self.model_loads = 1
        self.inference_runs = 0
        self.ort_version = ort.__version__
        try:
            from importlib.metadata import version

            self.bp_version = version("basic-pitch")
        except Exception:  # pragma: no cover - metadata missing in an odd install
            self.bp_version = None

    def model_output(self, wav: Path, view_sha256: Optional[str], cache_dir: Optional[Path]) -> tuple:
        """(output dict of float32 arrays, cached: bool)."""
        import numpy as np

        cache_file = None
        if cache_dir is not None and view_sha256:
            cache_file = cache_dir / f"{view_sha256}__{self.model_sha256[:12]}.npz"
            if cache_file.is_file():
                try:
                    with np.load(cache_file) as z:
                        return {k: np.array(z[k]) for k in ("note", "onset", "contour")}, True
                except Exception as e:  # a corrupt cache entry is recomputed, never fatal
                    log.warning("ignoring unreadable model-output cache %s: %s", cache_file, e)
        from basic_pitch import inference

        out = inference.run_inference(str(wav), self.model)
        self.inference_runs += 1
        out = {k: np.asarray(out[k], dtype=np.float32) for k in ("note", "onset", "contour")}
        if cache_file is not None:
            buf = io.BytesIO()
            np.savez(buf, **out)
            atomic.write_bytes(cache_file, buf.getvalue())
        return out, False

    @staticmethod
    def notes(output: Dict[str, Any], params: Dict[str, Any], *, allow_default_min_len: bool) -> List[Dict[str, Any]]:
        from basic_pitch import note_creation

        min_len = min_note_len_for(params["min_note_frames"], allow_default_min_len=allow_default_min_len)
        fresh = {k: v.copy() for k, v in output.items()}  # constrain_frequency writes into its inputs
        _midi, events = note_creation.model_output_to_notes(
            fresh,
            onset_thresh=params["onset_threshold"],
            frame_thresh=params["frame_threshold"],
            infer_onsets=True,
            min_note_len=min_len,
            min_freq=params["minimum_frequency"],
            max_freq=params["maximum_frequency"],
            include_pitch_bends=True,
            multiple_pitch_bends=params["multiple_pitch_bends"],
            melodia_trick=params["melodia_trick"],
        )
        return events_to_notes(list(events))


def _audio_seconds(wav: Path) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(str(wav))
        return info.frames / float(info.samplerate)
    except Exception as e:
        log.warning("could not read duration of %s: %s", wav, e)
        return None


def _limit_threads(threads: int) -> Any:
    """BLAS/OpenMP pools of numpy/scipy pinned too (threadpoolctl ships with scikit-learn in bp310)."""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=int(threads))
    except Exception:  # pragma: no cover
        import contextlib

        return contextlib.nullcontext()


# ------------------------------------------------------------------------------------------ handler


def _probe() -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "ok", "mode": "probe", "python_version": list(sys.version_info[:3]),
                           "executable": sys.executable}
    try:
        import onnxruntime as ort

        out["onnxruntime_version"] = ort.__version__
    except Exception as e:
        out["onnxruntime_error"] = f"{type(e).__name__}: {e}"
    try:
        from importlib.metadata import version

        out["basic_pitch_version"] = version("basic-pitch")
        from basic_pitch import ICASSP_2022_MODEL_PATH

        out["model_path"] = str(ICASSP_2022_MODEL_PATH)
    except Exception as e:
        out["basic_pitch_error"] = f"{type(e).__name__}: {e}"
    return out


def _run(request: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    mode = str(request.get("mode", "transcribe"))
    allow_default = bool(request.get("allow_default_min_len", False))
    threads = int(request.get("threads", 4))
    if threads < 1:
        raise ParamError(f"threads must be >= 1, got {threads}")
    views = list(request.get("views") or [])
    if not views:
        raise ParamError("request has no views")
    if mode == "sweep":
        settings = [normalize_params(s, allow_default_min_len=allow_default) for s in (request.get("settings") or [])]
        if not settings:
            raise ParamError("sweep mode needs a non-empty 'settings' list")
    else:
        settings = [normalize_params(request.get("params"), allow_default_min_len=allow_default)]
    cache_dir = Path(request["model_output_cache"]) if request.get("model_output_cache") else None

    out: Dict[str, Any] = ctx.partial
    out.update({"status": "ok", "mode": mode, "threads": threads, "warnings": []})
    with _limit_threads(threads):
        t0 = time.perf_counter()
        runner = _Runner(threads)
        load_s = time.perf_counter() - t0
        backend = {"name": BACKEND, "version": runner.bp_version, "model": "icassp_2022/nmp.onnx",
                   "model_sha256": runner.model_sha256, "dtype_policy": "fp32", "device_used": "cpu"}
        out.update({"backend": backend, "model_path": str(runner.model_path), "model_sha256": runner.model_sha256,
                    "onnxruntime_version": runner.ort_version, "basic_pitch_version": runner.bp_version,
                    "frames_per_second": round(FRAMES_PER_SECOND, 4)})
        nb_backend = {"name": BACKEND, "version": runner.bp_version, "model": "icassp_2022/nmp.onnx",
                      "model_sha256": runner.model_sha256, "device": "cpu", "dtype": "fp32"}
        files: Dict[str, Dict[str, int]] = {}
        view_rows: List[Dict[str, Any]] = []
        out_dir = Path(request["out_dir"]) if request.get("out_dir") else None
        total_audio = 0.0
        total_proc = 0.0
        for v in views:
            vid, wav, out_path = str(v["id"]), Path(v["wav"]), Path(v["out"])
            vsha = v.get("view_sha256")
            t = time.perf_counter()
            output, cached = runner.model_output(wav, vsha, cache_dir)
            audio_s = _audio_seconds(wav)
            outs: List[str] = []
            n_notes: List[int] = []
            for k, params in enumerate(settings):
                notes = runner.notes(output, params, allow_default_min_len=allow_default)
                doc = noteset_doc(view_id=vid, view_sha256=vsha, backend=nb_backend, params=params,
                                  audio_s=audio_s, notes=notes)
                target = sweep_out_path(out_path, k) if mode == "sweep" else out_path
                data = dumps_noteset(doc).encode("utf-8")
                atomic.write_bytes(target, data)
                outs.append(str(target))
                n_notes.append(len(notes))
                if out_dir is not None:
                    try:
                        files[target.resolve().relative_to(out_dir.resolve()).as_posix()] = {"bytes": len(data)}
                    except ValueError:
                        files[str(target)] = {"bytes": len(data)}
            proc = time.perf_counter() - t
            total_proc += proc
            total_audio += audio_s or 0.0
            row: Dict[str, Any] = {"id": vid, "out": outs[0] if mode != "sweep" else str(out_path),
                                   "n_notes": n_notes[0] if mode != "sweep" else n_notes,
                                   "audio_s": None if audio_s is None else round(audio_s, 3),
                                   "process_s": round(proc, 3), "model_output_cached": cached}
            if mode == "sweep":
                row["outs"] = outs
            view_rows.append(row)
    first = settings[0]
    out.update({
        "views": view_rows,
        "files": files,
        "min_note_len_frames": min_note_len_for(first["min_note_frames"], allow_default_min_len=allow_default),
        "min_note_frames_kept": first["min_note_frames"],
        "settings": settings if mode == "sweep" else None,
        "model_loads": runner.model_loads,
        "inference_runs": runner.inference_runs,
        "timing": {"load_s": round(load_s, 3), "process_s": round(total_proc, 3), "audio_s": round(total_audio, 3),
                   "rtf": round(total_audio / total_proc, 3) if total_proc > 0 else None},
    })
    return out


def handle(request: Dict[str, Any], ctx: WorkerContext) -> Dict[str, Any]:
    mode = str(request.get("mode", "transcribe"))
    if mode == "probe":
        return _probe()
    if mode not in ("transcribe", "sweep"):
        raise ParamError(f"unknown mode {mode!r} (transcribe | sweep | probe)")
    return _run(request, ctx)


if __name__ == "__main__":
    worker_main(handle)
