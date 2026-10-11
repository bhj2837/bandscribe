"""Centre views of a stereo guitar stem (DESIGN S5 R2: "기타 스템에 CenterWide 를 돌려 centre 를 뽑는다"; E7, M6).

- ``mid``: (L + R) / 2, the plain matrix (keeps the hard-panned takes of a double at half level).
- ``ipd_ild``: an ADRess-style time-frequency mask on the mid signal: a bin counts as centre when its inter-channel
  level difference is small and its phase difference too; soft weights ``exp(-(ILD/ILD_SCALE_DB)^2) *
  cos(IPD/2)^2``, STFT ``N_FFT`` / ``HOP``.
- CenterWide BS-RoFormer models (``CENTRE_MODELS``; gilliaan, CC BY-NC-SA 4.0, ``bandscribe models fetch``): a
  learned split into ``center`` and ``wide``. They run on the SW separation worker (``sep_msst``, the vendored MSST
  BSRoformer) with their own yaml: chunk = the yaml's ``audio.chunk_size``, overlap = its ``inference.num_overlap``.
  Several inputs go through one worker call (laid out like the E6 / distractor batches: each input sees the chunk
  grid it would see alone); results are cached by (input pcm sha, model sha, chunk, overlap).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe import atomic

SR = 44100
N_FFT = 4096
HOP = 1024
ILD_SCALE_DB = 3.0
CENTRE_MODELS: dict[str, tuple[str, str]] = {  # id -> (model_sources ckpt name, yaml name)
    "cw_bs_v2": ("centerwide.bs_v2", "centerwide.bs_v2.yaml"),
    "cw_bs_v3beta": ("centerwide.bs_v3beta", "centerwide.bs_v3beta.yaml"),
}
CODE_VERSION = "1"
BATCH_S = 600.0


def mid(stereo: np.ndarray) -> np.ndarray:
    x = np.asarray(stereo, dtype=np.float32)
    x = x if x.shape[-1] == 2 else x.T
    return ((x[:, 0] + x[:, 1]) / 2).astype(np.float32)


def side(stereo: np.ndarray) -> np.ndarray:
    x = np.asarray(stereo, dtype=np.float32)
    x = x if x.shape[-1] == 2 else x.T
    return ((x[:, 0] - x[:, 1]) / 2).astype(np.float32)


def ipd_ild(stereo: np.ndarray, sr: int = SR) -> np.ndarray:
    """Soft centre mask applied to the mid signal (module doc) -> mono float32, same length."""
    from scipy import signal

    x = np.asarray(stereo, dtype=np.float32)
    x = x if x.shape[-1] == 2 else x.T
    n = x.shape[0]
    _f, _t, zl = signal.stft(x[:, 0], sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")
    _f, _t, zr = signal.stft(x[:, 1], sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")
    ild = 20 * np.log10((np.abs(zl) + 1e-9) / (np.abs(zr) + 1e-9))
    ipd = np.angle(zl * np.conj(zr))
    w = np.exp(-((ild / ILD_SCALE_DB) ** 2)) * np.cos(ipd / 2) ** 2
    _t, y = signal.istft(w * (zl + zr) / 2, sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")
    out = np.zeros(n, dtype=np.float32)
    m = min(n, y.size)
    out[:m] = y[:m]
    return out


def model_spec(model_id: str) -> dict[str, Any]:
    """Paths, pinned sha256, chunk and overlap of a CenterWide model (the files must have been fetched)."""
    from bandscribe import model_fetch
    from bandscribe.third_party.msst import loader

    if model_id not in CENTRE_MODELS:
        raise ValueError(f"unknown centre model {model_id!r}; known: {', '.join(CENTRE_MODELS)}")
    srcs = model_fetch.load_sources()
    ck, ym = (srcs[n] for n in CENTRE_MODELS[model_id])
    ckpt, yml = ck.dest_path(), ym.dest_path()
    for p, name in ((ckpt, CENTRE_MODELS[model_id][0]), (yml, CENTRE_MODELS[model_id][1])):
        if not p.is_file():
            raise FileNotFoundError(f"센터 모델 파일이 없습니다: {p} (bandscribe models fetch {name})")
    cfg = loader.load_yaml(yml)
    return {"id": model_id, "ckpt": ckpt, "ckpt_sha256": ck.sha256, "yaml": yml, "yaml_sha256": ym.sha256,
            "chunk": int(cfg["audio"]["chunk_size"]),
            "num_overlap": int((cfg.get("inference") or {}).get("num_overlap") or 2),
            "stems": loader.stem_order(cfg)}


def _key(sha: str, spec: dict[str, Any]) -> str:
    blob = json.dumps({"pcm": sha, "ckpt": spec["ckpt_sha256"], "yaml": spec["yaml_sha256"], "chunk": spec["chunk"],
                       "overlap": spec["num_overlap"], "v": CODE_VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def extract(items: Sequence[tuple[str, Path]], cfg: Any, model_id: str, cache_dir: Path, *,
            progress: Callable[[str], None] | None = None) -> tuple[dict[str, Path], dict[str, Any]]:
    """{pcm sha: stereo centre WAV} for (sha, stereo 44.1 kHz WAV) items with a CenterWide model (module doc)."""
    import soundfile as sf

    from bandscribe import audio as AU
    from bandscribe import vram
    from bandscribe.datasets import store
    from bandscribe.eval.distractors import batch_layout, batch_total
    from bandscribe.sep import run as SEP

    spec = model_spec(model_id)
    out: dict[str, Path] = {}
    todo: list[tuple[str, Path, Path]] = []
    for sha, wav in items:
        d = Path(cache_dir) / model_id / _key(sha, spec)
        if (d / "done.json").is_file():
            out[sha] = d / "center.wav"
        elif sha not in {t[0] for t in todo}:
            todo.append((sha, Path(wav), d))
    info: dict[str, Any] = {"model": model_id, "cached": len(out), "extracted": len(todo), "batches": 0,
                            "audio_s": 0.0, "rtf": [], "reserved_peak_mb": []}
    if not todo:
        return out, info
    chunk, overlap = spec["chunk"], spec["num_overlap"]
    step = chunk // overlap
    lengths = [sf.info(str(w)).frames for _s, w, _d in todo]
    label = f"chunk={chunk}"
    for b, lay in enumerate(batch_layout(lengths, chunk=chunk, step=step, max_samples=int(BATCH_S * SR))):
        total = batch_total(lengths, lay, chunk=chunk, step=step)
        mixbuf = np.zeros((total, 2), dtype=np.float32)
        for i, off in lay:
            y, _ = sf.read(str(todo[i][1]), dtype="float32", always_2d=True)
            mixbuf[off:off + len(y)] = y[:, :2]
        bdir = Path(store.make_scratch_dir(Path(cache_dir) / "tmp", f"centre{b}"))
        try:
            AU.write_f32(bdir / "mix.wav", mixbuf, SR)
            del mixbuf
            if progress:
                progress(f"센터 추출({model_id}) {b + 1}: 입력 {len(lay)}개, {total / SR / 60:.1f}분")
            request = {
                "format": "bandscribe.worker/1", "job": None, "mode": "separate", "out_dir": str(bdir.resolve()),
                "device": "auto",
                "vram": vram.build_vram_request(SEP.BACKEND, [label], cfg, force_rung=label),
                "determinism": True, "mix_wav": str((bdir / "mix.wav").resolve()),
                "model": {"ckpt": str(spec["ckpt"]), "ckpt_sha256": spec["ckpt_sha256"],
                          "config_yaml": str(spec["yaml"]), "arch": "bs_roformer"},
                "chunk_ladder": [chunk], "num_overlap": overlap, "batch_size": 1, "dtype": "upstream",
                "input_gain_db": 0.0, "outputs": {"stems_dir": "stems"},
            }
            res = vram.run_gpu_stage(SEP.BACKEND, [label], request, cfg, bdir / "_worker")
            output = res.output
            if output.get("status") != "ok":
                raise RuntimeError(f"센터 추출 워커가 실패했습니다: {output.get('status')!r}")
            timing = output.get("timing") or {}
            info["audio_s"] += float(timing.get("audio_s") or 0.0)
            info["rtf"].append(timing.get("rtf"))
            att = [a for a in ((output.get("vram") or {}).get("attempts") or []) if a.get("status") == "ok"]
            info["reserved_peak_mb"].append(att[-1].get("max_reserved_mb") if att else None)
            c, _ = sf.read(str(bdir / output["stems"]["center"]), dtype="float32", always_2d=True)
            for i, off in lay:
                sha, _w, d = todo[i]
                d.mkdir(parents=True, exist_ok=True)
                AU.write_f32(d / "center.wav", c[off:off + lengths[i]], SR)
                atomic.write_json(d / "done.json", {"sha": sha, "model": model_id, "chunk": chunk, "overlap": overlap})
                out[sha] = d / "center.wav"
            del c
            info["batches"] += 1
        finally:
            store.remove_tree(bdir)
    return out, info
