"""MuScriptor worker checks that need the gpu env but no CUDA and no weights (marker ``gpuenv``).

``probe`` reports the installed API without loading weights (M1_M2_SPEC 9.2 item 1): the MT3 group table must equal
``bandscribe.schema.instrumentation.MT3_GROUPS``, ``ProgressEvent`` comes from ``muscriptor.events`` and ``transcribe``
takes the keywords the worker passes. A bf16 request is refused before any model or CUDA work (sm_75).
"""
from __future__ import annotations

import inspect

import pytest

from bandscribe import gpu
from bandscribe.amt import instruments as I
from bandscribe.workers import amt_muscriptor as W

pytestmark = [pytest.mark.gpuenv]


def _run(req: dict, work):
    # use_lock=False: nothing here touches the GPU, so it must not queue behind (or block) GPU jobs.
    return gpu.run_worker("amt_muscriptor", req, work, timeout_s=600, use_lock=False)


def test_probe_reports_installed_api(tmp_path):
    res = _run({"mode": "probe"}, tmp_path / "_worker")
    assert res.ok, res.error
    out = res.output
    assert out["muscriptor_version"] == W.PINNED_VERSION
    assert out["n_groups"] == 35 and out["groups_match_bandscribe"] is True
    assert dict(out["groups"]) == dict(I.MT3_GROUPS)
    assert out["progress_event_module"] == "muscriptor.events"
    assert out["hf_hub_offline"] == "1"
    sig = out["transcribe_signature"]
    for kw in ("instruments", "prelude_forcing", "beam_size", "cfg_coef", "use_sampling", "batch_size",
               "no_eos_is_ok"):
        assert kw in sig, kw
    assert "weights_path" in out["load_model_signature"] and "device" in out["load_model_signature"]


@pytest.mark.parametrize("field", ["dtype", "autocast_dtype"])
def test_bf16_request_refused_before_any_work(tmp_path, field):
    req = {"format": "bandscribe.worker/1", "job": None, "out_dir": str(tmp_path), "device": "cpu",
           "model": {"size": "medium", "weights_path": str(tmp_path / "absent.safetensors"), "loader": "upstream"},
           "decode": {}, "dtype": "upstream",
           "views": [{"id": "v", "wav": str(tmp_path / "absent.wav"), "instruments": None,
                      "out": str(tmp_path / "o.json"), "view_sha256": None}]}
    req[field] = "bfloat16"
    res = _run(req, tmp_path / "_worker")
    assert not res.ok and "bfloat16" in (res.error or "").lower()
    assert not (tmp_path / "o.json").exists()


def test_worker_module_is_torch_free_at_import():
    src = inspect.getsource(W)
    head = src[: src.index("# ------------------------------------------------------------------------------------ pure helpers")]
    assert "import torch" not in head and "import muscriptor" not in head
