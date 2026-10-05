"""Port of MSST ``utils.model_utils.demix`` (generic branch) with CPU float32 overlap-add accumulators.

Upstream at the pinned commit (see VENDOR.md) keeps the whole track and its accumulators on the GPU when
they fit in half the free VRAM. VRAM is the scarce resource on this PC (RTX 2060 6 GB shared with the
desktop), so here the mix, ``result`` and ``counter`` stay in RAM (~0.6 GB for a 4-min song, M1_M2_SPEC A3)
and only one chunk batch at a time goes to the device. Everything else follows upstream exactly, so outputs
match upstream to float rounding (``tests/test_sep_upstream_parity.py``):

- step = chunk // num_overlap, fade = chunk // 10, border = chunk - step;
- reflect padding by ``border`` on both sides when length > 2 * border;
- a chunk shorter than the chunk size is padded with ``reflect`` if longer than half a chunk, else zeros;
- linear fade-in/out window; the first chunk has no fade-in, the last no fade-out (batch_size 1 semantics:
  with batch_size > 1 upstream decides this per batch, kept as is);
- AMP: ``torch.cuda.amp.autocast(enabled=use_amp)`` upstream, i.e. fp16 autocast on CUDA only
  (``torch.autocast("cuda", float16)`` here, same effect, not deprecated);
- result / counter, NaN -> 0 (positions no window covers).

``on_chunk(done_s, total_s)`` is called after every batch (the Watchdog hook, M1_M2_SPEC 8.2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn


def windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    """Linear fade-in over ``fade_size`` samples, ones, linear fade-out (upstream ``_getWindowingArray``)."""
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def demix(model: nn.Module, mix: np.ndarray, *, chunk_size: int, num_overlap: int, num_stems: int,
          device: torch.device, batch_size: int = 1, use_amp: bool = True,
          on_chunk: Callable[[float, float], None] | None = None, sample_rate: int = 44100) -> np.ndarray:
    """Separate ``mix`` (channels, time) float32 -> (num_stems, channels, time) float32 numpy array.

    ``mix`` is shared, not copied (``torch.from_numpy``; nothing here writes into it): the reflect padding
    below makes the one working copy. Peak RAM is the caller's mix + the padded copy + ``result`` (num_stems x
    the mix) + ``counter``; the padded copy is released before the final division (review 2026-10-05).
    """
    mix_t = torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32))  # CPU, shares the caller's array
    acc_device = torch.device("cpu")

    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix_t.shape[-1]
    windowing = windowing_array(chunk_size, fade_size).to(acc_device)
    padded = length_init > 2 * border and border > 0
    if padded:
        mix_t = nn.functional.pad(mix_t, (border, border), mode="reflect")
    total = mix_t.shape[1]
    total_s = length_init / float(sample_rate)

    amp = bool(use_amp) and device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        with torch.inference_mode():
            result = torch.zeros((num_stems,) + tuple(mix_t.shape), dtype=torch.float32, device=acc_device)
            counter = torch.zeros(total, dtype=torch.float32, device=acc_device)

            i = 0
            part = None
            batch_data: list[torch.Tensor] = []
            batch_locations: list[tuple[int, int]] = []
            while i < total:
                part = mix_t[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                if len(batch_data) >= batch_size or i >= total:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr).to(acc_device, torch.float32)

                    window = windowing.clone()  # clone(): upstream notes it fixes clicks at chunk edges
                    if i - step == 0:  # first chunk: no fade-in
                        window[:fade_size] = 1
                    elif i >= total:  # last chunk: no fade-out
                        window[-fade_size:] = 1

                    for j, (start, seg_len) in enumerate(batch_locations):
                        result[..., start:start + seg_len] += x[j, ..., :seg_len] * window[..., :seg_len]
                        counter[start:start + seg_len] += window[..., :seg_len]

                    batch_data.clear()
                    batch_locations.clear()
                    del x, arr
                    if on_chunk is not None:
                        done = min(i, total) / total * total_s if total else total_s
                        on_chunk(done, total_s)

            del mix_t, part, batch_data  # the padded working copy (and any chunk view of it) is not needed now
            estimated = result.div_(counter)
            del counter
            if padded:
                estimated = estimated[..., border:-border]
            # NaN (positions no window covers) -> 0, +-inf -> the largest finite float32, exactly as upstream's
            # np.nan_to_num; but in place: numpy's version builds several full-size boolean masks (isnan, isposinf,
            # isneginf and their temporaries), the peak of the whole worker on long songs (review 2026-10-05)
            torch.nan_to_num_(estimated, nan=0.0)
            out = estimated.numpy()  # a view of `result` (trimmed); no second copy
    return out


class IdentityStems(nn.Module):
    """Test "model": every stem = the input (selftest_identity; reconstruction error must be ~0)."""

    def __init__(self, num_stems: int) -> None:
        super().__init__()
        self.num_stems = num_stems

    def forward(self, x: torch.Tensor) -> Any:
        return x.unsqueeze(1).expand(x.shape[0], self.num_stems, *x.shape[1:]).clone()
