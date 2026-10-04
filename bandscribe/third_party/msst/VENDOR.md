# Vendored: Music-Source-Separation-Training (MSST)

- Upstream: https://github.com/ZFTurbo/Music-Source-Separation-Training
- Commit: `84b1eac0887756b4f1a9d7a1ff49105939749ed2` (2026-09-26)
- License: MIT (`LICENSE`, copied verbatim)
- Why vendored (DESIGN S3 [안2], M1_M2_SPEC 9.1): only the BS-RoFormer model code is needed. MSST's
  `inference.py` / `utils` pull librosa, matplotlib (module level), tqdm, omegaconf and ml_collections, none
  of which belong in the gpu env. The inference loop is our own port (`demix.py`), the config loader too
  (`loader.py`).

## Files

`tests/test_sep_vendor.py` checks the local sha256 column. "upstream" = the file at the commit above, fetched
from `raw.githubusercontent.com` on 2026-09-30.

| Local file | Upstream path | Upstream sha256 | Local sha256 |
|---|---|---|---|
| `LICENSE` | `LICENSE` | `3282dc057695ef5b9a64909a7092ca40b2c292c232580fc6ace6e5d665cc0207` | `3282dc057695ef5b9a64909a7092ca40b2c292c232580fc6ace6e5d665cc0207` |
| `models/bs_roformer/__init__.py` | `models/bs_roformer/__init__.py` | `fc236d3158ba954e77ec7f7d4cae8eec924bebc3150db4885ab946d693f1e862` | `886a86a81cc477443561877761ab3e362ccc94c37c98dacb6afb733385ae3c10` |
| `models/bs_roformer/bs_roformer.py` | `models/bs_roformer/bs_roformer.py` | `a6f670325cdb8a7a212914f62602119b23eb888df9bcfe4a85f9be74728b3ef4` | `238a5c184ef8a167ea131e77f55be4ec3ddeae715e9838150cba79492f293287` |
| `models/bs_roformer/attend.py` | `models/bs_roformer/attend.py` | `c7abbc40a3fd20ff7f6001fa9f8ee9ad5df6452272712a5e719b8f8468bf2223` | `e2da12060f57938ced42e3dc9bb18cbe30e81dae7c6a1d9fd246723be954c992` |

Not vendored, written here: `__init__.py` (package docstring + pinned commit), `models/__init__.py`,
`demix.py` (port of `utils/model_utils.py::demix`, generic branch), `loader.py` (yaml + strict weight load).

## Local patches (all verified necessary by the 2026-09-30 critique probe)

1. `models/bs_roformer/__init__.py` **trimmed** to `from .bs_roformer import BSRoformer`. Upstream also imports
   `bs_conformer`, `mel_band_roformer` and `mel_band_conformer`; the mel-band ones import librosa, which the
   gpu env does not have.
2. `models/bs_roformer/bs_roformer.py`: `from models.bs_roformer.attend import Attend` ->
   `from .attend import Attend` (package-relative; upstream assumes the repo root on `sys.path`).
3. `models/bs_roformer/attend.py`: the **CPU** path of `flash_attn` called the deprecated
   `torch.backends.cuda.sdp_kernel(...)` context manager, which emits a FutureWarning on every call; it now
   calls `F.scaled_dot_product_attention` directly (upstream's CPU config enabled every backend, so the
   computation is the same). The CUDA inference path is unchanged: it already uses
   `torch.nn.attention.sdpa_kernel` with cuDNN -> flash -> mem-efficient -> math priority; on sm_75 (RTX 2060)
   flash kernels do not exist and SDPA picks mem-efficient / math.
4. No module-level matplotlib / librosa anywhere (nothing to patch in the three files beyond 1).
5. 2026-10-04: the `# gtab local patch` comments in the three files now say `# bandscribe local patch`
   (project rename, docs/RENAME.md). Comment-only; the local sha256 column was updated for it.

`bs_roformer.py` keeps its optional `PoPE_pytorch` import inside `try/except` (not installed; the SW config
does not use PoPE).

## Behaviour notes

- `BSRoformer(zero_dc=True)` is the upstream default and is **kept**: the DC bin of every stem is zeroed in the
  iSTFT. Consequently `leftover = mix - sum(stems)` contains the mix's DC component. Recorded as
  `"zero_dc": true` in every `sep.json`.
- `demix.py` keeps the overlap-add buffers (`result`, `counter`) and the whole mix on the **CPU** in float32.
  Upstream HEAD (this commit) moved them onto the GPU when they fit in half the free VRAM; we do not copy that
  because VRAM is the scarce resource on this PC (M1_M2_SPEC A3). The arithmetic is otherwise identical
  (`tests/test_sep_upstream_parity.py` compares against a pristine checkout, SNR > 60 dB per stem).
- AMP: upstream `demix` wraps inference in `torch.cuda.amp.autocast(enabled=training.use_amp)` (deprecated API,
  fp16 on CUDA only); we use `torch.autocast("cuda", dtype=torch.float16)` - same effect.

## Updating

Fetch the three files at the new commit, re-apply the patches above, update the table (sha256 of both columns)
and `MSST_COMMIT` in `__init__.py`, and bump `sep.msst_commit` in `bandscribe/defaults.toml` (integrator) so every
`sep` stage key changes.
