"""Vendored model code from ZFTurbo/Music-Source-Separation-Training (MIT), pinned; see VENDOR.md.

Importing this package is cheap and torch-free. ``models.bs_roformer`` (torch, einops, beartype,
rotary-embedding-torch) and ``demix`` / ``loader`` (torch, pyyaml) are imported only by the ``sep_msst`` worker
in the gpu env - the torch boundary of M1_M2_SPEC 0 is extended to this package (A2).
"""

MSST_REPO = "https://github.com/ZFTurbo/Music-Source-Separation-Training"
MSST_COMMIT = "84b1eac0887756b4f1a9d7a1ff49105939749ed2"
