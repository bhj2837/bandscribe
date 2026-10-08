"""Bass (DESIGN 5, M5): F0 trackers, octave/pitch anchors, note clean-up and playing techniques.

- ``bandscribe.bass.f0``: the three F0 tracks on one 10 ms grid, per-note evidence, the 2-of-3 anchor vote,
  frame-level tracker accuracy.
- ``bandscribe.bass.clean``: the S11 note clean-up on MuScriptor's bass (anchor, kick rejection, fragment merge,
  slide chains, F0 gap fill, the AND policy) and technique suggestions (slide, vibrato, dead note).
- ``bandscribe.bass.stage``: the ``bass_f0`` (GPU) and ``bass`` (CPU) pipeline stages.
- ``bandscribe.bass.experiments``: E13 and the M5 measurements.
"""
