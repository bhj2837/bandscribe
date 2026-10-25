"""Silver tabs from tab videos (docs/TAB_VIDEO.md): read the tab a cover video shows into bars of (string, fret), in
play order, as a sheet part for ``bandscribe eval sheets``.

- ``glyphs``: digit / 'x' / bracket / rest recognition by font templates.
- ``staff``: staff lines, bar lines, box-first fret numbers, bar numbers in one image.
- ``video``: frames -> pages -> voted bars with numbers.
- ``sheet``: the sheet JSON, and the agreement of two tabs.
Videos, frames and the read tabs live under ``data/eval/tabvid/`` only (a transcriber's work; never committed).
"""
