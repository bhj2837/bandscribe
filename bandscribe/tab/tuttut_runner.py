"""tuttut baseline runner (E12). Runs in ``envs/tuttut`` (Python 3.10, tuttut 0.0.6 with its pinned numpy
1.23.2 / networkx 2.8.6 / pretty-midi 0.2.9; MIT, Nathan Candre): **stdlib + tuttut only, never import
bandscribe here.**

``python tuttut_runner.py request.json result.json`` where request = {"tuning": [open MIDI pitches, lowest
string first], "tracks": [{"id": str, "events": [[onset_s, [pitch, ...]], ...]}]} (events strictly increasing in
time, at least 10 ms apart) and result = {"tracks": [{"id", "events": [[[string, fret], ...] per event, or null],
"error": str | null, "seconds": float}]}. Strings are numbered 1 = highest (tuttut's index 0 = thinnest string).
Every note of an event is written at the event's onset, so tuttut sees the same chords as bandscribe's fretter.
"""

from __future__ import annotations

import json
import sys
import time
import traceback


def _tab(events, tuning_pitches):
    import numpy as np
    import pretty_midi
    from tuttut.logic.tab import Tab
    from tuttut.logic.theory import Tuning

    np.seterr(divide="ignore")
    names = [pretty_midi.note_number_to_name(p) for p in reversed(tuning_pitches)]  # thin to thick
    midi = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=25)
    for k, (on, pitches) in enumerate(events):
        nxt = events[k + 1][0] if k + 1 < len(events) else on + 0.5
        end = max(on + 0.01, min(nxt, on + 2.0))
        for p in sorted(set(int(x) for x in pitches)):
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=float(on), end=float(end)))
    midi.instruments.append(inst)
    tab = Tab("t", Tuning(names), midi)
    n_strings = len(tuning_pitches)
    out = []
    for measure in tab.tab["measures"]:
        for ev in measure["events"]:
            if "notes" in ev:
                out.append((int(ev["time_ticks"]), [[int(n["string"]) + 1, int(n["fret"])] for n in ev["notes"]]))
    out.sort(key=lambda x: x[0])
    ticks = [midi.time_to_tick(float(on)) for on, _p in events]
    by_tick = {t: sf for t, sf in out}
    result = []
    for t in ticks:
        sf = by_tick.get(t)
        result.append([s for s in sf if 1 <= s[0] <= n_strings] if sf is not None else None)
    return result


def main(argv):
    req = json.load(open(argv[1], encoding="utf-8"))
    tuning = [int(p) for p in req.get("tuning") or [40, 45, 50, 55, 59, 64]]
    res = []
    for tr in req["tracks"]:
        t0 = time.monotonic()
        try:
            res.append({"id": tr["id"], "events": _tab(tr["events"], tuning), "error": None,
                        "seconds": round(time.monotonic() - t0, 3)})
        except Exception as e:  # noqa: BLE001 - one failing track must not lose the others
            res.append({"id": tr["id"], "events": None, "error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc(limit=3), "seconds": round(time.monotonic() - t0, 3)})
    with open(argv[2], "w", encoding="utf-8") as f:
        json.dump({"format": "bandscribe.tuttut_result/1", "tracks": res}, f)


if __name__ == "__main__":
    main(sys.argv)
