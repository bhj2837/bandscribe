"""S3.5 v0 instrumentation estimator (M1_M2_SPEC 9.2 item 5, DESIGN S3.5). Hand rules, **[잠정]**; E3/E3b calibrate.

Why it exists: MuScriptor's ``instruments`` hard-masks every other class. Leave keys/synth out of the guitar-view
mask and keyboard notes are forced into guitar classes; leave clean guitar out and that guitar disappears. So the
guitar mask needs to know which classes actually play in the song, with evidence a person can check and fix.
Missing a keys/synth class is the safe failure (the guitar mask stays guitar-only, as in M2's "absent" case); a
false class costs guitar notes and reshuffles MuScriptor's guitar sub-class labels (review 2026-10-04: SC's false
electric_piano turned 1,805 clean-guitar notes into acoustic and added 18 % notes). The rules lean strict.

Evidence per MT3 class, combined with a noisy-OR ``p = 1 - prod(1 - w_s * e_s)``:

- **transcription** (weight 0.9), the strongest of the passes that covered the class:
  - ``presence_transcription`` (every profile): the piano+other MuScriptor pass, masked with keys/synth/strings
    (+ guitar), over sampled windows (``gtab.amt.presence``) or the whole song (``amt.presence_pass = "full"``).
    Evidence for the non-guitar classes of its mask only: share of the covered bars (bars at least half inside a
    window) with >= 2 notes of the class, ``e = clip(share / saturation, 0, 1)`` (saturation 0.25 for windows,
    which are picked where piano/other is active, 0.1 for the full pass), if the class has >=
    ``presence_min_notes`` notes **and** passes two guards (review 2026-10-04, SC's electric_piano: 37 notes in
    two adjacent windows of one section, piano stem active in 3 % of bars):
    - sections: in a sampled pass the class needs >= 4 notes in windows of at least ``min(2, sampled sections)``
      distinct sections (one burst is not a part);
    - stem agreement: one of the class's stems (piano for keys, other for strings, either for organ and synth: SW
      often puts pads in the piano stem) is active in >= ``STEM_AGREE_MIN`` (10 %) of the bars.
    Without any grid bar the windows themselves are the units.
  - ``bass_transcription`` (every profile): the full-length ``bass_mono`` pass (bass classes only), same rule as
    the mix pass. Without it both bass classes came from stem energy alone and were always both "present".
  - ``mix_transcription``: the all-class pass on the full mix (``eval`` only, baseline B0). **Not evidence** in
    production: ``eval`` must estimate exactly like ``quality`` (review 2026-10-04: otherwise ``eval`` changes the
    guitar mask, i.e. the system under test). It is reported with ``w = 0, used = false``; experiments that study
    the M2 estimator pass ``mix_as_evidence=True`` (share of bars with >= 2 notes, saturation 0.1, >=
    ``mix_class_min_notes``).
  Passes of the same model on overlapping audio are one term: ``max`` of their ``e``.
- **stem energy** (SW stems, ``energy.npz`` at 10 Hz, weight 0.6): share of bars whose stem RMS is above
  ``active_rel_db`` relative to the mix. piano -> keys classes, guitar -> guitar classes, other ->
  synth/organ/strings/brass/winds (weak, x0.5); bass -> bass classes, vocals -> voice, drums -> drums.
  **Attribution**: a stem says that *some* class of its set plays, not which one. When a transcription covered a
  class and saw the stem active in >= ``INFORMATIVE_MIN_BARS`` of its covered bars, the stem term counts only for
  the classes that transcription found (``e > 0``); the others get ``attributed = false, e = 0``. On 青と夏 the
  piano stem alone kept organ / chromatic_percussion "present" (p 0.63 / 0.51) with 0 notes in the presence pass.
  Stem activity outside the windows is reported per stem (``uncovered_active_ratio``): a part that plays only
  there is missed (the safe failure for the guitar mask; the windows sample every active stem at least once).
- **weights**: nominal when a transcription covered the class, including a presence pass that ran and was
  skipped (no window qualified: ``e = 0``, nothing covered, so the stem term stays attributed). Only classes no
  pass can cover (voice, drums, brass, winds; any class when no pass ran at all) scale the remaining weights by
  ``(0.9 + 0.6) / sum(available)``, capped at 0.9. So brass/winds (other stem x0.5) never reach 0.5 outside
  ``mix_as_evidence``: their families are reported but never "present" from stems alone.
- **hints** (``hints.instruments``): listed families ``p = max(p, 0.99)``; with an explicit list, unlisted
  families ``p = min(p, 0.05)`` - guitar excepted (a wrong hint must not delete a guitar).

``present = p >= instr.threshold`` (``instr.guitar_threshold`` for guitar classes, deliberately low). If the
guitar stem is active in >= ``instr.guitar_stem_active_ratio`` of the bars, all three guitar classes are present
(DESIGN S3.5 start value). The masks derived from this (``gtab.amt.instruments``) never drop a guitar class.

Deterministic: pure numpy on the inputs, floats rounded to 1e-4, dict order fixed (MT3 order / FAMILIES order).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gtab.amt import instruments as I

log = logging.getLogger(__name__)

W_TX = 0.9
W_MIX = W_TX  # name kept for callers of the M2 estimator
W_STEM = 0.6
W_CAP = 0.9  # a renormalised weight never exceeds the strongest nominal one
NOMINAL_WEIGHTS: dict[str, float] = {"transcription": W_TX, "stem_energy": W_STEM}
OTHER_STEM_SCALE = 0.5
MIX_SATURATION_RATIO = 0.1
PRESENCE_SATURATION_RATIO: dict[str, float] = {"sampled": 0.25, "full": 0.1}
SPAN_SLACK_S = 0.05  # a note within 50 ms of a window counts for it (latency correction shifts onsets)
WINDOW_ACTIVE_MIN_NOTES = 4  # a window "has" a class with this many notes of it (sections guard)
PRESENCE_MIN_SECTIONS = 2  # a sampled class needs active windows in this many distinct sections (fewer sampled: all)
STEM_AGREE_MIN = 0.1  # ... and one of its agreement stems active in >= 10 % of the bars
INFORMATIVE_MIN_BARS = 4  # a pass that saw a stem active in this many covered bars attributes that stem's term
HINT_PRESENT_P = 0.99
HINT_ABSENT_P = 0.05
KEYS_WARN_STEM_RATIO = 0.3

# stem -> (classes it evidences, scale)
STEM_EVIDENCE: dict[str, tuple[tuple[str, ...], float]] = {
    "piano": (tuple(I.FAMILIES["keys"]), 1.0),
    "guitar": (tuple(I.FAMILIES["guitar"]), 1.0),
    "other": (tuple(I.FAMILIES["synth"]) + ("organ",) + tuple(I.FAMILIES["strings"]) + tuple(I.FAMILIES["brass"])
              + tuple(I.FAMILIES["winds"]), OTHER_STEM_SCALE),
    "bass": (tuple(I.FAMILIES["bass"]), 1.0),
    "vocals": (("voice",), 1.0),
    "drums": (("drums",), 1.0),
}

DEFAULT_PARAMS: dict[str, Any] = {
    "threshold": 0.5, "guitar_threshold": 0.15, "guitar_stem_active_ratio": 0.05, "active_rel_db": -40.0,
    "mix_class_min_notes": 8, "presence_min_notes": 8,
}


def _r4(x: float) -> float:
    return round(float(x), 4)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def bars_from_grid(grid: Any) -> list[tuple[int, float, float]]:
    """[(bar, start_s, end_s)] from a Grid model or grid.json dict (musical bar numbers)."""
    out = []
    for b in _get(grid, "bars") or []:
        out.append((int(_get(b, "bar")), float(_get(b, "start_s")), float(_get(b, "end_s"))))
    out.sort()
    return out


def covered_bars(bars: Sequence[tuple[int, float, float]], spans: Sequence[Sequence[float]] | None) -> np.ndarray:
    """Bool per bar: at least half of the bar lies inside ``spans`` (None = every bar)."""
    if spans is None:
        return np.ones(len(bars), dtype=bool)
    out = np.zeros(len(bars), dtype=bool)
    for i, (_b, s, e) in enumerate(bars):
        inside = sum(max(0.0, min(e, float(z)) - max(s, float(a))) for a, z in spans)
        out[i] = e > s and inside >= 0.5 * (e - s)
    return out


def _bar_counts(notes: Sequence[Any], bars: Sequence[tuple[int, float, float]],
                spans: Sequence[Sequence[float]] | None) -> tuple[dict[str, np.ndarray], dict[str, int],
                                                                   dict[str, np.ndarray]]:
    """({class: notes per bar}, {class: n_notes}, {class: notes per span}); notes outside ``spans`` are ignored."""
    starts = np.array([b[1] for b in bars], dtype=float)
    ends = np.array([b[2] for b in bars], dtype=float)
    per_bar: dict[str, np.ndarray] = {}
    per_span: dict[str, np.ndarray] = {}
    n_notes: dict[str, int] = {}
    n_sp = 0 if spans is None else len(spans)
    for n in notes:
        cls = _get(n, "instrument")
        if cls is None or cls not in I.MT3_GROUPS:
            continue
        on = float(_get(n, "onset_s"))
        if spans is not None:  # a little slack: latency correction moves onsets by a few ms
            k_sp = next((k for k, (a, z) in enumerate(spans)
                         if float(a) - SPAN_SLACK_S <= on < float(z) + SPAN_SLACK_S), None)
            if k_sp is None:
                continue
            per_span.setdefault(cls, np.zeros(n_sp, dtype=np.int64))[k_sp] += 1
        n_notes[cls] = n_notes.get(cls, 0) + 1
        if len(bars) == 0:
            continue
        k = int(np.searchsorted(starts, on, side="right") - 1)
        if 0 <= k < len(bars) and on < ends[k]:
            per_bar.setdefault(cls, np.zeros(len(bars), dtype=np.int64))[k] += 1
    return per_bar, n_notes, per_span


def mix_evidence(notes: Sequence[Any], bars: Sequence[tuple[int, float, float]], *, min_notes: int,
                 classes: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """Per class (default every MT3 class) of a full-length pass: n_notes, bars with >= 2 notes, active_ratio,
    e (clip(ratio / 0.1)). Used for the mix pass and the bass pass."""
    per_bar, n_notes, _ = _bar_counts(notes, bars, None)
    out: dict[str, dict[str, Any]] = {}
    for cls in (I.MT3_GROUPS if classes is None else classes):
        cnt = per_bar.get(cls)
        active = int((cnt >= 2).sum()) if cnt is not None else 0
        ratio = active / len(bars) if len(bars) else 0.0
        nn = n_notes.get(cls, 0)
        e = min(1.0, ratio / MIX_SATURATION_RATIO) if nn >= min_notes else 0.0
        out[cls] = {"n_notes": nn, "active_bars": active, "active_ratio": _r4(ratio), "e": _r4(e)}
    return out


def presence_evidence(notes: Sequence[Any], bars: Sequence[tuple[int, float, float]], *,
                      spans: Sequence[Sequence[float]] | None, classes: Sequence[str], min_notes: int,
                      saturation: float, window_sections: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """Per class of ``classes``: the presence-pass evidence on the bars it covered (see the module docstring).

    ``spans`` None = the whole song was transcribed; ``window_sections`` = the home section id of each span
    (default: every span its own section). Keys: n_notes, covered_bars, active_bars, active_ratio, windows,
    active_windows (>= 4 notes in the window), active_window_ratio, sampled_sections, active_sections,
    sections_needed, saturation, e (0 unless n_notes >= min_notes and the sections guard holds). The stem-agreement
    guard needs the stems and is applied by ``estimate``.
    """
    cov = covered_bars(bars, spans)
    n_cov = int(cov.sum())
    per_bar, n_notes, per_span = _bar_counts(notes, bars, spans)
    n_win = 0 if spans is None else len(spans)
    ws = list(window_sections) if window_sections is not None and len(window_sections) == n_win else \
        [f"W{k + 1}" for k in range(n_win)]
    n_secs = len(set(ws))
    need = min(PRESENCE_MIN_SECTIONS, n_secs) if spans is not None else 0
    out: dict[str, dict[str, Any]] = {}
    for cls in classes:
        cnt = per_bar.get(cls)
        active = int(((cnt >= 2) & cov).sum()) if cnt is not None else 0
        sp = per_span.get(cls)
        act_k = [k for k in range(n_win) if sp is not None and sp[k] >= WINDOW_ACTIVE_MIN_NOTES]
        act_win = len(act_k)
        act_secs = len({ws[k] for k in act_k})
        if n_cov:
            ratio = active / n_cov
        elif n_win:  # no grid bar inside the windows: the windows are the units
            ratio = act_win / n_win
        else:
            ratio = 0.0
        nn = n_notes.get(cls, 0)
        e = min(1.0, ratio / saturation) if nn >= min_notes and saturation > 0 and act_secs >= need else 0.0
        out[cls] = {"n_notes": nn, "covered_bars": n_cov, "active_bars": active, "active_ratio": _r4(ratio),
                    "windows": n_win, "active_windows": act_win,
                    "active_window_ratio": _r4(act_win / n_win) if n_win else None,
                    "sampled_sections": n_secs if spans is not None else None,
                    "active_sections": act_secs if spans is not None else None,
                    "sections_needed": need if spans is not None else None,
                    "saturation": saturation, "e": _r4(e)}
    return out


def stem_bar_activity(energy: Mapping[str, np.ndarray], bars: Sequence[tuple[int, float, float]], *,
                      active_rel_db: float, fps: float = 10.0) -> dict[str, np.ndarray]:
    """Per stem: bool per bar, True if the stem's bar RMS is above ``active_rel_db`` relative to the mix's.

    Bar RMS comes from the 10 Hz dB series: mean power over the frames inside the bar.
    """
    mix = np.asarray(energy.get("mix"), dtype=np.float64) if "mix" in energy else None
    out: dict[str, np.ndarray] = {}
    if mix is None or len(bars) == 0:
        return out

    def bar_db(series: np.ndarray) -> np.ndarray:
        p = 10.0 ** (series / 10.0)
        vals = np.full(len(bars), -200.0)
        for i, (_b, s, e) in enumerate(bars):
            a, z = int(np.floor(s * fps)), int(np.ceil(e * fps))
            a, z = max(0, a), min(len(series), max(z, a + 1))
            if z > a:
                vals[i] = 10.0 * np.log10(np.mean(p[a:z]) + 1e-20)
        return vals

    mix_bar = bar_db(mix)
    for stem in STEM_EVIDENCE:
        if stem in energy:
            out[stem] = (bar_db(np.asarray(energy[stem], dtype=np.float64)) - mix_bar) > active_rel_db
    return out


def renormalised_weights(available: Sequence[str]) -> dict[str, float]:
    """Effective weight per available evidence type (``NOMINAL_WEIGHTS`` keys), scaled up when a type is missing."""
    avail = [a for a in NOMINAL_WEIGHTS if a in available]
    if not avail:
        return {}
    scale = sum(NOMINAL_WEIGHTS.values()) / sum(NOMINAL_WEIGHTS[a] for a in avail)
    return {a: _r4(min(W_CAP, NOMINAL_WEIGHTS[a] * scale)) if scale != 1.0 else NOMINAL_WEIGHTS[a] for a in avail}


def _stems_of(cls: str) -> list[str]:
    return [stem for stem, (evidenced, _scale) in STEM_EVIDENCE.items() if cls in evidenced]


def agreement_stems(cls: str) -> list[str]:
    """Stems that can confirm a presence-pass class: its evidencing stems, plus piano for synth classes (SW often
    puts pads and leads in the piano stem; the presence view is piano + other)."""
    out = _stems_of(cls)
    if I.FAMILY_OF.get(cls) == "synth" and "piano" not in out:
        out = ["piano"] + out
    return out


def estimate(*, profile: str, bars: Sequence[tuple[int, float, float]], mix_notes: Sequence[Any] | None,
             energy: Mapping[str, np.ndarray] | None, sections: Sequence[Any] | None, hint: Any,
             params: Mapping[str, Any], guitar_view: str = "guitar_mono", guitar_mask: str = "guitar+present",
             piano_other_mode: str = "keys+guitar", energy_fps: float = 10.0,
             presence: Mapping[str, Any] | None = None, bass_notes: Sequence[Any] | None = None,
             mix_as_evidence: bool = False) -> dict[str, Any]:
    """The ``instrumentation.json`` document (schema §6.5 ``Instrumentation``), built deterministically.

    ``profile`` is recorded only: every profile estimates the same way (``eval``'s mix pass is reported, not used;
    ``mix_as_evidence`` is for experiments). ``presence``: the piano+other presence transcription, ``{"mode":
    "sampled"|"full", "notes": [...], "spans": [[s, e], ...] | None (None = whole song), "window_sections": [id per
    span] | None, "classes": [...] (default: the non-guitar classes of the piano+other mask), "skipped": reason |
    None, "summary": {...} (copied into the document)}``. ``bass_notes``: the bass pass's notes (bass classes).
    """
    prm = dict(DEFAULT_PARAMS)
    prm.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})
    families_hint = I.parse_hint(hint)
    n_bars = len(bars)
    everywhere = np.ones(n_bars, dtype=bool)
    stem_act = stem_bar_activity(energy or {}, bars, active_rel_db=float(prm["active_rel_db"]), fps=energy_fps)
    stem_ratio = {s: (float(a.mean()) if len(a) else 0.0) for s, a in stem_act.items()}
    min_full = int(prm["mix_class_min_notes"])

    # transcription sources: name -> {"ev": {class: evidence}, "cov": covered bars, "notes", "spans", "used"}
    tx: dict[str, dict[str, Any]] = {}
    mix_info: dict[str, dict[str, Any]] | None = None
    if presence is not None:
        pres_classes = list(presence.get("classes") or I.presence_evidence_classes(piano_other_mode))
        if presence.get("skipped"):
            ev = {c: {"n_notes": 0, "covered_bars": 0, "e": 0.0, "skipped": presence.get("skipped")}
                  for c in pres_classes}
            tx["presence_transcription"] = {"ev": ev, "cov": np.zeros(n_bars, dtype=bool), "notes": [],
                                            "spans": [], "mode": presence.get("mode")}
        else:
            spans = presence.get("spans")
            ev = presence_evidence(presence.get("notes") or [], bars, spans=spans, classes=pres_classes,
                                   min_notes=int(prm["presence_min_notes"]),
                                   saturation=PRESENCE_SATURATION_RATIO["full" if spans is None else "sampled"],
                                   window_sections=presence.get("window_sections"))
            for c, e in ev.items():  # stem-agreement guard (needs stem energy; without it the guard is off)
                stems = [st for st in agreement_stems(c) if st in stem_ratio]
                agree = max((stem_ratio[st] for st in stems), default=None)
                e["stem_agree"] = None if agree is None else _r4(agree)
                if agree is not None and agree < STEM_AGREE_MIN:
                    e["e"] = 0.0
            tx["presence_transcription"] = {"ev": ev, "cov": covered_bars(bars, spans), "notes": presence.get("notes")
                                            or [], "spans": spans, "mode": "full" if spans is None else "sampled"}
    if bass_notes is not None:
        tx["bass_transcription"] = {"ev": mix_evidence(bass_notes, bars, min_notes=min_full, classes=I.BASS_CLASSES),
                                    "cov": everywhere, "notes": bass_notes, "spans": None}
    if mix_notes is not None:
        mixe = mix_evidence(mix_notes, bars, min_notes=min_full)
        if mix_as_evidence:
            tx["mix_transcription"] = {"ev": mixe, "cov": everywhere, "notes": mix_notes, "spans": None}
        else:
            mix_info = mixe

    classes: dict[str, Any] = {}
    warnings: list[str] = []
    guitar_stem_ratio = stem_ratio.get("guitar", 0.0)
    guitar_forced = guitar_stem_ratio >= float(prm["guitar_stem_active_ratio"])
    for cls in I.MT3_GROUPS:
        fam = I.FAMILY_OF.get(cls, "")
        sources: dict[str, dict[str, Any]] = {}
        covering = [name for name, src in tx.items() if cls in src["ev"]]
        e_tx = max((float(tx[name]["ev"][cls]["e"]) for name in covering), default=0.0)
        found = e_tx > 0.0
        stems_ev: dict[str, dict[str, Any]] = {}
        for stem in _stems_of(cls):
            if stem not in stem_ratio:
                continue
            _ev, scale = STEM_EVIDENCE[stem]
            act = stem_act[stem]
            item: dict[str, Any] = {"active_ratio": _r4(stem_ratio[stem]), "scale": scale}
            attributed = True
            if covering:
                cov = np.zeros(n_bars, dtype=bool)
                for name in covering:
                    cov |= tx[name]["cov"]
                seen = int((act & cov).sum())
                item["covered_active_bars"] = seen
                item["uncovered_active_ratio"] = _r4(float((act & ~cov).mean())) if n_bars else 0.0
                attributed = found or seen < INFORMATIVE_MIN_BARS
                item["attributed"] = attributed
            # organ is evidenced by both the piano and the other stem: each is its own noisy-OR term
            item["e"] = _r4(scale * stem_ratio[stem]) if attributed else 0.0
            stems_ev[stem] = item
        weights = renormalised_weights((["transcription"] if covering else []) + (["stem_energy"] if stems_ev else []))
        terms: list[float] = []
        for name in covering:
            src = dict(tx[name]["ev"][cls], w=weights["transcription"])
            if name == "presence_transcription":
                src["mode"] = tx[name]["mode"]
            sources[name] = src
        if covering:
            terms.append(weights["transcription"] * e_tx)
        if stems_ev:
            sources["stem_energy"] = {"w": weights["stem_energy"], "stems": stems_ev}
            terms += [weights["stem_energy"] * ev["e"] for ev in stems_ev.values()]
        if mix_info is not None:
            sources["mix_transcription"] = dict(mix_info[cls], w=0.0, used=False)
        p = 1.0 - float(np.prod([1.0 - t for t in terms])) if terms else 0.0
        if families_hint is not None:
            if fam in families_hint:
                p = max(p, HINT_PRESENT_P)
                sources["hint"] = {"families": list(families_hint), "effect": "listed"}
            elif fam != "guitar":
                p = min(p, HINT_ABSENT_P)
                sources["hint"] = {"families": list(families_hint), "effect": "not_listed"}
        thr = float(prm["guitar_threshold"] if fam == "guitar" else prm["threshold"])
        present = p >= thr
        if fam == "guitar" and guitar_forced:
            present = True
            sources["guitar_stem_rule"] = {"active_ratio": _r4(guitar_stem_ratio),
                                           "min_ratio": float(prm["guitar_stem_active_ratio"])}
        classes[cls] = {"p": _r4(min(1.0, max(0.0, p))), "present": bool(present), "sources": sources}

    families = {fam: _r4(max(classes[c]["p"] for c in members)) for fam, members in I.FAMILIES.items()}

    # per-section family activity: a family is active in a bar if its stem is active there or a transcription
    # that counts as evidence has >= 2 notes of an accepted (e > 0) class of it in a bar it covered.
    tx_notes = [(src["notes"], src["spans"], {c for c, ev in src["ev"].items() if ev["e"] > 0})
                for src in tx.values() if src["notes"]]
    fam_bar = _family_bar_activity(tx_notes, stem_act, bars)
    per_section = []
    bar_index = {b[0]: i for i, b in enumerate(bars)}
    for sec in sections or []:
        sid = str(_get(sec, "id"))
        sb, eb = int(_get(sec, "start_bar")), int(_get(sec, "end_bar"))
        idx = [bar_index[b] for b in range(sb, eb) if b in bar_index]  # half-open bar range
        fa = {fam: (_r4(float(np.mean(act[idx]))) if idx else 0.0) for fam, act in fam_bar.items()}
        per_section.append({"section": sid, "family_active": fa})

    keys_counts = [sum(int(src["ev"][c].get("n_notes", 0)) for c in I.FAMILIES["keys"] if c in src["ev"])
                   for src in tx.values() if any(c in src["ev"] for c in I.FAMILIES["keys"])]
    if stem_ratio.get("piano", 0.0) >= KEYS_WARN_STEM_RATIO and keys_counts:
        if max(keys_counts) < int(prm["mix_class_min_notes"]):
            warnings.append("건반(piano) 스템 에너지는 있는데 전사에 건반 음이 거의 없음: 건반이 있는지 확인하세요.")
    if presence is None and not mix_as_evidence:
        warnings.append("건반·신스 전사가 없어 스템 에너지와 힌트로만 편성을 정했습니다(가중치 재정규화).")
    elif presence is not None and presence.get("skipped") and \
            max(stem_ratio.get("piano", 0.0), stem_ratio.get("other", 0.0)) >= KEYS_WARN_STEM_RATIO:
        warnings.append("piano/other 스템에 소리가 있는데 건반·신스 확인 전사 구간을 고르지 못했습니다"
                        f"({presence.get('skipped')}): 건반·신스는 스템 에너지로만 판단했습니다.")
    if not stem_ratio:
        warnings.append("스템 에너지(energy.npz)가 없어 전사와 힌트로만 편성을 정했습니다.")

    doc_present = {c: classes[c]["present"] for c in classes}
    inst = {"classes": {c: {"present": v} for c, v in doc_present.items()}}
    guitar_pass = I.mask_guitar_view(inst, guitar_mask)
    # Pass-1 views keep their own masks; the guitar pass (amt_gtr) reads "guitar_pass". The guitar view id is
    # listed too (spec example) unless it is a pass-1 view itself (E2: mix_mono), whose pass-1 mask stays.
    masks: dict[str, list[str] | None] = {}
    if guitar_view not in I.pass1_masks(piano_other_mode):
        masks[guitar_view] = guitar_pass
    masks.update(I.pass1_masks(piano_other_mode))
    masks["guitar_pass"] = guitar_pass
    thresholds = {k: prm[k] for k in DEFAULT_PARAMS}
    thresholds.update({"w_mix": W_TX, "w_transcription": W_TX, "w_stem": W_STEM, "w_cap": W_CAP,
                       "mix_saturation_ratio": MIX_SATURATION_RATIO,
                       "presence_saturation_ratio": dict(PRESENCE_SATURATION_RATIO),
                       "presence_min_sections": PRESENCE_MIN_SECTIONS, "window_active_min_notes":
                       WINDOW_ACTIVE_MIN_NOTES, "stem_agree_min": STEM_AGREE_MIN,
                       "informative_min_bars": INFORMATIVE_MIN_BARS, "mix_as_evidence": bool(mix_as_evidence),
                       "guitar_mask": guitar_mask, "piano_other_instruments": piano_other_mode})
    pres_summary = None
    if presence is not None:
        pev = tx["presence_transcription"]["ev"]
        rejected = {}
        for c, ev in pev.items():
            if ev.get("n_notes", 0) >= int(prm["presence_min_notes"]) and not ev["e"]:
                if ev.get("sections_needed") and (ev.get("active_sections") or 0) < ev["sections_needed"]:
                    rejected[c] = "sections"
                elif ev.get("stem_agree") is not None and ev["stem_agree"] < STEM_AGREE_MIN:
                    rejected[c] = "stem"
        pres_summary = dict(presence.get("summary") or {})
        pres_summary.update(mode=presence.get("mode"), skipped=presence.get("skipped"),
                            class_notes={c: pev[c]["n_notes"] for c in pev if pev[c]["n_notes"]},
                            rejected=rejected, present=[c for c in pev if classes[c]["present"]])
    return {
        "format": "gtab.instrumentation/1",
        "profile": profile,
        "classes": classes,
        "families": families,
        "masks": masks,
        "per_section": per_section,
        "thresholds": thresholds,
        "hints": {"instruments": hint if isinstance(hint, (str, type(None))) else list(hint),
                  "families": families_hint},
        "warnings": warnings,
        "presence": pres_summary,
    }


def _family_bar_activity(tx_notes: Sequence[tuple[Sequence[Any], Sequence[Sequence[float]] | None, set[str] | None]],
                         stem_act: Mapping[str, np.ndarray],
                         bars: Sequence[tuple[int, float, float]]) -> dict[str, np.ndarray]:
    n = len(bars)
    fam_act = {fam: np.zeros(n, dtype=bool) for fam in I.FAMILIES}
    for stem, (evidenced, _scale) in STEM_EVIDENCE.items():
        act = stem_act.get(stem)
        if act is None:
            continue
        fams = {I.FAMILY_OF[c] for c in evidenced}
        if stem == "other":
            continue  # too ambiguous for a family-level activity map (weak evidence only)
        for fam in fams:
            fam_act[fam] |= act
    if not n:
        return fam_act
    for notes, spans, only in tx_notes:
        if not notes:
            continue
        per_bar, _n, _sp = _bar_counts(notes, bars, spans)
        cov = covered_bars(bars, spans)
        for cls, c in per_bar.items():
            if only is not None and cls not in only:
                continue
            fam_act[I.FAMILY_OF[cls]] |= (c >= 2) & cov
    return fam_act


def estimated_families(instrumentation: Mapping[str, Any]) -> list[str]:
    """FAMILIES keys with at least one present class (the b13 comparison with ``families_present``)."""
    present = {c for c, ev in (instrumentation.get("classes") or {}).items()
               if (ev.get("present") if isinstance(ev, Mapping) else getattr(ev, "present", False))}
    return [fam for fam, members in I.FAMILIES.items() if any(c in present for c in members)]


def guitar_class_omissions(instrumentation: Mapping[str, Any]) -> int:
    """Guitar classes missing from the guitar-pass mask (must be 0 in every case; b13 / E3 constraint)."""
    masks = instrumentation.get("masks") or {}
    return I.guitar_omissions(masks.get("guitar_pass", I.mask_guitar_only()))
