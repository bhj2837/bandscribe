"""Synthetic check of *scenario-routing features* computed on a stereo guitar stem.
Question: can cheap stereo statistics tell apart A (uncorrelated doubles L/R + centre lead),
fake/ADT doubles, two players panned apart, centred parts (B), mono (F), and do they
flag the harmonised-twin-lead trap?  Synthetic, idealised guitar-only stem. Indicative only.
Reuses the signal generator from synth_exp.py.
"""
import sys, io
import numpy as np
from scipy import signal

import synth_exp as S   # guarded by __main__, safe to import
SR, N = S.SR, S.N
# stdout already wrapped by synth_exp

NFFT, HOP = 4096, 1024

def stft2(x):
    return S.stft(x[0]), S.stft(x[1])

def features(st):
    L, R = st[0], st[1]
    M, Sd = (L + R) / 2, (L - R) / 2
    em, es = np.mean(M ** 2), np.mean(Sd ** 2)
    side_mid_db = 10 * np.log10(es / (em + 1e-12) + 1e-12)
    # broadband zero-lag correlation ("phase meter")
    corr0 = np.sum(L * R) / np.sqrt(np.sum(L ** 2) * np.sum(R ** 2) + 1e-12)
    # max normalised cross-correlation over +-40 ms lags (excluding |lag|<1 ms)
    maxlag = int(0.04 * SR)
    c = signal.correlate(L, R, mode='full', method='fft')
    lags = np.arange(-len(L) + 1, len(L))
    sel = (np.abs(lags) <= maxlag) & (np.abs(lags) > int(0.001 * SR))
    norm = np.sqrt(np.sum(L ** 2) * np.sum(R ** 2)) + 1e-12
    k = np.argmax(np.abs(c[sel]))
    xc_off = c[sel][k] / norm
    xc_lag_ms = lags[sel][k] / SR * 1000
    # per-bin coherence (time-smoothed ~150 ms) weighted by power
    XL, XR = stft2(st)
    lam = np.exp(-HOP / SR / 0.15)
    sm = lambda A: signal.lfilter([1 - lam], [1, -lam], A, axis=-1)
    cross = sm(XL * np.conj(XR)); pl = sm(np.abs(XL) ** 2); pr = sm(np.abs(XR) ** 2)
    coh = np.abs(cross) / np.sqrt(pl * pr + 1e-20)
    w = np.abs(XL) ** 2 + np.abs(XR) ** 2
    coh_w = np.sum(coh * w) / np.sum(w)
    # ILD distribution (energy-weighted)
    ild = 20 * np.log10((np.abs(XL) + 1e-9) / (np.abs(XR) + 1e-9))
    f_centre = np.sum(w[np.abs(ild) < 3]) / np.sum(w)
    f_hard = np.sum(w[np.abs(ild) > 15]) / np.sum(w)
    # energy-weighted mean |ILD| of *coherent* bins (coh>0.9) -> where the correlated stuff sits
    cm = coh > 0.9
    ild_coh = np.sum((np.abs(ild) * w)[cm]) / (np.sum(w[cm]) + 1e-20) if cm.any() else np.nan
    f_coh = np.sum(w[cm]) / np.sum(w)
    return dict(SM=side_mid_db, r0=corr0, xcoff=xc_off, lag=xc_lag_ms, coh=coh_w,
                cen=f_centre, hard=f_hard, fcoh=f_coh, ildcoh=ild_coh)


def twin_lead(rng, semis=4, gain=9.0):
    notes = [(0.5, 329.63, 0.5), (1.0, 392.0, 0.25), (1.25, 440.0, 0.75), (2.0, 493.88, 1.0),
             (3.0, 587.33, 0.5), (3.5, 659.26, 1.0), (4.5, 587.33, 0.25), (4.75, 493.88, 0.25),
             (5.0, 440.0, 0.5), (5.5, 392.0, 0.5), (6.0, 329.63, 1.5)]
    buf = np.zeros(N + SR)
    for t0, f, d in notes:
        t0 += rng.normal(0, 0.008)
        S.place(buf, S.tone(f * 2 ** (semis / 12), d + 0.05, decay=1.2, rng=rng,
                            vib_depth_cents=25 if d >= 0.75 else 0), t0)
    y = S.amp_cab(buf[:N], gain, 5500.0, rng)
    return y / np.sqrt(np.mean(y ** 2)) * 0.1


def main():
    rng = np.random.default_rng(1)
    r1, r2, r3, r4 = (S.rhythm_take(np.random.default_rng(s)) for s in (11, 12, 13, 14))
    lead = S.lead_line(np.random.default_rng(21))
    lead2 = twin_lead(np.random.default_rng(22))
    z = np.zeros(N)
    hardL = lambda x: np.stack([x, z]); hardR = lambda x: np.stack([z, x])
    ctr = lambda x: S.pan(x, 0.0)
    d15 = int(0.015 * SR)
    adt = np.concatenate([np.zeros(d15), r1[:-d15]])
    sc = {}
    sc['A  doubles L/R + centre lead 0dB'] = hardL(r1) + hardR(r2) + ctr(lead)
    sc['A  doubles L/R, rhythm-only section'] = hardL(r1) + hardR(r2)
    sc['A  doubles L/R + centre lead -6dB'] = hardL(r1) + hardR(r2) + ctr(0.5 * lead)
    sc['A  doubles L/R + lead w/ stereo reverb'] = hardL(r1) + hardR(r2) + S.stereo_reverb(ctr(lead), rng=rng)
    sc['A  quad (2 hard + 2 at 60%) + centre lead'] = hardL(r1) + hardR(r2) + S.pan(r3, -0.6) + S.pan(r4, 0.6) + ctr(lead)
    sc['ADT fake double (15ms) L/R + centre lead'] = hardL(r1) + hardR(adt) + ctr(lead)
    sc['2 players: rhythm 70%L, lead 70%R'] = S.pan(r1, -0.7) + S.pan(lead, 0.7)
    sc['2 players: rhythm hard L, lead hard R'] = hardL(r1) + hardR(lead)
    sc['B  rhythm centre + lead centre'] = ctr(r1) + ctr(lead)
    sc['B  rhythm centre + lead 20%R'] = ctr(r1) + S.pan(lead, 0.2)
    sc['B  both centre + stereo reverb'] = S.stereo_reverb(ctr(r1) + ctr(lead), rng=rng)
    sc['F  mono sum of A'] = ctr((r1 + r2) / np.sqrt(2) + lead)
    sc['TRAP twin leads L/R (harm.) + doubles L/R'] = hardL(r1 + lead) + hardR(r2 + lead2)
    sc['TRAP twin leads L/R, no rhythm'] = hardL(lead) + hardR(lead2)
    hdr = f"{'scenario':44s} {'S/M dB':>7s} {'r0':>6s} {'xcOff':>6s} {'lag':>6s} {'coh':>5s} {'cen':>5s} {'hard':>5s} {'fcoh':>5s} {'|ILD|coh':>8s}"
    print(hdr)
    for k, st in sc.items():
        f = features(st)
        print(f"{k:44s} {f['SM']:7.1f} {f['r0']:6.2f} {f['xcoff']:6.2f} {f['lag']:6.1f} {f['coh']:5.2f} "
              f"{f['cen']:5.2f} {f['hard']:5.2f} {f['fcoh']:5.2f} {f['ildcoh']:8.1f}")


if __name__ == '__main__':
    main()
