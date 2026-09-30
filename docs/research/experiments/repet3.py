"""Harder variant: 3 repetitions of the riff (independent takes); lead present only in reps 2-3
(rep 1 = rhythm-only verse). Background model = median or min across aligned reps, optionally with
alignment error. Score only the lead-bearing reps. Synthetic, indicative."""
import numpy as np
import repet_exp as R
S = R.S; SR = R.SR; SEG = S.N

def bg_model(V, p, reps, how):
    Vs = np.stack([V[:, i * p:(i + 1) * p] for i in range(reps)], 1)
    W1 = np.median(Vs, 1) if how == 'median' else Vs.min(1)
    return np.tile(W1, (1, reps))

def run(misalign_ms, how):
    reps = 3
    takes = [S.rhythm_take(np.random.default_rng(300 + i)) for i in range(reps)]
    sh = int(misalign_ms / 1000 * SR)
    if sh:  # rep 3 played/aligned late by misalign_ms (beat-alignment error)
        takes[2] = np.concatenate([np.zeros(sh), takes[2][:-sh]])
    rhythm = np.concatenate(takes)
    lead = R.lead_random(np.random.default_rng(9), len(rhythm) / SR)
    lead[:SEG] = 0  # rep 1 rhythm-only
    mix = rhythm + lead
    X = R.stft(mix); V = np.abs(X)
    p = int(round(SEG / R.HOP)); T = p * reps
    Xc, Vc = X[:, :T], V[:, :T]
    W = np.minimum(bg_model(Vc, p, reps, how), Vc)
    Mb = W ** 2 / (W ** 2 + np.maximum(Vc - W, 0) ** 2 + 1e-12)
    fg = R.istft(np.pad(Xc * (1 - Mb), ((0, 0), (0, X.shape[1] - T))), len(mix))
    sl = slice(SEG, 3 * SEG)
    l = R.sdr_sir(fg[sl], lead[sl], rhythm[sl]); b = R.sdr_sir(mix[sl], lead[sl], rhythm[sl])
    print(f"{how:6s} misalign {misalign_ms:3d} ms | mix-as-lead SIR {b[1]:5.1f} -> fg lead SDR {l[0]:5.1f} SIR {l[1]:5.1f}")

for how in ('median', 'min'):
    for m in (0, 20, 50):
        run(m, how)
