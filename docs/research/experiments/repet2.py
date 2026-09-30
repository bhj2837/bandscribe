import numpy as np
import repet_exp as R
S = R.S; SR = R.SR

def repet_period(x, period_s, power=2):
    X = R.stft(x); V = np.abs(X)
    p = int(round(period_s * SR / R.HOP)); T = V.shape[1]; r = T // p
    Vs = V[:, :r * p].reshape(V.shape[0], r, p)
    W1 = np.median(Vs, axis=1)                    # (F,p)
    W = np.tile(W1, (1, r + 1))[:, :T]
    W = np.minimum(W, V)
    Mb = (W ** power) / (W ** power + np.maximum(V - W, 0) ** power + 1e-12)
    return R.istft(X * (1 - Mb), len(x)), R.istft(X * Mb, len(x))

rhythm = np.concatenate([S.rhythm_take(np.random.default_rng(100 + i)) for i in range(4)])
for lead_db in (0, -6):
    lead = R.lead_random(np.random.default_rng(7), len(rhythm) / SR) * 10 ** (lead_db / 20)
    mix = rhythm + lead
    fg, bg = repet_period(mix, 8.0)
    l = R.sdr_sir(fg, lead, rhythm); r = R.sdr_sir(bg, rhythm, lead)
    for k in (5, 10):
        fg2, bg2 = R.repet_sim(mix, k=k)
        l2 = R.sdr_sir(fg2, lead, rhythm)
        print(f"lead {lead_db:+d} dB  REPET-SIM k={k}: lead SDR {l2[0]:5.1f} SIR {l2[1]:5.1f}")
    print(f"lead {lead_db:+d} dB  REPET oracle 8s period: lead SDR {l[0]:5.1f} SIR {l[1]:5.1f} | rhythm SDR {r[0]:5.1f} SIR {r[1]:5.1f}")
