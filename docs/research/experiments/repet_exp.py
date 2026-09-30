"""Synthetic check: does repetition-based separation (REPET-SIM style) pull a non-repeating lead
out of a MONO guitar stem whose rhythm part repeats (scenario B/F, where stereo gives nothing)?
32 s: the same 4-bar power-chord pattern played 4 times (4 independent takes -> micro-timing
differs), plus a lead that does NOT repeat (random pentatonic line, sustained notes w/ vibrato).
Idealised (no drums/vocal bleed, perfectly repeating chords). Indicative only.
"""
import numpy as np
from scipy import signal
import synth_exp as S  # stdout wrapped there

SR = S.SR
SEG = S.N  # 8 s


def lead_random(rng, dur_s):
    n = int(dur_s * SR)
    buf = np.zeros(n + SR)
    scale = [329.63, 392.0, 440.0, 493.88, 587.33, 659.26, 783.99, 880.0]
    t = 0.3
    while t < dur_s - 0.5:
        d = rng.choice([0.25, 0.5, 0.75, 1.0, 1.5], p=[0.3, 0.3, 0.15, 0.15, 0.1])
        f = rng.choice(scale)
        S.place(buf, S.tone(f, d + 0.05, decay=1.2, rng=rng, vib_depth_cents=25 if d >= 0.75 else 0), t)
        t += d + rng.choice([0.0, 0.25, 0.5], p=[0.6, 0.3, 0.1])
    y = S.amp_cab(buf[:n], 9.0, 5500.0, rng)
    return y / np.sqrt(np.mean(y ** 2)) * 0.1


NFFT, HOP = 4096, 1024
def stft(x): return signal.stft(x, SR, nperseg=NFFT, noverlap=NFFT - HOP, window='hann')[2]
def istft(X, n):
    y = signal.istft(X, SR, nperseg=NFFT, noverlap=NFFT - HOP, window='hann')[1]
    out = np.zeros(n); out[:min(n, len(y))] = y[:n]; return out


def repet_sim(x, k=20, min_dist_s=1.0, power=2):
    X = stft(x); V = np.abs(X)
    Vn = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    C = Vn.T @ Vn
    T = V.shape[1]; md = int(min_dist_s * SR / HOP)
    W = np.zeros_like(V)
    for j in range(T):
        c = C[j].copy()
        c[max(0, j - md):j + md + 1] = -np.inf  # exclude own neighbourhood
        idx = np.argsort(c)[-k:]
        W[:, j] = np.median(V[:, idx], axis=1)
    W = np.minimum(W, V)
    Mb = (W ** power) / (W ** power + np.maximum(V - W, 0) ** power + 1e-12)  # soft mask
    n = len(x)
    return istft(X * (1 - Mb), n), istft(X * Mb, n)


def sdr_sir(est, tgt, intf):
    # projection-based: est = a*tgt + b*intf + e
    A = np.stack([tgt, intf], 1)
    coef, *_ = np.linalg.lstsq(A, est, rcond=None)
    s_t = coef[0] * tgt; s_i = coef[1] * intf; e = est - s_t - s_i
    sdr = 10 * np.log10(np.sum(s_t ** 2) / (np.sum((s_i + e) ** 2) + 1e-12))
    sir = 10 * np.log10(np.sum(s_t ** 2) / (np.sum(s_i ** 2) + 1e-12))
    return sdr, sir


def main():
    rhythm = np.concatenate([S.rhythm_take(np.random.default_rng(100 + i)) for i in range(4)])
    n = len(rhythm)
    for lead_db in (0, -6):
        lead = lead_random(np.random.default_rng(7), n / SR) * 10 ** (lead_db / 20)
        mix = rhythm + lead
        base = sdr_sir(mix, lead, rhythm)
        fg, bg = repet_sim(mix)
        l = sdr_sir(fg, lead, rhythm)
        r = sdr_sir(bg, rhythm, lead)
        print(f"lead {lead_db:+d} dB | unprocessed mix as lead: SDR {base[0]:5.1f} SIR {base[1]:5.1f} | "
              f"REPET-SIM foreground (lead): SDR {l[0]:5.1f} SIR {l[1]:5.1f} | background (rhythm): SDR {r[0]:5.1f} SIR {r[1]:5.1f}")
    # sanity: what if the rhythm does NOT repeat (4 different chord progressions)? -> use transposed takes
    print("-- control: rhythm pattern changes every 8 s (no repetition within the clip) --")
    rh2 = []
    for i, semis in enumerate([0, 5, 2, 7]):
        r = S.rhythm_take(np.random.default_rng(200 + i))
        # crude transposition by resampling (changes timbre slightly, fine for a control)
        f = 2 ** (semis / 12)
        r = signal.resample(r, int(len(r) / f))
        r = np.pad(r, (0, SEG - len(r)))[:SEG] if len(r) < SEG else r[:SEG]
        rh2.append(r)
    rh2 = np.concatenate(rh2)
    lead = lead_random(np.random.default_rng(7), len(rh2) / SR)
    fg, bg = repet_sim(rh2 + lead)
    l = sdr_sir(fg, lead, rh2)
    print(f"lead  +0 dB | REPET-SIM foreground (lead): SDR {l[0]:5.1f} SIR {l[1]:5.1f}")


if __name__ == '__main__':
    main()
