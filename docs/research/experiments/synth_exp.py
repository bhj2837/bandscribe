"""Synthetic sanity check: can stereo-position masks split a centred lead guitar
from double-tracked hard-panned rhythm guitars (scenario A), and where do they break?

Everything here is synthetic (additive guitar-ish tones + tanh distortion + cab LPF).
The mixture contains only guitars, i.e. it models an *ideal* guitar stem coming out of
a separator. Numbers are indicative only.
"""
import sys, io
import numpy as np
from scipy import signal

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
SR = 44100
DUR = 8.0
N = int(SR * DUR)
rng_global = np.random.default_rng(0)


def tone(f, dur, decay, rng, vib_depth_cents=0.0, vib_rate=5.5, nharm=40):
    n = int(dur * SR)
    t = np.arange(n) / SR
    if vib_depth_cents > 0:
        # vibrato ramps in after 150 ms
        ramp = np.clip((t - 0.15) / 0.2, 0, 1)
        inst_f = f * 2 ** (vib_depth_cents / 1200 * ramp * np.sin(2 * np.pi * vib_rate * t))
    else:
        inst_f = np.full(n, f)
    phase = 2 * np.pi * np.cumsum(inst_f) / SR
    y = np.zeros(n)
    for k in range(1, nharm + 1):
        if k * f > 0.45 * SR:
            break
        amp = (1.0 / k) * np.exp(-t * decay * (1 + 0.15 * k))
        y += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    # pick transient
    y[: int(0.004 * SR)] += rng.normal(0, 0.3, int(0.004 * SR)) * np.linspace(1, 0, int(0.004 * SR))
    return y


def amp_cab(x, gain, lpf_hz, rng):
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
    y = np.tanh(gain * x)
    b, a = signal.butter(4, lpf_hz / (SR / 2), 'low')
    y = signal.lfilter(b, a, y)
    b, a = signal.butter(2, 90 / (SR / 2), 'high')
    y = signal.lfilter(b, a, y)
    return y


def place(buf, x, start):
    s = int(start * SR)
    if s < 0:
        x = x[-s:]; s = 0
    e = min(len(buf), s + len(x))
    buf[s:e] += x[: e - s]


def rhythm_take(rng, jitter_ms=8.0, detune_cents=4.0, gain=6.0, lpf=5000.0):
    """Power-chord eighth notes, E5-G5-A5-C5, 2 s each, 120 bpm."""
    roots = [82.41, 98.00, 110.00, 130.81]
    buf = np.zeros(N + SR)
    det = 2 ** (rng.normal(0, detune_cents) / 1200)
    for bar, r in enumerate(roots):
        for e in range(8):
            t0 = bar * 2.0 + e * 0.25 + rng.normal(0, jitter_ms / 1000)
            for s_i, mult in enumerate([1.0, 1.5, 2.0]):
                note = tone(r * mult * det, 0.26, decay=9.0, rng=rng)
                place(buf, note * (0.9 if s_i else 1.0), t0 + 0.004 * s_i)
    y = amp_cab(buf[:N], gain, lpf, rng)
    return y / np.sqrt(np.mean(y ** 2)) * 0.1


def lead_line(rng, gain=9.0, lpf=5500.0):
    notes = [(0.5, 329.63, 0.5), (1.0, 392.0, 0.25), (1.25, 440.0, 0.75), (2.0, 493.88, 1.0),
             (3.0, 587.33, 0.5), (3.5, 659.26, 1.0), (4.5, 587.33, 0.25), (4.75, 493.88, 0.25),
             (5.0, 440.0, 0.5), (5.5, 392.0, 0.5), (6.0, 329.63, 1.5)]
    buf = np.zeros(N + SR)
    for t0, f, d in notes:
        place(buf, tone(f, d + 0.05, decay=1.2, rng=rng, vib_depth_cents=25 if d >= 0.75 else 0), t0)
    y = amp_cab(buf[:N], gain, lpf, rng)
    return y / np.sqrt(np.mean(y ** 2)) * 0.1


def stereo_reverb(x_stereo, rt60=1.2, wet_db=-12, rng=None):
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    env = np.exp(-6.9 * t / rt60)
    irL = rng.normal(0, 1, n) * env
    irR = rng.normal(0, 1, n) * env
    irL /= np.sqrt(np.sum(irL ** 2)); irR /= np.sqrt(np.sum(irR ** 2))
    mono = x_stereo.mean(axis=0)
    wet = np.stack([signal.fftconvolve(mono, irL)[:N], signal.fftconvolve(mono, irR)[:N]])
    return x_stereo + 10 ** (wet_db / 20) * wet


def pan(x, p):
    """constant-power pan, p in [-1 (L), 1 (R)] -> (2, n)"""
    th = (p + 1) * np.pi / 4
    return np.stack([np.cos(th) * x, np.sin(th) * x])


# ---------------- STFT helpers
NFFT, HOP = 4096, 1024
def stft(x):
    return signal.stft(x, SR, nperseg=NFFT, noverlap=NFFT - HOP, window='hann')[2]
def istft(X):
    y = signal.istft(X, SR, nperseg=NFFT, noverlap=NFFT - HOP, window='hann')[1]
    out = np.zeros(N); out[: min(N, len(y))] = y[:N]
    return out


def smooth_t(A, lam):
    """one-pole recursive smoothing along time (axis -1)"""
    return signal.lfilter([1 - lam], [1, -lam], A, axis=-1)


# ---------------- estimators: each returns a real mask on the (F,T) grid, applied to mid
def mask_ratio(XL, XR, p=2.0):
    a, b = np.abs(XL), np.abs(XR)
    g = np.minimum(a, b) / (np.maximum(a, b) + 1e-12)   # 1 = centre, 0 = hard pan (magnitude only)
    return g ** p


def mask_adress(XL, XR, beta=100, H=10):
    """ADRess-style: null of |R - g L| (left half) or |L - g R| (right half), g in [0,1].
    Returns a *magnitude* estimate and a centre indicator; we convert it to a mask on mid."""
    left_side = np.abs(XL) >= np.abs(XR)
    A = np.where(left_side, XL, XR)   # louder channel
    B = np.where(left_side, XR, XL)   # quieter channel
    g = np.arange(beta + 1) / beta
    # discrete search, vectorised over g in chunks
    best_i = np.zeros(A.shape, dtype=int)
    best_v = np.full(A.shape, np.inf)
    maxv = np.zeros(A.shape)
    for i, gi in enumerate(g):
        v = np.abs(B - gi * A)
        upd = v < best_v
        best_v = np.where(upd, v, best_v)
        best_i = np.where(upd, i, best_i)
        maxv = np.maximum(maxv, v)
    centre = (beta - best_i) <= H                  # null within H steps of g=1 (centre)
    mag = maxv - best_v                            # ADRess magnitude estimate
    mid_mag = np.abs(0.5 * (XL + XR)) + 1e-12
    return np.clip(centre * mag / (2 * mid_mag), 0, 1) * centre


def mask_coherence(XL, XR, lam=0.8, p=2.0):
    """Avendano-Jot style similarity with time smoothing: psi = 2|<LR*>| / (<|L|^2>+<|R|^2>)."""
    SLR = smooth_t(XL * np.conj(XR), lam)
    SLL = smooth_t(np.abs(XL) ** 2, lam)
    SRR = smooth_t(np.abs(XR) ** 2, lam)
    psi = 2 * np.abs(SLR) / (SLL + SRR + 1e-12)
    return np.clip(psi, 0, 1) ** p


def mask_ipd_ild(XL, XR, sig_db=3.0, q=4.0):
    ild = 20 * np.log10((np.abs(XL) + 1e-12) / (np.abs(XR) + 1e-12))
    ipd = np.angle(XL * np.conj(XR))
    return np.exp(-(ild / sig_db) ** 2) * ((1 + np.cos(ipd)) / 2) ** q


def mask_side_wiener(XL, XR, lam=0.7, fbins=9, alpha=1.0, floor=0.0):
    """Use |S|^2 (lead-free when the lead is centred and mono) as an estimate of the rhythm power in M.
    Valid when the two rhythm takes are uncorrelated: E|t1+t2|^2 = E|t1-t2|^2."""
    M = 0.5 * (XL + XR); S = 0.5 * (XL - XR)
    PS = smooth_t(np.abs(S) ** 2, lam)
    PS = signal.convolve2d(PS, np.ones((fbins, 1)) / fbins, mode='same')
    PM = smooth_t(np.abs(M) ** 2, lam)
    PL = np.maximum(PM - alpha * PS, 0)
    return np.maximum(PL / (PL + alpha * PS + 1e-12), floor)


def mask_combo(XL, XR):
    return np.sqrt(mask_side_wiener(XL, XR) * mask_ipd_ild(XL, XR))


def evaluate(name, lead_st, rhy_st, masks, extra_note=""):
    """lead_st, rhy_st: (2, N) stereo components (everything that is not lead counts as rhythm)."""
    X = lead_st + rhy_st
    XL, XR = stft(X[0]), stft(X[1])
    LL, LR_ = stft(lead_st[0]), stft(lead_st[1])
    RL, RR = stft(rhy_st[0]), stft(rhy_st[1])
    lead_mid, rhy_mid = 0.5 * (LL + LR_), 0.5 * (RL + RR)
    lead_side, rhy_side = 0.5 * (LL - LR_), 0.5 * (RL - RR)

    def e(Z):
        return float(np.sum(np.abs(Z) ** 2))

    rows = []
    # baseline: mid channel as lead estimate, side channel as rhythm estimate
    rows.append(("mid (L+R)/2", 10 * np.log10(e(lead_mid) / e(rhy_mid)), None))
    rows.append(("side (L-R)/2 [lead leak into rhythm est.]", 10 * np.log10((e(lead_side) + 1e-20) / (e(rhy_side) + 1e-20)), None))
    rows.append(("  rhythm energy kept in side vs in mid (dB)", 10 * np.log10((e(rhy_side) + 1e-20) / (e(rhy_mid) + 1e-20)), None))
    oracle = np.abs(lead_mid) ** 2 / (np.abs(lead_mid) ** 2 + np.abs(rhy_mid) ** 2 + 1e-20)
    allm = dict(masks)
    allm["oracle IRM on mid (upper bound)"] = lambda a, b: oracle
    ref = istft(lead_mid)
    for mname, fn in allm.items():
        m = fn(XL, XR)
        lead_pass, rhy_pass = e(m * lead_mid), e(m * rhy_mid)
        sir = 10 * np.log10(lead_pass / (rhy_pass + 1e-20))
        est = istft(m * 0.5 * (XL + XR))
        sdr = 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - est) ** 2))
        kept = 10 * np.log10(lead_pass / e(lead_mid))
        rows.append((mname, sir, (sdr, kept)))
    print(f"\n=== {name} {extra_note}")
    print(f"{'estimator':48s} {'lead/rhythm in output (dB)':>28s} {'SDR vs lead-mid (dB)':>22s} {'lead energy kept (dB)':>22s}")
    for r in rows:
        if r[2] is None:
            print(f"{r[0]:48s} {r[1]:28.1f}")
        else:
            print(f"{r[0]:48s} {r[1]:28.1f} {r[2][0]:22.1f} {r[2][1]:22.1f}")


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    t1 = rhythm_take(np.random.default_rng(11), gain=6.0, lpf=5000)
    t2 = rhythm_take(np.random.default_rng(12), gain=7.0, lpf=4500)   # second performance, different amp
    t3 = rhythm_take(np.random.default_rng(13), gain=5.0, lpf=5500)
    t4 = rhythm_take(np.random.default_rng(14), gain=6.5, lpf=4800)
    lead = lead_line(np.random.default_rng(21))

    masks = {
        "magnitude-ratio centre mask (level only)": mask_ratio,
        "ADRess-style null search (beta=100,H=10)": mask_adress,
        "Avendano-Jot coherence psi (lam=0.8)": mask_coherence,
        "instantaneous IPD+ILD mask": mask_ipd_ild,
        "side-PSD Wiener (S as rhythm-noise estimate)": mask_side_wiener,
        "geo-mean(side-PSD Wiener, IPD+ILD)": mask_combo,
    }
    lead_c = pan(lead, 0.0)
    # A0: double-tracked rhythm hard L/R, lead centre, dry
    rhyA = np.stack([t1, t2])
    evaluate("A0 double-tracked hard L/R + centred lead, dry", lead_c, rhyA, masks)
    # A0 quieter lead
    evaluate("A0b same, lead 6 dB quieter", 0.5 * lead_c, rhyA, masks)
    # A1: + stereo reverb on everything (reverb of rhythm counted as rhythm, of lead as 'lead' component)
    r = np.random.default_rng(5)
    lead_rev = stereo_reverb(lead_c, rng=np.random.default_rng(6))
    rhy_rev = stereo_reverb(rhyA, rng=np.random.default_rng(7))
    evaluate("A1 A0 + decorrelated stereo reverb (-12 dB wet, RT60 1.2 s)", lead_rev, rhy_rev, masks,
             "(lead reference includes its own reverb)")
    # A2: lead with ping-pong delay
    d1, d2 = int(0.375 * SR), int(0.5625 * SR)
    dl = np.zeros(N); dl[d1:] = lead[:-d1] * 0.35
    dr = np.zeros(N); dr[d2:] = lead[:-d2] * 0.35
    lead_pp = pan(lead, 0.0) + np.stack([dl, dr])
    evaluate("A2 A0 but lead has hard-panned ping-pong delay (-9 dB)", lead_pp, rhyA, masks)
    # A3: fake double: right = left delayed 12 ms (ADT-style), lead centre
    dd = int(0.012 * SR)
    t1d = np.zeros(N); t1d[dd:] = t1[:-dd]
    evaluate("A3 'fake double' (R = L delayed 12 ms)", lead_c, np.stack([t1, t1d]), masks)
    # A4: quad tracking: two hard, two at +-70 %
    rhyQ = np.stack([t1, t2]) + pan(t3, -0.7) + pan(t4, 0.7)
    evaluate("A4 quad-tracked rhythm (2 hard + 2 at 70 %)", lead_c, rhyQ, masks)
    # B: single rhythm take centred (mono) + lead centred
    evaluate("B single rhythm guitar centred + lead centred", lead_c, pan(t1, 0.0), masks)
    # B2: single rhythm panned 30 % left, lead centred
    evaluate("B2 single rhythm 30 % left + lead centred", lead_c, pan(t1, -0.3), masks)
    # C: lead panned 20 % right (common 'slightly off-centre' practice), rhythm hard L/R
    evaluate("C lead panned 20 % right + double-tracked hard L/R", pan(lead, 0.2), rhyA, masks)
