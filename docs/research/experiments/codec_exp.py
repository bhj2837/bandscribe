"""Synthetic check: what do YouTube-like lossy codecs (Opus ~128-160k, AAC 128k, MP3) do to
(1) waveform SNR, (2) the side signal S=(L-R)/2 used as a rhythm-guitar reference for
double-tracked mixes, (3) the stereo router features, (4) true peaks / overs.
Idealised guitar-only stereo stem from synth_exp. Indicative only (not a real mix, not SDR of a separator).
"""
import os, subprocess, tempfile
import numpy as np
from scipy.io import wavfile
from scipy import signal

import synth_exp as S          # wraps stdout to utf-8
import router_exp as R

SR, N = S.SR, S.N


def roundtrip(x, codec_args, ext, tmp):
    src = os.path.join(tmp, 'in.wav'); enc = os.path.join(tmp, 'enc.' + ext)
    wavfile.write(src, SR, x.T.astype(np.float32))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', src] + codec_args + [enc], check=True)
    # decode to float32 at 44.1k (soxr resampler), no clipping
    p = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', enc, '-af', 'aresample=44100:resampler=soxr',
                        '-f', 'f32le', '-acodec', 'pcm_f32le', '-ac', '2', '-'], check=True, capture_output=True)
    y = np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, 2).T.astype(np.float64)
    size_kbps = os.path.getsize(enc) * 8 / (N / SR) / 1000
    return y, size_kbps


def align(ref, y):
    m = min(ref.shape[1], y.shape[1])
    c = signal.correlate(y[0, :m] + y[1, :m], ref[0, :m] + ref[1, :m], mode='full', method='fft')
    lag = int(np.argmax(c) - (m - 1))
    if lag > 0:
        y = y[:, lag:]
    elif lag < 0:
        y = np.concatenate([np.zeros((2, -lag)), y], axis=1)
    if y.shape[1] < ref.shape[1]:
        y = np.concatenate([y, np.zeros((2, ref.shape[1] - y.shape[1]))], axis=1)
    return y[:, :ref.shape[1]], lag


def snr(ref, est):
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + 1e-20))


def main():
    r1, r2 = S.rhythm_take(np.random.default_rng(11)), S.rhythm_take(np.random.default_rng(12))
    lead = S.lead_line(np.random.default_rng(21))
    z = np.zeros(N)
    mix = np.stack([r1, z]) + np.stack([z, r2]) + S.pan(lead, 0.0)   # scenario A
    mix = mix / np.max(np.abs(mix)) * 10 ** (-0.1 / 20)               # loud master: peak -0.1 dBFS
    sig_S = (mix[0] - mix[1]) / 2
    codecs = {
        'opus 160k vbr (~yt 251 upper)': (['-c:a', 'libopus', '-b:a', '160k', '-vbr', 'on'], 'opus'),
        'opus 128k vbr (~yt 251)':       (['-c:a', 'libopus', '-b:a', '128k', '-vbr', 'on'], 'opus'),
        'opus 96k vbr':                  (['-c:a', 'libopus', '-b:a', '96k', '-vbr', 'on'], 'opus'),
        'aac 128k (ffmpeg native, ~yt 140)': (['-c:a', 'aac', '-b:a', '128k'], 'm4a'),
        'aac 256k (ffmpeg native, ~141)': (['-c:a', 'aac', '-b:a', '256k'], 'm4a'),
        'mp3 128k cbr (lame)':           (['-c:a', 'libmp3lame', '-b:a', '128k'], 'mp3'),
        'mp3 320k cbr (lame)':           (['-c:a', 'libmp3lame', '-b:a', '320k'], 'mp3'),
    }
    f0 = R.features(mix)
    print(f"reference (lossless)                    S/M {f0['SM']:6.1f} dB  coh {f0['coh']:.3f}  cen {f0['cen']:.2f}  hard {f0['hard']:.2f}  peak {np.max(np.abs(mix)):.3f}")
    print(f"{'codec':38s} {'kbps':>5s} {'lag':>5s} {'SNR L/R':>8s} {'SNR S':>6s} {'S/M':>6s} {'coh':>6s} {'cen':>5s} {'hard':>5s} {'peak':>6s} {'>1.0 %':>7s}")
    with tempfile.TemporaryDirectory() as tmp:
        for name, (args, ext) in codecs.items():
            y, kbps = roundtrip(mix, args, ext, tmp)
            y, lag = align(mix, y)
            s_snr = snr(sig_S, (y[0] - y[1]) / 2)
            lr = (snr(mix[0], y[0]) + snr(mix[1], y[1])) / 2
            f = R.features(y)
            over = 100 * np.mean(np.abs(y) > 1.0)
            print(f"{name:38s} {kbps:5.0f} {lag:5d} {lr:8.1f} {s_snr:6.1f} {f['SM']:6.1f} {f['coh']:6.3f} {f['cen']:5.2f} {f['hard']:5.2f} {np.max(np.abs(y)):6.3f} {over:7.3f}")


if __name__ == '__main__':
    main()
