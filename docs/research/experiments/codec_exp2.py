import numpy as np, tempfile
import synth_exp as S
import codec_exp as C
N=S.N
r1, r2 = S.rhythm_take(np.random.default_rng(11)), S.rhythm_take(np.random.default_rng(12))
lead = S.lead_line(np.random.default_rng(21)); z=np.zeros(N)
mix = np.stack([r1, z]) + np.stack([z, r2]) + S.pan(lead, 0.0)
# 'loud master': drive +12 dB into a hard clipper at 0 dBFS (crude brickwall), like loudness-war masters
x = mix/np.max(np.abs(mix))*10**(12/20); x = np.clip(x,-1,1)
print('crude loud master: RMS dBFS %.1f, peak %.3f' % (10*np.log10(np.mean(x**2)), np.max(np.abs(x))))
with tempfile.TemporaryDirectory() as tmp:
  for name,(args,ext) in {'opus 128k':(['-c:a','libopus','-b:a','128k'],'opus'),'opus 160k':(['-c:a','libopus','-b:a','160k'],'opus'),
                          'aac 128k':(['-c:a','aac','-b:a','128k'],'m4a'),'mp3 128k':(['-c:a','libmp3lame','-b:a','128k'],'mp3')}.items():
    y,k = C.roundtrip(x,args,ext,tmp); y,lag = C.align(x,y)
    n_over = int(np.sum(np.abs(y)>1.0))
    print(f'{name:10s} decoded float peak {np.max(np.abs(y)):.3f} ({20*np.log10(np.max(np.abs(y))):+.2f} dBFS), samples >1.0: {n_over} of {y.size} ({100*n_over/y.size:.2f}%)')
