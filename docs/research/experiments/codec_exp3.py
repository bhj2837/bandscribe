import numpy as np, tempfile
import synth_exp as S
import codec_exp as C
N=S.N
r1, r2 = S.rhythm_take(np.random.default_rng(11)), S.rhythm_take(np.random.default_rng(12))
lead = S.lead_line(np.random.default_rng(21)); z=np.zeros(N)
mix = np.stack([r1, z]) + np.stack([z, r2]) + S.pan(lead, 0.0)
mix = mix/np.max(np.abs(mix))
with tempfile.TemporaryDirectory() as tmp:
  for drive in (0.0, 3.0, 6.0):
    x = np.clip(mix*10**(drive/20)*10**(-0.1/20), -1, 1)
    print('drive +%.0f dB -> RMS %.1f dBFS, clipped %.2f%%' % (drive, 10*np.log10(np.mean(x**2)), 100*np.mean(np.abs(x)>=0.999)))
    for name,(args,ext) in {'opus 128k':(['-c:a','libopus','-b:a','128k'],'opus'),'aac 128k':(['-c:a','aac','-b:a','128k'],'m4a')}.items():
      y,k = C.roundtrip(x,args,ext,tmp); y,lag = C.align(x,y)
      n_over = int(np.sum(np.abs(y)>1.0))
      print(f'   {name:10s} decoded peak {20*np.log10(np.max(np.abs(y))):+.2f} dBFS, samples >1.0: {100*n_over/y.size:.2f}%')
