import numpy as np, tempfile
import synth_exp as S
import codec_exp as C
N=S.N
r1, r2 = S.rhythm_take(np.random.default_rng(11)), S.rhythm_take(np.random.default_rng(12))
lead = S.lead_line(np.random.default_rng(21)); z=np.zeros(N)
x = np.stack([r1, z]) + np.stack([z, r2]) + S.pan(lead, 0.0)
x = x/np.max(np.abs(x))*10**(-1/20)
with tempfile.TemporaryDirectory() as tmp:
  yo,_ = C.roundtrip(x,['-c:a','libopus','-b:a','128k'],'opus',tmp); yo,_=C.align(x,yo)
  ya,_ = C.roundtrip(x,['-c:a','aac','-b:a','128k'],'m4a',tmp); ya,_=C.align(x,ya)
  s = lambda y: (C.snr(x[0],y[0])+C.snr(x[1],y[1]))/2
  eo, ea = yo-x, ya-x
  rho = np.sum(eo*ea)/np.sqrt(np.sum(eo**2)*np.sum(ea**2))
  print(f'opus128 SNR {s(yo):.1f} dB | aac128 SNR {s(ya):.1f} dB | mean(opus,aac) SNR {s((yo+ya)/2):.1f} dB | error correlation {rho:.2f}')
