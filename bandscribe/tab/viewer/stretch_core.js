// Pitch shifter for slowed playback (M4b, DESIGN 7.2 "50–100 % 감속(피치 보존)").
//
// The engine plays every recording at `rate` (< 1), which lowers its pitch by that factor; this shifter raises the
// pitch back by `ratio` = 1 / rate. Two read taps run through a delay line `ratio` times faster than it is written,
// half a grain apart: each tap's delay falls by (ratio - 1) per sample and jumps back up when its sin² window is
// silent, so the two windows always sum to 1. When a tap jumps back, its new delay is chosen within `searchMs` so
// that what it is about to read lines up with what the other tap reads (largest cross-correlation, SOLA): without
// that the two taps cancel each other at some frequencies. The heard signal lags its input by `latency` on average,
// which the engine takes off the cursor. ES module shared by the AudioWorklet (stretch.worklet.js) and the Node test
// (tests/node/stretch_test.mjs).

export class Shifter {
  constructor(sampleRate, channels = 2, grainMs = 60, searchMs = 25) {
    this.sr = sampleRate;
    this.ch = channels;
    this.W = Math.max(256, Math.round((sampleRate * grainMs) / 1000));
    this.S = Math.max(0, Math.round((sampleRate * searchMs) / 1000));
    this.L = 1024;  // correlation span in read steps (every 4th): longer than a slowed low E's period
    let n = 1;
    while (n < 4 * (this.W + this.S) + 2 * this.L) n <<= 1;
    this.N = n;
    this.mask = n - 1;
    this.buf = Array.from({ length: channels }, () => new Float32Array(n));
    this.mono = new Float32Array(n);
    this.ratio = 1;
    this.reset();
  }

  setRatio(r) { this.ratio = Math.min(2, Math.max(1, Number(r) || 1)); }

  // seconds the heard signal lags its input, on average over a grain
  get latency() { return (this.W / 2 + this.S / 2 + 1) / this.sr; }

  reset() {
    for (const b of this.buf) b.fill(0);
    this.mono.fill(0);
    this.w = 0;
    const W = this.W;
    this.taps = [{ phase: 0, d: W }, { phase: 0.5, d: W - (W - 1) * 0.5 }];
  }

  _read(b, pos) {
    const i = Math.floor(pos);
    const f = pos - i;
    const a = b[i & this.mask];
    return a + (b[(i + 1) & this.mask] - a) * f;
  }

  // the restart delay in [W, W + S] whose read point lines up with the other tap's: the waveforms just before the
  // two read points (stepping back at `ratio`, so only written samples are read) correlate best
  _align(other) {
    const W = this.W, S = this.S, L = this.L, r = this.ratio, m = this.mono;
    if (S === 0) return W;
    const ro = this.w - other.d + 2 * this.N;
    const corr = (delta, stride) => {
      const rn = this.w - (W + delta) + 2 * this.N;
      let c = 0;
      for (let j = 0; j < L; j += stride) c += this._read(m, rn - j * r) * this._read(m, ro - j * r);
      return c;
    };
    // coarse (every 3rd delay, every 8th step), then the neighbours of the best one finely: about a quarter of a
    // full search, so one 128-frame block stays well inside its time budget
    let best = 0, bestC = -Infinity;
    for (let delta = 0; delta <= S; delta += 3) {
      const c = corr(delta, 8);
      if (c > bestC) { bestC = c; best = delta; }
    }
    let fine = best, fineC = -Infinity;
    for (let delta = Math.max(0, best - 2); delta <= Math.min(S, best + 2); delta++) {
      const c = corr(delta, 4);
      if (c > fineC) { fineC = c; fine = delta; }
    }
    return W + fine;
  }

  // inputs / outputs: arrays of Float32Array (one per channel); a missing input channel repeats the first
  process(inputs, outputs, frames) {
    const W = this.W;
    const step = (this.ratio - 1) / (W - 1);  // a grain lasts while the delay falls by W - 1
    const fall = this.ratio - 1;
    const [A, B] = this.taps;
    for (let k = 0; k < frames; k++) {
      const wi = this.w & this.mask;
      let s = 0;
      for (let c = 0; c < this.ch; c++) {
        const src = inputs[c] || inputs[0];
        const v = src ? src[k] : 0;
        this.buf[c][wi] = v;
        s += v;
      }
      this.mono[wi] = s / this.ch;
      const ga = Math.sin(Math.PI * A.phase) ** 2;
      const gb = 1 - ga;
      const ra = this.w - A.d + this.N;
      const rb = this.w - B.d + this.N;
      for (let c = 0; c < this.ch; c++) {
        const out = outputs[c];
        if (out) out[k] = ga * this._read(this.buf[c], ra) + gb * this._read(this.buf[c], rb);
      }
      this.w = (this.w + 1) & 0x3fffffff;
      for (const [t, o] of [[A, B], [B, A]]) {
        t.phase += step;
        t.d -= fall;
        if (t.phase >= 1) {
          t.phase -= 1;
          t.d = this._align(o);
        }
      }
    }
  }
}
