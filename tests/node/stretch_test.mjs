// M4b pitch check of the slowed-playback shifter (bandscribe/tab/viewer/stretch_core.js), run by
// tests/test_viewer_m4b.py: a tone played at `rate` (its pitch lowered by `rate`, as the engine plays a recording)
// goes through the shifter at ratio 1 / rate; the output's pitch is measured with a Goertzel scan (0.25-cent steps,
// ±60 cents around the original pitch). Prints one JSON object.
import { Shifter } from "../../bandscribe/tab/viewer/stretch_core.js";

const SR = 44100;
const goertzel = (x, f) => {
  const w = (2 * Math.PI * f) / SR, c = 2 * Math.cos(w);
  let s1 = 0, s2 = 0;
  for (let i = 0; i < x.length; i++) { const s0 = x[i] + c * s1 - s2; s2 = s1; s1 = s0; }
  return s1 * s1 + s2 * s2 - c * s1 * s2;
};
const peakCents = (x, f0) => {
  let best = 0, bp = -1;
  for (let c = -60; c <= 60; c += 0.25) {
    const p = goertzel(x, f0 * 2 ** (c / 1200));
    if (p > bp) { bp = p; best = c; }
  }
  return best;
};
const signal = (f, harmonics, rate, seconds) => {
  const n = Math.round(seconds * SR), x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let h = 1; h <= harmonics; h++) v += Math.sin((2 * Math.PI * f * rate * h * i) / SR) / h;
    x[i] = 0.3 * v;
  }
  return x;
};

const out = [];
for (const [name, f, harm] of [["sine440", 440, 1], ["harm110", 110, 6], ["low82", 82.41, 4]]) {
  for (const rate of [0.5, 0.6, 0.75, 0.9]) {
    const x = signal(f, harm, rate, 3.0);
    const y = new Float32Array(x.length);
    const sh = new Shifter(SR, 1);
    sh.setRatio(1 / rate);
    for (let i = 0; i < x.length; i += 128) {
      const m = Math.min(128, x.length - i);
      const yo = new Float32Array(m);
      sh.process([x.subarray(i, i + m)], [yo], m);
      y.set(yo, i);
    }
    const steady = y.subarray(Math.round(0.3 * SR));
    const rms = [];
    const fr = Math.round(0.05 * SR);
    for (let i = 0; i + fr <= steady.length; i += fr) {
      let s = 0;
      for (let j = 0; j < fr; j++) s += steady[i + j] ** 2;
      rms.push(Math.sqrt(s / fr));
    }
    let si = 0;
    for (let j = 0; j < x.length; j++) si += x[j] ** 2;
    const inRms = Math.sqrt(si / x.length);
    out.push({ name, rate, cents: peakCents(steady, f), rms_min: Math.min(...rms) / inRms,
               rms_max: Math.max(...rms) / inRms, latency_ms: sh.latency * 1000 });
  }
}
console.log(JSON.stringify(out));
