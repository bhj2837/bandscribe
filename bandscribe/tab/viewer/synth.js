// Tab synth for the A/B mixer (M4b, DESIGN 7.2): alphaTab plays either its synthesizer or an external recording, so
// each score track is rendered offline with alphaTab's Audio Export API (alphaTab 1.6+), aligned to the recording
// by the score's \sync points (useSyncPoints), one track at a time (the others at volume 0). alphaTab renders at
// EXPORT_SR (at 22.05 kHz its synthesizer turns to NaN within a second - measured 2026-10-17, alphaTab 1.8.4); the
// page keeps a mono copy at SYNTH_SR (half-band smoothing, every second sample) to halve the memory.
// A track is rendered again only when what it plays changed (`signature`).

export const EXPORT_SR = 44100;
export const SYNTH_SR = 22050;
let soundFont = null;

async function font() {
  if (!soundFont) {
    const r = await fetch("/alphatab/soundfont/sonivox.sf2");
    if (!r.ok) throw new Error(`soundfont: HTTP ${r.status}`);
    soundFont = new Uint8Array(await r.arrayBuffer());
  }
  return soundFont;
}

// what a track plays, from the page's note map: written time, length and pitch of each note written in it (every
// written place - a note copied into a second player's track counts there too)
export function signature(map, trackIndex) {
  const rows = [];
  const pre = `${trackIndex}:`;
  for (const [key, uid] of Object.entries(map.pos || {})) {
    if (!key.startsWith(pre)) continue;
    const n = map.notes[uid];
    if (n) rows.push(`${key}|${n.t}|${n.d}|${n.pitch}`);
  }
  rows.sort();
  const bars = (map.timeline && map.timeline.bars) || [];
  return `${bars.length}:${bars.length ? bars[bars.length - 1][2] : 0}#` + rows.join(",");
}

// one track of the loaded score -> a mono AudioBuffer on `ctx` (null when the export gives no samples)
export async function renderTrack(api, ctx, trackIndex, onProgress) {
  const opts = new alphaTab.synth.AudioExportOptions();
  opts.soundFonts = [await font()];
  opts.sampleRate = EXPORT_SR;
  opts.useSyncPoints = true;
  opts.masterVolume = 1;
  opts.metronomeVolume = 0;
  opts.trackVolume = new Map(api.score.tracks.map((t) => [t.index, t.index === trackIndex ? 1 : 0]));
  const exporter = await api.exportAudio(opts);
  const parts = [];
  let total = 0;
  try {
    for (;;) {
      const chunk = await exporter.render(4000);
      if (!chunk) break;
      parts.push(chunk.samples);
      total += chunk.samples.length;
      if (onProgress && chunk.endTime) onProgress(chunk.currentTime / chunk.endTime);
    }
  } finally {
    exporter.destroy();
  }
  const frames = Math.floor(total / 2);  // interleaved stereo
  if (frames < 4) return null;
  const mono = new Float32Array(frames);
  let k = 0, bad = 0;
  for (const p of parts) {
    for (let i = 0; i + 1 < p.length; i += 2) {
      const v = 0.5 * (p[i] + p[i + 1]);
      if (Number.isFinite(v)) mono[k++] = v; else { mono[k++] = 0; bad++; }
    }
  }
  const n = Math.floor(frames / 2);
  const buf = ctx.createBuffer(1, n, SYNTH_SR);
  const out = buf.getChannelData(0);
  for (let j = 0; j < n; j++) {
    const i = 2 * j;
    out[j] = 0.25 * (i > 0 ? mono[i - 1] : 0) + 0.5 * mono[i] + 0.25 * mono[i + 1];
  }
  buf.nonFinite = bad;  // samples the synthesizer gave as NaN / Infinity (written as 0)
  return buf;
}

// M4b acceptance ("원곡·스템·탭 신스 동기, 드리프트 50 ms 이하"): how far a rendered track's onsets sit from the
// written times of its notes on the recording (the note map's `t`). Onsets = peaks of the rise of the 1.5 ms
// envelope above 15 % of its 99.5th percentile; each written time takes the nearest onset within 250 ms.
// -> {notes, matched, p05, median, p95 (ms), medianBy40s: [[start s, ms]]}
export function syncReport(map, trackIndex, buffer) {
  const x = buffer.getChannelData(0), sr = buffer.sampleRate, hop = 32;
  const nf = Math.floor(x.length / hop), env = new Float32Array(nf), on = new Float32Array(nf);
  for (let k = 0; k < nf; k++) { let a = 0; for (let j = 0; j < hop; j++) { const y = x[k * hop + j]; a += y * y; } env[k] = Math.sqrt(a / hop); }
  for (let k = 2; k < nf; k++) on[k] = Math.max(0, env[k] - env[k - 2]);
  const fps = sr / hop, sorted = Array.from(on).sort((a, b) => a - b), thr = 0.15 * sorted[Math.floor(0.995 * (nf - 1))];
  const peaks = [];
  for (let k = 1; k < nf - 1; k++) if (on[k] > thr && on[k] >= on[k - 1] && on[k] > on[k + 1]) peaks.push(k / fps);
  const times = [...new Set(Object.entries(map.at).filter(([, a]) => a[0] === trackIndex).map(([u]) => map.notes[u].t))]
    .sort((a, b) => a - b);
  const d = [];
  let j = 0;
  for (const t of times) {
    while (j < peaks.length - 1 && peaks[j + 1] <= t) j++;
    const near = [peaks[j], peaks[j + 1]].filter((p) => p !== undefined)
      .reduce((m, p) => (Math.abs(p - t) < Math.abs(m - t) ? p : m), Infinity);
    if (Math.abs(near - t) < 0.25) d.push([t, near - t]);
  }
  const ms = d.map((r) => r[1]).sort((a, b) => a - b);
  const q = (f) => (ms.length ? Math.round(ms[Math.floor(f * (ms.length - 1))] * 1000) : null);
  const by = [];
  for (let s0 = 0; s0 < buffer.duration; s0 += 40) {
    const z = d.filter((r) => r[0] >= s0 && r[0] < s0 + 40).map((r) => r[1]).sort((a, b) => a - b);
    if (z.length) by.push([s0, Math.round(z[Math.floor(z.length / 2)] * 1000)]);
  }
  return { notes: times.length, matched: d.length, p05: q(0.05), median: q(0.5), p95: q(0.95), medianBy40s: by };
}
