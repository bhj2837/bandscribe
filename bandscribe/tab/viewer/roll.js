// The viewer's lower panel (M4b, DESIGN 7.2 "화면 구성"):
// - part band: the whole song, bar by bar - sections, the router's window labels, and one lane per part (Line) that
//   is filled where the part plays (darker = more notes); the piano roll's window is outlined. Click = seek.
// - piano roll: the constant-Q image of a recording (the guitar stem by default) with the score's notes on it in
//   their track colour, opacity = the chosen confidence; notes the transcribers heard but the run did not keep are
//   outlined (dashed). It follows the cursor; click a note to select it, elsewhere to seek.

export const LINE_COLORS = { "1": "#ff6b35", "2": "#4dabf7", "3": "#69db7c", "4": "#da77f2" };
const BASS_COLOR = "#ffd43b";
const UNASSIGNED = "#adb5bd";
const ROUTER_COLORS = { R0: "#868e96", R1: "#4dabf7", R2: "#69db7c", R3: "#ff922b" };
const TILE = 2048;

const lut = (() => {  // dark-to-light ramp with a little warmth, for a dark roll in both themes
  const t = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const v = (i / 255) ** 1.6;
    t[i * 4] = 18 + v * 232; t[i * 4 + 1] = 18 + v * 214; t[i * 4 + 2] = 22 + v * 180; t[i * 4 + 3] = 255;
  }
  return t;
})();

export function noteColor(n) {
  if (n.track === "bass") return BASS_COLOR;
  return LINE_COLORS[n.line] || UNASSIGNED;
}

export function confOf(n, kind) {
  if (kind === "part") return n.p === null || n.p === undefined ? null : Number(n.p);
  if (kind === "fret") return n.fc === null || n.fc === undefined ? null : Number(n.fc);
  return null;
}

export class Lower {
  constructor(band, roll, { onSeek, onPick }) {
    this.band = band;
    this.roll = roll;
    this.onSeek = onSeek;
    this.onPick = onPick;
    this.span = 8;          // seconds in the roll
    this.lead = 0.25;       // the cursor sits a quarter in
    this.conf = "part";
    this.cqt = null;        // {head, tiles}
    this.notes = [];        // [{uid, t, d, pitch, color, conf}] by t
    this.cands = [];
    this.timeline = { bars: [], sections: [], lines: [] };
    this.router = [];
    this.duration = 0;
    this.time = 0;
    this.selected = null;
    roll.addEventListener("click", (e) => this._rollClick(e));
    band.addEventListener("click", (e) => this._bandClick(e));
    roll.addEventListener("wheel", (e) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      this.span = Math.min(30, Math.max(2, this.span * (e.deltaY > 0 ? 1.25 : 0.8)));
      this.draw(this.time);
    }, { passive: false });
    new ResizeObserver(() => { this._fit(); this.drawBand(); this.draw(this.time); }).observe(roll);
    this._fit();
  }

  _fit() {
    for (const c of [this.band, this.roll]) {
      const r = c.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      c.width = Math.max(1, Math.round(r.width * dpr));
      c.height = Math.max(1, Math.round(r.height * dpr));
    }
  }

  setCqt(bytes, head) {
    const tiles = [];
    for (let f0 = 0; f0 < head.frames; f0 += TILE) {
      const w = Math.min(TILE, head.frames - f0);
      const cv = document.createElement("canvas");
      cv.width = w; cv.height = head.bins;
      const g = cv.getContext("2d");
      const img = g.createImageData(w, head.bins);
      for (let b = 0; b < head.bins; b++) {
        const row = (head.bins - 1 - b) * w;  // canvas row 0 = highest bin
        const src = b * head.frames + f0;
        for (let x = 0; x < w; x++) {
          const v = bytes[src + x] * 4, o = (row + x) * 4;
          img.data[o] = lut[v]; img.data[o + 1] = lut[v + 1]; img.data[o + 2] = lut[v + 2]; img.data[o + 3] = 255;
        }
      }
      g.putImageData(img, 0, 0);
      tiles.push({ f0, w, cv });
    }
    this.cqt = { head, tiles };
    this.draw(this.time);
  }

  setDoc(map, duration) {
    this.duration = duration || this.duration;
    this.timeline = (map && map.timeline) || this.timeline;
    const out = [];
    for (const [uid, n] of Object.entries((map && map.notes) || {})) {
      if (n.t === undefined) continue;
      out.push({ uid, t: n.t, d: Math.max(0.03, n.d || 0), pitch: n.pitch, color: noteColor(n), n });
    }
    out.sort((a, b) => a.t - b.t);
    this.notes = out;
    this.maxD = out.reduce((m, x) => Math.max(m, x.d), 0.1);
    this.drawBand();
    this.draw(this.time);
  }

  setCandidates(list) {
    this.cands = (list || []).map((c) => ({ ...c, d: Math.max(0.03, c.off - c.on) })).sort((a, b) => a.on - b.on);
    this.maxCandD = this.cands.reduce((m, x) => Math.max(m, x.d), 0.1);
    this.draw(this.time);
  }

  setRouter(windows) { this.router = windows || []; this.drawBand(); }

  // --------------------------------------------------------------------------------------- geometry

  _window(t) { const a = t - this.lead * this.span; return [a, a + this.span]; }

  _range() {
    const h = this.cqt ? this.cqt.head : { min_midi: 28, bins: 216, bins_per_semitone: 3 };
    // bin 1 is centred on min_midi (viewer_media.cqt_image), so the image spans min_midi - 0.5 .. + bins / bps
    return { lo: h.min_midi - 0.5, semis: h.bins / h.bins_per_semitone };
  }

  _y(pitch) {
    const { lo, semis } = this._range();
    return this.roll.height * (1 - (pitch - lo) / semis);  // the centre of the semitone's band
  }

  _first(list, key, t) {  // first index whose key >= t
    let lo = 0, hi = list.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (list[m][key] < t) lo = m + 1; else hi = m; }
    return lo;
  }

  // ----------------------------------------------------------------------------------------- drawing

  draw(t) {
    this.time = t;
    const c = this.roll, g = c.getContext("2d");
    const W = c.width, H = c.height;
    const [a, b] = this._window(t);
    const x = (tt) => ((tt - a) / this.span) * W;
    g.fillStyle = "#121214";
    g.fillRect(0, 0, W, H);
    if (this.cqt) {
      const { head, tiles } = this.cqt;
      const f0 = a * head.fps, f1 = b * head.fps;
      for (const tl of tiles) {
        const s0 = Math.max(f0, tl.f0), s1 = Math.min(f1, tl.f0 + tl.w);
        if (s1 <= s0) continue;
        g.drawImage(tl.cv, s0 - tl.f0, 0, s1 - s0, head.bins, ((s0 - f0) / (f1 - f0)) * W, 0,
                    ((s1 - s0) / (f1 - f0)) * W, H);
      }
    }
    // octave lines (C) and bar lines
    const { semis, lo } = this._range();
    g.strokeStyle = "rgba(255,255,255,.08)";
    g.lineWidth = 1;
    for (let p = Math.ceil(lo / 12) * 12; p < lo + semis; p += 12) {
      const y = Math.round(this._y(p) + H / semis / 2) + 0.5;
      g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke();
    }
    g.strokeStyle = "rgba(255,255,255,.18)";
    for (const bar of this.timeline.bars || []) {
      if (bar[1] < a || bar[1] > b) continue;
      const xx = Math.round(x(bar[1])) + 0.5;
      g.beginPath(); g.moveTo(xx, 0); g.lineTo(xx, H); g.stroke();
    }
    const nh = Math.max(2, H / semis - 1);
    // candidates: dashed outlines
    g.setLineDash([3, 2]);
    g.strokeStyle = "rgba(255,255,255,.55)";
    for (let i = this._first(this.cands, "on", a - this.maxCandD); i < this.cands.length && this.cands[i].on < b; i++) {
      const n = this.cands[i];
      g.strokeRect(x(n.on) + 0.5, this._y(n.pitch) - nh / 2 + 0.5, Math.max(2, (n.d / this.span) * W), nh);
    }
    g.setLineDash([]);
    // the score's notes
    for (let i = this._first(this.notes, "t", a - this.maxD); i < this.notes.length && this.notes[i].t < b; i++) {
      const n = this.notes[i];
      const cf = confOf(n.n, this.conf);
      g.globalAlpha = cf === null ? 0.9 : 0.2 + 0.8 * Math.max(0, Math.min(1, cf));
      g.fillStyle = n.color;
      const xx = x(n.t), w = Math.max(2, (n.d / this.span) * W), y = this._y(n.pitch) - nh / 2;
      g.fillRect(xx, y, w, nh);
      if (n.uid === this.selected) {
        g.globalAlpha = 1;
        g.strokeStyle = "#ff3b30";
        g.lineWidth = 2;
        g.strokeRect(xx - 2, y - 2, w + 4, nh + 4);
        g.lineWidth = 1;
      }
    }
    g.globalAlpha = 1;
    const px = Math.round(x(t)) + 0.5;
    g.strokeStyle = "#fff";
    g.beginPath(); g.moveTo(px, 0); g.lineTo(px, H); g.stroke();
    this._bandCursor(a, b, t);
  }

  _laneRows() {
    const rows = [{ key: "sec", label: "구간" }, { key: "router", label: "라우터" }];
    for (const ln of this.timeline.lines || []) rows.push({ key: `line:${ln.number}`, label: ln.short || `Gtr ${ln.number}`,
                                                           color: LINE_COLORS[ln.number] || UNASSIGNED });
    rows.push({ key: "line:u", label: "미배정", color: UNASSIGNED });
    rows.push({ key: "bass", label: "베이스", color: BASS_COLOR });
    return rows;
  }

  drawBand() {
    const c = this.band, g = c.getContext("2d");
    const W = c.width, H = c.height, D = this.duration || 1;
    const dpr = window.devicePixelRatio || 1;
    const rows = this._laneRows(), rh = H / rows.length, lw = 64 * dpr;
    const x = (t) => lw + (t / D) * (W - lw);
    this._bandGeom = { lw, W, D };
    const cv = document.createElement("canvas");
    cv.width = W; cv.height = H;
    const o = cv.getContext("2d");
    o.fillStyle = "#18181b";
    o.fillRect(0, 0, W, H);
    o.font = `${11 * dpr}px system-ui, "Malgun Gothic", sans-serif`;
    o.textBaseline = "middle";
    const bars = this.timeline.bars || [];
    const barOf = (t) => {
      let lo = 0, hi = bars.length;
      while (lo < hi) { const m = (lo + hi) >> 1; if (bars[m][1] <= t) lo = m + 1; else hi = m; }
      return Math.max(0, lo - 1);
    };
    const counts = new Map();
    for (const n of this.notes) {
      const key = n.n.track === "bass" ? "bass" : `line:${n.n.line && n.n.line !== "u" ? n.n.line : "u"}`;
      const k = `${key}|${barOf(n.t)}`;
      counts.set(k, (counts.get(k) || 0) + 1);
    }
    rows.forEach((r, i) => {
      const y = i * rh;
      o.fillStyle = "#a3a29b";
      o.fillText(r.label, 4 * dpr, y + rh / 2);
      if (r.key === "sec") {
        for (const s of this.timeline.sections || []) {
          o.fillStyle = "rgba(255,255,255,.07)";
          o.fillRect(x(s.start_s) + 1, y + 1, x(s.end_s) - x(s.start_s) - 2, rh - 2);
          o.fillStyle = "#ecebe6";
          o.save(); o.beginPath(); o.rect(x(s.start_s), y, x(s.end_s) - x(s.start_s), rh); o.clip();
          o.fillText(String(s.label || s.id || ""), x(s.start_s) + 3 * dpr, y + rh / 2);
          o.restore();
        }
      } else if (r.key === "router") {
        for (const w of this.router) {
          o.fillStyle = ROUTER_COLORS[w.label] || "#666";
          o.globalAlpha = 0.75;
          o.fillRect(x(w.start_s), y + 2, Math.max(1, x(w.end_s) - x(w.start_s) - 1), rh - 4);
          o.globalAlpha = 1;
        }
      } else {
        bars.forEach((bar, bi) => {
          const n = counts.get(`${r.key}|${bi}`) || 0;
          if (!n) return;
          o.globalAlpha = Math.min(1, 0.25 + n / 12);
          o.fillStyle = r.color;
          o.fillRect(x(bar[1]), y + 2, Math.max(1, x(bar[2]) - x(bar[1]) - 0.5), rh - 4);
        });
        o.globalAlpha = 1;
      }
    });
    this._bandImg = cv;
    this._bandCursor(...this._window(this.time), this.time);
  }

  _bandCursor(a, b, t) {
    if (!this._bandImg || !this._bandGeom) return;
    const g = this.band.getContext("2d"), { lw, W, D } = this._bandGeom, H = this.band.height;
    const x = (tt) => lw + (Math.max(0, Math.min(D, tt)) / D) * (W - lw);
    g.drawImage(this._bandImg, 0, 0);
    g.strokeStyle = "rgba(255,255,255,.7)";
    g.strokeRect(x(a) + 0.5, 0.5, Math.max(2, x(b) - x(a)), H - 1);
    g.fillStyle = "#fff";
    g.fillRect(Math.round(x(t)), 0, 1.5, H);
  }

  // ------------------------------------------------------------------------------------------ clicks

  _rollClick(e) {
    const r = this.roll.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const [a] = this._window(this.time);
    const t = a + fx * this.span;
    const { lo, semis } = this._range();
    const pitch = lo + (1 - fy) * semis;
    let best = null, bd = Infinity;
    for (let i = this._first(this.notes, "t", t - this.maxD); i < this.notes.length && this.notes[i].t <= t; i++) {
      const n = this.notes[i];
      if (t > n.t + n.d + 0.02) continue;
      const d = Math.abs(n.pitch - pitch);
      if (d < 1.2 && d < bd) { bd = d; best = n; }
    }
    if (best) this.onPick(best.uid, best.t);
    else this.onSeek(Math.max(0, t));
  }

  _bandClick(e) {
    if (!this._bandGeom) return;
    const r = this.band.getBoundingClientRect(), { lw, W, D } = this._bandGeom;
    const px = (e.clientX - r.left) * (this.band.width / r.width);
    if (px < lw) return;
    this.onSeek(((px - lw) / (W - lw)) * D);
  }
}
