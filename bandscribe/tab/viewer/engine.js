// The viewer's playback engine (M4b, DESIGN 7.2 "A/B 재생"): every recording (original, stems, practice mixes) and
// every tab-synth track (rendered offline by synth.js) is an AudioBuffer started on one AudioContext clock with the
// same `when` and offset, so they stay sample-locked. It stands in for the <audio> element of M3/M4a (currentTime,
// paused, duration, play(), pause(), events "play" / "pause" / "seeked" / "loadedmetadata" / "ended") so the
// alphaTab cursor (EnabledExternalMedia) follows it the same way.
//
//   recording sources -> gain -> [rec group] -> view matrix (stereo / L / R / M / S) -> bus
//   synth tracks      -> gain -> [synth group] ----------------------------------- -> bus -> [pitch shifter] -> out
//
// Slowed playback: the buffers play at `rate` and the shifter (stretch.worklet.js) raises the pitch back by 1 / rate.
// A/B: in A/B mode the synth group plays and holding the A/B key switches to the recordings; outside it the synth
// is heard only with "synthWithRec". Loop: the buffer
// sources' own loop points (sample-accurate, no gap). Count-in: clicks at the local beat before the next beat.

const RAMP_S = 0.006;
const START_AHEAD_S = 0.05;
export const VIEWS = { stereo: [1, 0, 0, 1], left: [1, 0, 1, 0], right: [0, 1, 0, 1], mid: [0.5, 0.5, 0.5, 0.5],
                       side: [0.5, -0.5, 0.5, -0.5] };

export class Engine extends EventTarget {
  constructor() {
    super();
    const ctx = new AudioContext({ latencyHint: "interactive" });
    this.ctx = ctx;
    this.out = ctx.createGain();
    this.out.connect(ctx.destination);
    this.bus = ctx.createGain();
    this.bus.connect(this.out);
    this.groups = { rec: ctx.createGain(), synth: ctx.createGain() };
    // the view matrix of the recordings: out L = a L + b R, out R = c L + d R
    this.split = ctx.createChannelSplitter(2);
    this.merge = ctx.createChannelMerger(2);
    this.mat = [0, 1, 2, 3].map(() => ctx.createGain());
    this.groups.rec.connect(this.split);
    this.split.connect(this.mat[0], 0); this.split.connect(this.mat[1], 1);
    this.split.connect(this.mat[2], 0); this.split.connect(this.mat[3], 1);
    this.mat[0].connect(this.merge, 0, 0); this.mat[1].connect(this.merge, 0, 0);
    this.mat[2].connect(this.merge, 0, 1); this.mat[3].connect(this.merge, 0, 1);
    this.merge.connect(this.bus);
    this.groups.synth.connect(this.bus);
    this.setView("stereo");
    this.sources = new Map();   // id -> {id, label, group, url, buffer, loading, gain, on, solo, vol, node}
    this.rate = 1;
    this.duration = 0;
    this.paused = true;
    this._pos = 0;              // position while paused (s, recording time)
    this._t0 = 0;               // ctx time the sources started at
    this._p0 = 0;               // their start offset
    this.loop = null;           // [a, b] (s) or null
    this.countIn = false;
    this.beats = [];            // recording times of the beats (count-in)
    this.bars = [];             // [bar, start_s, end_s, n_beats]
    this.ab = false;
    this.abHeld = false;
    this.synthWithRec = false;  // outside A/B mode the synth plays only when asked to
    this.shifter = null;        // AudioWorkletNode (loaded on the first slowed play)
    this.shifterLatency = 0.0425;
    this.clicks = [];
    this._timer = 0;
    this._req = 0;              // bumped by every play / pause: a play still awaiting its setup gives way
    this._shifterP = null;
  }

  // ------------------------------------------------------------------------------------------- sources

  add(src) {
    const g = this.ctx.createGain();
    g.connect(this.groups[src.group]);
    const s = { vol: 1, on: false, solo: false, buffer: null, loading: null, node: null, ...src, gain: g };
    this.sources.set(s.id, s);
    this._gains();
    if (s.on && s.url) this.load(s.id).catch((e) => this._emit("error", e));
    return s;
  }

  // a source that is no longer there (a score track an edit removed): silenced and dropped
  remove(id) {
    const s = this.sources.get(id);
    if (!s) return;
    if (s.node) this._stopNode(s);
    s.gain.disconnect();
    this.sources.delete(id);
    this._gains();
    this._emit("sources");
  }

  // a buffer shorter than the recording (a synth track that ends at the last bar) padded with silence, so a loop
  // past its end does not wrap at its own end (AudioBufferSourceNode caps loopEnd at the buffer's duration)
  _pad(buf) {
    if (!this.duration || buf.duration >= this.duration - 1e-3) return buf;
    const nb = this.ctx.createBuffer(buf.numberOfChannels, Math.ceil(this.duration * buf.sampleRate), buf.sampleRate);
    for (let c = 0; c < buf.numberOfChannels; c++) nb.copyToChannel(buf.getChannelData(c), c);
    return nb;
  }

  async load(id) {
    const s = this.sources.get(id);
    if (!s || s.buffer || !s.url) return s && s.buffer;
    if (!s.loading) {
      s.loading = (async () => {
        const r = await fetch(s.url);
        if (!r.ok) throw new Error(`${s.label}: HTTP ${r.status}`);
        const buf = await this.ctx.decodeAudioData(await r.arrayBuffer());
        this.setBuffer(id, buf);
        return buf;
      })();
      s.loading.catch(() => { s.loading = null; });
      this._emit("sources");
    }
    return s.loading;
  }

  // a new buffer for a source (a synth track re-rendered after an edit): it joins the playback at once
  setBuffer(id, buffer) {
    const s = this.sources.get(id);
    if (!s) return;
    if (s.node) { this._stopNode(s); }
    s.loading = null;
    if (s.group === "rec" && buffer.duration > this.duration) {
      this.duration = buffer.duration;
      for (const o of this.sources.values()) {  // synth tracks rendered before the recording was decoded
        if (o === s || !o.buffer || o.group !== "synth") continue;
        const padded = this._pad(o.buffer);
        if (padded !== o.buffer) {
          o.buffer = padded;
          if (o.node) { this._stopNode(o); this._join(o); }
        }
      }
      this._emit("loadedmetadata");
    }
    s.buffer = s.group === "synth" ? this._pad(buffer) : buffer;
    if (!this.paused) this._join(s);
    this._emit("sources");
  }

  set(id, patch) {
    const s = this.sources.get(id);
    if (!s) return;
    Object.assign(s, patch);
    if (s.on && !s.buffer && s.url) this.load(s.id).catch((e) => this._emit("error", e));
    this._gains();
    this._emit("sources");
  }

  _audible(s) {
    const anySolo = [...this.sources.values()].some((x) => x.solo);
    return s.on && (!anySolo || s.solo) ? s.vol : 0;
  }

  _ramp(param, v) {
    const t = this.ctx.currentTime;
    param.cancelScheduledValues(t);
    param.setTargetAtTime(v, t, RAMP_S);
  }

  _gains() {
    for (const s of this.sources.values()) this._ramp(s.gain.gain, this._audible(s));
    const rec = !this.ab || this.abHeld ? 1 : 0;
    const syn = this.ab ? (this.abHeld ? 0 : 1) : (this.synthWithRec ? 1 : 0);
    this._ramp(this.groups.rec.gain, rec);
    this._ramp(this.groups.synth.gain, syn);
  }

  setView(name) {
    const m = VIEWS[name] || VIEWS.stereo;
    this.view = VIEWS[name] ? name : "stereo";
    this.mat.forEach((g, i) => this._ramp(g.gain, m[i]));
  }

  setAB(on) { this.ab = !!on; this._gains(); this._emit("ab"); }
  setSynthWithRec(on) { this.synthWithRec = !!on; this._gains(); }
  holdAB(held) { if (this.abHeld !== !!held) { this.abHeld = !!held; this._gains(); this._emit("ab"); } }

  // ------------------------------------------------------------------------------------------ position

  get lag() {
    const out = (this.ctx.outputLatency || 0) + (this.ctx.baseLatency || 0);
    return (out + (this.rate < 1 ? this.shifterLatency : 0)) * this.rate;
  }

  _wrap(raw) {
    if (this.loop && raw >= this.loop[1]) {
      const [a, b] = this.loop;
      return a + ((raw - b) % (b - a));
    }
    return raw;
  }

  // the position the sources are playing at ctx time `t` (no output latency)
  _at(t) {
    if (this.paused) return this._pos;
    return this._wrap(this._p0 + Math.max(0, t - this._t0) * this.rate);
  }

  // what is heard now (the cursor): the playing position minus the output and shifter latency
  get currentTime() {
    if (this.paused) return this._pos;
    const el = this.ctx.currentTime - this._t0;
    if (el <= 0) return this._p0;  // count-in, or the start just scheduled
    const raw = this._p0 + el * this.rate - this.lag;
    if (raw < this._p0) return this._p0;  // the start position is not heard yet
    return Math.min(this.duration || Infinity, this._wrap(raw));
  }

  set currentTime(t) { this.seek(t); }

  get playbackRate() { return this.rate; }
  set playbackRate(v) { this.setRate(v); }

  // ------------------------------------------------------------------------------------------- transport

  _ensureShifter(rate) {
    if (!this._shifterP) {
      this._shifterP = (async () => {
        await this.ctx.audioWorklet.addModule("/static/stretch.worklet.js");
        this.shifter = new AudioWorkletNode(this.ctx, "bandscribe-stretch", {
          numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2], processorOptions: { ratio: 1 / rate } });
        this.shifter.connect(this.out);
      })();
      this._shifterP.catch(() => { this._shifterP = null; });
    }
    return this._shifterP;
  }

  _route() {
    this.bus.disconnect();
    if (this.rate < 1 && this.shifter) {
      this.shifter.port.postMessage({ ratio: 1 / this.rate, reset: true });
      this.bus.connect(this.shifter);
    } else {
      this.bus.connect(this.out);
    }
  }

  _stopNode(s) {
    try { s.node.stop(); } catch (e) { /* not started */ }
    s.node.disconnect();
    s.node = null;
  }

  _node(s, when, offset) {
    const n = this.ctx.createBufferSource();
    n.buffer = s.buffer;
    n.playbackRate.value = this.rate;
    if (this.loop) { n.loop = true; n.loopStart = this.loop[0]; n.loopEnd = this.loop[1]; }
    n.connect(s.gain);
    if (offset < s.buffer.duration || this.loop) n.start(when, Math.min(offset, Math.max(0, s.buffer.duration - 1e-3)));
    s.node = n;
  }

  // a source that got its buffer while playing starts in step with the others (not before the others: during a
  // count-in they start at _t0)
  _join(s) {
    const when = Math.max(this._t0, this.ctx.currentTime + START_AHEAD_S);
    this._node(s, when, this._at(when));
  }

  _startAll(when, offset) {
    for (const s of this.sources.values()) {
      if (s.node) this._stopNode(s);
      if (s.buffer) this._node(s, when, offset);
    }
    this._t0 = when;
    this._p0 = offset;
  }

  _stopAll() {
    for (const s of this.sources.values()) if (s.node) this._stopNode(s);
    for (const c of this.clicks) { try { c.stop(); } catch (e) { /* done */ } }
    this.clicks = [];
  }

  _beatAfter(t) {
    const b = this.beats;
    let lo = 0, hi = b.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (b[m] < t - 1e-3) lo = m + 1; else hi = m; }
    return lo;
  }

  _barBeats(t) {
    const bar = this.bars.find((x) => x[1] <= t + 1e-3 && t < x[2]);
    return bar ? Math.max(2, Math.min(8, bar[3])) : 4;
  }

  _click(when, accent) {
    if (this.rate < 1 && this.shifter) when += this.shifterLatency;  // the music comes out of the shifter that late
    const o = this.ctx.createOscillator(), g = this.ctx.createGain();
    o.frequency.value = accent ? 1568 : 1046;
    g.gain.setValueAtTime(0, when);
    g.gain.linearRampToValueAtTime(accent ? 0.5 : 0.35, when + 0.002);
    g.gain.exponentialRampToValueAtTime(0.001, when + 0.06);
    o.connect(g); g.connect(this.out);
    o.start(when); o.stop(when + 0.08);
    this.clicks.push(o);
  }

  async play() {
    if (!this.paused || !this.duration) return;
    const req = ++this._req;
    await this.ctx.resume();
    if (this.rate < 1) await this._ensureShifter(this.rate);
    if (req !== this._req || !this.paused) return;  // paused (or played again) while setting up
    this._route();
    let pos = this._pos >= this.duration - 0.05 && !this.loop ? 0 : this._pos;
    if (this.loop && (pos < this.loop[0] || pos >= this.loop[1])) pos = this.loop[0];
    const now = this.ctx.currentTime + START_AHEAD_S;
    let when = now;
    const j = this._beatAfter(pos);
    if (this.countIn && this.beats.length > 1 && j < this.beats.length) {
      const k = Math.min(j + 1, this.beats.length - 1);
      const ibi = Math.max(0.15, (this.beats[k] - this.beats[k - 1]) / this.rate);
      const n = this._barBeats(this.beats[j]);
      const beatAt = (this.beats[j] - pos) / this.rate;  // the next beat sounds this long after the start
      when = now + Math.max(0, n * ibi - beatAt);
      for (let i = n; i >= 1; i--) this._click(when + beatAt - i * ibi, i === n);
    }
    this.paused = false;
    this._startAll(when, pos);
    clearInterval(this._timer);
    this._timer = setInterval(() => this._watch(), 100);
    this._emit("play");
  }

  _watch() {
    if (this.paused || this.loop) return;
    if (this._at(this.ctx.currentTime) >= this.duration) { this.pause(); this._pos = this.duration; this._emit("ended"); }
  }

  pause() {
    this._req++;
    if (this.paused) return;
    this._pos = Math.max(0, this.currentTime);
    this.paused = true;
    this._stopAll();
    clearInterval(this._timer);
    this._emit("pause");
  }

  seek(t) {
    const pos = Math.max(0, Math.min(this.duration || 0, Number(t) || 0));
    if (this.paused) {
      this._pos = pos;
    } else {
      const when = this.ctx.currentTime + START_AHEAD_S;
      this._stopAll();
      if (this.shifter && this.rate < 1) this.shifter.port.postMessage({ reset: true });
      this._startAll(when, this.loop && (pos < this.loop[0] || pos >= this.loop[1]) ? this.loop[0] : pos);
    }
    this._emit("seeked");
  }

  // the sources restart at the position they were playing (not the heard one, which lags), with no await between
  // stopping and starting
  async setRate(v) {
    const r = Math.min(1, Math.max(0.5, Number(v) || 1));
    if (r === this.rate) return;
    if (r < 1) await this._ensureShifter(r);
    if (r === this.rate) return;
    const now = this.ctx.currentTime;
    const playing = !this.paused;
    const pos = playing ? this._at(now) : this._pos;
    if (playing) this._stopAll();
    this.rate = r;
    this._route();
    if (playing) this._startAll(now + START_AHEAD_S, pos); else this._pos = pos;
    this._emit("ratechange");
  }

  setLoop(a, b) {
    let loop = null;
    if (a !== null && a !== undefined && b !== null && b !== undefined) {
      const a2 = Math.max(0, Number(a)), b2 = Math.min(this.duration || Number(b), Number(b));
      if (b2 > a2 + 0.05) loop = [a2, b2];  // checked after clamping: a loop is never empty or inverted
    }
    const playing = !this.paused;
    const pos = playing ? this._at(this.ctx.currentTime) : this._pos;
    this.loop = loop;
    if (playing) {
      this._stopAll();
      this._startAll(this.ctx.currentTime + START_AHEAD_S, loop && (pos < loop[0] || pos >= loop[1]) ? loop[0] : pos);
    }
    this._emit("loop");
  }

  _emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail })); }
}
