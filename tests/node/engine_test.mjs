// M4b engine checks (bandscribe/tab/viewer/engine.js) on a fake AudioContext whose clock the test moves; run by
// tests/test_viewer_m4b.py. Prints {check: true/false, ...} as JSON.
const nodes = [];
class Param {
  constructor(v = 1) { this.value = v; }
  setTargetAtTime(v) { this.value = v; }
  cancelScheduledValues() {}
  setValueAtTime(v) { this.value = v; }
  linearRampToValueAtTime() {}
  exponentialRampToValueAtTime() {}
}
class Node_ {
  constructor(kind) { this.kind = kind; this.gain = new Param(); this.out = []; nodes.push(this); }
  connect(n) { this.out.push(n); }
  disconnect() { this.out = []; this.disconnected = true; }
}
class Buffer_ {
  constructor(ch, n, sr) { this.numberOfChannels = ch; this.length = n; this.sampleRate = sr; this.duration = n / sr;
                          this.data = Array.from({ length: ch }, () => new Float32Array(n)); }
  getChannelData(c) { return this.data[c]; }
  copyToChannel(src, c) { this.data[c].set(src); }
}
class Ctx {
  constructor() { this.currentTime = 0; this.outputLatency = 0.05; this.baseLatency = 0.01; this.state = "running";
                  this.destination = new Node_("dest"); this.audioWorklet = { addModule: async () => {} }; }
  createGain() { return new Node_("gain"); }
  createChannelSplitter() { return new Node_("split"); }
  createChannelMerger() { return new Node_("merge"); }
  createOscillator() { const o = new Node_("osc"); o.frequency = new Param(); o.start = () => {}; o.stop = () => {}; return o; }
  createBufferSource() {
    const n = new Node_("src");
    n.playbackRate = new Param();
    n.start = (when, offset) => { n.when = when; n.offset = offset; };
    n.stop = () => { n.stopped = true; };
    return n;
  }
  createBuffer(ch, n, sr) { return new Buffer_(ch, n, sr); }
  async resume() {}
}
globalThis.AudioContext = Ctx;
globalThis.AudioWorkletNode = class extends Node_ { constructor() { super("worklet"); this.port = { postMessage() {} }; } };
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};

const { Engine } = await import("../../bandscribe/tab/viewer/engine.js");
const out = {};
const close = (a, b, e = 1e-6) => Math.abs(a - b) <= e;

// a recording of 100 s and a synth track that ends at 90 s
const e = new Engine();
e.add({ id: "mix", label: "mix", group: "rec" });
e.add({ id: "stem", label: "stem", group: "rec" });
e.add({ id: "syn", label: "syn", group: "synth", on: true });
e.setBuffer("mix", new Buffer_(2, 4410000, 44100));
e.setBuffer("syn", new Buffer_(1, 22050 * 90, 22050));
out.synth_padded_to_recording = close(e.sources.get("syn").buffer.duration, 100, 1e-3);

// count-in: the sources start in the future; a source decoded meanwhile starts with them, at the same offset
e.beats = Array.from({ length: 200 }, (_, i) => i * 0.5);
e.bars = Array.from({ length: 50 }, (_, i) => [i + 1, i * 2, i * 2 + 2, 4]);
e.countIn = true;
e._pos = 10;
await e.play();
const t0 = e._t0;
out.count_in_starts_later = t0 > e.ctx.currentTime + 1.5;
e.ctx.currentTime = 0.5;  // halfway through the count-in
e.setBuffer("stem", new Buffer_(2, 4410000, 44100));
const st = e.sources.get("stem").node;
out.join_during_count_in_waits = close(st.when, t0) && close(st.offset, 10);
out.position_holds_during_count_in = e.currentTime === 10;

// playing: the heard position lags the playing one by the output latency
e.ctx.currentTime = t0 + 2;
out.heard_position = close(e.currentTime, 12 - 0.06, 1e-9);

// rate change restarts at the playing position, not the heard one
await e.setRate(0.5);
const m = e.sources.get("mix").node;
out.rate_change_keeps_position = close(m.offset, 12) && m.playbackRate.value === 0.5;
await e.setRate(1);

// a loop near the end is clamped before it is checked; an empty one is refused; wrapping never gives NaN
e.setLoop(99.0, 101.0);
out.loop_clamped = e.loop && close(e.loop[1], 100);
e.setLoop(99.98, 100.5);
out.empty_loop_refused = e.loop === null;
e.setLoop(20, 22);
e.ctx.currentTime += 10;
out.loop_wraps = Number.isFinite(e.currentTime) && e.currentTime >= 20 && e.currentTime < 22;
e.setLoop(null);

// a removed source stops and is disconnected
const synNode = e.sources.get("syn").node;
const synGain = e.sources.get("syn").gain;
e.remove("syn");
out.removed_source_stops = synNode.stopped === true && synGain.disconnected === true && !e.sources.has("syn");

// a pause while play() is still setting up wins
e.pause();
const p = e.play();
e.pause();
await p;
out.pause_during_setup_wins = e.paused === true;

console.log(JSON.stringify(out));
