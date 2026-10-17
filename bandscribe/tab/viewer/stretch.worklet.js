// AudioWorklet of the slowed-playback pitch shifter (stretch_core.js): the ratio (= 1 / rate) comes as
// processorOptions.ratio and later as {ratio} messages ({reset} clears the delay line).
import { Shifter } from "./stretch_core.js";

class StretchProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.shift = new Shifter(sampleRate, 2);
    const o = (options && options.processorOptions) || {};
    if (o.ratio) this.shift.setRatio(o.ratio);  // a message may arrive only after the first blocks
    this.port.onmessage = (e) => {
      if (e.data && e.data.ratio) this.shift.setRatio(e.data.ratio);
      if (e.data && e.data.reset) this.shift.reset();
    };
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const inp = inputs[0] || [];
    this.shift.process(inp, out, out[0] ? out[0].length : 128);
    return true;
  }
}

registerProcessor("bandscribe-stretch", StretchProcessor);
