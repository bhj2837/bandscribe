// bandscribe helper around @coderline/alphatab 1.8.4 (MPL-2.0). Local use only; run through bandscribe.eval.node
// (winjob.run_captured), never through npm.cmd.
//
//   node bandscribe_score.mjs import <in.gp|gpx|gp5|...> <out.json> [--encoding cp949]
//   node bandscribe_score.mjs render <in> <out.wav> [--soundfont <sf2>] [--sample-rate 44100]
//   node bandscribe_score.mjs checktex <in.alphatex> <out.json>
//   node bandscribe_score.mjs version
//
// `import` writes the neutral RefScore JSON (format "bandscribe.refscore/1", see bandscribe/eval/refscore.py) with the
// bandscribe conventions: string 1 = HIGHEST-pitched string, tuning[i] = open pitch of string i+1, pitch stored
// explicitly per note. alphaTab facts (checked on 1.8.4, 2026-09-30):
//   - Note.string counts from the LOWEST string (1 = bottom tab line)  -> bandscribe string = nStrings - s + 1
//   - Staff.tuning is already ordered top tab line first (= highest string first) -> emitted as is
//   - Note.realValue = fret + capo + tuning (capo included)            -> emitted as the reference pitch
//   - MasterBar.repeatCount = total play count of the repeat (GP5 stores it; GP3/4 are converted +1)
//   - beat.playbackStart / playbackDuration are MIDI ticks (960 per quarter) relative to the bar
// `render` builds the MIDI with MidiFileGenerator and synthesises it headless with AlphaSynth.exportAudio
// (the exporter is returned synchronously), writing a 32-bit float stereo WAV.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, basename, extname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ALPHATAB_DIR = join(HERE, "node_modules", "@coderline", "alphatab");
const alphaTab = await import(pathToFileURL(join(ALPHATAB_DIR, "dist", "alphaTab.mjs")).href);
const PKG = JSON.parse(readFileSync(join(ALPHATAB_DIR, "package.json"), "utf8"));

// Python codec names -> WHATWG TextDecoder labels (Node's TextDecoder has no "cp949"; "euc-kr" is UHC/cp949).
const ENCODING_LABELS = { cp949: "euc-kr", "euc-kr": "euc-kr", cp1252: "windows-1252", latin1: "windows-1252", "utf-8": "utf-8", utf8: "utf-8" };

function parseArgs(argv) {
  const pos = [];
  const opts = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) { opts[argv[i].slice(2)] = argv[i + 1]; i++; } else pos.push(argv[i]);
  }
  return { pos, opts };
}

function loadScore(file, encoding) {
  const settings = new alphaTab.Settings();
  if (encoding) settings.importer.encoding = ENCODING_LABELS[encoding.toLowerCase()] ?? encoding;
  const data = new Uint8Array(readFileSync(file));
  return { score: alphaTab.importer.ScoreLoader.loadScoreFromBytes(data, settings), settings };
}

function barTempo(mb) {
  const autos = mb.tempoAutomations ?? [];
  const first = autos.find((a) => a.ratioPosition === 0) ?? null;
  return first ? first.value : null;
}

function cmdImport(input, output, encoding) {
  const { score } = loadScore(input, encoding);
  const tracks = [];
  const notes = [];
  let graceSkipped = 0;
  score.tracks.forEach((tr, ti) => {
    const st = tr.staves[0];
    const tuning = Array.from(st.tuning ?? []);
    const n = tuning.length;
    tracks.push({
      index: ti + 1, name: tr.name, percussion: !!st.isPercussion, n_strings: n, tuning,
      capo: st.capo | 0, program: tr.playbackInfo ? tr.playbackInfo.program : null,
    });
    for (const bar of st.bars) {
      for (const voice of bar.voices) {
        for (const beat of voice.beats) {
          if (beat.isRest || !beat.notes.length) continue;
          if (beat.graceType && beat.graceType !== 0) { graceSkipped += beat.notes.length; continue; }
          const tup = beat.tupletNumerator > 0 && beat.tupletDenominator > 0 && beat.tupletNumerator !== beat.tupletDenominator
            ? [beat.tupletNumerator, beat.tupletDenominator] : null;
          for (const note of beat.notes) {
            if (!note.isStringed) continue;
            notes.push({
              track: ti + 1, bar: bar.index + 1, voice: voice.index + 1,
              start: beat.playbackStart, dur: beat.playbackDuration,
              string: n - note.string + 1, fret: note.fret, pitch: note.realValue,
              tie: !!note.isTieDestination, dead: !!note.isDead, tuplet: tup,
            });
          }
        }
      }
    }
  });
  const masterbars = score.masterBars.map((mb) => ({
    index: mb.index + 1, numerator: mb.timeSignatureNumerator, denominator: mb.timeSignatureDenominator,
    repeat_open: !!mb.isRepeatStart, repeat_count: mb.repeatCount | 0, alternate: mb.alternateEndings | 0,
    tempo_bpm: barTempo(mb), start: mb.start, length: mb.calculateDuration(), anacrusis: !!mb.isAnacrusis,
  }));
  const ext = extname(input).slice(1).toLowerCase();
  const out = {
    format: "bandscribe.refscore/1",
    source: { file: basename(input), format: ext, importer: `alphatab-${PKG.version}`, encoding: encoding ?? null },
    title: score.title ?? "", artist: score.artist ?? "", tempo_bpm: score.tempo,
    tracks, masterbars, notes, warnings: graceSkipped ? [`grace notes skipped: ${graceSkipped}`] : [],
  };
  writeFileSync(output, JSON.stringify(out, null, 1) + "\n", "utf8");
}

function floatWav(samples, sampleRate, channels) {
  const dataBytes = samples.length * 4;
  const buf = Buffer.alloc(12 + 26 + 12 + 8 + dataBytes);
  let o = 0;
  buf.write("RIFF", o); o += 4; buf.writeUInt32LE(buf.length - 8, o); o += 4; buf.write("WAVE", o); o += 4;
  buf.write("fmt ", o); o += 4; buf.writeUInt32LE(18, o); o += 4;
  buf.writeUInt16LE(3, o); o += 2; buf.writeUInt16LE(channels, o); o += 2; buf.writeUInt32LE(sampleRate, o); o += 4;
  buf.writeUInt32LE(sampleRate * channels * 4, o); o += 4; buf.writeUInt16LE(channels * 4, o); o += 2;
  buf.writeUInt16LE(32, o); o += 2; buf.writeUInt16LE(0, o); o += 2;
  buf.write("fact", o); o += 4; buf.writeUInt32LE(4, o); o += 4; buf.writeUInt32LE(samples.length / channels, o); o += 4;
  buf.write("data", o); o += 4; buf.writeUInt32LE(dataBytes, o); o += 4;
  for (let i = 0; i < samples.length; i++, o += 4) buf.writeFloatLE(samples[i], o);
  return buf;
}

class NullOutput {
  // Minimal ISynthOutput: exportAudio renders offline and never plays through the output.
  constructor(sampleRate) { this.sampleRate = sampleRate; const ev = () => ({ on() {}, off() {} }); this.ready = ev(); this.samplesPlayed = ev(); this.sampleRequest = ev(); }
  open() {} play() {} destroy() {} pause() {} addSamples() {} resetSamples() {} activate() {}
  async enumerateOutputDevices() { return []; } async setOutputDevice() {} async getOutputDevice() { return null; }
}

function cmdRender(input, output, soundfont, sampleRate) {
  const sf = soundfont ?? join(ALPHATAB_DIR, "dist", "soundfont", "sonivox.sf2");
  if (!existsSync(sf)) {
    process.stderr.write(`SOUNDFONT_MISSING ${sf}\n`);
    process.exit(3);
  }
  const { score, settings } = loadScore(input, null);
  const midiFile = new alphaTab.midi.MidiFile();
  const handler = new alphaTab.midi.AlphaSynthMidiFileHandler(midiFile, true);
  const generator = new alphaTab.midi.MidiFileGenerator(score, settings, handler);
  generator.generate();
  const synth = new alphaTab.synth.AlphaSynth(new NullOutput(sampleRate), 500);
  const options = new alphaTab.synth.AudioExportOptions();
  options.soundFonts = [new Uint8Array(readFileSync(sf))];
  options.sampleRate = sampleRate;
  options.useSyncPoints = false;
  const exporter = synth.exportAudio(options, midiFile, [], generator.transpositionPitches);
  const chunks = [];
  let total = 0;
  for (let guard = 0; guard < 1e6; guard++) {
    const chunk = exporter.render(500);
    if (!chunk) break;
    chunks.push(chunk.samples);
    total += chunk.samples.length;
  }
  const all = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { all.set(c, off); off += c.length; }
  writeFileSync(output, floatWav(all, sampleRate, 2));
  process.stdout.write(JSON.stringify({ frames: total / 2, sample_rate: sampleRate, soundfont: basename(sf) }) + "\n");
}

// `checktex`: parse an alphaTex file with alphaTab's own importer (the viewer's parser) and report what it built:
// master bars, sync points and per track bars / beats / notes / ties / technique marks, or the parse error.
function cmdCheckTex(input, output) {
  const tex = readFileSync(input, "utf8");
  const res = { format: "bandscribe.alphatex_check/1", alphatab: PKG.version, ok: false, error: null,
                master_bars: 0, sync_points: 0, sync_ms: [], tracks: [] };
  try {
    const score = alphaTab.importer.ScoreLoader.loadAlphaTex(tex, new alphaTab.Settings());
    res.ok = true;
    res.master_bars = score.masterBars.length;
    const sync = score.exportFlatSyncPoints();
    res.sync_points = sync.length;
    res.sync_ms = sync.map((p) => [p.barIndex, p.millisecondOffset]);
    for (const tr of score.tracks) {
      const st = tr.staves[0];
      let beats = 0, notes = 0, ties = 0, slides = 0, dead = 0, ghost = 0, vibrato = 0;
      for (const b of st.bars) for (const v of b.voices) for (const be of v.beats) {
        beats++;
        notes += be.notes.length;
        for (const n of be.notes) {
          if (n.isTieDestination) ties++;
          if (n.slideOutType) slides++;  // M5 bass techniques
          if (n.isDead) dead++;
          if (n.isGhost) ghost++;
          if (n.vibrato) vibrato++;
        }
      }
      res.tracks.push({ name: tr.name, tabs: !!st.showTablature, score: !!st.showStandardNotation, bars: st.bars.length,
                        beats, notes, ties, slides, dead, ghost, vibrato, tuning: Array.from(st.tuning ?? []),
                        capo: st.capo | 0 });
    }
  } catch (e) {
    res.error = String(e && e.message ? e.message : e);
  }
  writeFileSync(output, JSON.stringify(res));
}

const [cmd, ...rest] = process.argv.slice(2);
const { pos, opts } = parseArgs(rest);
try {
  if (cmd === "import" && pos.length === 2) cmdImport(pos[0], pos[1], opts.encoding ?? null);
  else if (cmd === "render" && pos.length === 2) cmdRender(pos[0], pos[1], opts.soundfont ?? null, Number(opts["sample-rate"] ?? 44100));
  else if (cmd === "checktex" && pos.length === 2) cmdCheckTex(pos[0], pos[1]);
  else if (cmd === "version") process.stdout.write(JSON.stringify({ alphatab: PKG.version, node: process.version }) + "\n");
  else { process.stderr.write("usage: bandscribe_score.mjs import <in> <out.json> | render <in> <out.wav> | version\n"); process.exit(2); }
} catch (e) {
  process.stderr.write(`ERROR ${e && e.stack ? e.stack : e}\n`);
  process.exit(1);
}
