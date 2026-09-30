// Run with: node docs-site/live.test.mjs
// PLAN Phase 2 step P2 tests for the browser LIF engine (lif.js) and live interface
// (live-interface.js). Exits non-zero if any test fails. Tolerances are fixed by docs/PLAN.md.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { performance } from "node:perf_hooks";
import { LiveBrain } from "./lif.js";
import { inputRates, Readout, makeLevel, observe } from "./live-interface.js";
import { decodeCsr } from "./live-loader.js";
import { createEngine } from "./engine.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const LIVE = join(__dirname, "live");

let failures = 0;
function report(name, ok, detail) {
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}: ${detail}`);
}

function toArrayBuffer(buf) {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

async function loadFromDisk() {
  const model = JSON.parse(await readFile(join(LIVE, "model.json"), "utf-8"));
  const [ip, ix, w] = await Promise.all(
    ["csr_indptr.bin", "csr_indices.bin", "csr_weights.bin"].map((f) => readFile(join(LIVE, f)))
  );
  return { model, csr: decodeCsr(model, toArrayBuffer(ip), toArrayBuffer(ix), toArrayBuffer(w)) };
}

const fmt = (x, d = 2) => Number(x).toFixed(d);

// One closed-loop episode: obs from the START of each frame -> rates -> brain.run -> readout
// -> flap -> engine step (same 25 ms latency as interface.py play_batch).
function playEpisode(brain, model, level, { params = model.interface, brainSeed, inputGain = 1 } = {}) {
  const game = model.game;
  const stepsPerFrame = Math.round(game.frame_ms / model.constants.dt);
  brain.reset(brainSeed);
  const readout = new Readout(params, model.roles, game.frame_ms);
  const engine = createEngine(game, level);
  const rates = new Float32Array(model.n_neurons);
  let frames = 0;
  while (frames < game.max_frames) {
    const obs = observe(level, game, engine.frame, engine.birdY);
    inputRates(obs, params, model.roles, game, { out: rates, inputGain });
    const counts = brain.run(stepsPerFrame, rates);
    const flap = readout.step(counts);
    const st = engine.step(flap);
    frames++;
    if (!st.alive) break;
  }
  return { score: engine.score, frames };
}

function isolatedModel(model) {
  return { ...model, n_neurons: 1, nnz: 0 };
}

async function main() {
  const { model, csr } = await loadFromDisk();
  const N = model.n_neurons;
  const c = model.constants;
  const dt = c.dt;
  const roles = model.roles;
  const isInput = new Uint8Array(N);
  for (const i of roles.input_L) isInput[i] = 1;
  for (const i of roles.input_R) isInput[i] = 1;
  const nonInputCount = N - isInput.reduce((a, b) => a + b, 0);

  // ---- Test 1: isolated neuron, constant current ----
  {
    const emptyCsr = { indptr: new Int32Array([0, 0]), indices: new Uint16Array(0), weights: new Int16Array(0) };
    const secs = 5;
    const steps = Math.round((secs * 1000) / dt);
    const zero = new Float32Array(1);
    const b10 = new LiveBrain(isolatedModel(model), emptyCsr, { seed: 1, iExt: new Float64Array([10]) });
    const rate10 = b10.run(steps, zero)[0] / secs;
    const tSpike = -c.t_mbr * Math.log(1 - (c.v_th - c.v_rst) / 10);
    const analytic = 1000 / (c.t_rfc + tSpike);
    const b5 = new LiveBrain(isolatedModel(model), emptyCsr, { seed: 1, iExt: new Float64Array([5]) });
    const n5 = b5.run(steps, zero)[0];
    const err = Math.abs(rate10 - analytic) / analytic;
    report(
      "T1 isolated neuron",
      err <= 0.05 && n5 === 0,
      `I=10 mV -> ${fmt(rate10, 3)} Hz vs analytic ${fmt(analytic, 3)} Hz (err ${fmt(err * 100)}%, tol 5%); I=5 mV -> ${n5} spikes in ${secs} s`
    );
  }

  // ---- Test 2: zero input on the real network ----
  {
    const brain = new LiveBrain(model, csr, { seed: 2 });
    const zero = new Float32Array(N);
    const counts = brain.run(Math.round(5000 / dt), zero);
    let total = 0;
    for (let i = 0; i < N; i++) total += counts[i];
    report("T2 zero input", total === 0, `${total} spikes across ${N} neurons in 5 simulated s`);
  }

  // ---- Test 3: delay -- a forced spike arrives exactly D steps later ----
  {
    const D = Math.round(c.t_dly / dt);
    // pre-synaptic neuron: the output seed with the most out-edges (plenty of targets to check)
    let pre = roles.output[0];
    for (const i of roles.output) if (csr.indptr[i + 1] - csr.indptr[i] > csr.indptr[pre + 1] - csr.indptr[pre]) pre = i;
    const brain = new LiveBrain(model, csr, { seed: 3 });
    const zero = new Float32Array(N);
    const eG = Math.exp(-dt / c.tau);
    brain.iExt[pre] = 1000; // one step of huge current: v jumps past threshold at step 0
    const first = brain.run(1, zero)[pre];
    brain.iExt[pre] = 0;
    const targets = [];
    for (let p = csr.indptr[pre]; p < csr.indptr[pre + 1]; p++) targets.push([csr.indices[p], csr.weights[p] * c.w_syn]);
    let arrivalStep = -1;
    let earlyLeak = false;
    let valueOk = true;
    for (let step = 1; step <= D + 2; step++) {
      const counts = brain.run(1, zero);
      let anyNonzero = false;
      for (const [j, w] of targets) {
        if (brain.g[j] !== 0 || counts[j] > 0) anyNonzero = true;
        if (step === D && counts[j] === 0 && Math.abs(brain.g[j] - w * eG) > 1e-9) valueOk = false;
      }
      if (anyNonzero && arrivalStep < 0) arrivalStep = step;
      if (anyNonzero && step < D) earlyLeak = true;
    }
    // chunked run() calls must equal one long call (ring buffer keyed on the absolute step)
    const lc4Rates = new Float32Array(N);
    for (const i of roles.lc4) lc4Rates[i] = 150;
    const a = new LiveBrain(model, csr, { seed: 33 });
    const one = Int32Array.from(a.run(1000, lc4Rates));
    const b = new LiveBrain(model, csr, { seed: 33 });
    const chunked = new Int32Array(N);
    for (let k = 0; k < 8; k++) {
      const cc = b.run(125, lc4Rates);
      for (let i = 0; i < N; i++) chunked[i] += cc[i];
    }
    let same = true;
    for (let i = 0; i < N; i++) if (one[i] !== chunked[i]) same = false;
    let nSpk = 0;
    for (let i = 0; i < N; i++) nSpk += one[i];
    report(
      "T3 delay",
      first === 1 && arrivalStep === D && !earlyLeak && valueOk && same,
      `forced spike of neuron ${pre} at step 0 (${targets.length} targets) first reaches targets at step ${arrivalStep} (want D=${D}); ` +
        `g at arrival = w*e_g: ${valueOk}; 8x run(125) == run(1000): ${same} (${nSpk} spikes compared)`
    );
  }

  // ---- Test 4: open-loop reference (LC4 at 150 Hz, 1 s, 40 trials) ----
  {
    const ref = model.reference;
    const trials = 40;
    const steps = Math.round(1000 / dt);
    const rates = new Float32Array(N);
    for (const i of roles.lc4) rates[i] = 150;
    const outSum = new Float64Array(roles.output.length);
    let netSum = 0;
    const brain = new LiveBrain(model, csr, { seed: 0 });
    for (let t = 0; t < trials; t++) {
      brain.reset(40000 + t);
      const counts = brain.run(steps, rates); // 1 s window, so counts == Hz
      roles.output.forEach((i, k) => (outSum[k] += counts[i]));
      let s = 0;
      for (let i = 0; i < N; i++) if (!isInput[i]) s += counts[i];
      netSum += s / nonInputCount;
    }
    let ok = true;
    const rows = [];
    roles.output.forEach((i, k) => {
      const js = outSum[k] / trials;
      const py = ref.output_rates_hz[k];
      const tolAbs = py < 20 ? 3 : 0.07 * py;
      const pass = Math.abs(js - py) <= tolAbs;
      if (!pass) ok = false;
      rows.push(
        `    ${roles.output_labels[k].padEnd(8)} js ${fmt(js).padStart(7)}  py ${fmt(py).padStart(7)}  diff ${fmt(((js - py) / py) * 100).padStart(6)}%  ${pass ? "ok" : "OUT OF TOLERANCE"}`
      );
    });
    const net = netSum / trials;
    const netErr = Math.abs(net - ref.network_rate_hz) / ref.network_rate_hz;
    if (netErr > 0.07) ok = false;
    report(
      "T4 open-loop reference",
      ok,
      `40 trials (seeds 40000-40039); network rate js ${fmt(net, 4)} Hz vs py ${ref.network_rate_hz} Hz (diff ${fmt(((net - ref.network_rate_hz) / ref.network_rate_hz) * 100)}%, tol 7%)\n` +
        rows.join("\n")
    );
  }

  // ---- Test 5: closed loop on 10 fresh levels, plus two brain-free controls ----
  {
    const brain = new LiveBrain(model, csr, { seed: 0 });
    const scores = [];
    const lines = [];
    const zeroW = { ...model.interface, w: model.interface.w.map(() => 0) };
    const ctlW = [];
    const ctlDrive = [];
    for (let s = 1; s <= 10; s++) {
      const level = makeLevel(s, model.game);
      const r = playEpisode(brain, model, level, { brainSeed: 1000 + s });
      scores.push(r.score);
      lines.push(`level ${s}: ${r.score} (${r.frames} frames)`);
      ctlW.push(playEpisode(brain, model, level, { params: zeroW, brainSeed: 1000 + s }).score);
      ctlDrive.push(playEpisode(brain, model, level, { brainSeed: 1000 + s, inputGain: 0 }).score);
    }
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const wOk = ctlW.every((x) => x === 0);
    const dOk = ctlDrive.every((x) => x === 0);
    report(
      "T5 closed loop",
      mean >= 15 && wOk && dOk,
      `mean score ${fmt(mean)} (need >= 15; Python source run held-out mean ${model.interface.source_heldout_mean})\n    ` +
        lines.join("; ") +
        `\n    readout weights zeroed: scores [${ctlW.join(", ")}]; visual drive off: scores [${ctlDrive.join(", ")}]`
    );
  }

  // ---- Test 6: pokes ----
  {
    const steps = Math.round(1000 / dt);
    const rates = new Float32Array(N);
    for (const i of roles.lc4) rates[i] = 150;
    const brain = new LiveBrain(model, csr, { seed: 0 });
    const trials = 10;
    let dnIntact = 0, dnCut = 0, netIntact = 0, netNoInh = 0;
    const silenced = new Uint8Array(N);
    for (const i of roles.dnp01) silenced[i] = 1;
    const netRate = (counts) => {
      let s = 0;
      for (let i = 0; i < N; i++) if (!isInput[i]) s += counts[i];
      return s / nonInputCount;
    };
    for (let t = 0; t < trials; t++) {
      brain.reset(60000 + t);
      brain.setLesions({ silenced: null, inhibitionOff: false });
      let counts = brain.run(steps, rates);
      for (const i of roles.dnp01) dnIntact += counts[i];
      netIntact += netRate(counts);

      brain.reset(60000 + t);
      brain.setLesions({ silenced, inhibitionOff: false });
      counts = brain.run(steps, rates);
      for (const i of roles.dnp01) dnCut += counts[i];

      brain.reset(60000 + t);
      brain.setLesions({ silenced: null, inhibitionOff: true });
      counts = brain.run(steps, rates);
      netNoInh += netRate(counts);
    }
    brain.setLesions({ silenced: null, inhibitionOff: false });
    // closed-loop check too: DNp01 silenced while the fly plays level 1
    const level = makeLevel(1, model.game);
    brain.reset(7);
    brain.setLesions({ silenced, inhibitionOff: false });
    const game = model.game;
    const r = new Float32Array(N);
    let dnPlay = 0;
    for (let f = 0; f < 400; f++) {
      inputRates(observe(level, game, 0, game.start_y), model.interface, roles, game, { out: r });
      const cc = brain.run(125, r);
      for (const i of roles.dnp01) dnPlay += cc[i];
    }
    netIntact /= trials;
    netNoInh /= trials;
    report(
      "T6 pokes",
      dnCut === 0 && dnPlay === 0 && dnIntact > 0 && netNoInh > netIntact,
      `DNp01 spikes (LC4 150 Hz, ${trials} x 1 s): intact ${dnIntact}, cut ${dnCut}; cut under level-1 start-frame input drive (400 frames): ${dnPlay}; ` +
        `network rate intact ${fmt(netIntact, 3)} Hz -> inhibitionOff ${fmt(netNoInh, 3)} Hz`
    );
  }

  // ---- Test 7: performance, ms per 125-step frame with the fly playing ----
  {
    const timeFrames = (lesions, nFrames) => {
      const brain = new LiveBrain(model, csr, { seed: 77 });
      brain.setLesions(lesions);
      const game = model.game;
      const readout = new Readout(model.interface, roles, game.frame_ms);
      const rates = new Float32Array(N);
      let levelSeed = 1;
      let level = makeLevel(levelSeed, game);
      let engine = createEngine(game, level);
      const times = [];
      let spikes = 0;
      for (let f = 0; f < nFrames; f++) {
        if (!engine.alive || engine.frame >= game.max_frames) {
          level = makeLevel(++levelSeed, game); // bird died: next level, same running brain
          engine = createEngine(game, level);
          readout.reset();
        }
        inputRates(observe(level, game, engine.frame, engine.birdY), model.interface, roles, game, { out: rates });
        const t0 = performance.now();
        const counts = brain.run(125, rates);
        times.push(performance.now() - t0);
        for (let i = 0; i < N; i++) spikes += counts[i];
        engine.step(readout.step(counts));
      }
      return { times, spikes };
    };
    const median = (a) => {
      const s = [...a].sort((x, y) => x - y);
      return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2;
    };
    timeFrames({ silenced: null, inhibitionOff: false }, 400); // JIT warm-up
    timeFrames({ silenced: null, inhibitionOff: true }, 400);
    const intact = timeFrames({ silenced: null, inhibitionOff: false }, 200);
    const noInh = timeFrames({ silenced: null, inhibitionOff: true }, 200);
    const mI = median(intact.times);
    const mN = median(noInh.times);
    const rI = intact.spikes / (N * 200 * 0.025);
    const rN = noInh.spikes / (N * 200 * 0.025);
    report(
      "T7 performance",
      mI <= 10 && mN <= 10,
      `median ms per 125-step frame (200 frames, fly playing): intact ${fmt(mI, 3)} ms (mean rate ${fmt(rI, 2)} Hz), ` +
        `inhibitionOff ${fmt(mN, 3)} ms (mean rate ${fmt(rN, 2)} Hz); budget 10 ms`
    );
  }

  console.log(failures === 0 ? "ALL 7 TESTS PASSED" : `${failures} TEST(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
