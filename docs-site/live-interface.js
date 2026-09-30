// Browser port of the Step 5 interface (src/flybrain/interface.py) for live mode (PLAN P2).
// The left/right somaSide split that routes "bird below gap" vs "bird above gap" drive is an
// arbitrary interface convention, not fly biology (see interface.py docstring).

import { makeUniform } from "./lif.js";

const RATE_MAX_HZ = 300.0;

function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-x));
}

/**
 * Per-neuron Poisson rates (Hz) for one frame, exactly interface.py compute_input_rates:
 *   proximity = exp(-max(dx, 0) / lam); offset = (bird_y - gap_y) / (gap_height / 2)
 *   rate_L = r0 + G * proximity * sigmoid(kappa * offset)   -> roles.input_L
 *   rate_R = r0 + G * proximity * sigmoid(-kappa * offset)  -> roles.input_R
 *   clipped to [0, 300] Hz. Pokes: inputGain multiplies the final rates; blindLeft/blindRight
 *   zero that side. Every other neuron gets 0.
 * @param {{bird_y:number, next_pipe_dx:number, next_gap_y:number}} obs
 * @param {object} params - model.json `interface`
 * @param {object} roles - model.json `roles`
 * @param {object} game - model.json `game`
 * @param {{inputGain?:number, blindLeft?:boolean, blindRight?:boolean,
 *          out?:Float32Array, nNeurons?:number}} [opts] - pass `out` (reused, no allocation)
 *          or `nNeurons` so the array length is known.
 * @returns {Float32Array}
 */
export function inputRates(obs, params, roles, game, opts = {}) {
  const { inputGain = 1, blindLeft = false, blindRight = false } = opts;
  let out = opts.out;
  if (!out) {
    if (!opts.nNeurons) throw new Error("inputRates needs opts.out or opts.nNeurons");
    out = new Float32Array(opts.nNeurons);
  } else {
    out.fill(0);
  }
  const dx = Math.max(obs.next_pipe_dx, 0);
  const offset = (obs.bird_y - obs.next_gap_y) / (game.gap_height / 2);
  const proximity = Math.exp(-dx / params.lam);
  let rateL = params.r0 + params.G * proximity * sigmoid(params.kappa * offset);
  let rateR = params.r0 + params.G * proximity * sigmoid(-params.kappa * offset);
  rateL = Math.min(Math.max(rateL, 0), RATE_MAX_HZ) * inputGain;
  rateR = Math.min(Math.max(rateR, 0), RATE_MAX_HZ) * inputGain;
  if (blindLeft) rateL = 0;
  if (blindRight) rateR = 0;
  const L = roles.input_L, R = roles.input_R;
  for (let k = 0; k < L.length; k++) out[L[k]] = rateL;
  for (let k = 0; k < R.length; k++) out[R[k]] = rateR;
  return out;
}

/** Spike-trace readout (interface.py update_trace + flap_decision). */
export class Readout {
  /**
   * @param {object} params - model.json `interface` (w, tau_ms, bias)
   * @param {object} roles - model.json `roles` (output: the 10 output seeds in readout order)
   * @param {number} frameMs - simulated ms per game frame (game.frame_ms)
   */
  constructor(params, roles, frameMs) {
    this.w = Float64Array.from(params.w);
    this.bias = params.bias;
    this.output = Int32Array.from(roles.output);
    this.decay = Math.exp(-frameMs / params.tau_ms);
    this.trace = new Float64Array(this.output.length);
    this.value = 0;
  }

  reset() {
    this.trace.fill(0);
    this.value = 0;
  }

  /** trace <- trace * exp(-frame_ms / tau) + output counts; flap = trace . w + bias > 0. */
  step(counts) {
    let s = 0;
    for (let k = 0; k < this.trace.length; k++) {
      this.trace[k] = this.trace[k] * this.decay + counts[this.output[k]];
      s += this.trace[k] * this.w[k];
    }
    this.value = s + this.bias;
    return this.value > 0;
  }
}

/**
 * A fresh level in replay/best.json's `level` format ({x0, gap_centre} by x0 ascending) so
 * engine.js can play it. Gap centres uniform in [gap_height/2 + gap_margin,
 * height - gap_height/2 - gap_margin]; pipes every pipe_spacing from first_pipe_x; enough pipes
 * that a pipe is always ahead of the bird for all max_frames (plus a few for on-screen look-ahead).
 * @param {number} seed - integer
 * @param {object} game - model.json `game`
 */
export function makeLevel(seed, game) {
  const uniform = makeUniform(seed);
  const lo = game.gap_height / 2 + game.gap_margin;
  const hi = game.height - game.gap_height / 2 - game.gap_margin;
  const travel = game.max_frames * game.pipe_speed;
  // last pipe index still ahead of the bird (right edge >= bird_x - bird_radius) at max_frames
  const kLast = Math.ceil(
    (travel + game.bird_x - game.bird_radius - game.pipe_width - game.first_pipe_x) / game.pipe_spacing
  );
  const n = Math.max(kLast, 0) + 4;
  const level = new Array(n);
  for (let k = 0; k < n; k++) {
    level[k] = { x0: game.first_pipe_x + k * game.pipe_spacing, gap_centre: lo + (hi - lo) * uniform() };
  }
  return level;
}

/**
 * The Python Obs fields the interface needs, for a level played by engine.js at `frame`
 * (game.py _make_obs: next pipe = first whose right edge >= bird_x - bird_radius).
 */
export function observe(level, game, frame, birdY) {
  const edge = game.bird_x - game.bird_radius;
  for (let k = 0; k < level.length; k++) {
    const x = level[k].x0 - frame * game.pipe_speed;
    if (x + game.pipe_width >= edge) {
      return { bird_y: birdY, next_pipe_dx: x - game.bird_x, next_gap_y: level[k].gap_centre };
    }
  }
  throw new Error("no pipe ahead of the bird");
}
