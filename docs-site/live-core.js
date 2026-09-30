// Pure logic behind live mode and race mode (PLAN P3): endless levels, the game wrapper that
// extends them lazily, poke -> lesion mapping, and the speed governor. No DOM here, so
// live-app.js drives it in the browser and a Node harness can check it directly.

import { makeUniform } from "./lif.js";
import { createEngine } from "./engine.js";

// Pipes kept generated beyond the next one the bird has to pass.
const PIPES_AHEAD = 8;

/** The model.json `game` config with the frame cap removed: live and race levels are endless.
 * (engine.js never reads max_frames; live-interface.js makeLevel does, so it is not used here.) */
export function endlessConfig(game) {
  return { ...game, max_frames: Infinity };
}

/**
 * A level of any length, reproducible from its seed. Gap centres come from the same seeded
 * stream and the same formula as makeLevel() in live-interface.js, drawn one per pipe in order,
 * so the first n pipes equal makeLevel(seed, game)'s first n pipes; the rest are generated on
 * demand as the bird gets near them.
 */
export class EndlessLevel {
  constructor(seed, game) {
    this.seed = seed;
    this.game = game;
    this.uniform = makeUniform(seed);
    this.lo = game.gap_height / 2 + game.gap_margin;
    this.hi = game.height - game.gap_height / 2 - game.gap_margin;
    /** @type {Array<{x0:number, gap_centre:number}>} */
    this.pipes = [];
    this.ensure(PIPES_AHEAD + 4);
  }

  /** Make sure pipes 0..n-1 exist. */
  ensure(n) {
    const g = this.game;
    while (this.pipes.length < n) {
      const k = this.pipes.length;
      this.pipes.push({
        x0: g.first_pipe_x + k * g.pipe_spacing,
        gap_centre: this.lo + (this.hi - this.lo) * this.uniform(),
      });
    }
  }

  /** Index range [kMin, kMax] of pipes that can be on screen at `frame` (view width in game px). */
  visibleRange(frame, viewWidth) {
    const g = this.game;
    const shift = frame * g.pipe_speed;
    const kMin = Math.max(0, Math.floor((shift - g.first_pipe_x - g.pipe_width) / g.pipe_spacing));
    const kMax = Math.max(kMin, Math.ceil((shift + viewWidth - g.first_pipe_x) / g.pipe_spacing));
    this.ensure(kMax + 1);
    return [kMin, kMax];
  }
}

/**
 * One bird on an EndlessLevel. Physics is engine.js, unchanged: createEngine copies the level
 * with `level.map(...)` once, so this hands it a `map` shim and keeps the copied array, then
 * appends new pipes to that array (same {x0, gapTop, gapBottom} shape) as the level grows.
 * Several games (human and fly in a race) can share one EndlessLevel.
 */
export class EndlessGame {
  constructor(level, game) {
    this.level = level;
    this.game = game;
    this._nextIdx = 0;
    this._enginePipes = null;
    const view = {
      map: (fn) => {
        this._enginePipes = level.pipes.map(fn);
        return this._enginePipes;
      },
    };
    this.engine = createEngine(game, view);
    if (!Array.isArray(this._enginePipes) || this._enginePipes.length !== level.pipes.length) {
      throw new Error("engine.js no longer copies its level via .map(); EndlessGame cannot extend it");
    }
  }

  _sync() {
    const src = this.level.pipes;
    const dst = this._enginePipes;
    const half = this.game.gap_height / 2;
    for (let k = dst.length; k < src.length; k++) {
      dst.push({ x0: src[k].x0, gapTop: src[k].gap_centre - half, gapBottom: src[k].gap_centre + half });
    }
  }

  /** Move the pointer to the first pipe whose right edge has not yet passed the bird (game.py _make_obs). */
  _advance() {
    const g = this.game;
    const edge = g.bird_x - g.bird_radius;
    const shift = this.engine.frame * g.pipe_speed;
    for (;;) {
      this.level.ensure(this._nextIdx + PIPES_AHEAD);
      if (this.level.pipes[this._nextIdx].x0 - shift + g.pipe_width >= edge) return;
      this._nextIdx++;
    }
  }

  /** The interface's view of the world at the start of this frame. */
  observe() {
    this._advance();
    const g = this.game;
    const p = this.level.pipes[this._nextIdx];
    const x = p.x0 - this.engine.frame * g.pipe_speed;
    return { bird_y: this.engine.birdY, next_pipe_dx: x - g.bird_x, next_gap_y: p.gap_centre };
  }

  step(flap) {
    // keep enough pipes ahead of the bird for both scoring and collision checks
    this._advance();
    this._sync();
    return this.engine.step(flap);
  }

  get alive() { return this.engine.alive; }
  get score() { return this.engine.score; }
  get frame() { return this.engine.frame; }
  get birdY() { return this.engine.birdY; }
}

// ---------------- pokes ----------------

export function defaultPokes() {
  return { blindLeft: false, blindRight: false, cutGiantFiber: false, gain: 1, inhibitionOff: false };
}

/** Pokes -> LiveBrain.setLesions argument. `buf` is a reusable Uint8Array(n_neurons). */
export function buildLesions(pokes, roles, buf) {
  buf.fill(0);
  let any = false;
  if (pokes.cutGiantFiber) {
    for (const i of roles.dnp01) buf[i] = 1;
    any = true;
  }
  return { silenced: any ? buf : null, inhibitionOff: !!pokes.inhibitionOff };
}

/** Pokes -> the option fields inputRates() takes (mutates and returns `opts`). */
export function fillInputOpts(opts, pokes) {
  opts.inputGain = pokes.gain;
  opts.blindLeft = pokes.blindLeft;
  opts.blindRight = pokes.blindRight;
  return opts;
}

export function activePokes(pokes) {
  const out = [];
  if (pokes.blindLeft) out.push("left eye blind");
  if (pokes.blindRight) out.push("right eye blind");
  if (pokes.cutGiantFiber) out.push("giant fiber cut");
  if (pokes.gain < 1) out.push(`eyes dimmed to ${Math.round(pokes.gain * 100)}%`);
  if (pokes.inhibitionOff) out.push("inhibition removed");
  return out;
}

export function pokeSummary(pokes) {
  const a = activePokes(pokes);
  return a.length ? `Pokes: ${a.join(", ")}` : "Pokes: none";
}

// ---------------- stats ----------------

/** Game time survived, e.g. "12.4 s". frame_ms is brain time per frame, so this is game time, not wall-clock. */
export function formatSurvival(frames, frameMs) {
  return `${((frames * frameMs) / 1000).toFixed(1)} s`;
}

// ---------------- speed governor ----------------

/**
 * Keeps the simulation from starving the page. Tracks the average compute time of one game
 * step and caps the effective speed so simulation stays within `budgetShare` of wall-clock
 * time. Frames are never skipped: a slower speed just means game time advances more slowly.
 */
export class SpeedGovernor {
  constructor({ stepMs = 1000 / 30, budgetShare = 0.6, alpha = 0.1, warmup = 8, minSpeed = 0.05 } = {}) {
    this.stepMs = stepMs;
    this.budgetShare = budgetShare;
    this.alpha = alpha;
    this.warmup = warmup;
    this.minSpeed = minSpeed;
    this.reset();
  }

  reset() {
    this.avgMs = 0;
    this.samples = 0;
  }

  /** Record the compute time (ms) of one game step. The first few are ignored (JIT warm-up). */
  record(ms) {
    this.samples++;
    if (this.samples <= this.warmup) return;
    this.avgMs = this.avgMs === 0 ? ms : this.avgMs + this.alpha * (ms - this.avgMs);
  }

  /** Highest speed multiplier this device sustains within the compute budget. */
  maxSpeed() {
    if (this.avgMs <= 0) return Infinity;
    return Math.max(this.minSpeed, (this.budgetShare * this.stepMs) / this.avgMs);
  }

  effective(requested) {
    return Math.min(requested, this.maxSpeed());
  }

  isSlowed(requested) {
    return this.effective(requested) < requested * 0.98;
  }
}
