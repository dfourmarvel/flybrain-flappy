// Re-run a logged race (race_finished event properties) and compare it with what was claimed.
// Mirrors live-app.js startLevel/stepFrame exactly: same level, same brain seed, same pokes at
// the same fly frames. Used by tools/verify-race.mjs and race.test.mjs (Node only).

import { LiveBrain } from "./lif.js";
import { inputRates, Readout } from "./live-interface.js";
import { EndlessLevel, EndlessGame, endlessConfig, buildLesions, fillInputOpts, defaultPokes } from "./live-core.js";
import { raceCode, handicaps, brainSeed, RACE_LOG_FORMAT } from "./race-log.js";

function flyRunner(model, csr, seed, level, game) {
  const brain = new LiveBrain(model, csr, { seed: 0 });
  const readout = new Readout(model.interface, model.roles, model.game.frame_ms);
  const rates = new Float32Array(model.n_neurons);
  const silBuf = new Uint8Array(model.n_neurons);
  const opts = { out: rates };
  const steps = Math.round(model.game.frame_ms / model.constants.dt);
  const fly = new EndlessGame(level, game);
  const setPokes = (pokes) => {
    brain.setLesions(buildLesions(pokes, model.roles, silBuf));
    fillInputOpts(opts, pokes);
  };
  const resetBrain = () => {
    brain.reset(brainSeed(seed));
    readout.reset();
  };
  const step = () => {
    const obs = fly.observe();
    inputRates(obs, model.interface, model.roles, model.game, opts);
    fly.step(readout.step(brain.run(steps, rates)));
  };
  return { fly, setPokes, resetBrain, step };
}

/** The unpoked fly on level `seed`, run until it crashes or `maxFrames` pass. */
export function fairFlyRun(model, csr, seed, maxFrames = 200000) {
  const game = endlessConfig(model.game);
  const r = flyRunner(model, csr, seed, new EndlessLevel(seed, game), game);
  r.setPokes(defaultPokes());
  r.resetBrain();
  while (r.fly.alive && r.fly.frame < maxFrames) r.step();
  return { pipes: r.fly.score, frames: r.fly.frame, alive: r.fly.alive };
}

// Longer than any real run; stops a forged frame count from hanging the checker.
export const MAX_REPLAY_FRAMES = 200000;

/**
 * @param {object} ev race_finished event properties
 * @returns {{ok: boolean, checks: {name: string, claimed: any, replayed: any, ok: boolean}[]}}
 */
export function replayRace(model, csr, ev) {
  if (ev.format !== RACE_LOG_FORMAT) throw new Error(`race log format ${ev.format}, this checker reads ${RACE_LOG_FORMAT}`);
  for (const k of ["human_frames", "fly_frames"]) {
    if (!Number.isInteger(ev[k]) || ev[k] < 0 || ev[k] > MAX_REPLAY_FRAMES) throw new Error(`${k} = ${ev[k]} is not a plausible frame count`);
  }
  const game = endlessConfig(model.game);
  const level = new EndlessLevel(ev.seed, game);

  const human = new EndlessGame(level, game);
  const flaps = new Set(ev.flaps);
  while (human.alive && human.frame < ev.human_frames) human.step(flaps.has(human.frame));

  // startLevel: the carried-over pokes are already set, then the brain is reset for the level.
  const log = ev.poke_log;
  const r = flyRunner(model, csr, ev.seed, level, game);
  r.setPokes(log[0].p);
  r.resetBrain();
  let li = 0;
  const applyDue = () => {
    for (; li < log.length && log[li].f === r.fly.frame; li++) {
      r.setPokes(log[li].p);
      if (log[li].r) r.resetBrain();
    }
  };
  applyDue();
  while (r.fly.alive && r.fly.frame < ev.fly_frames) {
    r.step();
    applyDue();
  }
  const fly = r.fly;

  const hc = handicaps({ pokeLog: log });
  const checks = [
    ["human pipes", ev.human_pipes, human.score],
    ["human frames", ev.human_frames, human.frame],
    ["human still flying", ev.human_alive, human.alive],
    ["fly pipes", ev.fly_pipes, fly.score],
    ["fly frames", ev.fly_frames, fly.frame],
    ["fly still flying", ev.fly_alive, fly.alive],
    ["fly handicapped", ev.fly_handicapped, hc.length > 0],
    ["race finished", ev.finished, !human.alive && !fly.alive],
    ["outcome", ev.outcome, human.score > fly.score ? "human_won" : human.score < fly.score ? "fly_won" : "tie"],
    [
      "race code",
      ev.race_code,
      raceCode({ seed: ev.seed, flaps: ev.flaps, pokeLog: log, humanFrames: human.frame, flyFrames: fly.frame }),
    ],
  ].map(([name, claimed, replayed]) => ({ name, claimed, replayed, ok: claimed === replayed }));
  return { ok: checks.every((c) => c.ok), checks, handicaps: hc };
}
