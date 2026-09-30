// Live mode and race mode UI (PLAN P3). app.js owns the mode switch and calls enter()/leave();
// everything here runs the live brain, draws the shared game canvas and the live activity panel.
// Pure logic (endless levels, pokes, speed governor) lives in live-core.js.

import { loadLiveModel } from "./live-loader.js";
import { LiveBrain } from "./lif.js";
import { inputRates, Readout } from "./live-interface.js";
import {
  EndlessLevel,
  EndlessGame,
  endlessConfig,
  defaultPokes,
  buildLesions,
  fillInputOpts,
  pokeSummary,
  activePokes,
  formatSurvival,
  SpeedGovernor,
} from "./live-core.js?v=6";
import { sizeGameCanvas, drawWorld, drawPipe, drawBird, drawTag, BIRD } from "./sprites.js?v=7";
import { newRaceLog, logPokes, handicaps, raceCode, brainSeed, RACE_LOG_FORMAT } from "./race-log.js?v=1";
import { track } from "./analytics.js?v=1";

const BEST_PIPES_KEY = "flybrain-flappy-best-pipes";

// Display cadence: rendered game frames per second at 1x. A rendering choice, NOT game.frame_ms
// (that is simulated brain time per game frame).
const STEP_MS = 1000 / 30;
// Longest a single animation frame may spend computing before it yields to the browser.
const MAX_RAF_MS = 16;
// Frames of history shown by the scrolling panels.
const HISTORY = 240;
// Spikes per frame that map to full brightness in the input raster.
const RASTER_FULL = 5;

const $ = (id) => document.getElementById(id);

function parseColor(cssColor) {
  const hex = cssColor.replace("#", "");
  return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
}

function readBest() {
  try {
    const n = parseInt(window.localStorage.getItem(BEST_PIPES_KEY) ?? "0", 10);
    return Number.isFinite(n) ? n : 0;
  } catch {
    return 0;
  }
}

function writeBest(n) {
  try {
    window.localStorage.setItem(BEST_PIPES_KEY, String(n));
  } catch {
    // storage can throw (private mode, quota); never blocks play
  }
}

const pipesText = (n) => `${n} ${n === 1 ? "pipe" : "pipes"}`;

/**
 * @param {{canvas: HTMLCanvasElement, viewWidth: number, getCss: (name:string)=>string,
 *          prefersReducedMotion: boolean}} opts
 */
export function createLiveApp({ canvas, viewWidth, getCss, prefersReducedMotion }) {
  const els = {
    status: $("live-status"),
    liveStats: $("live-stats"),
    livePipes: $("live-pipes"),
    liveTime: $("live-time"),
    gameTimeNote: $("game-time-note"),
    raceTimeNote: $("race-time-note"),
    raceHumanPipes: $("race-human-pipes"),
    raceHumanTime: $("race-human-time"),
    raceFlyPipes: $("race-fly-pipes"),
    raceFlyTime: $("race-fly-time"),
    bestLine: $("best-score-line"),
    gameOver: $("game-over-line"),
    btnPlay: $("btn-live-play"),
    btnNewLevel: $("btn-new-level"),
    btnRestartLevel: $("btn-restart-level"),
    btnFlap: $("btn-flap"),
    levelNumber: $("level-number"),
    speedGroup: $("live-speed-group"),
    speedBtns: Array.from(document.querySelectorAll(".live-speed-btn")),
    slowNote: $("slow-note"),
    raceHint: $("race-hint"),
    pokeLeft: $("poke-blind-left"),
    pokeRight: $("poke-blind-right"),
    pokeGF: $("poke-cut-gf"),
    pokeInh: $("poke-no-inhibition"),
    pokeGain: $("poke-gain"),
    pokeGainValue: $("poke-gain-value"),
    btnResetBrain: $("btn-reset-brain"),
    pokeSummary: $("poke-summary"),
    inputCanvas: $("live-input-canvas"),
    inputHint: $("live-input-hint"),
    dnBars: $("live-dn-bars"),
    gfCanvas: $("live-gf-canvas"),
    flapDot: $("live-flap-dot"),
    flapText: $("live-flap-text"),
    screen: canvas.parentElement,
    screenCta: $("screen-cta"),
    screenGo: $("screen-go"),
    screenSame: $("screen-same"),
    screenPause: $("screen-pause"),
  };
  const controlEls = [
    els.btnPlay, els.btnNewLevel, els.btnRestartLevel, els.btnFlap, ...els.speedBtns,
    els.pokeLeft, els.pokeRight, els.pokeGF, els.pokeInh, els.pokeGain, els.btnResetBrain,
  ];

  const ctx = canvas.getContext("2d");
  const inputCtx = els.inputCanvas.getContext("2d");
  const gfCtx = els.gfCanvas.getContext("2d");
  const scratch = document.createElement("canvas");
  const scratchCtx = scratch.getContext("2d");

  const st = {
    kind: null, // "live" | "race" | null (another mode is showing)
    loaded: false,
    failed: false,
    status: "Loading the live model…",
    model: null,
    brain: null,
    readout: null,
    game: null, // model.game with the frame cap removed
    steps: 125,
    rates: null,
    inputOpts: null,
    silBuf: null,
    pokes: defaultPokes(),
    gov: new SpeedGovernor({ stepMs: STEP_MS }),
    seed: 0,
    levelKind: null, // kind the current level was built for
    level: null,
    fly: null,
    human: null,
    playing: false,
    started: false, // race: the human has pressed Start
    resumeOnEnter: false,
    speed: 1,
    acc: 0,
    lastTs: null,
    raf: 0,
    slowUntil: 0,
    flapQueued: false,
    bestPipes: readBest(),
    outcomeKey: "",
    race: null, // race-log.js log of the current race, for analytics and replay checks
    liveSent: false,
    // activity history
    inputRows: null,
    raster: null, // Uint8Array(HISTORY * nRows), column-major ring
    pushed: 0,
    rendered: 0,
    gfR: new Uint8Array(HISTORY),
    gfL: new Uint8Array(HISTORY),
    outCounts: null,
    outMax: null,
    flap: false,
    dirty: true,
  };

  // ---------------- small helpers ----------------

  function setText(el, text) {
    if (el.textContent !== text) el.textContent = text;
  }

  function setStatus(text) {
    st.status = text;
    if (st.kind) setText(els.status, text);
  }

  function setControlsDisabled(disabled) {
    for (const el of controlEls) el.disabled = disabled;
  }

  const isRace = () => st.kind === "race";

  // ---------------- loading ----------------

  async function load() {
    setStatus("Loading the live model…");
    try {
      const { model, csr } = await loadLiveModel("live/");
      st.model = model;
      st.game = endlessConfig(model.game);
      st.steps = Math.round(model.game.frame_ms / model.constants.dt);
      st.brain = new LiveBrain(model, csr, { seed: 0 });
      st.readout = new Readout(model.interface, model.roles, model.game.frame_ms);
      st.rates = new Float32Array(model.n_neurons);
      st.silBuf = new Uint8Array(model.n_neurons);
      st.inputOpts = { out: st.rates };
      st.inputRows = Int32Array.from([...model.roles.input_L, ...model.roles.input_R].sort((a, b) => a - b));
      st.raster = new Uint8Array(HISTORY * st.inputRows.length);
      st.outCounts = new Int32Array(model.roles.output.length);
      st.outMax = new Float64Array(model.roles.output.length).fill(3);
      buildDnBars();
      els.inputCanvas.width = HISTORY;
      els.inputCanvas.height = Math.max(st.inputRows.length, 1);
      scratch.width = HISTORY;
      scratch.height = els.inputCanvas.height;
      els.gfCanvas.width = HISTORY;
      els.gfCanvas.height = 40;
      const g = model.game;
      const note = `Game time, not wall-clock: ${g.frame_ms} ms of simulated brain time per frame, at any speed.`;
      els.gameTimeNote.textContent = note;
      els.raceTimeNote.textContent = note;
      els.inputHint.textContent =
        `One row per input neuron, one column per frame (last ${HISTORY}), newest on the right. ` +
        `Darker = more spikes.`;
      st.loaded = true;
      setControlsDisabled(false);
      setStatus("");
      syncPokeUI();
      if (st.kind) beginKind();
    } catch (err) {
      st.failed = true;
      setStatus("The live model could not be loaded. The recorded replay still works.");
      console.error("flybrain-flappy live model failed to load:", err);
    }
  }

  // ---------------- enter / leave ----------------

  function enter(kind) {
    st.kind = kind;
    setText(els.status, st.status);
    els.btnFlap.hidden = kind !== "race";
    els.raceHint.hidden = kind !== "race";
    if (kind === "race") {
      canvas.setAttribute("tabindex", "0");
      canvas.setAttribute(
        "aria-label",
        "Flappy Bird game canvas. Your red bird races the fly's yellow bird. Press Space, click or tap to flap. Live text scores are below."
      );
    } else {
      canvas.removeAttribute("tabindex");
      canvas.setAttribute(
        "aria-label",
        "Flappy Bird game canvas. The fly's bird moves through a column of pipes; live text scores are below."
      );
    }
    sizeGameCanvas(canvas, viewWidth, (st.game ?? { height: 512 }).height);
    if (st.loaded) beginKind();
    else {
      hideScreenControls();
      drawBlank();
    }
  }

  function leave() {
    st.resumeOnEnter = st.playing;
    pause();
    st.kind = null;
    hideScreenControls();
    els.status.textContent = "";
    canvas.removeAttribute("tabindex");
  }

  // Start a fresh level for this kind, or resume the one that was showing when the mode was left.
  function beginKind() {
    if (st.level && st.levelKind === st.kind) {
      sizeGameCanvas(canvas, viewWidth, st.game.height);
      st.outcomeKey = "";
      render(true);
      if (st.resumeOnEnter && !allDone()) play();
    } else {
      startLevel(newSeed());
    }
  }

  // ---------------- levels ----------------

  function newSeed() {
    let s;
    do s = 1 + Math.floor(Math.random() * 9999);
    while (s === st.seed);
    return s;
  }

  function startLevel(seed) {
    if (st.race && st.race.started && !st.race.sent) sendRace(true);
    st.race = null; // so this pause() is not counted against the abandoned race
    pause();
    st.race = isRace() ? newRaceLog(seed, st.pokes) : null;
    st.liveSent = false;
    st.seed = seed;
    st.level = new EndlessLevel(seed, st.game);
    st.levelKind = st.kind;
    st.fly = new EndlessGame(st.level, st.game);
    st.human = isRace() ? new EndlessGame(st.level, st.game) : null;
    st.brain.reset(brainSeed(seed));
    st.readout.reset();
    st.started = false;
    st.acc = 0;
    st.lastTs = null;
    st.flapQueued = false;
    st.outcomeKey = "";
    st.flyVy = 0;
    st.humanVy = 0;
    sizeGameCanvas(canvas, viewWidth, st.game.height);
    clearActivity();
    els.gameOver.hidden = true;
    els.gameOver.textContent = "";
    setText(els.levelNumber, `#${seed}`);
    updateBestLine();
    render(true);
    // Live plays at once unless the visitor prefers reduced motion; a race always waits for Start.
    if (st.kind === "live" && !prefersReducedMotion) play();
  }

  // ---------------- pokes ----------------

  function applyLesions() {
    if (!st.brain) return;
    st.brain.setLesions(buildLesions(st.pokes, st.model.roles, st.silBuf));
    fillInputOpts(st.inputOpts, st.pokes);
  }

  function syncPokeUI() {
    const p = st.pokes;
    const toggles = [
      [els.pokeLeft, p.blindLeft],
      [els.pokeRight, p.blindRight],
      [els.pokeGF, p.cutGiantFiber],
      [els.pokeInh, p.inhibitionOff],
    ];
    for (const [btn, on] of toggles) {
      btn.setAttribute("aria-pressed", String(on));
      btn.classList.toggle("is-on", on);
      btn.querySelector(".toggle-state").textContent = on ? "on" : "off";
    }
    const pct = Math.round(p.gain * 100);
    els.pokeGain.value = String(pct);
    setText(els.pokeGainValue, `${pct}%`);
    setText(els.pokeSummary, pokeSummary(p));
    els.pokeSummary.classList.toggle("is-active", activePokes(p).length > 0);
    applyLesions();
  }

  // Pokes change the fly's run, so a race logs each one against the fly frame it takes effect on.
  function notePokes(reset) {
    if (st.race && st.fly && st.fly.alive) logPokes(st.race, st.fly.frame, st.pokes, reset);
  }

  function togglePoke(key) {
    st.pokes[key] = !st.pokes[key];
    syncPokeUI();
    notePokes(false);
    track("poke_changed", { poke: key, on: st.pokes[key], mode: st.kind });
  }

  function resetBrain() {
    st.pokes = defaultPokes();
    syncPokeUI();
    if (st.brain) {
      st.brain.reset(brainSeed(st.seed));
      st.readout.reset();
    }
    notePokes(true);
    track("brain_reset", { mode: st.kind });
  }

  // ---------------- play / pause / loop ----------------

  const flyDone = () => !st.fly || !st.fly.alive;
  const humanDone = () => !st.human || !st.human.alive;
  const allDone = () => (isRace() ? flyDone() && humanDone() : flyDone());

  function play() {
    if (!st.loaded || allDone()) return;
    if (st.race && !st.race.started) {
      st.race.started = true;
      track("race_started", { seed: st.seed });
    }
    st.playing = true;
    st.started = true;
    st.lastTs = null;
    syncControls();
    if (isRace()) canvas.focus({ preventScroll: true }); // so Space flaps instead of re-pressing a button
    scheduleLoop();
  }

  function pause() {
    if (st.playing && st.race && !allDone()) st.race.pauses++;
    st.playing = false;
    if (st.raf) {
      cancelAnimationFrame(st.raf);
      st.raf = 0;
    }
    syncControls();
  }

  function scheduleLoop() {
    if (!st.raf) st.raf = requestAnimationFrame(tick);
  }

  const requestedSpeed = () => (isRace() && !humanDone() ? 1 : st.speed);

  // One game frame: obs -> input rates -> 125 brain steps -> readout -> flap -> engine step.
  function stepFrame() {
    const fly = st.fly;
    if (fly.alive) {
      const m = st.model;
      const obs = fly.observe();
      inputRates(obs, m.interface, m.roles, m.game, st.inputOpts);
      const counts = st.brain.run(st.steps, st.rates);
      const flap = st.readout.step(counts);
      const y0 = fly.birdY;
      fly.step(flap);
      st.flyVy = fly.birdY - y0;
      pushActivity(counts, flap);
    }
    if (isRace() && st.human.alive) {
      const f = st.flapQueued;
      st.flapQueued = false;
      if (f && st.race) st.race.flaps.push(st.human.frame);
      const y0 = st.human.birdY;
      st.human.step(f);
      st.humanVy = st.human.birdY - y0;
    }
  }

  function tick(ts) {
    st.raf = 0;
    if (!st.playing || !st.kind) return;
    const dt = st.lastTs === null ? 0 : Math.min(ts - st.lastTs, 100);
    st.lastTs = ts;
    const req = requestedSpeed();
    const eff = st.gov.effective(req);
    st.acc += dt * eff;

    let cappedOut = false;
    const t0 = performance.now();
    while (st.acc >= STEP_MS && !allDone()) {
      const flyRuns = st.fly.alive; // only frames that run the brain say anything about compute cost
      const s0 = performance.now();
      stepFrame();
      const now = performance.now();
      if (flyRuns) st.gov.record(now - s0);
      st.acc -= STEP_MS;
      if (now - t0 > MAX_RAF_MS && st.acc >= STEP_MS) {
        cappedOut = true; // yield to the browser; the remaining frames run next tick, none are skipped
        break;
      }
    }
    if (cappedOut) st.acc = Math.min(st.acc, STEP_MS);
    if (cappedOut || st.gov.isSlowed(req)) st.slowUntil = ts + 2500;
    setText(els.slowNote, ts < st.slowUntil ? "Slowed to fit this device." : "");

    if (allDone()) st.playing = false;
    render(false);
    if (st.playing) scheduleLoop();
    else syncControls();
  }

  // ---------------- rendering ----------------

  function drawBlank() {
    drawWorld(ctx, viewWidth, (st.game ?? { height: 512 }).height, 0);
  }

  function drawOneBird(game, vy, colors, label, labelBelow, displayFrame) {
    if (!game.alive && game.frame !== displayFrame) return; // a crashed bird only shows at the crash
    const g = st.game;
    drawBird(ctx, g.bird_x, game.birdY, g.bird_radius, colors, vy, game.frame, game.alive);
    if (label) drawTag(ctx, g.bird_x, game.birdY, label, colors.body, labelBelow, g.bird_radius);
  }

  function drawScene() {
    const g = st.game;
    const w = viewWidth;
    const h = g.height;
    const displayFrame = st.human ? Math.max(st.fly.frame, st.human.frame) : st.fly.frame;
    drawWorld(ctx, w, h, displayFrame * g.pipe_speed);
    const [k0, k1] = st.level.visibleRange(displayFrame, w);
    for (let k = k0; k <= k1; k++) {
      const p = st.level.pipes[k];
      const x = p.x0 - displayFrame * g.pipe_speed;
      if (x + g.pipe_width < 0 || x > w) continue;
      drawPipe(ctx, x, g.pipe_width, p.gap_centre - g.gap_height / 2, p.gap_centre + g.gap_height / 2, h);
    }
    const race = isRace();
    drawOneBird(st.fly, st.flyVy, BIRD.fly, race ? "Fly" : "", false, displayFrame);
    if (st.human) drawOneBird(st.human, st.humanVy, BIRD.human, "You", true, displayFrame);
  }

  function updateStats() {
    const ms = st.model.game.frame_ms;
    if (isRace()) {
      setText(els.raceHumanPipes, String(st.human.score));
      setText(els.raceHumanTime, formatSurvival(st.human.frame, ms));
      setText(els.raceFlyPipes, String(st.fly.score));
      setText(els.raceFlyTime, formatSurvival(st.fly.frame, ms));
    } else {
      setText(els.livePipes, String(st.fly.score));
      setText(els.liveTime, formatSurvival(st.fly.frame, ms));
    }
  }

  // Result line; rewritten only when a bird crashes so the live region announces each event once.
  function updateOutcome() {
    const key = `${st.fly.alive ? 1 : 0}${st.human ? (st.human.alive ? 1 : 0) : "-"}`;
    if (key === st.outcomeKey) return;
    st.outcomeKey = key;
    const ms = st.model.game.frame_ms;
    const fly = `${pipesText(st.fly.score)}, ${formatSurvival(st.fly.frame, ms)}`;
    let text = "";
    if (!isRace()) {
      if (!st.fly.alive) {
        text = `The fly crashed on level #${st.seed}: ${fly} of game time.`;
        if (!st.liveSent) {
          st.liveSent = true;
          track("live_run_finished", {
            seed: st.seed,
            fly_pipes: st.fly.score,
            fly_frames: st.fly.frame,
            pokes: activePokes(st.pokes),
          });
        }
      }
    } else {
      const you = st.human ? `${pipesText(st.human.score)}, ${formatSurvival(st.human.frame, ms)}` : "";
      if (!st.human.alive && !st.fly.alive) {
        const r = st.race;
        if (!r.sent) sendRace(true);
        const hc = handicaps(r);
        text = `You: ${you} · Fly: ${fly}. ${
          st.human.score > st.fly.score ? "You passed more pipes." : st.human.score < st.fly.score ? "The fly passed more pipes." : "Same number of pipes."
        } Race code ${r.code}.${hc.length ? ` The fly was handicapped (${hc.join(", ")}), so this was not a fair race.` : ""}`;
      } else if (!st.human.alive) {
        text = `You crashed: ${you}. The fly is still flying; speed is unlocked so you can skip ahead.`;
      } else if (!st.fly.alive) {
        text = `The fly crashed: ${fly}. You are still flying.`;
      }
      if (!st.human.alive && st.human.score > st.bestPipes) {
        st.bestPipes = st.human.score;
        writeBest(st.bestPipes);
      }
      updateBestLine();
    }
    els.gameOver.hidden = text === "";
    els.gameOver.textContent = text;
  }

  function updateBestLine() {
    setText(els.bestLine, `Your best on this device: ${pipesText(st.bestPipes)} (kept in this browser).`);
  }

  // ---------------- race analytics ----------------

  function raceProps() {
    const r = st.race;
    const h = st.human;
    const f = st.fly;
    const hc = handicaps(r);
    return {
      race_code: raceCode({ seed: r.seed, flaps: r.flaps, pokeLog: r.pokeLog, humanFrames: h.frame, flyFrames: f.frame }),
      format: RACE_LOG_FORMAT,
      seed: r.seed,
      finished: !h.alive && !f.alive,
      outcome: h.score > f.score ? "human_won" : h.score < f.score ? "fly_won" : "tie",
      human_pipes: h.score,
      human_frames: h.frame,
      human_alive: h.alive,
      fly_pipes: f.score,
      fly_frames: f.frame,
      fly_alive: f.alive,
      fly_handicapped: hc.length > 0,
      handicaps: hc,
      pauses: r.pauses,
      flaps: r.flaps,
      poke_log: r.pokeLog,
    };
  }

  // The final event goes once per race: when both birds are down, or with finished=false if the
  // visitor starts another level mid-race. Leaving the page sends a partial one that does not end
  // the race, because a back-button return (bfcache) resumes it.
  function sendRace(final) {
    const props = raceProps();
    if (final) {
      st.race.code = props.race_code;
      st.race.sent = true;
    }
    track("race_finished", props, { beacon: !final });
  }

  function render(force) {
    if (!st.level) return;
    drawScene();
    updateStats();
    updateOutcome();
    if (force || st.dirty) drawActivity();
    syncControls();
  }

  // ---------------- activity panel ----------------

  function clearActivity() {
    st.pushed = 0;
    st.rendered = 0;
    st.raster.fill(0);
    st.gfR.fill(0);
    st.gfL.fill(0);
    st.outCounts.fill(0);
    st.outMax.fill(3);
    st.flap = false;
    inputCtx.clearRect(0, 0, els.inputCanvas.width, els.inputCanvas.height);
    st.dirty = true;
  }

  function pushActivity(counts, flap) {
    const rows = st.inputRows;
    const n = rows.length;
    const col = st.pushed % HISTORY;
    const base = col * n;
    for (let r = 0; r < n; r++) {
      const a = (counts[rows[r]] * 255) / RASTER_FULL;
      st.raster[base + r] = a > 255 ? 255 : a;
    }
    const out = st.model.roles.output;
    for (let k = 0; k < out.length; k++) {
      const c = counts[out[k]];
      st.outCounts[k] = c;
      st.outMax[k] = Math.max(st.outMax[k] * 0.999, c, 3);
    }
    const gf = st.model.roles.dnp01;
    st.gfR[col] = Math.min(255, counts[gf[0]]);
    st.gfL[col] = Math.min(255, counts[gf[1]]);
    st.flap = flap;
    st.pushed++;
    st.dirty = true;
  }

  function buildDnBars() {
    const labels = st.model.roles.output_labels;
    els.dnBars.innerHTML = "";
    labels.forEach((label, i) => {
      const wrap = document.createElement("div");
      wrap.className = "dn-bar";
      const track = document.createElement("div");
      track.className = "dn-bar-track";
      const fill = document.createElement("div");
      fill.className = "dn-bar-fill" + (label.startsWith("DNp01") ? " is-gf" : "");
      fill.style.height = "0%";
      track.appendChild(fill);
      const text = document.createElement("div");
      text.className = "dn-bar-label";
      text.textContent = label;
      wrap.appendChild(track);
      wrap.appendChild(text);
      els.dnBars.appendChild(wrap);
    });
    st.dnFills = Array.from(els.dnBars.querySelectorAll(".dn-bar-fill"));
    st.dnPct = new Array(labels.length).fill(-1);
  }

  function drawActivity() {
    st.dirty = false;
    const n = st.inputRows.length;

    // input raster: shift the picture left by the new columns, then paint them at the right edge
    const fresh = Math.min(st.pushed - st.rendered, HISTORY);
    if (fresh > 0) {
      const w = HISTORY;
      scratchCtx.clearRect(0, 0, w, n);
      scratchCtx.drawImage(els.inputCanvas, 0, 0);
      inputCtx.clearRect(0, 0, w, n);
      inputCtx.drawImage(scratch, -fresh, 0);
      const img = inputCtx.createImageData(fresh, n);
      const [fr, fg, fb] = parseColor(getCss("--sig-input"));
      for (let c = 0; c < fresh; c++) {
        const col = (st.pushed - fresh + c) % HISTORY;
        for (let r = 0; r < n; r++) {
          const i = (r * fresh + c) * 4;
          img.data[i] = fr;
          img.data[i + 1] = fg;
          img.data[i + 2] = fb;
          img.data[i + 3] = st.raster[col * n + r];
        }
      }
      inputCtx.putImageData(img, w - fresh, 0);
    }
    st.rendered = st.pushed;

    // escape-neuron bars: spikes in the latest frame, scaled to each neuron's recent maximum
    for (let k = 0; k < st.dnFills.length; k++) {
      const pct = Math.round(Math.min(100, (st.outCounts[k] / st.outMax[k]) * 100));
      if (pct !== st.dnPct[k]) {
        st.dnPct[k] = pct;
        st.dnFills[k].style.height = `${pct}%`;
      }
    }

    // giant fiber trace: R solid, L faded, newest on the right
    const gw = els.gfCanvas.width;
    const gh = els.gfCanvas.height;
    gfCtx.clearRect(0, 0, gw, gh);
    const count = Math.min(st.pushed, HISTORY);
    if (count > 0) {
      let maxV = 3;
      for (let i = 0; i < HISTORY; i++) maxV = Math.max(maxV, st.gfR[i], st.gfL[i]);
      gfCtx.strokeStyle = getCss("--sig-gf");
      gfCtx.lineWidth = 1;
      const trace = (series, alpha) => {
        gfCtx.globalAlpha = alpha;
        gfCtx.beginPath();
        for (let i = 0; i < count; i++) {
          const v = series[(st.pushed - count + i) % HISTORY];
          const x = gw - count + i;
          const y = gh - (v / maxV) * (gh - 2) - 1;
          if (i === 0) gfCtx.moveTo(x, y);
          else gfCtx.lineTo(x, y);
        }
        gfCtx.stroke();
      };
      trace(st.gfR, 1);
      trace(st.gfL, 0.55);
      gfCtx.globalAlpha = 1;
    }

    els.flapDot.classList.toggle("is-active", st.flap);
    setText(els.flapText, st.flap ? "yes" : "no");
  }

  // ---------------- controls ----------------

  function syncControls() {
    if (!st.loaded) return;
    const race = isRace();
    const over = allDone();
    let label;
    if (race) label = !st.started ? "Start race" : st.playing ? "Pause" : "Resume";
    else label = st.playing ? "Pause" : "Play";
    setText(els.btnPlay, label);
    els.btnPlay.disabled = over;
    els.btnFlap.disabled = !(race && st.playing && st.human && st.human.alive);
    els.speedGroup.hidden = race && !humanDone() ? true : false;
    syncScreenControls(label, over);
    for (const b of els.speedBtns) {
      const on = parseFloat(b.dataset.speed) === st.speed;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-pressed", String(on));
    }
  }

  function hideScreenControls() {
    els.screenCta.hidden = true;
    els.screenPause.hidden = true;
  }

  // The on-game buttons: Play/Start/Resume in the middle while stopped, New/Same level after the
  // game is over, and a pause button while it runs.
  function syncScreenControls(label, over) {
    if (!st.kind) return;
    els.screenPause.hidden = !st.playing;
    els.screenCta.hidden = st.playing;
    setText(els.screenGo, over ? "New level" : label);
    els.screenSame.hidden = !over;
  }

  // Keep the whole game in view when play starts from a control below it.
  function revealScreen() {
    els.screen.scrollIntoView({ block: "nearest", behavior: prefersReducedMotion ? "auto" : "smooth" });
  }

  function requestFlap() {
    if (isRace() && st.playing && st.human && st.human.alive) st.flapQueued = true;
  }

  function wire() {
    els.btnPlay.addEventListener("click", () => {
      if (st.playing) pause();
      else {
        play();
        revealScreen();
      }
    });
    // app.js drives the same on-game buttons in replay mode, when st.kind is null
    els.screenGo.addEventListener("click", () => {
      if (!st.kind || !st.loaded) return;
      if (allDone()) startLevel(newSeed());
      else play();
    });
    // The on-game buttons are pointer-only duplicates, hidden from assistive tech: a click must
    // not leave focus on them.
    for (const b of [els.screenGo, els.screenSame, els.screenPause]) {
      b.addEventListener("mousedown", (e) => e.preventDefault());
    }
    els.screenSame.addEventListener("click", () => {
      if (st.kind && st.loaded) startLevel(st.seed);
    });
    els.screenPause.addEventListener("click", () => {
      if (st.kind) pause();
    });
    els.btnNewLevel.addEventListener("click", () => startLevel(newSeed()));
    els.btnRestartLevel.addEventListener("click", () => startLevel(st.seed));
    els.btnFlap.addEventListener("click", requestFlap);
    for (const b of els.speedBtns) {
      b.addEventListener("click", () => {
        st.speed = parseFloat(b.dataset.speed);
        syncControls();
      });
    }

    els.pokeLeft.addEventListener("click", () => togglePoke("blindLeft"));
    els.pokeRight.addEventListener("click", () => togglePoke("blindRight"));
    els.pokeGF.addEventListener("click", () => togglePoke("cutGiantFiber"));
    els.pokeInh.addEventListener("click", () => togglePoke("inhibitionOff"));
    els.pokeGain.addEventListener("input", () => {
      st.pokes.gain = Number(els.pokeGain.value) / 100;
      syncPokeUI();
      notePokes(false);
    });
    els.pokeGain.addEventListener("change", () => {
      track("poke_changed", { poke: "gain", value: st.pokes.gain, mode: st.kind });
    });
    els.btnResetBrain.addEventListener("click", resetBrain);
    window.addEventListener("pagehide", () => {
      if (st.race && st.race.started && !st.race.sent) sendRace(false);
    });

    // In a race that has not started yet, the first tap on the game starts it.
    canvas.addEventListener("pointerdown", () => {
      if (isRace() && st.loaded && !st.started && !allDone()) play();
      requestFlap();
    });

    // Space flaps only in Race, only for the human, and only when focus is on the game or the
    // page itself, so Space still presses whichever button has focus.
    window.addEventListener("keydown", (e) => {
      if (e.code !== "Space" || !isRace()) return;
      const t = e.target;
      const onControl = t instanceof Element && t !== canvas && t.closest("button, input, select, textarea, a, summary");
      if (onControl) return;
      if (!st.playing) {
        if (t !== canvas) return;
        e.preventDefault(); // Space on the game must never scroll the page away from it
        if (st.loaded && !st.started && !allDone()) play();
        return;
      }
      e.preventDefault();
      if (!e.repeat) requestFlap();
    });
  }

  wire();
  setControlsDisabled(true);

  return { load, enter, leave };
}
