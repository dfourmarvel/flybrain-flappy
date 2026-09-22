// Web demo for flybrain-flappy (Step 10). No build step, no framework.
// Everything about the level, physics and neuron activity comes from replay/best.json
// at runtime -- nothing about frame count, score or a GameConfig value is assumed here.

import { createEngine } from "./engine.js";

const VIEW_WIDTH = 400; // logical canvas width in game px; not a GameConfig field, a rendering choice.
const BEST_SCORE_KEY = "flybrain-flappy-best-score";

// Display/animation cadence, in ms per rendered game-step. This is a rendering choice, NOT
// config.frame_ms -- that field is simulated brain time (how much brain time one game frame
// represents to the neuron simulator), not a display frame duration. Never conflate the two.
const DISPLAY_STEP_MS = 1000 / 30; // ~30 rendered game-steps/sec at 1x, independent of the data.

const prefersReducedMotion =
  window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const els = {
  canvas: document.getElementById("game-canvas"),
  scoreLine: document.getElementById("score-line"),
  flyScoreText: document.getElementById("fly-score-text"),
  humanScoreLine: document.getElementById("human-score-line"),
  humanScoreText: document.getElementById("human-score-text"),
  gameOverLine: document.getElementById("game-over-line"),
  rasterCanvas: document.getElementById("raster-canvas"),
  rasterPlayhead: document.getElementById("raster-playhead"),
  inputBlock: document.getElementById("input-block"),
  inputRasterCanvas: document.getElementById("input-raster-canvas"),
  inputPlayhead: document.getElementById("input-playhead"),
  gfCanvas: document.getElementById("gf-canvas"),
  gfPlayhead: document.getElementById("gf-playhead"),
  dnBars: document.getElementById("dn-bars"),
  placeholderBanner: document.getElementById("placeholder-banner"),
  connectomeExplainer: document.getElementById("connectome-explainer"),
  dataCreditLine: document.getElementById("data-credit-line"),
  flapDot: document.getElementById("flap-dot"),
  flapText: document.getElementById("flap-text"),
  modeWatch: document.getElementById("mode-watch"),
  modePlay: document.getElementById("mode-play"),
  watchControls: document.getElementById("watch-controls"),
  playControls: document.getElementById("play-controls"),
  btnPlayPause: document.getElementById("btn-play-pause"),
  btnRestart: document.getElementById("btn-restart"),
  btnFlap: document.getElementById("btn-flap"),
  btnPlayRestart: document.getElementById("btn-play-restart"),
  speedBtns: Array.from(document.querySelectorAll(".speed-btn")),
  metaGrid: document.getElementById("meta-grid"),
  bestScoreLine: document.getElementById("best-score-line"),
};

const ctx = els.canvas.getContext("2d");
const rasterCtx = els.rasterCanvas.getContext("2d");
const inputRasterCtx = els.inputRasterCanvas.getContext("2d");
const gfCtx = els.gfCanvas.getContext("2d");

let data = null; // parsed replay JSON
let config = null;
let level = null;
let isRealConnectome = true;

// Index groups by role, valid after main() loads data. Falls back to the pre-`activity.roles`
// convention (first 10 = output-seed) when the field is absent, so an old replay.json still works.
let roleIdx = { output: [], input: [], inter: [] };

let mode = "watch"; // "watch" | "play"

// --- watch-mode state ---
const watch = {
  frameIdx: 0, // 0-based index into fly.* arrays (frame number = frameIdx + 1)
  playing: !prefersReducedMotion,
  speed: 1,
  acc: 0,
  lastTs: null,
};

// --- play-mode state ---
const play = {
  engine: null,
  running: false,
  acc: 0,
  lastTs: null,
  flapQueued: false,
  over: false,
  bestScore: readBestScore(),
};

function readBestScore() {
  try {
    const raw = window.localStorage.getItem(BEST_SCORE_KEY);
    const n = raw === null ? 0 : parseInt(raw, 10);
    return Number.isFinite(n) ? n : 0;
  } catch {
    return 0;
  }
}

function writeBestScore(n) {
  try {
    window.localStorage.setItem(BEST_SCORE_KEY, String(n));
  } catch {
    // localStorage can throw (private mode, quota, disabled) -- ignore, never blocks play.
  }
}

async function main() {
  // The replay is ~0.6 MB: until it arrives, every control is inert, so disable them rather
  // than silently swallowing clicks (a click on "Play" before load did nothing at all).
  const allControls = document.querySelectorAll("button");
  allControls.forEach((b) => { b.disabled = true; });
  const playLabel = els.btnPlayPause.textContent;
  els.btnPlayPause.textContent = "Loading…";

  const res = await fetch("replay/best.json");
  data = await res.json();
  config = data.config;
  level = data.level;

  // meta.is_real_connectome is new; fall back to meta.network for older replay files that
  // predate the field (this file has always distinguished "real" vs "control_NN" networks).
  isRealConnectome =
    typeof data.meta.is_real_connectome === "boolean"
      ? data.meta.is_real_connectome
      : data.meta.network === "real";

  computeRoleGroups();
  applyConnectomeProvenance();

  els.canvas.width = VIEW_WIDTH;
  els.canvas.height = config.height;

  buildMetaGrid();
  buildDnBars();
  drawInputRasterStatic();
  drawRasterStatic();
  drawGfStatic();
  updateBestScoreLine();

  allControls.forEach((b) => { b.disabled = false; });
  els.btnPlayPause.textContent = playLabel;

  setMode("watch");
  wireControls();

  if (watch.playing) {
    requestAnimationFrame(watchLoop);
  } else {
    renderWatchFrame(); // draw frame 0 statically, no animation, per reduced-motion gate
  }
}

// Groups activity.* neuron indices by activity.roles (new field). Never groups by position.
function computeRoleGroups() {
  const roles = data.activity.roles;
  const n = data.activity.ids.length;
  const out = [];
  const inp = [];
  const inter = [];

  if (Array.isArray(roles) && roles.length === n) {
    for (let i = 0; i < n; i++) {
      if (roles[i] === "output_seed") out.push(i);
      else if (roles[i] === "input_seed") inp.push(i);
      else inter.push(i);
    }
  } else {
    // Legacy fallback (no activity.roles field): old schema's documented convention was
    // "the 10 output-seed neurons first ... then the kept interneurons".
    for (let i = 0; i < n; i++) {
      if (i < 10) out.push(i);
      else inter.push(i);
    }
  }

  roleIdx = { output: out, input: inp, inter: inter };
  els.inputBlock.hidden = inp.length === 0;
}

// Shows/hides the placeholder banner and neutralises provenance text that would otherwise
// claim a real fly brain when this replay came from a synthetic placeholder network.
function applyConnectomeProvenance() {
  if (isRealConnectome) {
    els.placeholderBanner.hidden = true;
    els.placeholderBanner.textContent = "";

    els.connectomeExplainer.innerHTML =
      `The <a href="https://github.com/dfourmarvel/flybrain-flappy" rel="noopener">connectome</a> — ` +
      `the wiring diagram of a real fruit fly's brain, down to individual synapses — is never ` +
      `changed. A small sub-circuit around the fly's visual looming-detection and escape pathways ` +
      `runs exactly as measured. The only thing that was <strong>fitted</strong> to this game is a ` +
      `thin interface: how the game's pixels turn into simulated visual input, and how a ` +
      `spike-count readout turns into a flap decision. The brain itself did not learn Flappy Bird — ` +
      `the interface around it did.`;

    els.dataCreditLine.innerHTML =
      `Connectome data: ${escapeHtml(data.meta.connectome)}, Janelia / Google, CC-BY. See ` +
      `<a href="https://github.com/dfourmarvel/flybrain-flappy/blob/main/docs/DATA.md" rel="noopener">docs/DATA.md</a>.`;
  } else {
    els.placeholderBanner.hidden = false;
    els.placeholderBanner.textContent =
      "Placeholder data: this run used a synthetic test network, not a real fly connectome. " +
      "The score and neuron activity on this page are for testing the page, not biology.";

    els.connectomeExplainer.textContent =
      "This particular replay used a synthetic placeholder network to test this page before a " +
      "trained fly run was available — not a real connectome. In the finished project, the " +
      "connectome is a real fruit fly's wiring diagram and is never changed; only a thin interface " +
      "between the game and the brain is fitted. Nothing below is a claim about real fly biology.";

    els.dataCreditLine.textContent =
      "This replay used placeholder data, not connectome data — no data credit applies here.";
  }
}

// ---------------- meta / explainer ----------------

function buildMetaGrid() {
  const rows = [
    ["Connectome", data.meta.connectome],
    ["Neurons (sub-circuit)", data.meta.n_neurons],
    ["Synaptic edges", data.meta.n_edges],
    ["Network", data.meta.network],
    ["Held-out mean score", data.meta.heldout_mean],
    ["This episode's score", data.meta.score],
    ["Recorded frames", data.meta.frames],
  ];
  els.metaGrid.innerHTML = rows
    .map(
      ([label, value]) =>
        `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value))}</dd></div>`
    )
    .join("");
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

function updateBestScoreLine() {
  els.bestScoreLine.textContent = `Your best on this device: ${play.bestScore} (kept in this browser only, never sent anywhere).`;
}

// ---------------- canvas: game rendering ----------------

function pipeXAtFrame(pipe, f) {
  return pipe.x0 - f * config.pipe_speed;
}

function drawGame(frameNumber, birdY, birdColor, ghostY) {
  const w = els.canvas.width;
  const h = els.canvas.height;
  ctx.clearRect(0, 0, w, h);

  // pipes: solid body with a bright gap-facing edge, so the gap reads at a glance
  const body = "#233348";
  const edge = "#6E93B5";
  for (const pipe of level) {
    const x = pipeXAtFrame(pipe, frameNumber);
    if (x + config.pipe_width < 0 || x > w) continue;
    const gapTop = pipe.gap_centre - config.gap_height / 2;
    const gapBottom = pipe.gap_centre + config.gap_height / 2;
    ctx.fillStyle = body;
    ctx.fillRect(x, 0, config.pipe_width, gapTop);
    ctx.fillRect(x, gapBottom, config.pipe_width, h - gapBottom);
    ctx.fillStyle = edge;
    ctx.fillRect(x, gapTop - 6, config.pipe_width, 6);
    ctx.fillRect(x, gapBottom, config.pipe_width, 6);
  }

  // ghost bird (fly, in play mode)
  if (ghostY !== undefined && ghostY !== null) {
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = getCss("--fly");
    ctx.beginPath();
    ctx.arc(config.bird_x, ghostY, config.bird_radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // main bird
  ctx.fillStyle = birdColor;
  ctx.beginPath();
  ctx.arc(config.bird_x, birdY, config.bird_radius, 0, Math.PI * 2);
  ctx.fill();
}

let cssVarCache = null;
function getCss(name) {
  if (!cssVarCache) cssVarCache = getComputedStyle(document.documentElement);
  return cssVarCache.getPropertyValue(name).trim();
}

// ---------------- activity panel ----------------

// Draws a raster (one column per frame, one row per neuron index in `indices`) into `canvasEl`
// via `ctxRef`, coloured by `colorVar`. Shared by the interneuron raster and the input-seed raster.
function drawRasterFromIndices(canvasEl, ctxRef, indices, colorVar) {
  const counts = data.activity.counts;
  const frames = counts.length;
  const n = indices.length;
  canvasEl.width = Math.max(frames, 1);
  canvasEl.height = Math.max(n, 1);

  if (n === 0) return;

  let globalMax = 0;
  for (const row of counts) {
    for (const i of indices) globalMax = Math.max(globalMax, row[i]);
  }

  const img = ctxRef.createImageData(canvasEl.width, canvasEl.height);
  const [fr, fg, fb] = parseColor(getCss(colorVar));
  for (let f = 0; f < frames; f++) {
    for (let ni = 0; ni < n; ni++) {
      const v = counts[f][indices[ni]];
      const alpha = globalMax > 0 ? v / globalMax : 0;
      const idx = (ni * frames + f) * 4;
      img.data[idx] = fr;
      img.data[idx + 1] = fg;
      img.data[idx + 2] = fb;
      img.data[idx + 3] = Math.round(alpha * 255);
    }
  }
  ctxRef.putImageData(img, 0, 0);
}

function drawRasterStatic() {
  drawRasterFromIndices(els.rasterCanvas, rasterCtx, roleIdx.inter, "--fly");
}

function drawInputRasterStatic() {
  drawRasterFromIndices(els.inputRasterCanvas, inputRasterCtx, roleIdx.input, "--ok");
}

function drawGfStatic() {
  const counts = data.gf.counts;
  const frames = counts.length;
  els.gfCanvas.width = Math.max(frames, 1);
  els.gfCanvas.height = 40;

  gfCtx.clearRect(0, 0, els.gfCanvas.width, els.gfCanvas.height);
  const maxV = counts.reduce((m, row) => Math.max(m, row[0], row[1]), 0) || 1;

  gfCtx.strokeStyle = getCss("--gf");
  gfCtx.lineWidth = 1;
  drawTrace(counts.map((r) => r[0]), maxV); // R
  gfCtx.globalAlpha = 0.55;
  drawTrace(counts.map((r) => r[1]), maxV); // L
  gfCtx.globalAlpha = 1;
}

function drawTrace(series, maxV) {
  const h = els.gfCanvas.height;
  gfCtx.beginPath();
  series.forEach((v, i) => {
    const y = h - (v / maxV) * (h - 2) - 1;
    if (i === 0) gfCtx.moveTo(i, y);
    else gfCtx.lineTo(i, y);
  });
  gfCtx.stroke();
}

function parseColor(cssColor) {
  // cssColor is a hex string like "#22D3EE"
  const hex = cssColor.replace("#", "");
  return [
    parseInt(hex.slice(0, 2), 16),
    parseInt(hex.slice(2, 4), 16),
    parseInt(hex.slice(4, 6), 16),
  ];
}

function buildDnBars() {
  const indices = roleIdx.output;
  const ids = indices.map((i) => data.activity.ids[i]);
  const types = indices.map((i) => data.activity.types[i]);
  els.dnBars.innerHTML = "";
  ids.forEach((id, i) => {
    const wrap = document.createElement("div");
    wrap.className = "dn-bar";
    const track = document.createElement("div");
    track.className = "dn-bar-track";
    const fill = document.createElement("div");
    fill.className = "dn-bar-fill";
    fill.style.height = "0%";
    fill.dataset.idx = String(i);
    track.appendChild(fill);
    const label = document.createElement("div");
    label.className = "dn-bar-label";
    label.textContent = types[i] ?? String(id);
    wrap.appendChild(track);
    wrap.appendChild(label);
    els.dnBars.appendChild(wrap);
  });

  const counts = data.activity.counts;
  const perNeuronMax = indices.map((neuronIdx) =>
    counts.reduce((m, row) => Math.max(m, row[neuronIdx]), 0)
  );
  els.dnBars._indices = indices;
  els.dnBars._perNeuronMax = perNeuronMax;
}

function updateDnBars(frameIdx) {
  const counts = data.activity.counts;
  const row = counts[frameIdx];
  const indices = els.dnBars._indices;
  const perNeuronMax = els.dnBars._perNeuronMax;
  const fills = els.dnBars.querySelectorAll(".dn-bar-fill");
  fills.forEach((fill, i) => {
    const max = perNeuronMax[i] || 1;
    const v = row ? row[indices[i]] : 0;
    const pct = row ? Math.min(100, (v / max) * 100) : 0;
    fill.style.height = `${pct}%`;
  });
}

function updatePlayheads(frameIdx, totalFrames) {
  const pct = totalFrames > 1 ? frameIdx / (totalFrames - 1) : 0;
  els.rasterPlayhead.style.left = `${pct * 100}%`;
  els.gfPlayhead.style.left = `${pct * 100}%`;
  els.inputPlayhead.style.left = `${pct * 100}%`;
}

function updateFlapIndicator(isFlap) {
  els.flapDot.classList.toggle("is-active", !!isFlap);
  els.flapText.textContent = isFlap ? "yes" : "no";
}

// ---------------- watch mode ----------------

function setFlyScoreText(score) {
  if (els.flyScoreText.textContent !== String(score)) {
    els.flyScoreText.textContent = String(score);
  }
}

function renderWatchFrame() {
  const i = watch.frameIdx;
  const frameNumber = i + 1;
  const birdY = data.fly.bird_y[i];
  drawGame(frameNumber, birdY, getCss("--fly"));
  setFlyScoreText(data.fly.score[i]);
  updateDnBars(i);
  updatePlayheads(i, data.meta.frames);
  updateFlapIndicator(data.fly.flap[i] === 1);
}

function watchLoop(ts) {
  if (mode !== "watch") return;
  if (!watch.playing) {
    watch.lastTs = null;
    return;
  }
  if (watch.lastTs === null) watch.lastTs = ts;
  const dt = ts - watch.lastTs;
  watch.lastTs = ts;
  watch.acc += dt * watch.speed;

  const stepMs = DISPLAY_STEP_MS;
  const total = data.meta.frames;
  let advanced = false;
  while (watch.acc >= stepMs) {
    watch.acc -= stepMs;
    if (watch.frameIdx < total - 1) {
      watch.frameIdx += 1;
      advanced = true;
    } else {
      watch.playing = false;
      els.btnPlayPause.textContent = "Play";
      break;
    }
  }
  if (advanced || watch.frameIdx === 0) renderWatchFrame();
  requestAnimationFrame(watchLoop);
}

function restartWatch() {
  watch.frameIdx = 0;
  watch.acc = 0;
  watch.lastTs = null;
  renderWatchFrame();
}

// ---------------- play mode ----------------

function startPlay() {
  play.engine = createEngine(config, level);
  play.running = true;
  play.over = false;
  play.acc = 0;
  play.lastTs = null;
  play.flapQueued = false;
  els.gameOverLine.hidden = true;
  els.humanScoreLine.hidden = false;
  els.humanScoreText.textContent = "0";
  requestAnimationFrame(playLoop);
}

function requestFlap() {
  if (mode === "play" && play.running) {
    play.flapQueued = true;
  }
}

function playLoop(ts) {
  if (mode !== "play" || !play.running) return;
  if (play.lastTs === null) play.lastTs = ts;
  const dt = ts - play.lastTs;
  play.lastTs = ts;
  play.acc += dt;

  const stepMs = DISPLAY_STEP_MS;
  while (play.acc >= stepMs) {
    play.acc -= stepMs;
    const flap = play.flapQueued;
    play.flapQueued = false;
    const obs = play.engine.step(flap);
    if (els.humanScoreText.textContent !== String(obs.score)) {
      els.humanScoreText.textContent = String(obs.score);
    }
    if (!obs.alive) {
      endPlay();
      break;
    }
  }

  if (play.running) {
    renderPlayFrame();
    requestAnimationFrame(playLoop);
  }
}

function renderPlayFrame() {
  const f = play.engine.frame;
  const ghostIdx = f - 1;
  const ghostY =
    ghostIdx >= 0 && ghostIdx < data.fly.bird_y.length ? data.fly.bird_y[ghostIdx] : null;
  drawGame(f, play.engine.birdY, getCss("--human"), ghostY);
}

function endPlay() {
  play.running = false;
  renderPlayFrame();
  const humanScore = play.engine.score;
  const flyScore = data.meta.score;

  if (humanScore > play.bestScore) {
    play.bestScore = humanScore;
    writeBestScore(humanScore);
  }
  updateBestScoreLine();

  els.gameOverLine.hidden = false;
  const sourceLine = isRealConnectome
    ? `The fly's score came from a frozen real connectome with only its interface fitted -- ` +
      `no training happened during this replay; its held-out average across 20 seeds was ${data.meta.heldout_mean}.`
    : `The fly's score came from a synthetic placeholder network, not a real connectome -- ` +
      `this replay is for testing the page, not a biology result.`;
  els.gameOverLine.textContent = `You: ${humanScore} · Fly: ${flyScore}. ${sourceLine}`;
}

// ---------------- mode + controls ----------------

function setMode(newMode) {
  mode = newMode;
  els.modeWatch.setAttribute("aria-pressed", String(newMode === "watch"));
  els.modePlay.setAttribute("aria-pressed", String(newMode === "play"));
  els.watchControls.hidden = newMode !== "watch";
  els.playControls.hidden = newMode !== "play";
  els.scoreLine.hidden = newMode !== "watch";
  els.humanScoreLine.hidden = newMode !== "play" || !play.engine;
  els.gameOverLine.hidden = true;

  if (newMode === "watch") {
    play.running = false;
    renderWatchFrame();
  } else {
    startPlay();
  }
}

function wireControls() {
  els.modeWatch.addEventListener("click", () => setMode("watch"));
  els.modePlay.addEventListener("click", () => setMode("play"));

  els.btnPlayPause.addEventListener("click", () => {
    watch.playing = !watch.playing;
    els.btnPlayPause.textContent = watch.playing ? "Pause" : "Play";
    if (watch.playing) {
      watch.lastTs = null;
      requestAnimationFrame(watchLoop);
    }
  });

  els.btnRestart.addEventListener("click", () => {
    restartWatch();
  });

  els.speedBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      watch.speed = parseFloat(btn.dataset.speed);
      els.speedBtns.forEach((b) => {
        b.classList.toggle("is-active", b === btn);
        b.setAttribute("aria-pressed", String(b === btn));
      });
    });
  });

  els.btnFlap.addEventListener("click", requestFlap);
  els.btnPlayRestart.addEventListener("click", startPlay);

  els.canvas.addEventListener("pointerdown", () => {
    if (mode === "play") requestFlap();
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && mode === "play") {
      e.preventDefault(); // space must flap, never scroll the page
      if (!e.repeat) requestFlap();
    }
  });
}

main().catch((err) => {
  console.error("flybrain-flappy demo failed to load:", err);
});
