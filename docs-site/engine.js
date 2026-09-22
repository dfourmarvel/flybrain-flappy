// ES module reimplementation of src/flybrain/game.py physics.
// No RNG anywhere: pipe layout comes from `level` (see replay/best.json meta.schema).
//
// Per-frame order matches Game.step() in game.py exactly:
//   (1) flap sets vy  (2) gravity, capped  (3) y += vy  (4) pipes move (implicit, via x0 - f*speed)
//   (5)/(6) spawn/drop are bookkeeping only in Python; here the full `level` is precomputed so every
//           pipe's position at any frame is x0 - f*pipe_speed, which is identical physics.
//   (7) score  (8) collision  (9) frame increment

/**
 * @param {object} config - all 15 GameConfig fields.
 * @param {Array<{x0:number, gap_centre:number}>} level - pipes ordered by x0 ascending.
 */
export function createEngine(config, level) {
  const pipes = level.map((p) => ({
    x0: p.x0,
    gapTop: p.gap_centre - config.gap_height / 2,
    gapBottom: p.gap_centre + config.gap_height / 2,
  }));

  let frame = 0; // number of steps taken so far (matches Python's obs.frame after increment)
  let birdY = config.start_y;
  let birdVy = 0;
  let score = 0;
  let alive = true;
  let scoreIdx = 0; // next unscored pipe index into `pipes`

  const threshold = config.bird_x - config.bird_radius;

  function pipeXAtFrame(pipe, f) {
    return pipe.x0 - f * config.pipe_speed;
  }

  function circleRectOverlap(rectX, rectY, rectW, rectH) {
    const closestX = Math.max(rectX, Math.min(config.bird_x, rectX + rectW));
    const closestY = Math.max(rectY, Math.min(birdY, rectY + rectH));
    const dx = config.bird_x - closestX;
    const dy = birdY - closestY;
    return Math.sqrt(dx * dx + dy * dy) < config.bird_radius;
  }

  function checkCollision(f) {
    if (birdY - config.bird_radius <= 0) return true;
    if (birdY + config.bird_radius >= config.height) return true;

    for (const pipe of pipes) {
      const x = pipeXAtFrame(pipe, f);
      // Cheap reject: only pipes anywhere near the bird can possibly overlap.
      if (x + config.pipe_width < config.bird_x - config.bird_radius - 1) continue;
      if (x > config.bird_x + config.bird_radius + 1) continue;
      if (circleRectOverlap(x, 0, config.pipe_width, pipe.gapTop)) return true;
      if (
        circleRectOverlap(
          x,
          pipe.gapBottom,
          config.pipe_width,
          config.height - pipe.gapBottom
        )
      )
        return true;
    }
    return false;
  }

  /**
   * Advance one frame. Returns the post-step state (mirrors Python's Obs for the fields the page needs).
   */
  function step(flap) {
    if (!alive) {
      return { bird_y: birdY, bird_vy: birdVy, score, alive, frame };
    }

    // (1) flap
    if (flap) birdVy = config.flap_velocity;

    // (2) gravity, capped
    birdVy = Math.min(birdVy + config.gravity, config.max_fall_speed);

    // (3) position
    birdY += birdVy;

    // (4) pipes move: implicit via pipeXAtFrame(pipe, frame + 1) below

    const f = frame + 1;

    // (7) scoring: first unscored pipe's right edge must clear the threshold
    while (
      scoreIdx < pipes.length &&
      pipeXAtFrame(pipes[scoreIdx], f) + config.pipe_width < threshold
    ) {
      score += 1;
      scoreIdx += 1;
    }

    // (8) collision
    if (checkCollision(f)) {
      alive = false;
    }

    // (9) frame increment
    frame = f;

    return { bird_y: birdY, bird_vy: birdVy, score, alive, frame };
  }

  function getPipeScreenPositions(f) {
    return pipes.map((p) => ({
      x: pipeXAtFrame(p, f),
      gapTop: p.gapTop,
      gapBottom: p.gapBottom,
    }));
  }

  return {
    step,
    getPipeScreenPositions,
    get birdY() {
      return birdY;
    },
    get birdVy() {
      return birdVy;
    },
    get score() {
      return score;
    },
    get alive() {
      return alive;
    },
    get frame() {
      return frame;
    },
  };
}
