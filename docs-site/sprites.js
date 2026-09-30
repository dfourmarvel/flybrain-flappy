// Drawing only: the game canvas in the style of the original Flappy Bird. Nothing here touches
// physics or scoring. Every shape stays inside the collision geometry it stands for (pipe
// rectangles, bird circle), so a bird is never drawn overlapping a pipe it did not hit.

// Strip of ground drawn below the play area. The engine crashes the bird at y = height, which is
// exactly the top of this strip, so the ground is where the bird visibly lands.
export const GROUND_H = 56;

// Backing-store scale: the canvas keeps logical game coordinates but draws at 2x for sharp edges.
const SCALE = 2;

const C = {
  sky: "#4EC0CA",
  cloud: "#E9FCD9",
  city: "#D3F1DB",
  cityWindow: "#B6E6C6",
  bush: "#5EE270",
  bushDark: "#4AC25A",
  outline: "#543847",
  pipe: "#73BF2E",
  pipeLight: "#9CE659",
  pipeDark: "#558022",
  grass: "#73BF2E",
  grassLight: "#9CE659",
  ground: "#DED895",
  groundDark: "#D0C874",
  eye: "#FFFFFF",
  pupil: "#1B1B1B",
  beak: "#F75B1C",
  beakLight: "#FB9A4B",
};

export const BIRD = {
  fly: { body: "#F8C630", belly: "#FBE08A", wing: "#FFF4C2" },
  human: { body: "#E8533B", belly: "#F59A8B", wing: "#FDE1DA" },
};

/** Size the canvas for a play area of width x height (logical px) plus the ground strip. */
export function sizeGameCanvas(canvas, width, height) {
  canvas.width = width * SCALE;
  canvas.height = (height + GROUND_H) * SCALE;
  canvas.style.aspectRatio = `${width} / ${height + GROUND_H}`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
  return ctx;
}

// Background tiles are drawn once per play-area size and reused every frame.
let tileCache = null;
const TILE_W = 288;

function buildTiles(height) {
  const mk = (h) => {
    const c = document.createElement("canvas");
    c.width = TILE_W * SCALE;
    c.height = h * SCALE;
    const x = c.getContext("2d");
    x.setTransform(SCALE, 0, 0, SCALE, 0, 0);
    return [c, x];
  };

  // clouds + city skyline, one tile that repeats
  const skyH = 150;
  const [far, f] = mk(skyH);
  f.fillStyle = C.cloud;
  for (let i = 0; i < 9; i++) {
    const cx = i * 34 + 10;
    const r = 18 + ((i * 7) % 11);
    f.beginPath();
    f.arc(cx, 48 - r / 3, r, 0, Math.PI * 2);
    f.fill();
  }
  f.fillRect(0, 48, TILE_W, skyH);
  const blocks = [[0, 34, 70], [30, 24, 92], [52, 38, 58], [88, 22, 84], [108, 30, 66],
    [136, 26, 98], [160, 40, 62], [198, 24, 88], [220, 34, 72], [252, 36, 60]];
  for (const [bx, bw, bh] of blocks) {
    f.fillStyle = C.city;
    f.fillRect(bx, skyH - bh, bw, bh);
    f.fillStyle = C.cityWindow;
    for (let wy = skyH - bh + 6; wy < skyH - 8; wy += 9) {
      for (let wx = bx + 4; wx < bx + bw - 5; wx += 8) f.fillRect(wx, wy, 4, 4);
    }
  }

  // bushes along the horizon
  const bushH = 34;
  const [near, n] = mk(bushH);
  for (let i = 0; i < 10; i++) {
    const cx = i * 30 + 8;
    const r = 14 + ((i * 5) % 9);
    n.fillStyle = C.bushDark;
    n.beginPath();
    n.arc(cx, bushH - r + 12, r + 1, 0, Math.PI * 2);
    n.fill();
    n.fillStyle = C.bush;
    n.beginPath();
    n.arc(cx, bushH - r + 13, r - 1, 0, Math.PI * 2);
    n.fill();
  }

  tileCache = { height, far, skyH, near, bushH };
}

function drawTiled(ctx, img, h, y, offset, width) {
  const start = -(((offset % TILE_W) + TILE_W) % TILE_W);
  for (let x = start; x < width; x += TILE_W) ctx.drawImage(img, x, y, TILE_W, h);
}

/**
 * Sky, parallax city and bushes, then the ground strip. `scroll` is the world distance in px
 * (frame x pipe speed); layers move at fractions of it.
 */
export function drawWorld(ctx, width, height, scroll) {
  if (!tileCache || tileCache.height !== height) buildTiles(height);
  const t = tileCache;
  ctx.fillStyle = C.sky;
  ctx.fillRect(0, 0, width, height);
  drawTiled(ctx, t.far, t.skyH, height - t.skyH, scroll * 0.15, width);
  drawTiled(ctx, t.near, t.bushH, height - t.bushH, scroll * 0.4, width);
  drawGround(ctx, width, height, scroll);
}

function drawGround(ctx, width, height, scroll) {
  const y = height;
  ctx.fillStyle = C.outline;
  ctx.fillRect(0, y, width, 2);
  // grass band with scrolling diagonal stripes, moving with the pipes
  const band = 12;
  ctx.fillStyle = C.grass;
  ctx.fillRect(0, y + 2, width, band);
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, y + 2, width, band);
  ctx.clip();
  ctx.fillStyle = C.grassLight;
  const period = 16;
  const off = -((scroll % period) + period) % period;
  for (let x = off - band; x < width + band; x += period) {
    ctx.beginPath();
    ctx.moveTo(x, y + 2 + band);
    ctx.lineTo(x + band, y + 2);
    ctx.lineTo(x + band + 8, y + 2);
    ctx.lineTo(x + 8, y + 2 + band);
    ctx.fill();
  }
  ctx.restore();
  ctx.fillStyle = C.pipeDark;
  ctx.fillRect(0, y + 2 + band, width, 3);
  ctx.fillStyle = C.ground;
  ctx.fillRect(0, y + 5 + band, width, GROUND_H - band - 5);
  ctx.fillStyle = C.groundDark;
  ctx.fillRect(0, y + GROUND_H - 6, width, 6);
}

/**
 * One pipe pair. The collision shape is the full rectangle [x, x + w] above gapTop and below
 * gapBottom; the lip spans that full width and the shaft is inset inside it.
 */
export function drawPipe(ctx, x, w, gapTop, gapBottom, height) {
  const lipH = 24;
  const inset = 3;
  const shaft = (y0, y1) => {
    if (y1 <= y0) return;
    const sx = x + inset;
    const sw = w - inset * 2;
    ctx.fillStyle = C.outline;
    ctx.fillRect(sx, y0, sw, y1 - y0);
    ctx.fillStyle = C.pipe;
    ctx.fillRect(sx + 2, y0, sw - 4, y1 - y0);
    ctx.fillStyle = C.pipeLight;
    ctx.fillRect(sx + 5, y0, 6, y1 - y0);
    ctx.fillStyle = C.pipeDark;
    ctx.fillRect(sx + sw - 10, y0, 6, y1 - y0);
  };
  const lip = (y0) => {
    ctx.fillStyle = C.outline;
    ctx.fillRect(x, y0, w, lipH);
    ctx.fillStyle = C.pipe;
    ctx.fillRect(x + 2, y0 + 2, w - 4, lipH - 4);
    ctx.fillStyle = C.pipeLight;
    ctx.fillRect(x + 5, y0 + 2, 6, lipH - 4);
    ctx.fillStyle = C.pipeDark;
    ctx.fillRect(x + w - 11, y0 + 2, 6, lipH - 4);
  };
  shaft(0, gapTop - lipH);
  lip(gapTop - lipH);
  shaft(gapBottom + lipH, height);
  lip(gapBottom);
}

/**
 * The bird, centred on its collision circle (radius r). `vy` tilts it (nose up when rising,
 * down when falling); `frame` drives the wing. A crashed bird is drawn faded.
 */
export function drawBird(ctx, x, y, r, colors, vy, frame, alive) {
  const tilt = Math.max(-0.45, Math.min(1.1, (vy || 0) * 0.09));
  const wingPhase = alive ? Math.floor(frame / 3) % 3 : 1;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(tilt);
  ctx.globalAlpha = alive ? 1 : 0.55;

  ctx.lineWidth = 2;
  ctx.strokeStyle = C.outline;

  // body
  ctx.fillStyle = colors.body;
  ctx.beginPath();
  ctx.arc(0, 0, r - 1, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = colors.belly;
  ctx.beginPath();
  ctx.arc(-1, 4, r * 0.55, 0, Math.PI);
  ctx.fill();

  // wing
  const wy = [-3, 1, 4][wingPhase];
  ctx.fillStyle = colors.wing;
  ctx.beginPath();
  ctx.ellipse(-r * 0.45, wy, r * 0.48, r * 0.3, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // eye
  ctx.fillStyle = C.eye;
  ctx.beginPath();
  ctx.arc(r * 0.32, -r * 0.3, r * 0.36, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = C.pupil;
  ctx.fillRect(r * 0.4, -r * 0.42, 2.5, 4);

  // beak: two lips, protruding at most 3 px past the circle
  ctx.fillStyle = C.beak;
  ctx.beginPath();
  ctx.roundRect(r * 0.2, r * 0.02, r * 0.8 + 3, r * 0.26, 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = C.beakLight;
  ctx.beginPath();
  ctx.roundRect(r * 0.15, r * 0.3, r * 0.75 + 2, r * 0.24, 2);
  ctx.fill();
  ctx.stroke();

  ctx.restore();
}

/** Small name tag above or below a bird (race mode). */
export function drawTag(ctx, x, y, text, color, below, r) {
  ctx.font = "700 12px 'Atkinson Hyperlegible Next', system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = below ? "top" : "bottom";
  const ty = below ? y + r + 6 : y - r - 6;
  ctx.lineWidth = 3;
  ctx.strokeStyle = C.outline;
  ctx.strokeText(text, x, ty);
  ctx.fillStyle = color;
  ctx.fillText(text, x, ty);
}
