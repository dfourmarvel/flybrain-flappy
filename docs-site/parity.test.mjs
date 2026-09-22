// Run with: node docs-site/parity.test.mjs
// Feeds the fly's recorded flap sequence into engine.js and checks the result matches the
// recorded replay exactly (within a small float tolerance for bird_y). This is the page's
// correctness contract: if it fails, the JS physics do not match Python's game.py.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createEngine } from "./engine.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

const TOLERANCE = 1e-3;

async function main() {
  const raw = await readFile(join(__dirname, "replay", "best.json"), "utf-8");
  const data = JSON.parse(raw);

  const engine = createEngine(data.config, data.level);

  const frames = data.meta.frames;
  const flaps = data.fly.flap;
  const expectedY = data.fly.bird_y;
  const expectedScore = data.fly.score;

  if (flaps.length !== frames || expectedY.length !== frames) {
    console.error(
      `FAIL: array length mismatch (meta.frames=${frames}, flap=${flaps.length}, bird_y=${expectedY.length})`
    );
    process.exit(1);
  }

  let maxDiff = 0;
  let maxDiffFrame = -1;
  const mismatches = [];

  for (let i = 0; i < frames; i++) {
    const flap = flaps[i] === 1;
    const obs = engine.step(flap);

    const diff = Math.abs(obs.bird_y - expectedY[i]);
    if (diff >= maxDiff) {
      maxDiff = diff;
      maxDiffFrame = i;
    }
    if (diff > TOLERANCE) {
      mismatches.push({ i, got: obs.bird_y, want: expectedY[i], diff });
    }
  }

  const finalScore = engine.score;
  const wantFinalScore = expectedScore[expectedScore.length - 1];

  let ok = true;

  if (mismatches.length > 0) {
    ok = false;
    console.error(`FAIL: ${mismatches.length} frame(s) exceed tolerance ${TOLERANCE}`);
    for (const m of mismatches.slice(0, 10)) {
      console.error(
        `  frame ${m.i}: got bird_y=${m.got}, want ${m.want}, diff=${m.diff}`
      );
    }
  }

  if (finalScore !== wantFinalScore) {
    ok = false;
    console.error(`FAIL: final score mismatch: got ${finalScore}, want ${wantFinalScore}`);
  }

  if (!ok) {
    process.exit(1);
  }

  console.log(
    `PASS: ${frames} frames match within ${TOLERANCE} (max diff ${maxDiff.toExponential(3)} at frame ${maxDiffFrame}); final score ${finalScore} === ${wantFinalScore}`
  );
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
