// Run with: node docs-site/race.test.mjs
// Races logged by the browser (fixtures/) must replay exactly in Node, and forged copies must not.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { decodeCsr } from "./live-loader.js";
import { replayRace } from "./race-replay.js";
import { newRaceLog, logPokes, handicaps } from "./race-log.js";
import { defaultPokes } from "./live-core.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const LIVE = join(__dirname, "live");

let failures = 0;
function report(name, ok, detail) {
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}: ${detail}`);
}

const model = JSON.parse(await readFile(join(LIVE, "model.json"), "utf-8"));
const bufs = await Promise.all(["csr_indptr.bin", "csr_indices.bin", "csr_weights.bin"].map((f) => readFile(join(LIVE, f))));
const csr = decodeCsr(model, ...bufs.map((b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));

const load = async (name) => JSON.parse(await readFile(join(__dirname, "fixtures", name), "utf-8"));
const failed = (res) => res.checks.filter((c) => !c.ok).map((c) => c.name).join(", ") || "none";

// Chrome race: pokes on and off mid-race, then "Reset brain" mid-race.
const poked = await load("race-poked.json");
const r1 = replayRace(model, csr, poked);
report("browser race with pokes replays exactly", r1.ok, `mismatches: ${failed(r1)}`);
report("mid-race pokes flagged as a handicap", r1.handicaps.includes("left eye blind") && r1.handicaps.includes("brain reset mid-race"), r1.handicaps.join(", "));

const unfinished = await load("race-unfinished.json");
const r2 = replayRace(model, csr, unfinished);
report("race abandoned mid-flight replays exactly", r2.ok, `mismatches: ${failed(r2)}`);

// Browser race: paused, giant fiber cut and restored in the same frame, resumed.
const blip = await load("race-gf-blip.json");
const r3 = replayRace(model, csr, blip);
report("same-frame giant fiber on/off replays exactly", r3.ok, `mismatches: ${failed(r3)}`);

// Cutting and restoring the giant fiber in one frame resets those neurons, so both entries stay.
const log = newRaceLog(1, defaultPokes());
const p = defaultPokes();
p.cutGiantFiber = true;
logPokes(log, 30, p, false);
p.cutGiantFiber = false;
logPokes(log, 30, p, false);
report("same-frame giant fiber on/off is kept in the log", log.pokeLog.length === 3 && handicaps(log).includes("giant fiber cut"), `${log.pokeLog.length} entries`);

// Forgeries: each must fail at least one check.
const forgeries = {
  "claimed more pipes": { ...poked, human_pipes: 20, outcome: "human_won" },
  "claimed the fly scored less": { ...poked, fly_pipes: 3, outcome: "human_won" },
  "extra flaps inserted": { ...poked, flaps: [...poked.flaps, 60, 70] },
  "pokes hidden from the log": { ...poked, poke_log: poked.poke_log.slice(0, 1), fly_handicapped: false },
  "unfinished race passed off as finished": { ...unfinished, finished: true },
};
for (const [name, ev] of Object.entries(forgeries)) {
  const res = replayRace(model, csr, ev);
  report(`forgery caught: ${name}`, !res.ok, `failed checks: ${failed(res)}`);
}

console.log(failures ? `\n${failures} test(s) failed` : "\nall race tests passed");
process.exit(failures ? 1 : 0);
